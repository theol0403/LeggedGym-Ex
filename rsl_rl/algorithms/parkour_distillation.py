import inspect
import math

import torch
import torch.nn as nn
import torch.optim as optim


class ParkourDistillation:
    """DAgger-style distillation with windowed BPTT through the GRU depth encoder.

    Collects live-graph student outputs across a BPTT window, computes loss and
    calls backward() at window boundaries, then accumulates gradients across the
    full rollout.  A single optimizer step happens after the entire rollout.

    This provides temporal gradient flow through the GRU (BPTT within each
    window) while bounding peak GPU memory to O(window_size) instead of
    O(num_steps_per_env).
    """

    def __init__(
        self,
        actor_critic,
        learning_rate=1e-3,
        action_loss_coef=1.0,
        yaw_loss_coef=1.0,
        scandot_loss_coef=0.0,
        max_grad_norm=1.0,
        bptt_window=24,
        lr_schedule="none",
        lr_schedule_max_iters=5000,
        lr_schedule_min_lr=1e-4,
        freeze_actor=False,
        actor_lr_scale=1.0,
        device="cpu",
    ):
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.action_loss_coef = action_loss_coef
        self.yaw_loss_coef = yaw_loss_coef
        self.scandot_loss_coef = scandot_loss_coef
        self.max_grad_norm = max_grad_norm
        self.bptt_window = bptt_window
        self._lr_schedule = lr_schedule
        self._lr_schedule_max_iters = lr_schedule_max_iters
        self._lr_schedule_min_lr = lr_schedule_min_lr
        self._freeze_actor = freeze_actor
        self._actor_lr_scale = actor_lr_scale
        self._iteration = 0
        if freeze_actor and hasattr(self.actor_critic, "actor"):
            for p in self.actor_critic.actor.parameters():
                p.requires_grad_(False)
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        extra_kwargs = {}
        if str(self.device).startswith("cuda"):
            try:
                if "fused" in inspect.signature(optim.Adam).parameters:
                    extra_kwargs["fused"] = True
            except (TypeError, ValueError):
                pass
        has_actor = hasattr(self.actor_critic, "actor") and self.actor_critic.actor is not None
        if has_actor and self._actor_lr_scale != 1.0 and not self._freeze_actor:
            encoder_params = list(self.actor_critic.recurrent_depth.parameters())
            actor_params = list(self.actor_critic.actor.parameters())
            encoder_lr = self.learning_rate
            actor_lr = self.learning_rate * self._actor_lr_scale
            param_groups = [
                {"params": encoder_params, "lr": encoder_lr, "initial_lr": encoder_lr},
                {"params": actor_params, "lr": actor_lr, "initial_lr": actor_lr},
            ]
            return optim.Adam(param_groups, **extra_kwargs)
        # For scandot student (no actor) or default case: optimize all owned parameters
        trainable = [p for p in self.actor_critic.parameters() if p.requires_grad]
        opt = optim.Adam(trainable, lr=self.learning_rate, **extra_kwargs)
        for pg in opt.param_groups:
            pg["initial_lr"] = self.learning_rate
        return opt

    def begin_rollout(self, num_steps: int):
        """Call at the start of each iteration before stepping the env."""
        self.optimizer.zero_grad(set_to_none=True)
        self._num_steps = num_steps
        self._num_windows = max(1, num_steps // self.bptt_window)
        self._actions_student_buf: list[torch.Tensor] = []
        self._actions_teacher_buf: list[torch.Tensor] = []
        self._yaw_student_buf: list[torch.Tensor] = []
        self._yaw_teacher_buf: list[torch.Tensor] = []
        self._scandot_pred_buf: list[torch.Tensor] = []
        self._scandot_gt_buf: list[torch.Tensor] = []
        self._step_in_window = 0
        self._accum_action_loss = 0.0
        self._accum_yaw_loss = 0.0
        self._accum_scandot_loss = 0.0
        self._window_count = 0

    def store_step(
        self,
        student_actions: torch.Tensor,
        teacher_actions: torch.Tensor,
        predicted_yaw: torch.Tensor,
        oracle_yaw: torch.Tensor,
        predicted_scandots: torch.Tensor = None,
        gt_scandots: torch.Tensor = None,
    ):
        """Append live-graph tensors; flush window when full."""
        self._actions_student_buf.append(student_actions)
        self._actions_teacher_buf.append(teacher_actions)
        self._yaw_student_buf.append(predicted_yaw)
        self._yaw_teacher_buf.append(oracle_yaw)
        if predicted_scandots is not None and gt_scandots is not None:
            self._scandot_pred_buf.append(predicted_scandots)
            self._scandot_gt_buf.append(gt_scandots)
        self._step_in_window += 1

        if self._step_in_window >= self.bptt_window:
            self._flush_window()

    def _flush_window(self):
        """Backward through the current window, accumulate gradients."""
        if not self._actions_student_buf:
            return

        actions_student = torch.cat(self._actions_student_buf, dim=0)
        actions_teacher = torch.cat(self._actions_teacher_buf, dim=0)
        yaw_student = torch.cat(self._yaw_student_buf, dim=0)
        yaw_teacher = torch.cat(self._yaw_teacher_buf, dim=0)

        action_loss = (actions_teacher.detach() - actions_student).norm(p=2, dim=1).mean()
        yaw_loss = (yaw_teacher.detach() - yaw_student).norm(p=2, dim=1).mean()
        loss = self.action_loss_coef * action_loss + self.yaw_loss_coef * yaw_loss

        # Scandot prediction loss (MSE between predicted and GT scandots)
        scandot_loss_val = 0.0
        if self._scandot_pred_buf and self.scandot_loss_coef > 0:
            scandots_pred = torch.cat(self._scandot_pred_buf, dim=0)
            scandots_gt = torch.cat(self._scandot_gt_buf, dim=0)
            scandot_loss = (scandots_gt.detach() - scandots_pred).pow(2).mean()
            loss = loss + self.scandot_loss_coef * scandot_loss
            scandot_loss_val = scandot_loss.item()

        (loss / self._num_windows).backward()

        self._accum_action_loss += action_loss.item()
        self._accum_yaw_loss += yaw_loss.item()
        self._accum_scandot_loss += scandot_loss_val
        self._window_count += 1

        self._actions_student_buf.clear()
        self._actions_teacher_buf.clear()
        self._yaw_student_buf.clear()
        self._yaw_teacher_buf.clear()
        self._scandot_pred_buf.clear()
        self._scandot_gt_buf.clear()
        self._step_in_window = 0

        self.actor_critic.detach_gru_hidden()

    def _update_lr(self):
        if self._lr_schedule == "cosine":
            progress = min(1.0, self._iteration / max(1, self._lr_schedule_max_iters))
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        elif self._lr_schedule == "linear":
            progress = min(1.0, self._iteration / max(1, self._lr_schedule_max_iters))
            scale = 1.0 - progress
        else:
            return
        for pg in self.optimizer.param_groups:
            base_lr = pg.get("initial_lr", self.learning_rate)
            pg["lr"] = self._lr_schedule_min_lr + (base_lr - self._lr_schedule_min_lr) * scale

    def finish_rollout(self) -> dict:
        """Flush any remaining steps, clip gradients, optimizer step."""
        self._flush_window()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._iteration += 1
        self._update_lr()

        n = max(self._window_count, 1)
        mean_action = self._accum_action_loss / n
        mean_yaw = self._accum_yaw_loss / n
        mean_scandot = self._accum_scandot_loss / n
        return {
            "loss": mean_action + mean_yaw + mean_scandot,
            "action_loss": mean_action,
            "yaw_loss": mean_yaw,
            "scandot_loss": mean_scandot,
        }
