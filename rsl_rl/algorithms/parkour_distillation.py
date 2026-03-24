import inspect

import torch
import torch.nn as nn
import torch.optim as optim


class ParkourDistillation:
    """DAgger-style distillation with per-step gradient accumulation.

    Instead of storing error tensors across the entire rollout (which keeps all
    computation graphs alive and causes OOM), we call backward() at each step,
    scaling the loss by 1/num_steps so the accumulated gradient equals the batch
    gradient.  The optimizer step happens once after the full rollout.
    """

    def __init__(
        self,
        actor_critic,
        learning_rate=1e-3,
        action_loss_coef=1.0,
        latent_loss_coef=0.25,
        yaw_loss_coef=1.0,
        max_grad_norm=1.0,
        device="cpu",
    ):
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.action_loss_coef = action_loss_coef
        self.latent_loss_coef = latent_loss_coef
        self.yaw_loss_coef = yaw_loss_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        optimizer_kwargs = {"lr": self.learning_rate}
        if str(self.device).startswith("cuda"):
            try:
                if "fused" in inspect.signature(optim.Adam).parameters:
                    optimizer_kwargs["fused"] = True
            except (TypeError, ValueError):
                pass
        return optim.Adam(self.actor_critic.parameters(), **optimizer_kwargs)

    def act(self, observations, student_depth):
        return self.actor_critic.act(observations, student_depth)

    def begin_rollout(self):
        """Call at the start of each iteration before stepping the env."""
        self.optimizer.zero_grad(set_to_none=True)
        self._step_count = 0
        self._accum_action_loss = 0.0
        self._accum_latent_loss = 0.0
        self._accum_yaw_loss = 0.0

    def accumulate_step(
        self,
        student_actions,
        teacher_actions,
        student_latent,
        teacher_latent,
        predicted_yaw,
        oracle_yaw,
        num_steps,
    ):
        """Compute per-step loss, backward (scaled by 1/num_steps), free the graph."""
        action_loss = (teacher_actions - student_actions).norm(p=2, dim=1).mean()
        latent_loss = (teacher_latent - student_latent).norm(p=2, dim=1).mean()
        yaw_loss = (oracle_yaw - predicted_yaw).norm(p=2, dim=1).mean()

        loss = (
            self.action_loss_coef * action_loss
            + self.latent_loss_coef * latent_loss
            + self.yaw_loss_coef * yaw_loss
        )

        (loss / num_steps).backward()

        self._step_count += 1
        self._accum_action_loss += action_loss.item()
        self._accum_latent_loss += latent_loss.item()
        self._accum_yaw_loss += yaw_loss.item()

    def finish_rollout(self) -> dict:
        """Clip gradients and optimizer step after the full rollout."""
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        n = max(self._step_count, 1)
        mean_action = self._accum_action_loss / n
        mean_latent = self._accum_latent_loss / n
        mean_yaw = self._accum_yaw_loss / n
        return {
            "loss": mean_action + mean_latent + mean_yaw,
            "action_loss": mean_action,
            "latent_loss": mean_latent,
            "yaw_loss": mean_yaw,
        }
