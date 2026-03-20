import inspect

import torch
import torch.nn as nn
import torch.optim as optim


class ParkourDistillation:
    def __init__(
        self,
        actor_critic,
        learning_rate=1e-3,
        action_loss_coef=1.0,
        latent_loss_coef=0.25,
        max_grad_norm=1.0,
        device="cpu",
    ):
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.action_loss_coef = action_loss_coef
        self.latent_loss_coef = latent_loss_coef
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

    def act(self, observations, student_depth, obs_history):
        return self.actor_critic.act(observations, student_depth, obs_history)

    def update(self, observations, student_depth, obs_history, teacher_actions, teacher_latent):
        student_latent = self.actor_critic.infer_student_latent(
            observations,
            student_depth,
            obs_history,
        )
        student_actions = self.actor_critic.actor_from_latent(observations, student_latent)
        action_loss = nn.functional.mse_loss(student_actions, teacher_actions)
        latent_loss = nn.functional.mse_loss(student_latent, teacher_latent)
        loss = self.action_loss_coef * action_loss + self.latent_loss_coef * latent_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return student_actions.detach(), {
            "loss": float(loss.item()),
            "action_loss": float(action_loss.item()),
            "latent_loss": float(latent_loss.item()),
        }
