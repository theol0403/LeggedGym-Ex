import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_ts import PPO_TS
from rsl_rl.modules import ActorCriticParkourStudent
from rsl_rl.storage import RolloutStorageTS


class PPO_ParkourStudent(PPO_TS):
    """PPO with depth-encoder distillation for parkour student training.

    RL updates flow through actor + critic + scandot_encoder.
    A separate supervised MSE loss trains the depth_encoder CNN to match
    the scandot_encoder latent (detached target).
    """

    actor_critic: ActorCriticParkourStudent

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        use_spo=False,
        device="cpu",
        encoder_lr=1e-3,
        num_encoder_epochs=1,
    ):
        # Skip PPO_TS.__init__ to avoid referencing history_encoder directly
        PPO.__init__(
            self,
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            use_spo,
            device,
        )
        self.encoder_lr = encoder_lr
        self.num_encoder_epochs = num_encoder_epochs

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        # RL parameters: actor + critic + scandot_encoder + action noise
        self.rl_parameters = (
            list(self.actor_critic.actor.parameters())
            + list(self.actor_critic.critic.parameters())
            + list(self.actor_critic.scandot_encoder.parameters())
            + [self.actor_critic.std]
        )
        self.optimizer = optim.Adam(self.rl_parameters, lr=learning_rate)

        # Depth encoder optimizer (named for PPO_TS.update() compatibility)
        self.history_encoder_optimizer = optim.Adam(
            self.actor_critic.depth_encoder.parameters(), lr=encoder_lr
        )
        self.transition = RolloutStorageTS.Transition()

    def _compute_encoder_loss(self, obs_histories_batch, privileged_obs_batch, terminated_batch):
        """Supervised MSE: depth_encoder output vs detached scandot_encoder output."""
        depth_image = obs_histories_batch.view(-1, *self.actor_critic.depth_image_shape)
        encoder_predictions = self.actor_critic.depth_encoder(depth_image)
        with torch.no_grad():
            encoder_targets = self.actor_critic.scandot_encoder(privileged_obs_batch)
        return nn.functional.mse_loss(
            encoder_predictions * terminated_batch,
            encoder_targets * terminated_batch,
        )
