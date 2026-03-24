import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation
from .mlp_utils import build_mlp


class ActorCriticParkour(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_scandots,
        scandot_start_idx=None,
        scandot_encoder_hidden_dims=[128, 64, 32],
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        clip_actions=100.0,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(
                "ActorCriticParkour received unexpected arguments: "
                + str(sorted(kwargs.keys()))
            )
        super().__init__()

        if num_scandots <= 0 or num_scandots >= num_actor_obs:
            raise ValueError(
                f"ActorCriticParkour expects 0 < num_scandots < num_actor_obs, got "
                f"num_scandots={num_scandots}, num_actor_obs={num_actor_obs}."
            )
        if scandot_start_idx is None:
            scandot_start_idx = num_actor_obs - num_scandots
        scandot_start_idx = int(scandot_start_idx)
        if scandot_start_idx < 0 or scandot_start_idx + num_scandots > num_actor_obs:
            raise ValueError(
                "ActorCriticParkour received an invalid scandot slice: "
                f"start={scandot_start_idx}, num_scandots={num_scandots}, num_actor_obs={num_actor_obs}."
            )

        activation_layer = get_activation(activation)
        self.num_scandots = int(num_scandots)
        self.scandot_start_idx = scandot_start_idx
        self.scandot_end_idx = scandot_start_idx + self.num_scandots
        self.num_prop_obs = int(num_actor_obs - num_scandots)

        self.scandot_encoder = build_mlp(
            input_dim=self.num_scandots,
            hidden_dims=scandot_encoder_hidden_dims,
            output_dim=None,
            activation=activation_layer,
            final_activation=True,
        )
        self.scandot_latent_dim = scandot_encoder_hidden_dims[-1]

        self.actor = build_mlp(
            input_dim=self.num_prop_obs + self.scandot_latent_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation=activation_layer,
        )
        self.actor.add_module(
            "clip_actions",
            nn.Hardtanh(min_val=-clip_actions, max_val=clip_actions),
        )
        self.critic = build_mlp(
            input_dim=num_critic_obs,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation_layer,
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _split_actor_obs(self, observations):
        prop_obs = torch.cat(
            (
                observations[:, : self.scandot_start_idx],
                observations[:, self.scandot_end_idx :],
            ),
            dim=-1,
        )
        scandot_obs = observations[:, self.scandot_start_idx : self.scandot_end_idx]
        return prop_obs, scandot_obs

    def _actor_mean(self, observations):
        prop_obs, scandot_obs = self._split_actor_obs(observations)
        scandot_latent = self.scandot_encoder(scandot_obs)
        actor_input = torch.cat((prop_obs, scandot_latent), dim=-1)
        return self.actor(actor_input)

    def infer_scandot_latent(self, observations):
        _, scandot_obs = self._split_actor_obs(observations)
        return self.scandot_encoder(scandot_obs)

    def update_distribution(self, observations):
        mean = self._actor_mean(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self._actor_mean(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
