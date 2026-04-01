"""ActorCriticRMA: RMA-style actor-critic matching the CAI23sbP/Isaaclab_Parkour checkpoint.

Architecture:
  Actor = scan_encoder (Tanh final) + priv_encoder + history_encoder + actor_backbone
  Critic = MLP on full 753-dim obs
  Estimator = separate MLP, loaded from estimator_state_dict

State dict keys match the CAI23sbP checkpoint exactly:
  actor.scan_encoder.{0,2,4}.*
  actor.priv_encoder.{0,2}.*
  actor.history_encoder.encoder.0.*, .conv_layers.{0,2}.*, .linear_output.0.*
  actor.actor_backbone.{0,2,4,6}.*
  critic.{0,2,4,6}.*
  std
  estimator.{0,2,4}.*  (in estimator_state_dict)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class StateHistoryEncoder(nn.Module):
    """Temporal 1D conv encoder: 10 frames of 53-dim prop -> 20-dim latent."""

    def __init__(self, num_prop: int = 53, channel_size: int = 10,
                 num_frames: int = 10, latent_dim: int = 20):
        super().__init__()
        self.num_prop = num_prop
        self.num_frames = num_frames
        # Per-frame linear: 53 -> channel_size*3 (=30), with ELU
        # NOTE: CAI23sbP passes the global activation (ELU) to StateHistoryEncoder.
        # All activations here must be ELU to match the trained weights.
        self.encoder = nn.Sequential(
            nn.Linear(num_prop, channel_size * 3),
            nn.ELU(),
        )
        # 1D convolutions on sequence (includes Flatten at the end)
        # conv1(k=4, s=2): 10 -> 4, conv2(k=2, s=1): 4 -> 3
        # Flatten: channel_size * 3 = 30
        self.conv_layers = nn.Sequential(
            nn.Conv1d(channel_size * 3, channel_size * 2, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv1d(channel_size * 2, channel_size, kernel_size=2, stride=1),
            nn.ELU(),
            nn.Flatten(),
        )
        # Final linear: channel_size*3 (=30) -> latent_dim (=20), with ELU
        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, latent_dim),
            nn.ELU(),
        )

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_history: (B, num_frames * num_prop) flattened history, oldest first.

        Returns:
            latent: (B, latent_dim)
        """
        B = obs_history.shape[0]
        # Reshape to (B, num_frames, num_prop)
        x = obs_history.view(B, self.num_frames, self.num_prop)
        # Per-frame encoding: (B, num_frames, 30)
        x = self.encoder(x)
        # Conv1d expects (B, C, L): transpose to (B, 30, num_frames)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        # Flatten and project
        x = x.flatten(1)
        return self.linear_output(x)


class Actor(nn.Module):
    """RMA Actor with scan_encoder (Tanh), priv_encoder, history_encoder, actor_backbone."""

    def __init__(
        self,
        num_prop: int = 53,
        num_scan: int = 132,
        num_priv_explicit: int = 9,
        num_priv_latent: int = 29,
        num_hist: int = 10,
        num_actions: int = 12,
        scan_encoder_dims: list = [128, 64, 32],
        priv_encoder_dims: list = [64, 20],
        actor_hidden_dims: list = [512, 256, 128],
        history_encoder_channel_size: int = 10,
        activation: str = "elu",
    ):
        super().__init__()
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_priv_explicit = num_priv_explicit
        self.num_priv_latent = num_priv_latent
        self.num_hist = num_hist
        self.num_actions = num_actions

        act = nn.ELU()

        # Scan encoder: 132 -> 128 -> 64 -> 32, with Tanh on final layer
        scan_layers = []
        prev = num_scan
        for i, dim in enumerate(scan_encoder_dims):
            scan_layers.append(nn.Linear(prev, dim))
            if i < len(scan_encoder_dims) - 1:
                scan_layers.append(nn.ELU())
            else:
                scan_layers.append(nn.Tanh())
            prev = dim
        self.scan_encoder = nn.Sequential(*scan_layers)
        self.scan_latent_dim = scan_encoder_dims[-1]

        # Priv encoder: 29 -> 64 (ELU) -> 20
        priv_layers = []
        prev = num_priv_latent
        for dim in priv_encoder_dims:
            priv_layers.append(nn.Linear(prev, dim))
            priv_layers.append(nn.ELU())
            prev = dim
        self.priv_encoder = nn.Sequential(*priv_layers)
        self.priv_latent_dim = priv_encoder_dims[-1]

        # History encoder
        self.history_encoder = StateHistoryEncoder(
            num_prop=num_prop,
            channel_size=history_encoder_channel_size,
            num_frames=num_hist,
            latent_dim=self.priv_latent_dim,  # matches priv_encoder output = 20
        )

        # Actor backbone: (prop + scan_latent + priv_explicit + latent) -> actions
        # 53 + 32 + 9 + 20 = 114
        backbone_input_dim = num_prop + self.scan_latent_dim + num_priv_explicit + self.priv_latent_dim
        backbone_layers = []
        prev = backbone_input_dim
        for dim in actor_hidden_dims:
            backbone_layers.append(nn.Linear(prev, dim))
            backbone_layers.append(nn.ELU())
            prev = dim
        backbone_layers.append(nn.Linear(prev, num_actions))
        self.actor_backbone = nn.Sequential(*backbone_layers)

    def infer_priv_latent(self, obs: torch.Tensor) -> torch.Tensor:
        """Use ground-truth privileged info (training with sim access)."""
        start = self.num_prop + self.num_scan + self.num_priv_explicit
        priv_latent = obs[:, start:start + self.num_priv_latent]
        return self.priv_encoder(priv_latent)

    def infer_hist_latent(self, obs: torch.Tensor) -> torch.Tensor:
        """Use history encoder (deployment / student distillation)."""
        hist_start = self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent
        obs_history = obs[:, hist_start:hist_start + self.num_hist * self.num_prop]
        return self.history_encoder(obs_history)

    def forward(self, obs: torch.Tensor, hist_encoding: bool = False,
                scandots_latent: torch.Tensor = None) -> torch.Tensor:
        obs_prop = obs[:, :self.num_prop]
        obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]

        if scandots_latent is None:
            scan_latent = self.scan_encoder(obs_scan)
        else:
            scan_latent = scandots_latent

        obs_priv_explicit = obs[:, self.num_prop + self.num_scan:
                                   self.num_prop + self.num_scan + self.num_priv_explicit]

        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)

        backbone_input = torch.cat([obs_prop, scan_latent, obs_priv_explicit, latent], dim=1)
        return self.actor_backbone(backbone_input)


class ActorCriticRMA(nn.Module):
    """Full ActorCriticRMA matching the CAI23sbP checkpoint format."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,  # 753
        num_critic_obs: int,  # 753
        num_actions: int = 12,
        num_prop: int = 53,
        num_scan: int = 132,
        num_priv_explicit: int = 9,
        num_priv_latent: int = 29,
        num_hist: int = 10,
        scan_encoder_dims: list = [128, 64, 32],
        priv_encoder_dims: list = [64, 20],
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        history_encoder_channel_size: int = 10,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        clip_actions: float = 100.0,
        **kwargs,
    ):
        if kwargs:
            print(f"ActorCriticRMA ignoring unexpected kwargs: {sorted(kwargs.keys())}")
        super().__init__()

        self.actor = Actor(
            num_prop=num_prop,
            num_scan=num_scan,
            num_priv_explicit=num_priv_explicit,
            num_priv_latent=num_priv_latent,
            num_hist=num_hist,
            num_actions=num_actions,
            scan_encoder_dims=scan_encoder_dims,
            priv_encoder_dims=priv_encoder_dims,
            actor_hidden_dims=actor_hidden_dims,
            history_encoder_channel_size=history_encoder_channel_size,
        )

        # Critic: full obs (753) -> 1
        critic_layers = []
        prev = num_critic_obs
        for dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev, dim))
            critic_layers.append(nn.ELU())
            prev = dim
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action std (learned per-action)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        # Store dims for obs slicing
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_priv_explicit = num_priv_explicit
        self.num_priv_latent = num_priv_latent

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

    def update_distribution(self, observations):
        mean = self.actor(observations, hist_encoding=False)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=True):
        return self.actor(observations, hist_encoding=hist_encoding)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def infer_scandot_latent(self, observations):
        """Encode scandots from full observation (for student distillation compatibility)."""
        scandots = observations[:, self.num_prop:self.num_prop + self.num_scan]
        return self.actor.scan_encoder(scandots)


class DefaultEstimator(nn.Module):
    """Estimator MLP: prop(53) -> priv_explicit(9). Loaded from estimator_state_dict."""

    def __init__(self, num_prop: int = 53, num_priv_explicit: int = 9,
                 hidden_dims: list = [128, 64]):
        super().__init__()
        layers = []
        prev = num_prop
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ELU())
            prev = dim
        layers.append(nn.Linear(prev, num_priv_explicit))
        self.estimator = nn.Sequential(*layers)

    def forward(self, prop_obs: torch.Tensor) -> torch.Tensor:
        return self.estimator(prop_obs)
