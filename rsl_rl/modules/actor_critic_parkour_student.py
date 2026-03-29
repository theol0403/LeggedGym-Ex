"""Student policy: depth encoder (CNN + GRU) replaces teacher scandot encoder."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .actor_critic import get_activation
from .mlp_utils import build_mlp


class DepthBackbone58x87(nn.Module):
    """CNN depth encoder matching Extreme Parkour (58 x 87) -> 32-dim features."""

    def __init__(self, output_dim=32, activation=nn.ELU, in_channels=1):
        super().__init__()
        act = activation()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            act,
            nn.Conv2d(32, 64, kernel_size=3),
            act,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            act,
            nn.Linear(128, output_dim),
        )

    def forward(self, depth_bchw: torch.Tensor) -> torch.Tensor:
        """depth_bchw: (B, C, H, W) with H=58, W=87."""
        return self.net(depth_bchw)


class RecurrentDepthEncoder(nn.Module):
    """CNN + proprio fusion + GRU + Tanh output (latent + yaw), matching the paper."""

    def __init__(
        self,
        num_proprio_for_combo: int,
        depth_backbone_output_dim: int = 32,
        gru_hidden_dim: int = 512,
        latent_dim: int = 32,
        yaw_dim: int = 2,
        activation=nn.ELU,
        depth_in_channels: int = 1,
    ):
        super().__init__()
        act = activation()
        self.latent_dim = latent_dim
        self.yaw_dim = yaw_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.depth_backbone = DepthBackbone58x87(
            output_dim=depth_backbone_output_dim, activation=activation,
            in_channels=depth_in_channels,
        )
        self.combination_mlp = nn.Sequential(
            nn.Linear(depth_backbone_output_dim + num_proprio_for_combo, 128),
            act,
            nn.Linear(128, 32),
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=gru_hidden_dim, num_layers=1, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, latent_dim + yaw_dim),
            nn.Tanh(),
        )

    def init_hidden(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device, dtype=dtype)

    def forward(
        self,
        depth_bchw: torch.Tensor,
        proprio_masked: torch.Tensor,
        hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent (B, latent_dim), yaw_raw (B, yaw_dim) in [-1, 1] from Tanh,
            new_hidden (1, B, gru_hidden_dim)
        """
        backbone_feat = self.depth_backbone(depth_bchw)
        combo = self.combination_mlp(torch.cat((backbone_feat, proprio_masked), dim=-1))
        seq = combo.unsqueeze(1)
        out, new_hidden = self.rnn(seq, hidden)
        flat = self.output_mlp(out.squeeze(1))
        latent = flat[:, : self.latent_dim]
        yaw_raw = flat[:, self.latent_dim :]
        return latent, yaw_raw, new_hidden


class ActorCriticParkourStudent(nn.Module):
    """Student actor: proprio + depth latent (replaces scandot latent). GRU carries temporal state."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        num_teacher_actor_obs,
        student_depth_shape=(1, 58, 87),
        depth_backbone_output_dim=32,
        gru_hidden_dim=512,
        yaw_output_dim=2,
        yaw_scale=1.5,
        heading_command_indices=(4, 5),
        actor_hidden_dims=(512, 256, 128),
        activation="elu",
        clip_actions=100.0,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(
                "ActorCriticParkourStudent received unexpected arguments: " + str(sorted(kwargs.keys()))
            )
        super().__init__()

        activation_layer = get_activation(activation)
        self.num_obs = int(num_actor_obs)
        self.num_teacher_actor_obs = int(num_teacher_actor_obs)
        self.latent_dim = int(depth_backbone_output_dim)
        self.student_depth_shape = tuple(student_depth_shape)
        self.yaw_output_dim = int(yaw_output_dim)
        self.yaw_scale = float(yaw_scale)
        self.heading_command_indices = tuple(heading_command_indices)

        c, h, w = self.student_depth_shape
        if h != 58 or w != 87:
            raise ValueError(
                f"DepthBackbone58x87 expects shape (C,58,87); got student_depth_shape={self.student_depth_shape}."
            )

        self.recurrent_depth = RecurrentDepthEncoder(
            num_proprio_for_combo=self.num_obs,
            depth_backbone_output_dim=depth_backbone_output_dim,
            gru_hidden_dim=gru_hidden_dim,
            latent_dim=self.latent_dim,
            yaw_dim=self.yaw_output_dim,
            activation=type(activation_layer),
            depth_in_channels=c,
        )

        self.actor = build_mlp(
            input_dim=self.num_obs + self.latent_dim,
            hidden_dims=list(actor_hidden_dims),
            output_dim=num_actions,
            activation=activation_layer,
        )
        self.actor.add_module(
            "clip_actions",
            nn.Hardtanh(min_val=-clip_actions, max_val=clip_actions),
        )

        self.register_buffer("_gru_hidden", torch.empty(0), persistent=False)

    def _ensure_hidden(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self._gru_hidden.numel() == 0 or self._gru_hidden.shape[1] != batch:
            self._gru_hidden = self.recurrent_depth.init_hidden(batch, device, dtype)
        return self._gru_hidden

    def reset_gru(self, env_ids: Optional[torch.Tensor] = None):
        """Zero GRU hidden for given env indices (or all)."""
        if self._gru_hidden.numel() == 0:
            return
        if env_ids is None or len(env_ids) == 0:
            self._gru_hidden.zero_()
        else:
            self._gru_hidden[:, env_ids, :] = 0.0

    def detach_gru_hidden(self):
        if self._gru_hidden.numel() > 0:
            self._gru_hidden = self._gru_hidden.detach()

    def proprio_for_depth_encoder(self, observations: torch.Tensor) -> torch.Tensor:
        """Mask heading errors (indices 4:6) so depth must predict yaw."""
        out = observations.clone()
        out[:, self.heading_command_indices[0] : self.heading_command_indices[1] + 1] = 0.0
        return out

    def predicted_yaw_scaled(self, yaw_raw: torch.Tensor) -> torch.Tensor:
        return self.yaw_scale * yaw_raw

    def apply_mts_yaw_to_obs(
        self,
        observations: torch.Tensor,
        yaw_scaled: torch.Tensor,
        oracle_heading: torch.Tensor,
        yaw_threshold: float,
    ) -> torch.Tensor:
        """Mixture of Teacher and Student: use predicted yaw in obs when close to oracle."""
        out = observations.clone()
        diff = torch.norm(yaw_scaled - oracle_heading, dim=-1)
        ok = diff < yaw_threshold
        idx0, idx1 = self.heading_command_indices
        out[ok, idx0] = yaw_scaled[ok, 0]
        out[ok, idx1] = yaw_scaled[ok, 1]
        return out

    def forward_depth(
        self,
        student_depth: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        student_depth: (B, 1, 58, 87)
        Returns latent, yaw_raw (Tanh), yaw_scaled (for loss / obs).
        """
        proprio_masked = self.proprio_for_depth_encoder(observations)
        hidden = self._ensure_hidden(observations.shape[0], observations.device, observations.dtype)
        latent, yaw_raw, new_h = self.recurrent_depth(student_depth, proprio_masked, hidden)
        self._gru_hidden = new_h
        yaw_scaled = self.predicted_yaw_scaled(yaw_raw)
        return latent, yaw_raw, yaw_scaled

    def actor_from_latent(self, observations: torch.Tensor, student_latent: torch.Tensor) -> torch.Tensor:
        return self.actor(torch.cat((observations, student_latent), dim=-1))

    def act(self, observations: torch.Tensor, student_depth: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.forward_depth(student_depth, observations)
        return self.actor_from_latent(observations, latent)

    def act_inference(self, observations: torch.Tensor, student_depth: torch.Tensor) -> torch.Tensor:
        return self.act(observations, student_depth)

    def load_teacher_actor_weights(self, teacher_state_dict):
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in teacher_state_dict.items()
            if key.startswith("actor.")
        }
        missing, unexpected = self.actor.load_state_dict(actor_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Teacher actor initialization mismatch. "
                f"Missing keys: {missing}. Unexpected keys: {unexpected}."
            )
