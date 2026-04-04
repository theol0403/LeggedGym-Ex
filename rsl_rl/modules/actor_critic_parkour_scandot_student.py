"""Scandot-prediction student: depth encoder predicts 132 scandots directly, frozen teacher actor produces actions."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .actor_critic import get_activation
from .actor_critic_parkour_student import RecurrentDepthEncoder


class ActorCriticParkourScandotStudent(nn.Module):
    """Direct scandot prediction student.

    The depth encoder (CNN + GRU) predicts 132 scandot heights + 2 yaw values
    from depth images and proprioception.  A *frozen* teacher actor (held as a
    non-owned reference) then maps full obs (with predicted scandots inserted)
    -> actions.

    Key differences from ActorCriticParkourStudent:
      - latent_dim = num_scandots (132) instead of 32
      - The actor is the teacher's full pipeline (frozen, external)
      - Training loss targets: MSE on predicted scandots + yaw loss (+
        optional action loss)
      - Teacher weights are NOT part of this module's state_dict /
        parameters; they live in the runner's ``self.teacher``
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        num_teacher_actor_obs,
        student_depth_shape=(1, 58, 87),
        depth_backbone_output_dim=32,
        gru_hidden_dim=512,
        num_scandots=132,
        yaw_output_dim=2,
        yaw_scale=1.5,
        heading_command_indices=(6, 7),
        activation="elu",
        clip_actions=100.0,
        backbone_type="cnn",
        **kwargs,
    ):
        if kwargs:
            raise TypeError(
                "ActorCriticParkourScandotStudent received unexpected arguments: "
                + str(sorted(kwargs.keys()))
            )
        super().__init__()

        activation_layer = get_activation(activation)
        self.num_obs = int(num_actor_obs)          # prop_dim (53)
        self.num_teacher_actor_obs = int(num_teacher_actor_obs)  # 753
        self.num_scandots = int(num_scandots)      # 132
        self.num_actions = int(num_actions)         # 12
        self.student_depth_shape = tuple(student_depth_shape)
        self.yaw_output_dim = int(yaw_output_dim)
        self.yaw_scale = float(yaw_scale)
        self.heading_command_indices = tuple(heading_command_indices)
        self.prop_dim = self.num_obs  # alias for clarity

        c, h, w = self.student_depth_shape
        if h != 58 or w != 87:
            raise ValueError(
                f"DepthBackbone58x87 expects shape (C,58,87); "
                f"got student_depth_shape={self.student_depth_shape}."
            )

        # Depth encoder predicts scandots (132) + yaw (2) = 134 total
        self.recurrent_depth = RecurrentDepthEncoder(
            num_proprio_for_combo=self.num_obs,
            depth_backbone_output_dim=depth_backbone_output_dim,
            gru_hidden_dim=gru_hidden_dim,
            latent_dim=self.num_scandots,
            yaw_dim=self.yaw_output_dim,
            activation=type(activation_layer),
            depth_in_channels=c,
            backbone_type=backbone_type,
        )

        # Teacher reference (set via set_teacher). We use object.__setattr__
        # to bypass nn.Module.__setattr__ which would register it as a
        # submodule and include it in state_dict/parameters.
        object.__setattr__(self, "_teacher_ref", None)

        self.register_buffer("_gru_hidden", torch.empty(0), persistent=False)

    # ------------------------------------------------------------------
    # Teacher management
    # ------------------------------------------------------------------

    def set_teacher(self, teacher_model):
        """Store a reference to the frozen teacher (ActorCriticRMA).

        The teacher is NOT registered as a submodule, so its parameters
        stay out of this module's state_dict / optimizer.
        """
        object.__setattr__(self, "_teacher_ref", teacher_model)

    def _teacher_act(self, teacher_obs: torch.Tensor) -> torch.Tensor:
        """Call the frozen teacher's forward with hist_encoding=True."""
        if self._teacher_ref is None:
            raise RuntimeError(
                "Teacher not set. Call set_teacher() before forward."
            )
        return self._teacher_ref.act_inference(teacher_obs, hist_encoding=True)

    # Alias kept for compatibility with the runner's load path
    def load_teacher_actor_weights(self, teacher_state_dict):
        """No-op: teacher weights are loaded by the runner into self._teacher."""
        pass

    # ------------------------------------------------------------------
    # GRU state management
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Observation helpers (same interface as ActorCriticParkourStudent)
    # ------------------------------------------------------------------

    def proprio_for_depth_encoder(self, observations: torch.Tensor) -> torch.Tensor:
        """Mask heading errors so depth must predict yaw."""
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

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def forward_depth(
        self,
        student_depth: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the depth encoder to predict scandots and yaw.

        Args:
            student_depth: (B, 1, 58, 87) depth image.
            observations: (B, prop_dim) proprioception.

        Returns:
            predicted_scandots: (B, 132) in [-1, 1] from Tanh.
            yaw_raw: (B, 2) in [-1, 1] from Tanh.
            yaw_scaled: (B, 2) scaled yaw prediction.
        """
        proprio_masked = self.proprio_for_depth_encoder(observations)
        hidden = self._ensure_hidden(
            observations.shape[0], observations.device, observations.dtype
        )
        predicted_scandots, yaw_raw, new_h = self.recurrent_depth(
            student_depth, proprio_masked, hidden
        )
        self._gru_hidden = new_h
        yaw_scaled = self.predicted_yaw_scaled(yaw_raw)
        return predicted_scandots, yaw_raw, yaw_scaled

    def construct_teacher_obs(
        self, prop_obs: torch.Tensor, predicted_scandots: torch.Tensor
    ) -> torch.Tensor:
        """Build the full observation for the teacher actor by concatenating prop + scandots.

        Note: this only provides prop(53) + scandots(132) = 185 dims.
        The teacher's act_inference with hist_encoding=True will slice
        the appropriate portions. The runner is responsible for providing
        the full 753-dim teacher_actor_obs when calling _teacher_act directly.
        """
        return torch.cat((prop_obs, predicted_scandots), dim=-1)

    def act(self, observations: torch.Tensor, student_depth: torch.Tensor) -> torch.Tensor:
        """Full forward: depth -> predicted scandots -> teacher actor -> actions."""
        predicted_scandots, _, _ = self.forward_depth(student_depth, observations)
        teacher_obs = self.construct_teacher_obs(observations, predicted_scandots)
        return self._teacher_act(teacher_obs)

    def act_inference(self, observations: torch.Tensor, student_depth: torch.Tensor) -> torch.Tensor:
        return self.act(observations, student_depth)
