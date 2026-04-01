"""Observation layout for parkour environments (CAI23sbP / RMA format).

Prop obs (53 dims):
  [0:3]   ang_vel * 0.25
  [3:5]   roll, pitch (IMU)
  [5:6]   zero placeholder
  [6:7]   delta_yaw (current target heading error)
  [7:8]   delta_next_yaw (next target heading error)
  [8:10]  zeros placeholder
  [10:11] vel_cmd_x (goal speed)
  [11:12] is_not_parkour_flat flag
  [12:13] is_parkour_flat flag
  [13:25] joint_pos - default (12)
  [25:37] joint_vel * 0.05 (12)
  [37:49] prev_actions (12)
  [49:53] contact_fill (4, filtered, centered +/-0.5)

Full teacher obs (753 dims):
  prop(53) + scandots(132) + priv_explicit(9) + priv_latent(29) + history(530)
"""

from __future__ import annotations

from dataclasses import dataclass


NUM_PROP = 53
NUM_PRIV_EXPLICIT = 9
NUM_PRIV_LATENT = 29
NUM_HIST_FRAMES = 10


def _num_scandot_points(terrain_cfg) -> int:
    return len(terrain_cfg.scandots.points_x) * len(terrain_cfg.scandots.points_y)


@dataclass(frozen=True)
class ParkourObservationSpec:
    num_actions: int
    num_scandots: int
    num_prop: int = NUM_PROP
    num_priv_explicit: int = NUM_PRIV_EXPLICIT
    num_priv_latent: int = NUM_PRIV_LATENT
    num_hist_frames: int = NUM_HIST_FRAMES
    num_foot_contacts: int = 4

    @property
    def prop_dim(self) -> int:
        return self.num_prop

    @property
    def history_dim(self) -> int:
        return self.num_hist_frames * self.num_prop

    @property
    def full_obs_dim(self) -> int:
        """Full teacher observation: prop + scandots + priv_explicit + priv_latent + history."""
        return (
            self.num_prop
            + self.num_scandots
            + self.num_priv_explicit
            + self.num_priv_latent
            + self.history_dim
        )

    @property
    def actor_dim(self) -> int:
        """Teacher actor sees the full observation."""
        return self.full_obs_dim

    @property
    def teacher_actor_dim(self) -> int:
        return self.full_obs_dim

    @property
    def critic_dim(self) -> int:
        """Critic also sees the full observation in the RMA setup."""
        return self.full_obs_dim

    # --- Slice properties for the prop portion ---
    @property
    def ang_vel_slice(self) -> slice:
        return slice(0, 3)

    @property
    def imu_slice(self) -> slice:
        return slice(3, 5)

    @property
    def yaw_delta_slice(self) -> slice:
        """Current and next heading errors at [6] and [7]."""
        return slice(6, 8)

    @property
    def vel_cmd_slice(self) -> slice:
        return slice(10, 11)

    @property
    def terrain_flag_slice(self) -> slice:
        return slice(11, 13)

    @property
    def dof_pos_slice(self) -> slice:
        return slice(13, 13 + self.num_actions)

    @property
    def dof_vel_slice(self) -> slice:
        start = 13 + self.num_actions
        return slice(start, start + self.num_actions)

    @property
    def actions_slice(self) -> slice:
        start = 13 + 2 * self.num_actions
        return slice(start, start + self.num_actions)

    @property
    def foot_contacts_slice(self) -> slice:
        start = 13 + 3 * self.num_actions
        return slice(start, start + self.num_foot_contacts)

    # --- Slices within full obs ---
    @property
    def scandots_slice(self) -> slice:
        return slice(self.num_prop, self.num_prop + self.num_scandots)

    @property
    def priv_explicit_slice(self) -> slice:
        start = self.num_prop + self.num_scandots
        return slice(start, start + self.num_priv_explicit)

    @property
    def priv_latent_slice(self) -> slice:
        start = self.num_prop + self.num_scandots + self.num_priv_explicit
        return slice(start, start + self.num_priv_latent)

    @property
    def history_slice(self) -> slice:
        start = self.num_prop + self.num_scandots + self.num_priv_explicit + self.num_priv_latent
        return slice(start, start + self.history_dim)

    # Heading command indices (for student yaw masking)
    @property
    def heading_command_indices(self) -> tuple:
        return (6, 7)

    @classmethod
    def from_cfg(cls, cfg) -> "ParkourObservationSpec":
        return cls(
            num_actions=len(cfg.asset.dof_names),
            num_scandots=_num_scandot_points(cfg.terrain),
        )


def parkour_actor_obs_dim(cfg) -> int:
    return ParkourObservationSpec.from_cfg(cfg).actor_dim


def parkour_critic_obs_dim(cfg) -> int:
    return ParkourObservationSpec.from_cfg(cfg).critic_dim


def parkour_prop_obs_dim(cfg) -> int:
    return ParkourObservationSpec.from_cfg(cfg).prop_dim
