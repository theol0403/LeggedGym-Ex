from __future__ import annotations

from dataclasses import dataclass


def _num_scandot_points(terrain_cfg) -> int:
    return len(terrain_cfg.scandots.points_x) * len(terrain_cfg.scandots.points_y)


def _num_contact_states(asset_cfg) -> int:
    return len(asset_cfg.contact_state_link_names)


@dataclass(frozen=True)
class ParkourObservationSpec:
    num_actions: int
    num_scandots: int
    num_contact_states: int
    num_goal_terms: int = 7
    num_gravity: int = 3
    num_ang_vel: int = 3
    num_foot_contacts: int = 4
    num_base_lin_vel: int = 3

    @property
    def privileged_dynamics_dim(self) -> int:
        return 1 + 1 + 3 + 2 + self.num_actions + self.num_actions

    @property
    def prop_dim(self) -> int:
        """Proprioceptive-only observation dimension (no scandots)."""
        return (
            self.num_goal_terms
            + self.num_gravity
            + self.num_ang_vel
            + self.num_actions
            + self.num_actions
            + self.num_actions
            + self.num_foot_contacts
        )

    @property
    def actor_dim(self) -> int:
        return self.prop_dim + self.num_scandots

    @property
    def critic_dim(self) -> int:
        return (
            self.actor_dim
            + self.num_base_lin_vel
            + self.privileged_dynamics_dim
            + self.num_contact_states
        )

    @property
    def command_slice(self) -> slice:
        return slice(0, self.num_goal_terms)

    @property
    def gravity_slice(self) -> slice:
        start = self.command_slice.stop
        return slice(start, start + self.num_gravity)

    @property
    def ang_vel_slice(self) -> slice:
        start = self.gravity_slice.stop
        return slice(start, start + self.num_ang_vel)

    @property
    def dof_pos_slice(self) -> slice:
        start = self.ang_vel_slice.stop
        return slice(start, start + self.num_actions)

    @property
    def dof_vel_slice(self) -> slice:
        start = self.dof_pos_slice.stop
        return slice(start, start + self.num_actions)

    @property
    def actions_slice(self) -> slice:
        start = self.dof_vel_slice.stop
        return slice(start, start + self.num_actions)

    @property
    def foot_contacts_slice(self) -> slice:
        start = self.actions_slice.stop
        return slice(start, start + self.num_foot_contacts)

    @property
    def scandots_slice(self) -> slice:
        start = self.foot_contacts_slice.stop
        return slice(start, start + self.num_scandots)

    @classmethod
    def from_cfg(cls, cfg) -> "ParkourObservationSpec":
        return cls(
            num_actions=len(cfg.asset.dof_names),
            num_scandots=_num_scandot_points(cfg.terrain),
            num_contact_states=_num_contact_states(cfg.asset),
        )


def parkour_actor_obs_dim(cfg) -> int:
    return ParkourObservationSpec.from_cfg(cfg).actor_dim


def parkour_critic_obs_dim(cfg) -> int:
    return ParkourObservationSpec.from_cfg(cfg).critic_dim


def parkour_prop_obs_dim(cfg) -> int:
    return ParkourObservationSpec.from_cfg(cfg).prop_dim
