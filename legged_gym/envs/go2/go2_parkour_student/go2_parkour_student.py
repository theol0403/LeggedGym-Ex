import torch

from legged_gym.envs.base.legged_robot_parkour import LeggedRobotParkour
from legged_gym.envs.base.parkour_observation import ParkourObservationSpec
from legged_gym.utils.math_utils import torch_rand_float


class Go2ParkourStudent(LeggedRobotParkour):
    """Parkour environment for depth-based student training.

    Actor obs are proprioceptive-only (no scandots).  Scandots are provided
    as privileged_obs for the scandot_encoder supervision target.  Depth images
    from the simulator camera are provided as the encoder input (stored in the
    ``depth_obs`` buffer and returned in the obs_history slot for TSRunner
    compatibility).
    """

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_depth_obs = self.cfg.env.num_depth_obs
        self.num_latent_dims = self.cfg.env.num_latent_dims
        self.num_critic_obs = self.cfg.env.num_critic_obs
        # Alias so TSRunner can read it as num_history_obs
        self.num_history_obs = self.num_depth_obs

    def _init_buffers(self):
        self.obs_spec = ParkourObservationSpec.from_cfg(self.cfg)
        super()._init_buffers()

        self.depth_obs = torch.zeros(
            self.num_envs, self.num_depth_obs,
            device=self.device, dtype=torch.float,
        )
        self.critic_obs_buf = torch.zeros(
            self.num_envs, self.num_critic_obs,
            device=self.device, dtype=torch.float,
        )

    def compute_observations(self):
        foot_contacts = (
            self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 1.0
        ).float()
        scandots_cfg = self.cfg.terrain.scandots
        scandot_obs = torch.clip(
            self.simulator.base_pos[:, 2:3]
            - scandots_cfg.base_height_offset
            - self.simulator.scandot_heights,
            scandots_cfg.clip_min,
            scandots_cfg.clip_max,
        )

        # Proprioceptive actor obs (no scandots)
        prop_parts = (
            self.commands * self.commands_scale,
            self.simulator.projected_gravity,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            foot_contacts,
        )
        self.obs_buf = torch.cat(prop_parts, dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # Privileged obs = scandots (for scandot_encoder supervision)
        self.privileged_obs_buf = scandot_obs

        # Depth obs from simulator depth camera
        self.depth_obs = self.simulator._depth_images[:, 0].reshape(self.num_envs, -1)

        # Critic obs = full teacher-style privileged obs
        actor_obs_with_scandots = torch.cat(prop_parts + (scandot_obs,), dim=-1)
        self.critic_obs_buf = torch.cat((
            actor_obs_with_scandots,
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self._get_privileged_dynamics(),
            self.simulator.link_contact_states,
        ), dim=-1)

    def step(self, actions):
        actions = self._pre_sim_step(actions)
        self.simulator.step(actions)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf, self.privileged_obs_buf, self.depth_obs,
            self.critic_obs_buf, self.rew_buf, self.reset_buf, self.extras,
        )

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, priv, depth, critic, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, priv, depth, critic

    def get_observations(self):
        return self.obs_buf, self.privileged_obs_buf, self.depth_obs, self.critic_obs_buf

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.depth_obs[env_ids] = 0.0

    def _reset_dofs(self, env_ids):
        dof_pos = torch.zeros(
            (len(env_ids), self.num_actions),
            dtype=torch.float, device=self.device, requires_grad=False,
        )
        dof_vel = torch.zeros(
            (len(env_ids), self.num_actions),
            dtype=torch.float, device=self.device, requires_grad=False,
        )
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + torch_rand_float(
            -0.2, 0.2, (len(env_ids), 4), self.device
        )
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + torch_rand_float(
            -0.4, 0.4, (len(env_ids), 4), self.device
        )
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + torch_rand_float(
            -0.4, 0.4, (len(env_ids), 4), self.device
        )
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _validate_configured_observation_dims(self):
        if self.num_obs != self.obs_spec.prop_dim:
            raise RuntimeError(
                f"Configured student actor obs dim {self.num_obs} does not match "
                f"parkour prop spec {self.obs_spec.prop_dim}."
            )

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # Student obs_buf layout: commands(7) gravity(3) ang_vel(3) dof_pos(12) dof_vel(12) actions(12) contacts(4)
        offset = 0
        noise_vec[offset:offset + 7] = 0.0  # commands
        offset += 7
        noise_vec[offset:offset + 3] = noise_scales.gravity * noise_level
        offset += 3
        noise_vec[offset:offset + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        offset += 3
        noise_vec[offset:offset + 12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        offset += 12
        noise_vec[offset:offset + 12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        offset += 12
        noise_vec[offset:offset + 12] = 0.0  # actions
        offset += 12
        noise_vec[offset:offset + 4] = 0.0  # foot contacts
        return noise_vec
