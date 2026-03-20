import torch

from legged_gym.envs.base.legged_robot_parkour import LeggedRobotParkour
from legged_gym.envs.base.parkour_observation import ParkourObservationSpec
from legged_gym.utils.math_utils import torch_rand_float


class Go2ParkourStudent(LeggedRobotParkour):
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_teacher_actor_obs = self.cfg.env.num_teacher_actor_obs
        self.num_history_obs = self.cfg.env.num_history_obs
        self.num_latent_dims = self.cfg.env.num_latent_dims
        self.student_depth_shape = tuple(self.cfg.env.student_depth_shape)

    def _init_buffers(self):
        self.obs_spec = ParkourObservationSpec.from_cfg(self.cfg)
        super()._init_buffers()

        self.teacher_actor_obs_buf = self.privileged_obs_buf
        self.obs_history = torch.zeros(
            self.num_envs,
            self.num_history_obs,
            device=self.device,
            dtype=torch.float,
        )
        self.student_depth = self.simulator.get_depth_images()
        if tuple(self.student_depth.shape[1:]) != self.student_depth_shape:
            raise RuntimeError(
                f"Simulator depth shape {tuple(self.student_depth.shape[1:])} does not match "
                f"configured student depth shape {self.student_depth_shape}."
            )

    def _push_obs_history(self):
        self.obs_history[:, :-self.num_obs].copy_(self.obs_history[:, self.num_obs :].clone())
        self.obs_history[:, -self.num_obs :].copy_(self.obs_buf)

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
        actor_obs = torch.cat(prop_parts, dim=-1)

        if self.add_noise:
            actor_obs += (2 * torch.rand_like(actor_obs) - 1) * self.noise_scale_vec

        self.obs_buf = actor_obs
        self.teacher_actor_obs_buf = torch.cat(prop_parts + (scandot_obs,), dim=-1)
        self.privileged_obs_buf = self.teacher_actor_obs_buf
        self._push_obs_history()

    def step(self, actions):
        actions = self._pre_sim_step(actions)
        self.simulator.step(actions)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.teacher_actor_obs_buf = torch.clip(self.teacher_actor_obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = self.teacher_actor_obs_buf
        return (
            self.obs_buf,
            self.teacher_actor_obs_buf,
            self.obs_history,
            self.student_depth,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, teacher_actor_obs, obs_history, student_depth, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, teacher_actor_obs, obs_history, student_depth

    def get_observations(self):
        return self.obs_buf, self.teacher_actor_obs_buf, self.obs_history, self.student_depth

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.obs_history[env_ids] = 0.0
        self.student_depth[env_ids] = 0.0

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
        if self.num_teacher_actor_obs != self.obs_spec.teacher_actor_dim:
            raise RuntimeError(
                f"Configured teacher actor obs dim {self.num_teacher_actor_obs} does not match "
                f"parkour actor spec {self.obs_spec.teacher_actor_dim}."
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
