import torch
import torch.nn.functional as F

from legged_gym.envs.base.legged_robot_parkour import LeggedRobotParkour


class Go2ParkourStudent(LeggedRobotParkour):
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_teacher_actor_obs = self.cfg.env.num_teacher_actor_obs
        self.student_depth_shape = tuple(self.cfg.env.student_depth_shape)
        self._depth_render_interval = max(1, int(self.cfg.sensor.depth_camera_config.decimation))
        self._depth_noise_level = float(getattr(self.cfg.sensor, "depth_noise_level", 0.0))
        self._use_inferred_depth = bool(
            getattr(self.cfg.sensor.depth_estimation, "enabled", False)
        )

    def _init_buffers(self):
        super()._init_buffers()
        if self._use_inferred_depth:
            self.student_depth = torch.zeros(
                self.num_envs, *self.student_depth_shape,
                device=self.device, dtype=torch.float,
            )
        else:
            self.student_depth = self.simulator.get_depth_images()
            if tuple(self.student_depth.shape[1:]) != self.student_depth_shape:
                raise RuntimeError(
                    f"Simulator depth shape {tuple(self.student_depth.shape[1:])} does not match "
                    f"configured student depth shape {self.student_depth_shape}."
                )
        self._depth_rendered_this_step = False

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.simulator.post_physics_step()
        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if ((self.common_step_counter - 1) % self._depth_render_interval) == 0:
            self.simulator.update_sensors()
            self._refresh_student_depth()
            self._apply_depth_noise()
            self._depth_rendered_this_step = True
        else:
            self._depth_rendered_this_step = False

        self.compute_observations()

        if self.debug:
            self.simulator.draw_debug_vis()
        if self.debug_sensor_images:
            self.simulator.draw_debug_sensor_images()

    def _refresh_student_depth(self):
        """Populate student_depth from the configured source after update_sensors."""
        if not self._use_inferred_depth:
            return
        inferred = self.simulator.get_inferred_depth_images()
        if inferred is not None:
            self.student_depth[:] = self._process_inferred_depth(inferred)

    def _process_inferred_depth(self, inferred_depth):
        """Resize and normalize inferred depth to student_depth_shape [-0.5, 0.5]."""
        _, h, w = self.student_depth_shape
        depth = inferred_depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        dmin = depth.amin(dim=(-2, -1), keepdim=True)
        dmax = depth.amax(dim=(-2, -1), keepdim=True)
        return (depth - dmin) / (dmax - dmin + 1e-6) - 0.5

    def _apply_depth_noise(self):
        if self._depth_noise_level > 0:
            self.student_depth += self._depth_noise_level * 2 * (
                torch.rand(1, device=self.device) - 0.5
            )

    def compute_observations(self):
        prop_obs, scandot_obs = self._compute_proprioception_and_scandots()

        if self.add_noise:
            prop_noise = (2 * torch.rand_like(prop_obs) - 1) * self.noise_scale_vec
            scandot_noise_scale = self.cfg.noise.noise_scales.scandots * self.cfg.noise.noise_level
            scandot_noise = (2 * torch.rand_like(scandot_obs) - 1) * scandot_noise_scale
            self.obs_buf = prop_obs + prop_noise
            self.teacher_actor_obs_buf = torch.cat(
                (prop_obs + prop_noise, scandot_obs + scandot_noise), dim=-1
            )
        else:
            self.obs_buf = prop_obs
            self.teacher_actor_obs_buf = torch.cat((prop_obs, scandot_obs), dim=-1)

    def step(self, actions):
        actions = self._pre_sim_step(actions)
        self.simulator.step(actions)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.teacher_actor_obs_buf = torch.clip(self.teacher_actor_obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = self.teacher_actor_obs_buf

        depth_updated = torch.full(
            (self.num_envs,),
            self._depth_rendered_this_step,
            device=self.device,
            dtype=torch.bool,
        )
        return (
            self.obs_buf,
            self.teacher_actor_obs_buf,
            self.student_depth,
            depth_updated,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, teacher_actor_obs, student_depth, depth_updated, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, teacher_actor_obs, student_depth, depth_updated

    def get_observations(self):
        depth_updated = torch.full(
            (self.num_envs,),
            self._depth_rendered_this_step,
            device=self.device,
            dtype=torch.bool,
        )
        return self.obs_buf, self.teacher_actor_obs_buf, self.student_depth, depth_updated

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.student_depth[env_ids] = 0.0

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
        noise_vec[self.obs_spec.command_slice] = 0.0
        noise_vec[self.obs_spec.gravity_slice] = noise_scales.gravity * noise_level
        noise_vec[self.obs_spec.ang_vel_slice] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[self.obs_spec.dof_pos_slice] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[self.obs_spec.dof_vel_slice] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[self.obs_spec.actions_slice] = 0.0
        noise_vec[self.obs_spec.foot_contacts_slice] = 0.0
        return noise_vec
