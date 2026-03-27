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
        self._depth_buffer_len = int(self.cfg.sensor.depth_camera_config.num_history)

        # Experiment flags (set via cfg.experiment_flags dict by train_experiment.py)
        flags = getattr(self.cfg, "experiment_flags", {}) or {}
        self._experiment_force_ema_norm = flags.get("force_ema_norm", False)
        self._experiment_random_affine_depth = flags.get("random_affine_depth", False)
        self._experiment_norm_mode = flags.get("norm_mode", None)
        self._experiment_ema_alpha = flags.get("ema_alpha", None)
        self._experiment_ema_quantiles = flags.get("ema_quantiles", None)
        self._experiment_invert_relative_depth = flags.get("invert_relative_depth", False)

    def _init_buffers(self):
        super()._init_buffers()
        _, h, w = self.student_depth_shape  # always (1, H, W) for the network
        if self._use_inferred_depth:
            # Manual 2-frame buffer for inferred depth
            self._depth_buffer = torch.zeros(
                self.num_envs, self._depth_buffer_len, h, w,
                device=self.device, dtype=torch.float,
            )
            self._init_ema_buffers()
        elif self._experiment_force_ema_norm:
            # O1-gt-ema: use GT depth from simulator, but re-normalize via EMA
            self._depth_buffer = self.simulator.get_depth_images()
            self._init_ema_buffers()
        else:
            self._depth_buffer = self.simulator.get_depth_images()
            if self._depth_buffer.shape[1] != self._depth_buffer_len:
                raise RuntimeError(
                    f"Simulator depth buffer length {self._depth_buffer.shape[1]} "
                    f"does not match configured num_history={self._depth_buffer_len}."
                )
            expected_hw = (h, w)
            if tuple(self._depth_buffer.shape[2:]) != expected_hw:
                raise RuntimeError(
                    f"Simulator depth resolution {tuple(self._depth_buffer.shape[2:])} "
                    f"does not match configured student depth resolution {expected_hw}."
                )
        # student_depth exposes the previous frame (index 1) when buffer_len>=2,
        # matching extreme-parkour which passes depth_buffer[:, -2].
        # Simulator buffer layout: index 0 = newest, index 1 = previous.
        self._prev_frame_idx = min(1, self._depth_buffer_len - 1)
        self.student_depth = self._depth_buffer[:, self._prev_frame_idx : self._prev_frame_idx + 1]
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

    def _init_ema_buffers(self):
        """Initialize EMA normalization state for inferred or force-EMA depth."""
        self._depth_ema_lo = torch.zeros(1, device=self.device)
        self._depth_ema_hi = torch.ones(1, device=self.device)
        self._depth_ema_initialized = False
        # For meanstd normalization mode
        self._depth_ema_mean = torch.zeros(1, device=self.device)
        self._depth_ema_std = torch.ones(1, device=self.device)

    def _refresh_student_depth(self):
        """Populate depth buffer from the configured source after update_sensors."""
        if self._experiment_force_ema_norm:
            # O1-gt-ema: simulator updated buffer in-place with linear norm.
            # Convert back to meters and re-normalize via EMA.
            near = self.cfg.sensor.depth_camera_config.near_clip
            far = self.cfg.sensor.depth_camera_config.far_clip
            linear_depth = self._depth_buffer.clone()
            # Undo linear norm: linear = (meters - near) / (far - near) - 0.5
            meters = (linear_depth + 0.5) * (far - near) + near
            # Re-normalize the newest frame through the EMA pipeline
            newest = meters[:, 0]  # (N, H, W)
            processed = self._process_inferred_depth(newest)  # (N, 1, H, W)
            self._depth_buffer[:, 0:1] = processed
            if self._depth_buffer_len > 1:
                # Also re-normalize the previous frame
                prev = meters[:, 1]
                processed_prev = self._process_inferred_depth(prev)
                self._depth_buffer[:, 1:2] = processed_prev
            return

        if self._experiment_random_affine_depth:
            # O2-gt-affine: simulator updated buffer in-place with linear norm.
            # Apply random per-batch affine perturbation.
            eps_s = 0.3 * (2 * torch.rand(1, device=self.device) - 1)  # scale noise
            eps_b = 0.2 * (2 * torch.rand(1, device=self.device) - 1)  # bias noise
            self._depth_buffer[:] = ((1 + eps_s) * self._depth_buffer + eps_b).clamp(-0.5, 0.5)
            return

        if not self._use_inferred_depth:
            # Direct depth: simulator already updated self._depth_buffer in-place
            return
        inferred = self.simulator.get_inferred_depth_images()
        if inferred is not None:
            if self._experiment_invert_relative_depth:
                inferred = inferred.max() - inferred
            processed = self._process_inferred_depth(inferred)
            # Shift buffer: copy current (index 0) to previous (index 1), insert new at 0
            if self._depth_buffer_len > 1:
                self._depth_buffer[:, 1:] = self._depth_buffer[:, :-1].clone()
            self._depth_buffer[:, 0:1] = processed

    def _process_inferred_depth(self, inferred_depth):
        """Crop, resize, and normalize inferred depth to (N, 1, H, W) in [-0.5, 0.5].

        Monocular depth estimators output arbitrary absolute scale (even "metric"
        models are miscalibrated on synthetic renders).  We use EMA-smoothed p2/p98
        percentiles across batches to give a stable, consistent normalization that
        preserves spatial structure without per-frame jitter.

        Supports experiment flag overrides for normalization mode, EMA alpha, and
        quantile bounds.
        """
        depth_cfg = self.cfg.sensor.depth_camera_config
        _, h, w = self.student_depth_shape

        # Crop to match GT depth FOV (skip if already correct size, e.g. O1-gt-ema)
        height, width = inferred_depth.shape[-2:]
        top = int(getattr(depth_cfg, "crop_top", 0))
        bottom = int(getattr(depth_cfg, "crop_bottom", 0))
        left = int(getattr(depth_cfg, "crop_left", 0))
        right = int(getattr(depth_cfg, "crop_right", 0))
        end_h = height - bottom if bottom > 0 else height
        end_w = width - right if right > 0 else width
        if (end_h - top) != height or (end_w - left) != width:
            depth = inferred_depth[:, top:end_h, left:end_w]
        else:
            depth = inferred_depth

        # Select normalization mode
        norm_mode = self._experiment_norm_mode or "ema"
        alpha = self._experiment_ema_alpha or 0.02

        if norm_mode == "affine_fit":
            # N1-affine-fit: pre-computed affine alignment from metric outdoor
            depth = 0.088 * depth + 0.858
            depth = depth.clamp(0, 2.0) / 2.0 - 0.5
        elif norm_mode == "meanstd":
            # N5-meanstd: running mean/std normalization
            batch_mean = depth.mean()
            batch_std = depth.std().clamp(min=1e-3)
            if not self._depth_ema_initialized:
                self._depth_ema_mean.fill_(batch_mean.item())
                self._depth_ema_std.fill_(batch_std.item())
                self._depth_ema_initialized = True
            else:
                self._depth_ema_mean.lerp_(batch_mean, alpha)
                self._depth_ema_std.lerp_(batch_std, alpha)
            depth = torch.tanh((depth - self._depth_ema_mean) / (2 * self._depth_ema_std)) * 0.5
        else:
            # Default EMA percentile normalization
            q_lo, q_hi = self._experiment_ema_quantiles or (0.02, 0.98)
            batch_lo = torch.quantile(depth, q_lo)
            batch_hi = torch.quantile(depth, q_hi)
            if not self._depth_ema_initialized:
                self._depth_ema_lo.fill_(batch_lo.item())
                self._depth_ema_hi.fill_(batch_hi.item())
                self._depth_ema_initialized = True
            else:
                self._depth_ema_lo.lerp_(batch_lo, alpha)
                self._depth_ema_hi.lerp_(batch_hi, alpha)
            span = (self._depth_ema_hi - self._depth_ema_lo).clamp(min=1e-3)
            depth = ((depth - self._depth_ema_lo) / span).clamp(0, 1) - 0.5

        depth = depth.unsqueeze(1)
        if depth.shape[-2:] != (h, w):
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        return depth

    def _apply_depth_noise(self):
        if self._depth_noise_level > 0:
            self._depth_buffer += self._depth_noise_level * 2 * (
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
            self._depth_buffer[env_ids] = 0.0

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
