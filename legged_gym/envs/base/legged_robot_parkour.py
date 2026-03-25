import torch

from legged_gym import SIMULATOR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.parkour_observation import ParkourObservationSpec
from legged_gym.utils.math_utils import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, torch_rand_float
from legged_gym.utils.parkour_terrain import PARKOUR_FAMILY_IDS


class LeggedRobotParkour(LeggedRobot):
    SECTION_STAIRS = 2

    def _parse_cfg(self, cfg):
        if SIMULATOR != "genesis":
            raise RuntimeError("LeggedRobotParkour is only supported with SIMULATOR=genesis.")
        super()._parse_cfg(cfg)
        self.parkour_cfg = self.cfg.terrain.parkour

    def _init_buffers(self):
        self.obs_spec = ParkourObservationSpec.from_cfg(self.cfg)
        super()._init_buffers()
        if self.simulator.lane_waypoints is None:
            raise RuntimeError("LeggedRobotParkour requires simulator-owned parkour lane metadata.")
        if self.simulator.scandot_heights is None:
            raise RuntimeError("LeggedRobotParkour requires scandot terrain observations to be enabled.")
        self._validate_configured_observation_dims()

        self.commands_scale = torch.tensor(
            [
                self.obs_scales.goal_pos,
                self.obs_scales.goal_pos,
                self.obs_scales.goal_pos,
                self.obs_scales.goal_pos,
                self.obs_scales.heading,
                self.obs_scales.heading,
                self.obs_scales.goal_speed,
            ],
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )
        self.tile_half_extent = torch.tensor(
            [0.5 * self.cfg.terrain.terrain_length, 0.5 * self.cfg.terrain.terrain_width],
            device=self.device,
            dtype=torch.float,
        )
        self.env_ids_long = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        self.goal_speed_targets = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.active_waypoint_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.waypoint_dwell_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.waypoints_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_along_course = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.progress_delta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.progress_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.course_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.waypoint_reached_events = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.success_events = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.local_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.motion_penalty_scale = torch.ones(self.num_envs, device=self.device, dtype=torch.float)

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= self.simulator.parkour_out_of_lane_buf
        self.reset_buf |= self.simulator.global_out_of_bounds_buf
        self.reset_buf |= self.course_success

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        progress_before_reset = self.progress_ratio[env_ids].clone()
        waypoints_before_reset = self.waypoints_reached[env_ids].clone().float()
        success_before_reset = self.course_success[env_ids].clone().float()
        family_before_reset = self.simulator.lane_family[env_ids].clone().float()
        row_before_reset = self.simulator.lane_difficulty_row[env_ids].clone().float()
        out_of_lane_before_reset = self.simulator.parkour_out_of_lane_buf[env_ids].clone().float()
        global_out_of_bounds_before_reset = self.simulator.global_out_of_bounds_buf[env_ids].clone().float()

        super().reset_idx(env_ids)

        self.extras["episode"]["progress_ratio"] = torch.mean(progress_before_reset)
        self.extras["episode"]["waypoints_reached"] = torch.mean(waypoints_before_reset)
        self.extras["episode"]["success"] = torch.mean(success_before_reset)
        self.extras["episode"]["terrain_family"] = torch.mean(family_before_reset)
        self.extras["episode"]["terrain_row"] = torch.mean(row_before_reset)
        self.extras["episode"]["out_of_lane"] = torch.mean(out_of_lane_before_reset)
        self.extras["episode"]["global_out_of_bounds"] = torch.mean(global_out_of_bounds_before_reset)

        self.active_waypoint_idx[env_ids] = 0
        self.waypoint_dwell_steps[env_ids] = 0
        self.waypoints_reached[env_ids] = 0
        self.progress_along_course[env_ids] = 0.0
        self.progress_delta[env_ids] = 0.0
        self.progress_ratio[env_ids] = 0.0
        self.course_success[env_ids] = False
        self.waypoint_reached_events[env_ids] = 0.0
        self.success_events[env_ids] = 0.0
        self._update_local_base_positions(env_ids)
        self._update_current_section_state(env_ids)
        self._update_command_targets(env_ids)
        self.simulator.update_scandot_heights()

    def _compute_proprioception_and_scandots(self):
        foot_contacts = (
            self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 1.0
        ).float()
        scandots_cfg = self.cfg.terrain.scandots
        scandot_obs = torch.clip(
            self.simulator.base_pos[:, 2:3] - scandots_cfg.base_height_offset - self.simulator.scandot_heights,
            scandots_cfg.clip_min,
            scandots_cfg.clip_max,
        )
        prop_obs = torch.cat(
            (
                self.commands * self.commands_scale,
                self.simulator.projected_gravity,
                self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
                self.simulator.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                foot_contacts,
            ),
            dim=-1,
        )
        return prop_obs, scandot_obs

    def compute_observations(self):
        prop_obs, scandot_obs = self._compute_proprioception_and_scandots()
        actor_obs = torch.cat((prop_obs, scandot_obs), dim=-1)
        self._validate_runtime_observation_dims(actor_obs, self.obs_spec.actor_dim, "actor")

        if self.num_privileged_obs is not None:
            privileged_parts = (
                actor_obs,
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                self._get_privileged_dynamics(),
                self.simulator.link_contact_states,
            )
            self.privileged_obs_buf = torch.cat(privileged_parts, dim=-1)
            self._validate_runtime_observation_dims(self.privileged_obs_buf, self.obs_spec.critic_dim, "critic")

        self.obs_buf = actor_obs
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _post_physics_step_callback(self):
        self.waypoint_reached_events.zero_()
        self.success_events.zero_()

        self._update_local_base_positions()
        self._update_current_section_state()
        self.simulator.update_scandot_heights()
        self._update_command_targets()
        advance_ids = self._update_parkour_progress()
        if len(advance_ids) > 0:
            self._update_command_targets(advance_ids)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self.simulator.push_robots()
        if self.cfg.domain_rand.push_links and (self.common_step_counter % self.cfg.domain_rand.push_links_interval == 0):
            self.simulator.push_links()

    def _resample_commands(self, env_ids):
        low, high = self.cfg.commands.goal_speed_range
        self.goal_speed_targets[env_ids] = torch_rand_float(
            low, high, (len(env_ids), 1), self.device
        ).squeeze(1)

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return

        move_up = self.course_success[env_ids] | (
            self.progress_ratio[env_ids] >= self.parkour_cfg.curriculum_progress_up_threshold
        )
        move_down = (~move_up) & (
            (self.progress_ratio[env_ids] <= self.parkour_cfg.curriculum_progress_down_threshold)
            | self.simulator.parkour_out_of_lane_buf[env_ids]
            | self.simulator.global_out_of_bounds_buf[env_ids]
            | (self.fail_buf[env_ids] > 0)
        )
        self.simulator.update_terrain_curriculum(env_ids, move_up, move_down)

    def _reset_root_states(self, env_ids):
        lane_spawn_pose = self.simulator.lane_spawn_pose[env_ids]
        safe_spawn_region = self.simulator.lane_safe_spawn_region[env_ids]

        spawn_local_x = torch_rand_float(
            0.0, 1.0, (len(env_ids), 1), self.device
        ).squeeze(1)
        spawn_local_y = torch_rand_float(
            0.0, 1.0, (len(env_ids), 1), self.device
        ).squeeze(1)
        spawn_local_x = safe_spawn_region[:, 0] + (safe_spawn_region[:, 1] - safe_spawn_region[:, 0]) * spawn_local_x
        spawn_local_y = safe_spawn_region[:, 2] + (safe_spawn_region[:, 3] - safe_spawn_region[:, 2]) * spawn_local_y

        tile_origin_xy = self.simulator.env_origins[env_ids, :2] - self.tile_half_extent

        base_pos = torch.zeros((len(env_ids), 3), device=self.device, dtype=torch.float)
        base_pos[:, 0] = tile_origin_xy[:, 0] + spawn_local_x
        base_pos[:, 1] = tile_origin_xy[:, 1] + spawn_local_y
        base_pos[:, 2] = lane_spawn_pose[:, 2] + self.simulator.base_init_pos[2]

        roll = torch_rand_float(
            -self.cfg.init_state.roll_random_scale,
            self.cfg.init_state.roll_random_scale,
            (len(env_ids), 1),
            self.device,
        ).squeeze(1)
        pitch = torch_rand_float(
            -self.cfg.init_state.pitch_random_scale,
            self.cfg.init_state.pitch_random_scale,
            (len(env_ids), 1),
            self.device,
        ).squeeze(1)
        yaw = torch_rand_float(
            -self.cfg.init_state.yaw_random_scale,
            self.cfg.init_state.yaw_random_scale,
            (len(env_ids), 1),
            self.device,
        ).squeeze(1)
        base_quat = quat_from_euler_xyz(roll, pitch, yaw)

        base_lin_vel = torch.zeros((len(env_ids), 3), device=self.device, dtype=torch.float)
        base_ang_vel = torch.zeros((len(env_ids), 3), device=self.device, dtype=torch.float)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _reset_dofs(self, env_ids):
        dof_pos = torch.zeros(
            (len(env_ids), self.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        dof_vel = torch.zeros(
            (len(env_ids), self.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
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
        noise_vec[self.obs_spec.scandots_slice] = noise_scales.scandots * noise_level
        return noise_vec

    def _get_privileged_dynamics(self):
        return torch.cat(
            (
                self.simulator.dr_friction_values - self.friction_value_offset,
                self.simulator.dr_added_base_mass,
                self.simulator.dr_base_com_bias,
                self.simulator.dr_rand_push_vels[:, :2],
                self.simulator.dr_kp_scale - self.kp_scale_offset,
                self.simulator.dr_kd_scale - self.kd_scale_offset,
            ),
            dim=-1,
        )

    def _resolve_env_ids(self, env_ids=None):
        if env_ids is None:
            return self.env_ids_long
        return env_ids.to(dtype=torch.long, device=self.device)

    def _update_local_base_positions(self, env_ids=None):
        env_ids = self._resolve_env_ids(env_ids)
        tile_origin_xy = self.simulator.env_origins[:, :2] - self.tile_half_extent
        self.local_base_pos[env_ids, :2] = self.simulator.base_pos[env_ids, :2] - tile_origin_xy[env_ids]
        self.local_base_pos[env_ids, 2] = self.simulator.base_pos[env_ids, 2]

    def _update_current_section_state(self, env_ids=None):
        env_ids = self._resolve_env_ids(env_ids)
        section_bounds = self.simulator.lane_section_bounds[env_ids]
        valid_sections = section_bounds[:, :, 1] > section_bounds[:, :, 0]
        local_x = self.local_base_pos[env_ids, 0].unsqueeze(1)
        within_section = valid_sections & (local_x >= section_bounds[:, :, 0]) & (local_x <= section_bounds[:, :, 1])
        section_idx = within_section.long().argmax(dim=1)
        self.motion_penalty_scale[env_ids] = 1.0
        if within_section.any():
            matched_ids = env_ids[within_section.any(dim=1)]
            matched_section_idx = section_idx[within_section.any(dim=1)]
            matched_tags = self.simulator.lane_section_tags[matched_ids, matched_section_idx]
            dynamic_motion = self.simulator.lane_jump_expected_mask[matched_ids, matched_section_idx] | (
                matched_tags == self.SECTION_STAIRS
            )
            self.motion_penalty_scale[matched_ids[dynamic_motion]] = self.cfg.rewards.dynamic_motion_penalty_scale

    def _update_command_targets(self, env_ids=None):
        env_ids = self._resolve_env_ids(env_ids)
        current_goal_local, next_goal_local = self._get_goal_pair_local(env_ids)
        current_goal_delta_world = self._lane_local_to_world(current_goal_local, env_ids) - self.simulator.base_pos[env_ids]
        next_goal_delta_world = self._lane_local_to_world(next_goal_local, env_ids) - self.simulator.base_pos[env_ids]

        current_goal_delta_base = quat_rotate_inverse(self.simulator.base_quat[env_ids], current_goal_delta_world)
        next_goal_delta_base = quat_rotate_inverse(self.simulator.base_quat[env_ids], next_goal_delta_world)

        current_goal_heading_world = torch.atan2(current_goal_delta_world[:, 1], current_goal_delta_world[:, 0])
        next_goal_heading_world = torch.atan2(next_goal_delta_world[:, 1], next_goal_delta_world[:, 0])
        current_goal_heading_error = wrap_to_pi(current_goal_heading_world - self.simulator.base_euler[env_ids, 2])
        next_goal_heading_error = wrap_to_pi(next_goal_heading_world - self.simulator.base_euler[env_ids, 2])

        self.commands[env_ids, 0] = current_goal_delta_base[:, 0]
        self.commands[env_ids, 1] = current_goal_delta_base[:, 1]
        self.commands[env_ids, 2] = next_goal_delta_base[:, 0]
        self.commands[env_ids, 3] = next_goal_delta_base[:, 1]
        self.commands[env_ids, 4] = current_goal_heading_error
        self.commands[env_ids, 5] = next_goal_heading_error
        self.commands[env_ids, 6] = self.goal_speed_targets[env_ids]

    def _update_parkour_progress(self):
        spawn_xy = self.simulator.lane_spawn_pose[:, :2]
        terminal_xy = self.simulator.lane_terminal_goal[:, :2]
        course_delta = terminal_xy - spawn_xy
        course_length = torch.norm(course_delta, dim=1).clamp(min=1e-6)
        course_direction = course_delta / course_length.unsqueeze(1)

        progress = torch.sum((self.local_base_pos[:, :2] - spawn_xy) * course_direction, dim=1)
        progress = torch.clamp(progress, min=0.0)
        self.progress_delta[:] = (progress - self.progress_along_course) / self.dt
        self.progress_along_course[:] = progress
        self.progress_ratio[:] = torch.clamp(progress / course_length, 0.0, 1.25)

        reached_goal = torch.norm(self.commands[:, :2], dim=1) <= self.parkour_cfg.waypoint_radius
        self.waypoint_dwell_steps = torch.where(
            reached_goal,
            self.waypoint_dwell_steps + 1,
            torch.zeros_like(self.waypoint_dwell_steps),
        )
        reached_waypoint = self.waypoint_dwell_steps >= self.parkour_cfg.waypoint_dwell_steps
        reached_waypoint &= ~self.course_success
        advance_ids = self.env_ids_long[:0]

        if reached_waypoint.any():
            reached_ids = reached_waypoint.nonzero(as_tuple=False).flatten()
            last_waypoint = self.active_waypoint_idx[reached_ids] >= (
                self.simulator.lane_waypoint_counts[reached_ids] - 1
            )

            self.waypoint_reached_events[reached_ids] = 1.0
            self.waypoints_reached[reached_ids] += 1
            self.waypoint_dwell_steps[reached_ids] = 0

            if (~last_waypoint).any():
                advance_ids = reached_ids[~last_waypoint]
                self.active_waypoint_idx[advance_ids] += 1

            if last_waypoint.any():
                success_ids = reached_ids[last_waypoint]
                self.course_success[success_ids] = True
                self.success_events[success_ids] = 1.0

        return advance_ids

    def _get_goal_pair_local(self, env_ids=None):
        env_ids = self._resolve_env_ids(env_ids)
        waypoint_counts = self.simulator.lane_waypoint_counts[env_ids]
        current_idx = torch.minimum(self.active_waypoint_idx[env_ids], waypoint_counts - 1)
        next_idx = torch.minimum(current_idx + 1, waypoint_counts - 1)
        return (
            self.simulator.lane_waypoints[env_ids, current_idx],
            self.simulator.lane_waypoints[env_ids, next_idx],
        )

    def _lane_local_to_world(self, local_points, env_ids=None):
        env_ids = self._resolve_env_ids(env_ids)
        tile_origin_xy = self.simulator.env_origins[env_ids, :2] - self.tile_half_extent
        world_points = local_points.clone()
        world_points[:, 0] = local_points[:, 0] + tile_origin_xy[:, 0]
        world_points[:, 1] = local_points[:, 1] + tile_origin_xy[:, 1]
        return world_points

    def _reward_lin_vel_z(self):
        return super()._reward_lin_vel_z() * self.motion_penalty_scale

    def _reward_ang_vel_xy(self):
        return super()._reward_ang_vel_xy() * self.motion_penalty_scale

    def _reward_orientation(self):
        return super()._reward_orientation() * self.motion_penalty_scale

    def _reward_tracking_goal_vel(self):
        goal_distance = torch.norm(self.commands[:, :2], dim=1, keepdim=True)
        desired_goal_velocity = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        non_zero = goal_distance[:, 0] > 1e-6
        desired_goal_velocity[non_zero] = (
            self.commands[non_zero, :2] / goal_distance[non_zero]
        ) * self.commands[non_zero, 6:7]
        velocity_error = torch.sum(torch.square(desired_goal_velocity - self.simulator.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-velocity_error / self.cfg.rewards.goal_velocity_tracking_sigma)

    def _reward_tracking_goal_heading(self):
        return torch.exp(-torch.square(self.commands[:, 4]) / self.cfg.rewards.goal_heading_tracking_sigma)

    def _reward_progress_along_course(self):
        return torch.clamp(
            self.progress_delta,
            min=self.cfg.rewards.progress_speed_clip[0],
            max=self.cfg.rewards.progress_speed_clip[1],
        )

    def _reward_waypoint_reached_bonus(self):
        return self.waypoint_reached_events

    def _reward_success_bonus(self):
        return self.success_events

    def _reward_feet_stumble(self):
        stumble = torch.any(
            torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :2], dim=2)
            > 4 * torch.abs(self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2]),
            dim=1,
        )
        return stumble.float()

    def _reward_foot_clearance(self):
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm
            * torch.square(
                self.simulator.feet_pos[:, :, 2]
                - torch.mean(self.simulator.height_around_feet, dim=-1)
                - self.cfg.rewards.foot_clearance_target
                - self.cfg.rewards.foot_height_offset
            ),
            dim=-1,
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)

    def _reward_hip_pos(self):
        hip_joint_indices = self.cfg.asset.hip_joint_indices
        return torch.sum(
            torch.square(
                self.simulator.dof_pos[:, hip_joint_indices] - self.simulator.default_dof_pos[:, hip_joint_indices]
            ),
            dim=-1,
        )

    def _reward_feet_edge(self):
        foot_contacts = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 1.0
        local_foot_xy = self.simulator.feet_pos[:, :, :2] - (
            self.simulator.env_origins[:, None, :2] - self.tile_half_extent
        )
        px = torch.round(local_foot_xy[:, :, 0] / self.cfg.terrain.horizontal_scale).long()
        py = torch.round(local_foot_xy[:, :, 1] / self.cfg.terrain.horizontal_scale).long()
        px = torch.clamp(px, 0, self.simulator.lane_edge_masks.shape[1] - 1)
        py = torch.clamp(py, 0, self.simulator.lane_edge_masks.shape[2] - 1)
        env_ids = self.env_ids_long.unsqueeze(1).expand_as(px)
        edge_hits = self.simulator.lane_edge_masks[env_ids, px, py]
        obstacle_family_mask = (
            (self.simulator.lane_family == PARKOUR_FAMILY_IDS["hurdle_block"])
            | (self.simulator.lane_family == PARKOUR_FAMILY_IDS["gap"])
            | (self.simulator.lane_family == -1)  # mixed gauntlet
        )
        row_mask = self.simulator.lane_difficulty_row >= 1
        active_mask = (obstacle_family_mask & row_mask).float().unsqueeze(1)
        return torch.sum(edge_hits.float() * foot_contacts.float() * active_mask, dim=1)

    def _validate_configured_observation_dims(self):
        if self.num_obs != self.obs_spec.actor_dim:
            raise RuntimeError(
                f"Configured actor observation dim {self.num_obs} does not match parkour spec {self.obs_spec.actor_dim}."
            )
        if self.num_privileged_obs is not None and self.num_privileged_obs != self.obs_spec.critic_dim:
            raise RuntimeError(
                f"Configured critic observation dim {self.num_privileged_obs} does not match parkour spec {self.obs_spec.critic_dim}."
            )

    def _validate_runtime_observation_dims(self, obs_tensor, expected_dim, obs_name):
        if obs_tensor.shape[1] != expected_dim:
            raise RuntimeError(
                f"Runtime {obs_name} observation dim {obs_tensor.shape[1]} does not match expected dim {expected_dim}."
            )
