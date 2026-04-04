from legged_gym import *
from legged_gym.perception import DepthEstimatorError, create_depth_estimator
from legged_gym.simulator.simulator import Simulator
import cv2 as cv
import torch
import torch.nn.functional as F
import numpy as np
import os
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
if SIMULATOR == "genesis":
    import genesis as gs
    import genesis.utils.geom as gu

""" ********** Genesis Simulator ********** """
class GenesisSimulator(Simulator):
    def __init__(self, cfg, sim_params: dict, device, headless):
        self._sim_params = sim_params
        self._debug_vis_objects = []
        self._camera_window_initialized = False
        self._depth_estimator = None
        self._depth_estimator_backend_name = None
        self._depth_estimator_latency_ms = None
        self._depth_estimator_step = 0
        super().__init__(cfg, sim_params, device, headless)
    
    #----- Public methods -----#
    def step(self, actions):
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._last_base_ang_vel[:] = self._base_ang_vel[:]
        self._last_feet_vel[:] = self._feet_vel[:]
        self._last_dof_vel[:] = self._dof_vel[:]
        for substep_idx in range(self._cfg.control.decimation):
            update_viewer = not self._headless and substep_idx == self._cfg.control.decimation - 1
            self._torques = self._compute_torques(actions)
            self._robot.control_dofs_force(
                self._torques, self._dof_indices)
            self._scene.step(update_visualizer=update_viewer, refresh_visualizer=update_viewer)
            self._dof_pos[:] = self._robot.get_dofs_position(self._dof_indices)
            self._dof_vel[:] = self._robot.get_dofs_velocity(self._dof_indices)

    def post_physics_step(self):
        # prepare quantities
        self._base_pos[:] = self._robot.get_pos()
        self._update_bounds_state()
        base_quat_gs = self._robot.get_quat()
        base_vel = self._robot.get_vel()
        base_ang = self._robot.get_ang()
        self._dof_pos[:] = self._robot.get_dofs_position(self._dof_indices)
        self._dof_vel[:] = self._robot.get_dofs_velocity(self._dof_indices)
        self._link_contact_forces[:] = self._robot.get_links_net_contact_force()
        links_pos = self._robot.get_links_pos()
        links_quat_gs = self._robot.get_links_quat()
        links_vel = self._robot.get_links_vel()
        links_ang = self._robot.get_links_ang()

        self._base_quat_gs[:] = base_quat_gs
        self._base_quat[:, -1] = base_quat_gs[:, 0]   # wxyz to xyzw
        self._base_quat[:, :3] = base_quat_gs[:, 1:4] # wxyz to xyzw

        self._base_euler[:] = get_euler_xyz(self._base_quat)
        self._base_lin_vel[:] = quat_rotate_inverse(self._base_quat, base_vel)
        self._base_ang_vel[:] = quat_rotate_inverse(self._base_quat, base_ang)
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)
        self._rigid_body_states[:, :, :3] = links_pos
        self._rigid_body_states[:, :, 3:6] = links_quat_gs[:, :, 1:4]
        self._rigid_body_states[:, :, 6] = links_quat_gs[:, :, 0]
        self._rigid_body_states[:, :, 7:10] = links_vel
        self._rigid_body_states[:, :, 10:13] = links_ang
        self._feet_pos[:] = links_pos[:, self._feet_indices, :]
        self._feet_vel[:] = links_vel[:, self._feet_indices, :]
        self._key_body_pos[:] = links_pos[:, self._key_body_indices, :]
        # Link contact state
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = 1. * (torch.norm(
                self._link_contact_forces[:, self._contact_state_link_indices, :], dim=-1) > 1.)
        # update terrain heights info
        if self._cfg.terrain.measure_heights:
            self._update_surrounding_heights()
            if self._cfg.terrain.obtain_terrain_info_around_feet:
                self._calc_terrain_info_around_feet()
        
    def reset_idx(self, env_ids):
        # domain randomization
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self._cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(env_ids)
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(env_ids)
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(env_ids)
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(env_ids)
        
        self._last_dof_vel[env_ids] = 0.
        self._last_feet_vel[env_ids] = 0.
        self._last_base_lin_vel[env_ids] = 0.
        self._last_base_ang_vel[env_ids] = 0.
        self._parkour_out_of_lane_buf[env_ids] = False
        self._global_out_of_bounds_buf[env_ids] = False

    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        self._dof_pos[env_ids] = dof_pos[:]
        self._dof_vel[env_ids] = dof_vel[:]
        
        self._robot.set_dofs_position(
            position=self._dof_pos[env_ids],
            dofs_idx_local=self._dof_indices,
            zero_velocity=True,
            envs_idx=env_ids,
        )
        self._robot.zero_all_dofs_velocity(env_ids)

    def reset_root_states(self, 
                          env_ids, 
                          base_pos, 
                          base_quat, 
                          base_lin_vel_w, 
                          base_ang_vel_w):
        # base pos
        self._base_pos[env_ids, :] = base_pos[:]
        self._robot.set_pos(
            self._base_pos[env_ids], zero_velocity=False, envs_idx=env_ids)

        # base quat
        self._base_quat[env_ids, :] = base_quat[:]
        self._base_quat_gs[env_ids, 0] = self._base_quat[env_ids, 3]  # xyzw to wxyz
        self._base_quat_gs[env_ids, 1:4] = self._base_quat[env_ids, 0:3] # xyzw to wxyz
        self._robot.set_quat(
            self._base_quat_gs[env_ids], zero_velocity=False, envs_idx=env_ids)
        self._robot.zero_all_dofs_velocity(env_ids)

        # update projected gravity
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)

        # reset root states - velocity
        base_vel = torch.concat(
            [base_lin_vel_w, base_ang_vel_w], dim=1)
        self._robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[
                                     0, 1, 2, 3, 4, 5], envs_idx=env_ids)
        self._base_lin_vel[env_ids] = quat_rotate_inverse(self._base_quat[env_ids], self._robot.get_vel()[env_ids])
        self._base_ang_vel[env_ids] = quat_rotate_inverse(self._base_quat[env_ids], self._robot.get_ang()[env_ids])

    def update_sensors(self):
        sensor_frames = {}
        render_depth = self._cfg.sensor.add_depth
        render_rgb = self._requires_rgb_camera_stream()

        depth_frames, rgb_frames = self._render_scene_camera_streams(
            render_depth=render_depth,
            render_rgb=render_rgb,
        )
        if depth_frames is not None:
            sensor_frames["depth"] = self._update_depth_images(depth_frames)
        if rgb_frames is not None:
            self._update_rgb_images(rgb_frames)
            if getattr(self._cfg.sensor, "add_rgb", False):
                sensor_frames["rgb"] = self._rgb_images
        if getattr(self._cfg.sensor.depth_estimation, "enabled", False):
            inferred_depth_frames = self._update_inferred_depth_images(self._camera_render_env_indices)
            if inferred_depth_frames is not None:
                sensor_frames["inferred_depth"] = inferred_depth_frames
        return sensor_frames or None

    def update_terrain_curriculum(self, env_ids, move_up, move_down):
        if self._terrain_uses_parkour_metadata():
            parkour_cfg = self._cfg.terrain.parkour
            if parkour_cfg.force_row is None:
                self._terrain_levels[env_ids] += 1 * move_up - 1 * move_down
                self._terrain_levels[env_ids] = torch.where(
                    self._terrain_levels[env_ids] >= self._max_terrain_level,
                    torch.randint_like(self._terrain_levels[env_ids], self._max_terrain_level),
                    torch.clip(self._terrain_levels[env_ids], 0),
                )
            else:
                self._terrain_levels[env_ids] = int(parkour_cfg.force_row)
            if parkour_cfg.force_family is not None:
                forced_family = str(parkour_cfg.force_family)
                if forced_family not in self._terrain.family_bucket_ids:
                    raise ValueError(
                        f"Forced parkour family '{forced_family}' is not available in terrain columns "
                        f"{list(self._terrain.family_bucket_ids.keys())}"
                    )
                self._terrain_family_types[env_ids] = int(self._terrain.family_bucket_ids[forced_family])
            self._assign_variant_columns_from_family_buckets(env_ids)
            self._refresh_lane_metadata(env_ids)
            return

        self._terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self._terrain_levels[env_ids] = torch.where(self._terrain_levels[env_ids] >= self._max_terrain_level,
                                                   torch.randint_like(
                                                       self._terrain_levels[env_ids], self._max_terrain_level),
                                                   torch.clip(self._terrain_levels[env_ids], 0))  # (the minumum level is zero)
        self._env_origins[env_ids] = self._terrain_origins[self._terrain_levels[env_ids],
            self._terrain_types[env_ids]]
        self._refresh_lane_metadata(env_ids)

    def push_robots(self):
        max_push_vel_xy = self._cfg.domain_rand.max_push_vel_xy
        # in Genesis, base link also has DOF, it's 6DOF if not fixed.
        dofs_vel = self._robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = torch_rand_float(-max_push_vel_xy,
                                     max_push_vel_xy, (self._num_envs, 2), self._device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        dofs_vel[:, :2] += push_vel
        self._robot.set_dofs_velocity(dofs_vel)
    
    def push_links(self):
        max_force = self._cfg.domain_rand.max_push_force
        # apply random forces to the links of the robot
        push_force = torch.rand((self._num_envs, self._robot.n_links, 3), 
                                device=self._device) * 2 * max_force - max_force
        all_link_idx = [link.idx - self._robot.link_start for link in self._robot.links]
        self._scene.sim.rigid_solver.apply_links_external_force(
            push_force,
            links_idx=all_link_idx
        )

    def draw_debug_vis(self,
                       ref_key_body_pos=None):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # # draw height points
        # if not self._cfg.terrain.measure_heights:
        #     return
        self._clear_debug_vis_objects()
        if self._cfg.env.debug_draw_key_body_points:
            self._draw_key_body_points(ref_key_body_pos)

    def draw_debug_sensor_images(self):
        if not (
            self._cfg.sensor.add_depth
            or getattr(self._cfg.sensor, "add_rgb", False)
            or getattr(self._cfg.sensor.depth_estimation, "enabled", False)
        ):
            return None
        return self._draw_debug_sensor_images()

    def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray):
        if self._scene.viewer is None:
            return
        self._scene.viewer.set_camera_pose(pos=eye, lookat=target)

    def enable_viewer_follow(self):
        if self._scene.viewer is None:
            return
        self._scene.viewer.follow_entity(self._robot, smoothing=0.6, fix_orientation=False)
    
    #----- Protected methods -----#
    def _parse_cfg(self):
        self._debug = self._cfg.env.debug
        self._control_dt = self._cfg.sim.dt * self._cfg.control.decimation
        self._batch_dofs_links_info = self._cfg.domain_rand.randomize_joint_armature or \
                self._cfg.domain_rand.randomize_joint_friction or \
                self._cfg.domain_rand.randomize_joint_damping
        if not self._headless:
            rendered_envs_idx = list(self._cfg.viewer.rendered_envs_idx)
            rendered_envs_idx = [
                env_idx for env_idx in rendered_envs_idx if 0 <= env_idx < self._cfg.env.num_envs
            ]
            if not rendered_envs_idx and self._cfg.env.num_envs > 0:
                rendered_envs_idx = [0]
            self._cfg.viewer.rendered_envs_idx = rendered_envs_idx
            self._cfg.viewer.ref_env = min(
                max(int(self._cfg.viewer.ref_env), 0),
                max(self._cfg.env.num_envs - 1, 0),
            )
        self._rendered_env_indices = list(self._cfg.viewer.rendered_envs_idx)
        _needs_all_env_cameras = (
            self._cfg.sensor.add_depth
            or getattr(self._cfg.sensor.depth_estimation, "enabled", False)
        )
        if _needs_all_env_cameras:
            self._camera_render_env_indices = list(range(self._cfg.env.num_envs))
        elif self._uses_batch_camera_rendering():
            self._camera_render_env_indices = list(self._rendered_env_indices)
        else:
            self._camera_render_env_indices = []
    def _create_sim(self):
        enable_self_collision = not self._cfg.asset.self_collisions
        enable_visual_geometry = self._requires_visual_geometry()

        # create scene
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self._sim_params["dt"],
                substeps=self._sim_params["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                res=getattr(self._cfg.viewer, "resolution", None),
                max_FPS=getattr(self._cfg.viewer, "max_fps", None),
                camera_pos=np.array(self._cfg.viewer.pos),
                camera_lookat=np.array(self._cfg.viewer.lookat),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=self._camera_render_env_indices or self._cfg.viewer.rendered_envs_idx,
                shadow=False,
                ),
            rigid_options=gs.options.RigidOptions(
                dt=self._sim_params["dt"],
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=enable_self_collision,
                max_collision_pairs=self._cfg.sim.max_collision_pairs,
                IK_max_targets=self._cfg.sim.IK_max_targets,
                batch_dofs_info=self._batch_dofs_links_info,
                batch_links_info=self._batch_dofs_links_info,
            ),
            renderer=gs.renderers.BatchRenderer(use_rasterizer=True) if self._uses_batch_camera_rendering() else None,
            show_viewer=not self._headless,
        )

        # add terrain
        mesh_type = self._cfg.terrain.mesh_type
        if mesh_type == 'plane':
            self._gs_terrain = self._scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/plane/plane.urdf", 
                    fixed=True,
                    visualization=enable_visual_geometry)
                )
        elif mesh_type == 'heightfield':
            self._terrain = Terrain(self._cfg.terrain)
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            raise NotImplementedError("Trimesh terrain is not validated yet in Genesis, please use heightfield for now.")
            self._terrain = Terrain(self._cfg.terrain)
            self._create_trimesh()
        else:
            raise ValueError(f"Unsupported terrain mesh type: {mesh_type}")
        self._gs_terrain.set_friction(self._cfg.terrain.static_friction)
        # specify the boundary of the heightfield
        self._terrain_x_range = torch.zeros(2, device=self._device)
        self._terrain_y_range = torch.zeros(2, device=self._device)
        if self._cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            # give a small margin(1.0m)
            self._terrain_x_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_x_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_rows * self._cfg.terrain.terrain_length - 1.0
            self._terrain_y_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_y_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_cols * self._cfg.terrain.terrain_width - 1.0
        elif self._cfg.terrain.mesh_type == 'plane':  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self._terrain_x_range[0] = -self._cfg.terrain.plane_length/2+1
            self._terrain_x_range[1] = self._cfg.terrain.plane_length/2-1
            # the plane is a square
            self._terrain_y_range[0] = -self._cfg.terrain.plane_length/2+1
            self._terrain_y_range[1] = self._cfg.terrain.plane_length/2-1

    def _create_envs(self):
        # Create envs
        asset_path = self._cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self._robot = self._scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links=True,
                links_to_keep=self._cfg.asset.links_to_keep,
                pos=np.array(self._cfg.init_state.pos),
                quat=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
                fixed=self._cfg.asset.fix_base_link,
                visualization=self._requires_visual_geometry(),
            ),
            # visualize_contact=self._debug,
        )
        
        if self._uses_batch_camera_rendering():
            self._setup_scene_cameras()
        
        # build
        self._scene.build(n_envs=self._num_envs)
        if self._uses_batch_camera_rendering():
            self._attach_scene_cameras()

        self._get_env_origins()

        self._dof_names = self._cfg.asset.dof_names
        self._num_dof = len(self._cfg.asset.dof_names)

        # name to indices
        self._dof_indices = [self._robot.get_joint(
            name).dof_start for name in self._cfg.asset.dof_names]
        print(f"motor dof indices: {self._dof_indices}")
        
        # find indices of links specified in the config
        def find_link_indices(names, exact_match=False):
            return [
                link.idx - self._robot.link_start
                for link in self._robot.links
                if any((link.name == name) if exact_match else (name in link.name) for name in names)
            ]

        self._termination_contact_indices = find_link_indices(
            self._cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self._robot.links]
        print(f"all link names: {all_link_names}")
        print("termination link indices:", self._termination_contact_indices)
        self._penalized_contact_indices = find_link_indices(
            self._cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self._penalized_contact_indices}")
        self._feet_names = [
            link.name for link in self._robot.links if self._cfg.asset.foot_name in link.name]
        self._feet_indices = find_link_indices(self._feet_names)
        print(f"feet names: {self._feet_names}, feet link indices: {self._feet_indices}")
        assert len(self._feet_indices) > 0
        self._key_body_indices = find_link_indices(self._cfg.asset.key_bodies)
        print(f"key body link indices: {self._key_body_indices}")
        self._base_link_index = self._robot.base_link_idx - self._robot.link_start
        print(f"base link index: {self._base_link_index}")
        
        if self._cfg.asset.obtain_link_contact_states:
            self._contact_state_link_indices = find_link_indices(
                self._cfg.asset.contact_state_link_names,
                exact_match=True,
            )

        # dof position limits
        self._dof_pos_limits = torch.stack(
            self._robot.get_dofs_limit(self._dof_indices), dim=1)
        # Genesis don't provide api for accessing vel limits, so we set it here
        if len(self._cfg.asset.dof_vel_limits) != self._num_dof:
            raise ValueError(
                "Genesis requires cfg.asset.dof_vel_limits to match cfg.asset.dof_names. "
                f"Got {len(self._cfg.asset.dof_vel_limits)} limits for {self._num_dof} DOFs."
            )
        self._dof_vel_limits = torch.tensor(
            self._cfg.asset.dof_vel_limits,
            device=self._device,
        ).unsqueeze(0)
        self._torque_limits = self._robot.get_dofs_force_range(self._dof_indices)[
            1]
        for i in range(self._dof_pos_limits.shape[0]):
            # soft limits
            m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
            r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
            self._dof_pos_limits[i, 0] = (
                m - 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
            )
            self._dof_pos_limits[i, 1] = (
                m + 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
            )
            
        self._init_domain_params()
        # randomize friction
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self._num_envs))
        # randomize base mass
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self._num_envs))
        # randomize COM displacement
        if self._cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self._num_envs))
        # randomize joint armature
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(np.arange(self._num_envs))
        # randomize joint friction
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(np.arange(self._num_envs))
        # randomize joint damping
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(np.arange(self._num_envs))
        # randomize pd gain
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(np.arange(self._num_envs))
            
    def _init_buffers(self):
        self._base_init_pos = torch.tensor(
            self._cfg.init_state.pos, device=self._device
        )
        self._base_init_quat = torch.tensor(
            self._cfg.init_state.rot, device=self._device
        )
        self._base_lin_vel = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_ang_vel = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._last_base_lin_vel = torch.zeros_like(self._base_lin_vel)
        self._last_base_ang_vel = torch.zeros_like(self._base_ang_vel)
        self._projected_gravity = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self._device, dtype=torch.float).repeat(
            self._num_envs, 1
        )
        self._dof_pos = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._dof_vel = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._last_dof_vel = torch.zeros_like(self._dof_vel)
        self._base_pos = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_quat = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float)
        self._base_quat_gs = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float) # quaternion in genesis definition, wxyz
        self._base_euler = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._link_contact_forces = torch.zeros(
            (self._num_envs, self._robot.n_links, 3), device=self._device, dtype=torch.float
        )
        self._rigid_body_states = torch.zeros(
            (self._num_envs, self._robot.n_links, 13), device=self._device, dtype=torch.float
        )
        self._feet_pos = torch.zeros(
            (self._num_envs, len(self._feet_indices), 3), device=self._device, dtype=torch.float
        )
        self._feet_vel = torch.zeros(
            (self._num_envs, len(self._feet_indices), 3), device=self._device, dtype=torch.float
        )
        self._key_body_pos = torch.zeros(
            (self._num_envs, len(self._key_body_indices), 3), device=self._device, dtype=torch.float
        )
        self._last_feet_vel = torch.zeros_like(self._feet_vel)
        self._terrain_tile_half_extent = torch.tensor(
            [0.5 * self._cfg.terrain.terrain_length, 0.5 * self._cfg.terrain.terrain_width],
            device=self._device,
            dtype=torch.float,
        )
        self._parkour_out_of_lane_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.bool
        )
        self._global_out_of_bounds_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.bool
        )
        self._camera_render_env_ids = None
        if self._camera_render_env_indices:
            self._camera_render_env_ids = torch.as_tensor(
                self._camera_render_env_indices,
                device=self._device,
                dtype=torch.long,
            )
        if self._cfg.sensor.add_depth:
            depth_cfg = self._cfg.sensor.depth_camera_config
            depth_width, depth_height = self._processed_depth_resolution(depth_cfg)
            self._depth_images = torch.zeros(
                (
                    self._num_envs,
                    depth_cfg.num_history,
                    depth_height,
                    depth_width,
                ),
                device=self._device,
                dtype=torch.float,
            )
        if self._requires_rgb_camera_stream():
            self._rgb_images = torch.zeros(
                (
                    self._num_envs,
                    self._cfg.sensor.rgb_camera_config.resolution[1],
                    self._cfg.sensor.rgb_camera_config.resolution[0],
                    3,
                ),
                device=self._device,
                dtype=torch.uint8,
            )
        if getattr(self._cfg.sensor.depth_estimation, "enabled", False):
            self._inferred_depth_images = torch.zeros(
                (
                    self._num_envs,
                    self._cfg.sensor.rgb_camera_config.resolution[1],
                    self._cfg.sensor.rgb_camera_config.resolution[0],
                ),
                device=self._device,
                dtype=torch.float,
            )
        
        # Terrain information around feet
        if self._cfg.terrain.obtain_terrain_info_around_feet:
            self._normal_vector_around_feet = torch.zeros(
                self._num_envs, len(self._feet_indices) * 3, dtype=torch.float, device=self._device, requires_grad=False)
            self._height_around_feet = torch.zeros(
                self._num_envs, len(self._feet_indices), 9, dtype=torch.float, device=self._device, requires_grad=False)
        
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = torch.zeros(
                self._num_envs, len(self._contact_state_link_indices), dtype=torch.float, device=self._device, requires_grad=False)
        
        self._default_dof_pos = torch.tensor(
            [self._cfg.init_state.default_joint_angles[name]
                for name in self._cfg.asset.dof_names],
            device=self._device,
            dtype=torch.float,
        )
        self._default_dof_pos = self._default_dof_pos.unsqueeze(0)
        # PD control
        stiffness = self._cfg.control.stiffness
        damping = self._cfg.control.damping

        self._p_gains, self._d_gains = [], []
        for dof_name in self._cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self._p_gains.append(stiffness[key])
                    self._d_gains.append(damping[key])
        self._p_gains = torch.tensor(self._p_gains, device=self._device)
        self._d_gains = torch.tensor(self._d_gains, device=self._device)
        if self._batch_dofs_links_info:   
            self._p_gains = self._p_gains[None, :].repeat(self._num_envs, 1)
            self._d_gains = self._d_gains[None, :].repeat(self._num_envs, 1)
        self._robot.set_dofs_kp(self._p_gains, self._dof_indices)
        self._robot.set_dofs_kv(self._d_gains, self._dof_indices)

        # DC motor saturation parameters (optional, matches IsaacLab ParkourDCMotor)
        self._has_motor_saturation = (
            getattr(self._cfg.control, 'effort_limit', None) is not None
            and getattr(self._cfg.control, 'saturation_effort', None) is not None
            and getattr(self._cfg.control, 'velocity_limit', None) is not None
        )
        if self._has_motor_saturation:
            self._effort_limit = self._build_per_dof_tensor(self._cfg.control.effort_limit)
            self._saturation_effort = self._build_per_dof_tensor(self._cfg.control.saturation_effort)
            self._velocity_limit = self._build_per_dof_tensor(self._cfg.control.velocity_limit)
            self._zero_effort = torch.zeros_like(self._effort_limit)
            print(f"DC motor saturation enabled: effort_limit={self._effort_limit.tolist()}, "
                  f"saturation_effort={self._saturation_effort.tolist()}, "
                  f"velocity_limit={self._velocity_limit.tolist()}")

        self._init_height_points()

    def _terrain_uses_parkour_metadata(self):
        return bool(getattr(self._terrain, "metadata_enabled", False))

    def _apply_forced_parkour_indices(self):
        if not self._terrain_uses_parkour_metadata():
            return

        parkour_cfg = self._cfg.terrain.parkour
        if parkour_cfg.force_row is not None:
            forced_row = int(parkour_cfg.force_row)
            if forced_row < 0 or forced_row >= self._cfg.terrain.num_rows:
                raise ValueError(
                    f"Forced parkour row {forced_row} is outside [0, {self._cfg.terrain.num_rows - 1}]"
                )
            self._terrain_levels[:] = forced_row
        if parkour_cfg.force_family is not None:
            forced_family = str(parkour_cfg.force_family)
            if forced_family not in self._terrain.family_bucket_ids:
                raise ValueError(
                    f"Forced parkour family '{forced_family}' is not available in terrain columns "
                    f"{list(self._terrain.family_bucket_ids.keys())}"
                )
            self._terrain_family_types[:] = int(self._terrain.family_bucket_ids[forced_family])
        self._assign_variant_columns_from_family_buckets(torch.arange(self._num_envs, device=self._device))

    def _assign_variant_columns_from_family_buckets(self, env_ids):
        if not self._terrain_uses_parkour_metadata() or len(env_ids) == 0:
            return

        env_ids = env_ids.to(dtype=torch.long, device=self._device)
        for family_name, col_indices in self._terrain_family_variant_cols.items():
            family_bucket_id = int(self._terrain.family_bucket_ids[family_name])
            family_env_ids = env_ids[self._terrain_family_types[env_ids] == family_bucket_id]
            if len(family_env_ids) == 0:
                continue
            if col_indices.numel() == 1:
                self._terrain_types[family_env_ids] = col_indices[0]
                continue
            family_rank = torch.arange(family_env_ids.numel(), device=self._device, dtype=torch.long)
            variant_idx = (self._terrain_variant_cycle[family_env_ids] + family_rank) % col_indices.numel()
            self._terrain_types[family_env_ids] = col_indices[variant_idx]
            self._terrain_variant_cycle[family_env_ids] += 1

    def _refresh_lane_metadata(self, env_ids=None):
        if not self._terrain_uses_parkour_metadata():
            return

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        env_ids = env_ids.to(dtype=torch.long, device=self._device)
        rows = self._terrain_levels[env_ids].long()
        cols = self._terrain_types[env_ids].long()
        self._env_origins[env_ids] = self._terrain_origins[rows, cols]

        self._lane_family[env_ids] = self._terrain_lane_family[rows, cols]
        self._lane_difficulty_row[env_ids] = self._terrain_lane_difficulty_row[rows, cols]
        self._lane_spawn_pose[env_ids] = self._terrain_lane_spawn_pose[rows, cols]
        self._lane_safe_spawn_region[env_ids] = self._terrain_lane_safe_spawn_region[rows, cols]
        self._lane_bounds[env_ids] = self._terrain_lane_bounds[rows, cols]
        self._lane_waypoints[env_ids] = self._terrain_lane_waypoints[rows, cols]
        self._lane_waypoint_counts[env_ids] = self._terrain_lane_waypoint_counts[rows, cols]
        self._lane_terminal_goal[env_ids] = self._terrain_lane_terminal_goal[rows, cols]
        self._lane_section_bounds[env_ids] = self._terrain_lane_section_bounds[rows, cols]
        self._lane_section_tags[env_ids] = self._terrain_lane_section_tags[rows, cols]
        self._lane_jump_expected_mask[env_ids] = self._terrain_lane_jump_expected_mask[rows, cols]
        self._lane_edge_masks[env_ids] = self._terrain_lane_edge_masks[rows, cols]

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self._cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self._custom_origins = True
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self._cfg.terrain.max_init_terrain_level
            if not self._cfg.terrain.curriculum:
                max_init_level = self._cfg.terrain.num_rows - 1
            self._terrain_levels = torch.randint(
                0, max_init_level+1, (self._num_envs,), device=self._device)
            self._terrain_types = torch.div(torch.arange(self._num_envs, device=self._device), (
                self._num_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self._max_terrain_level = self._cfg.terrain.num_rows
            self._terrain_origins = torch.from_numpy(
                self._terrain.env_origins).to(self._device).to(torch.float)
            self._env_origins[:] = self._terrain_origins[self._terrain_levels,
                                                       self._terrain_types]
            if self._terrain_uses_parkour_metadata():
                self._terrain_family_variant_cols = {
                    family_name: torch.tensor(col_indices, device=self._device, dtype=torch.long)
                    for family_name, col_indices in self._terrain.family_variant_col_indices.items()
                }
                num_family_buckets = len(self._terrain.family_variant_col_indices)
                family_divisor = self._num_envs / max(num_family_buckets, 1)
                self._terrain_family_types = torch.div(
                    torch.arange(self._num_envs, device=self._device),
                    family_divisor,
                    rounding_mode='floor',
                ).to(torch.long)
                self._terrain_family_types = torch.clamp(self._terrain_family_types, max=num_family_buckets - 1)
                self._terrain_variant_cycle = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
                self._terrain_lane_family = torch.from_numpy(self._terrain.lane_family).to(self._device, dtype=torch.long)
                self._terrain_lane_difficulty_row = torch.from_numpy(self._terrain.lane_difficulty_row).to(self._device, dtype=torch.long)
                self._terrain_lane_spawn_pose = torch.from_numpy(self._terrain.lane_spawn_pose).to(self._device, dtype=torch.float)
                self._terrain_lane_safe_spawn_region = torch.from_numpy(self._terrain.lane_safe_spawn_region).to(self._device, dtype=torch.float)
                self._terrain_lane_bounds = torch.from_numpy(self._terrain.lane_bounds).to(self._device, dtype=torch.float)
                self._terrain_lane_waypoints = torch.from_numpy(self._terrain.lane_waypoints).to(self._device, dtype=torch.float)
                self._terrain_lane_waypoint_counts = torch.from_numpy(self._terrain.lane_waypoint_counts).to(self._device, dtype=torch.long)
                self._terrain_lane_terminal_goal = torch.from_numpy(self._terrain.lane_terminal_goal).to(self._device, dtype=torch.float)
                self._terrain_lane_section_bounds = torch.from_numpy(self._terrain.lane_section_bounds).to(self._device, dtype=torch.float)
                self._terrain_lane_section_tags = torch.from_numpy(self._terrain.lane_section_tags).to(self._device, dtype=torch.long)
                self._terrain_lane_jump_expected_mask = torch.from_numpy(self._terrain.lane_jump_expected_mask).to(self._device, dtype=torch.bool)
                self._terrain_lane_edge_masks = torch.from_numpy(self._terrain.lane_edge_masks).to(self._device, dtype=torch.bool)

                self._lane_family = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
                self._lane_difficulty_row = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
                self._lane_spawn_pose = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float)
                self._lane_safe_spawn_region = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float)
                self._lane_bounds = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float)
                self._lane_waypoints = torch.zeros(
                    (self._num_envs, self._terrain_lane_waypoints.shape[2], 3),
                    device=self._device,
                    dtype=torch.float,
                )
                self._lane_waypoint_counts = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
                self._lane_terminal_goal = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
                self._lane_section_bounds = torch.zeros(
                    (self._num_envs, self._terrain_lane_section_bounds.shape[2], 2),
                    device=self._device,
                    dtype=torch.float,
                )
                self._lane_section_tags = torch.zeros(
                    (self._num_envs, self._terrain_lane_section_tags.shape[2]),
                    device=self._device,
                    dtype=torch.long,
                )
                self._lane_jump_expected_mask = torch.zeros(
                    (self._num_envs, self._terrain_lane_jump_expected_mask.shape[2]),
                    device=self._device,
                    dtype=torch.bool,
                )
                self._lane_edge_masks = torch.zeros(
                    (
                        self._num_envs,
                        self._terrain_lane_edge_masks.shape[2],
                        self._terrain_lane_edge_masks.shape[3],
                    ),
                    device=self._device,
                    dtype=torch.bool,
                )

                self._apply_forced_parkour_indices()
                self._refresh_lane_metadata()
        else:
            self._custom_origins = False
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self._num_envs))
            num_rows = np.ceil(self._num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols), indexing='ij')
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = self._cfg.env.env_spacing
            if num_rows * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2 or \
                    num_cols * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2:
                spacing = min((self._cfg.terrain.plane_length / 2) / (num_rows-1),
                              (self._cfg.terrain.plane_length / 2) / (num_cols-1))
            self._env_origins[:, 0] = spacing * xx.flatten()[:self._num_envs]
            self._env_origins[:, 1] = spacing * yy.flatten()[:self._num_envs]
            self._env_origins[:, 2] = 0.
            self._env_origins[:, 0] -= self._cfg.terrain.plane_length / 4
            self._env_origins[:, 1] -= self._cfg.terrain.plane_length / 4

    def _calc_terrain_info_around_feet(self):
        """ Finds neighboring points around each foot for terrain height measurement."""
        # Foot positions
        foot_points = self._feet_pos + self._cfg.terrain.border_size
        foot_points = (foot_points/self._cfg.terrain.horizontal_scale).long()
        # px and py for 4 feet, num_envs*len(feet_indices)
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        # clip to the range of height samples
        px = torch.clip(px, 0, self._height_samples.shape[0]-2)
        py = torch.clip(py, 0, self._height_samples.shape[1]-2)
        # get heights around the feet, 9 points for each foot
        heights1 = self._height_samples[px-1, py]  # [x-0.1, y]
        heights2 = self._height_samples[px+1, py]  # [x+0.1, y]
        heights3 = self._height_samples[px, py-1]  # [x, y-0.1]
        heights4 = self._height_samples[px, py+1]  # [x, y+0.1]
        heights5 = self._height_samples[px, py]    # [x, y]
        heights6 = self._height_samples[px-1, py-1]  # [x-0.1, y-0.1]
        heights7 = self._height_samples[px+1, py+1]  # [x+0.1, y+0.1]
        heights8 = self._height_samples[px-1, py+1]  # [x-0.1, y+0.1]
        heights9 = self._height_samples[px+1, py-1]  # [x+0.1, y-0.1]
        # Calculate normal vectors around feet
        dx = ((heights2 - heights1) / (self._cfg.terrain.horizontal_scale * 2)).view(self._num_envs, -1)
        dy = ((heights4 - heights3) / (self._cfg.terrain.horizontal_scale * 2)).view(self._num_envs, -1)
        for i in range(len(self._feet_indices)):
            normal_vector = torch.cat((dx[:, i].unsqueeze(1), dy[:, i].unsqueeze(1), 
                -1*torch.ones_like(dx[:, i].unsqueeze(1))), dim=-1).to(self._device)
            normal_vector /= torch.norm(normal_vector, dim=-1, keepdim=True)
            self._normal_vector_around_feet[:, i*3:i*3+3] = normal_vector[:]
        # Calculate height around feet
        for i in range(9):
            self._height_around_feet[:, :, i] = eval(f'heights{i+1}').view(self._num_envs, -1)[:] * self._cfg.terrain.vertical_scale

    def _update_bounds_state(self):
        """Track terrain and lane-bound violations without mutating parkour course state."""
        x_out_of_bound = (self._base_pos[:, 0] >= self._terrain_x_range[1]) | (
            self._base_pos[:, 0] <= self._terrain_x_range[0])
        y_out_of_bound = (self._base_pos[:, 1] >= self._terrain_y_range[1]) | (
            self._base_pos[:, 1] <= self._terrain_y_range[0])
        self._global_out_of_bounds_buf[:] = x_out_of_bound | y_out_of_bound
        if not self._terrain_uses_parkour_metadata():
            self._parkour_out_of_lane_buf[:] = False
            env_ids = self._global_out_of_bounds_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) == 0:
                return
            self._base_pos[env_ids] = self.base_init_pos
            self._base_pos[env_ids] += self._env_origins[env_ids]
            self._robot.set_pos(
                self._base_pos[env_ids], zero_velocity=False, envs_idx=env_ids)
            return

        tile_origin_xy = self._env_origins[:, :2] - self._terrain_tile_half_extent
        local_base_xy = self._base_pos[:, :2] - tile_origin_xy
        x_out = (local_base_xy[:, 0] < self._lane_bounds[:, 0]) | (local_base_xy[:, 0] > self._lane_bounds[:, 1])
        y_out = (local_base_xy[:, 1] < self._lane_bounds[:, 2]) | (local_base_xy[:, 1] > self._lane_bounds[:, 3])
        self._parkour_out_of_lane_buf[:] = x_out | y_out

    def _check_base_pos_out_of_bound(self):
        self._update_bounds_state()

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self._cfg.control.action_scale
        # get two dimensional gains
        if self._p_gains.ndim == 1:
            self._p_gains = self._p_gains.unsqueeze(0).repeat(self._num_envs, 1)
            self._d_gains = self._d_gains.unsqueeze(0).repeat(self._num_envs, 1)
        torques = (
            self._kp_scale * self._p_gains * (actions_scaled +
                                    self._default_dof_pos - self._dof_pos)
            - self._kd_scale * self._d_gains * self._dof_vel
        )
        # Apply DC motor velocity-dependent torque saturation (matches IsaacLab ParkourDCMotor)
        if self._has_motor_saturation:
            vel_ratio = self._dof_vel / self._velocity_limit
            max_effort = self._saturation_effort * (1.0 - vel_ratio)
            max_effort = torch.clamp(max_effort, min=self._zero_effort, max=self._effort_limit)
            min_effort = self._saturation_effort * (-1.0 - vel_ratio)
            min_effort = torch.clamp(min_effort, min=-self._effort_limit, max=self._zero_effort)
            torques = torch.clamp(torques, min=min_effort, max=max_effort)
        return torques

    def _build_per_dof_tensor(self, param_dict):
        """Build a (num_dofs,) tensor from a dict mapping joint-name substrings to values."""
        values = []
        for dof_name in self._cfg.asset.dof_names:
            matched = False
            for key, val in param_dict.items():
                if key in dof_name:
                    values.append(val)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"No motor param match for DOF '{dof_name}' in {param_dict}")
        return torch.tensor(values, device=self._device, dtype=torch.float)

    def _init_domain_params(self):
        """ Initializes domain randomization parameters, which are used to randomize the environment."""
        self._friction_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._added_base_mass = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._rand_push_vels = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_com_bias = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_armature = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_friction = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_damping = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._kp_scale = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        self._kd_scale = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)

    def _randomize_friction(self, env_ids=None):
        ''' Randomize friction of all links'''
        min_friction, max_friction = self._cfg.domain_rand.friction_range

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, self._robot.n_links) \
            * (max_friction - min_friction) + min_friction
        self._friction_values[env_ids] = ratios[:,
                                                0].unsqueeze(1).detach().clone()

        self._robot.set_friction_ratio(
            ratios, torch.arange(0, self._robot.n_links), env_ids)

    def _randomize_base_mass(self, env_ids=None):
        ''' Randomize base mass'''
        min_mass, max_mass = self._cfg.domain_rand.added_mass_range
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * \
            (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        self._robot.set_mass_shift(added_mass, self._base_link_index, env_ids)

    def _randomize_com_displacement(self, env_ids):
        ''' Randomize center of mass displacement of the robot'''
        min_displacement_x, max_displacement_x = self._cfg.domain_rand.com_pos_x_range
        min_displacement_y, max_displacement_y = self._cfg.domain_rand.com_pos_y_range
        min_displacement_z, max_displacement_z = self._cfg.domain_rand.com_pos_z_range
        com_displacement = torch.zeros((len(env_ids), 1, 3), dtype=torch.float, device=self._device)

        com_displacement[:, 0, 0] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) \
            * (max_displacement_x - min_displacement_x) + min_displacement_x
        com_displacement[:, 0, 1] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) \
            * (max_displacement_y - min_displacement_y) + min_displacement_y
        com_displacement[:, 0, 2] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) \
            * (max_displacement_z - min_displacement_z) + min_displacement_z
        self._base_com_bias[env_ids] = com_displacement[:,
                                                        0, :].detach().clone()

        self._robot.set_COM_shift(
            com_displacement, self._base_link_index, env_ids)

    def _randomize_joint_armature(self, env_ids):
        min_armature, max_armature = self._cfg.domain_rand.joint_armature_range
        armature = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_armature - min_armature) + min_armature
        self._joint_armature[env_ids, 0] = armature.detach().clone()
        # [len(env_ids)] -> [len(env_ids), num_actions], all joints within an env have the same armature
        armature = armature.unsqueeze(1).repeat(1, self._num_actions)
        self._robot.set_dofs_armature(
            armature, self._dof_indices, envs_idx=env_ids) 
        # This armature will be Refreshed when envs are reset

    def _randomize_joint_friction(self, env_ids):
        min_friction, max_friction = self._cfg.domain_rand.joint_friction_range
        friction = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_friction - min_friction) + min_friction
        self._joint_friction[env_ids, 0] = friction.detach().clone()
        friction = friction.unsqueeze(1).repeat(1, self._num_actions)
        self._robot.set_dofs_frictionloss(
            friction, self._dof_indices, envs_idx=env_ids)

    def _randomize_joint_damping(self, env_ids):
        """ Randomize joint damping of the robot
        """
        min_damping, max_damping = self._cfg.domain_rand.joint_damping_range
        damping = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_damping - min_damping) + min_damping
        self._joint_damping[env_ids, 0] = damping.detach().clone()
        damping = damping.unsqueeze(1).repeat(1, self._num_actions)
        self._robot.set_dofs_damping(
            damping, self._dof_indices, envs_idx=env_ids)

    def _randomize_pd_gain(self, env_ids):
        self._kp_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kp_range[0], self._cfg.domain_rand.kp_range[1], (len(env_ids), self._num_actions), device=self._device)
        self._kd_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kd_range[0], self._cfg.domain_rand.kd_range[1], (len(env_ids), self._num_actions), device=self._device)
    
    def _ensure_depth_estimator(self):
        if self._depth_estimator is not None:
            return self._depth_estimator
        self._depth_estimator = create_depth_estimator(self._cfg.sensor, self._device)
        self._depth_estimator_backend_name = self._depth_estimator.backend_name
        return self._depth_estimator

    def _update_inferred_depth_images(self, rendered_env_indices):
        if not rendered_env_indices:
            return None

        self._depth_estimator_step += 1
        update_interval = max(1, int(getattr(self._cfg.sensor.depth_estimation, "update_interval", 1)))
        if self._depth_estimator_step > 1 and (self._depth_estimator_step - 1) % update_interval != 0:
            return self._inferred_depth_images

        estimator = self._ensure_depth_estimator()
        env_ids = sorted(int(env_idx) for env_idx in rendered_env_indices)
        rgb_input = self._rgb_images[env_ids]
        # Apply color jitter if experiment flag is set (domain invariance test)
        flags = getattr(self._cfg, "experiment_flags", None) or {}
        brightness_jitter = float(flags.get("brightness_jitter", 0.0))
        color_jitter = float(flags.get("color_jitter", 0.0))
        if brightness_jitter > 0 or color_jitter > 0:
            rgb_input = rgb_input.clone().float()
            if brightness_jitter > 0:
                factor = 1.0 + brightness_jitter * (2 * torch.rand(1, device=rgb_input.device) - 1)
                rgb_input = (rgb_input * factor).clamp_(0, 255)
            if color_jitter > 0:
                shifts = color_jitter * 255 * (2 * torch.rand(1, 1, 1, 3, device=rgb_input.device) - 1)
                rgb_input = (rgb_input + shifts).clamp_(0, 255)
            rgb_input = rgb_input.to(self._rgb_images.dtype)
        output = estimator.estimate({"rgb": rgb_input, "env_ids": env_ids})
        if output.depth.ndim != 3:
            raise RuntimeError(
                f"Unexpected inferred-depth shape from {output.backend_name}: {tuple(output.depth.shape)}"
            )
        env_ids_tensor = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        self._inferred_depth_images.index_copy_(0, env_ids_tensor, output.depth)
        self._depth_estimator_backend_name = output.backend_name
        self._depth_estimator_latency_ms = output.latency_ms
        return self._inferred_depth_images

    def _draw_debug_sensor_images(self):
        if self._headless:
            return

        panel_builders = self._debug_panel_builders()
        if not panel_builders:
            return

        env_panels = []
        for env_idx in self._rendered_env_indices:
            frames = [builder(env_idx) for builder in panel_builders]
            target_height = max(frame.shape[0] for frame in frames)
            padded_frames = [self._pad_debug_frame(frame, target_height) for frame in frames]
            env_panels.append(np.concatenate(padded_frames, axis=1))
        if not env_panels:
            return

        camera_canvas = self._tile_debug_panels(env_panels)
        if not self._camera_window_initialized:
            cv.namedWindow("Genesis Cameras", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            self._camera_window_initialized = True
        cv.imshow("Genesis Cameras", camera_canvas)
        cv.waitKey(1)
    
    def _draw_key_body_points(self, ref_key_body_pos=None):
        """ Draws key body points for debugging
        """
        if ref_key_body_pos is not None:
            debug_obj = self._scene.draw_debug_spheres(
                ref_key_body_pos.view(-1, 3), radius=0.03, color=(1, 0, 0, 1)
            )
            self._debug_vis_objects.append(debug_obj)
        else:
            pass

    def _clear_debug_vis_objects(self):
        for debug_obj in self._debug_vis_objects:
            self._scene.clear_debug_object(debug_obj)
        self._debug_vis_objects.clear()

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        terrain_kwargs = dict(
            pos=(-self._cfg.terrain.border_size, - \
                 self._cfg.terrain.border_size, 0.0),
            horizontal_scale=self._cfg.terrain.horizontal_scale,
            vertical_scale=self._cfg.terrain.vertical_scale,
            height_field=self._terrain.height_field_raw,
            visualization=self._requires_visual_geometry(),
        )
        entity_kwargs = {}
        if getattr(self._cfg.terrain, "add_texture", False):
            terrain_kwargs["uv_scale"] = getattr(self._cfg.terrain, "texture_uv_scale", 10.0)
            entity_kwargs["surface"] = self._create_terrain_surface()
        self._gs_terrain = self._scene.add_entity(
            gs.morphs.Terrain(**terrain_kwargs),
            **entity_kwargs,
        )
        self._height_samples = torch.tensor(self._terrain.heightsamples).view(
            self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
    
    def _create_terrain_surface(self):
        """Create a textured surface for terrain to give DA2 visual depth cues.

        Supports multiple texture modes via cfg.terrain.texture_mode:
          - checkerboard: grey concrete with dark grid lines (default, used during training)
          - solid_red/solid_green/solid_blue: solid color surfaces
          - random_color: random solid color (seeded by current time)
          - noise: random RGB noise texture
          - bricks: brick-like pattern with mortar lines
        """
        tex_size = 512
        mode = getattr(self._cfg.terrain, "texture_mode", "checkerboard")

        if mode == "checkerboard":
            img = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
            img[:] = [180, 175, 170]
            grid_spacing = 32
            for i in range(0, tex_size, grid_spacing):
                img[i:i+2, :] = [120, 115, 110]
                img[:, i:i+2] = [120, 115, 110]
            noise = np.random.RandomState(42).randint(-15, 16, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif mode.startswith("solid_"):
            color_map = {
                "solid_red": [200, 60, 60],
                "solid_green": [60, 180, 60],
                "solid_blue": [60, 60, 200],
                "solid_yellow": [200, 200, 60],
                "solid_white": [240, 240, 240],
                "solid_dark": [40, 40, 40],
            }
            color = color_map.get(mode, [180, 180, 180])
            img = np.full((tex_size, tex_size, 3), color, dtype=np.uint8)
        elif mode == "random_color":
            import time
            rng = np.random.RandomState(int(time.time()) % 2**31)
            color = rng.randint(30, 230, size=3).tolist()
            img = np.full((tex_size, tex_size, 3), color, dtype=np.uint8)
        elif mode == "noise":
            rng = np.random.RandomState(123)
            img = rng.randint(0, 256, (tex_size, tex_size, 3), dtype=np.uint8)
        elif mode == "bricks":
            img = np.full((tex_size, tex_size, 3), [180, 100, 70], dtype=np.uint8)
            brick_h, brick_w, mortar = 32, 64, 2
            for row in range(0, tex_size, brick_h):
                img[row:row+mortar, :] = [160, 160, 155]
                offset = (brick_w // 2) if ((row // brick_h) % 2) else 0
                for col in range(offset, tex_size, brick_w):
                    img[row:row+brick_h, col:col+mortar] = [160, 160, 155]
            noise = np.random.RandomState(42).randint(-10, 11, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            # Fallback: plain grey
            img = np.full((tex_size, tex_size, 3), [180, 175, 170], dtype=np.uint8)

        return gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(image_array=img),
        )

    def _create_trimesh(self):
        """ Adds a trimesh terrain to the simulation, sets parameters based on the cfg.
        """
        # export terrain mesh to {LEGGED_GYM_ROOT_DIR}/resources/terrains/trimesh_terrain.stl
        trimesh_terrain_path = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", "terrains", "trimesh_terrain.stl")
        self._terrain.terrain_mesh.export(trimesh_terrain_path)
        print(f"Exported terrain mesh to {trimesh_terrain_path}")
        
        # add terrain to the scene
        self._gs_terrain = self._scene.add_entity(
            gs.morphs.Mesh(
                file=trimesh_terrain_path,
                pos=(-self._cfg.terrain.border_size,
                     -self._cfg.terrain.border_size, 
                     0.0),
                fixed=True,
                convexify=False,
            ),
        )
        # save height samples for height sampling
        self._height_samples = torch.tensor(self._terrain.heightsamples).view(
            self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)

    def _uses_batch_camera_rendering(self):
        return self._cfg.sensor.add_depth or self._requires_rgb_camera_stream()

    def _requires_visual_geometry(self):
        return (not self._headless) or self._uses_batch_camera_rendering()

    def _requires_rgb_camera_stream(self):
        return getattr(self._cfg.sensor, "add_rgb", False) or getattr(
            self._cfg.sensor.depth_estimation,
            "enabled",
            False,
        )

    def _processed_depth_resolution(self, depth_cfg):
        resolution = getattr(depth_cfg, "processed_resolution", None)
        if resolution is None:
            resolution = depth_cfg.resolution
        return int(resolution[0]), int(resolution[1])

    def _camera_vertical_fov_deg(self, camera_cfg):
        width, height = camera_cfg.resolution
        half_horizontal = np.deg2rad(camera_cfg.horizontal_fov_deg) * 0.5
        half_vertical = np.arctan(np.tan(half_horizontal) * (height / width))
        return float(np.rad2deg(half_vertical * 2.0))

    def _camera_configs_match(self, lhs_cfg, rhs_cfg):
        keys = (
            "resolution",
            "horizontal_fov_deg",
            "link_idx_local",
            "pos",
            "euler",
            "near_plane",
            "far_plane",
        )
        return all(getattr(lhs_cfg, key) == getattr(rhs_cfg, key) for key in keys)

    def _validate_scene_camera_configs(self):
        if not self._uses_batch_camera_rendering():
            return
        if gs.platform != "Linux":
            raise RuntimeError("Genesis BatchRenderer camera rendering is only supported on Linux.")
        if str(self._device).startswith("cpu"):
            raise RuntimeError("Genesis BatchRenderer camera rendering requires a CUDA device.")
        if self._cfg.sensor.add_depth and self._requires_rgb_camera_stream():
            depth_res = tuple(self._cfg.sensor.depth_camera_config.resolution)
            rgb_res = tuple(self._cfg.sensor.rgb_camera_config.resolution)
            if depth_res != rgb_res:
                raise RuntimeError(
                    "BatchRenderer requires identical camera resolutions for all scene cameras. "
                    f"Got depth={depth_res}, rgb={rgb_res}."
                )

    def _add_scene_camera(self, camera_cfg):
        return self._scene.add_camera(
            res=camera_cfg.resolution,
            pos=camera_cfg.pos,
            lookat=(camera_cfg.pos[0] + 1.0, camera_cfg.pos[1], camera_cfg.pos[2]),
            up=(0.0, 0.0, 1.0),
            fov=self._camera_vertical_fov_deg(camera_cfg),
            near=camera_cfg.near_plane,
            far=camera_cfg.far_plane,
            debug=False,
        )

    def _setup_scene_cameras(self):
        self._validate_scene_camera_configs()
        self.scene_cameras = {}
        self._scene_camera_output_indices = {}
        if self._cfg.sensor.add_depth:
            self.scene_cameras["depth"] = self._add_scene_camera(self._cfg.sensor.depth_camera_config)
            self._scene_camera_output_indices["depth"] = self.scene_cameras["depth"].idx
        if self._requires_rgb_camera_stream():
            if self._cfg.sensor.add_depth and self._camera_configs_match(
                self._cfg.sensor.depth_camera_config,
                self._cfg.sensor.rgb_camera_config,
            ):
                self.scene_cameras["rgb"] = self.scene_cameras["depth"]
            else:
                self.scene_cameras["rgb"] = self._add_scene_camera(self._cfg.sensor.rgb_camera_config)
            self._scene_camera_output_indices["rgb"] = self.scene_cameras["rgb"].idx

    def _attach_scene_cameras(self):
        if not self._uses_batch_camera_rendering():
            return
        mounted_links = {}
        attached_camera_ids = set()
        for name, camera in self.scene_cameras.items():
            if camera.uid in attached_camera_ids:
                continue
            camera_cfg = self._cfg.sensor.depth_camera_config if name == "depth" else self._cfg.sensor.rgb_camera_config
            link_idx_local = int(camera_cfg.link_idx_local)
            if link_idx_local not in mounted_links:
                mounted_links[link_idx_local] = self._get_robot_link_by_local_idx(link_idx_local)
            camera.attach(
                mounted_links[link_idx_local],
                self._camera_offset_transform(camera_cfg.pos, camera_cfg.euler),
            )
            camera.move_to_attach()
            attached_camera_ids.add(camera.uid)

    def _render_scene_camera_streams(self, render_depth: bool, render_rgb: bool):
        if not self._uses_batch_camera_rendering():
            return None, None
        if not render_depth and not render_rgb:
            return None, None

        rgb_out, depth_out, _, _ = self._scene.render_all_cameras(
            rgb=render_rgb,
            depth=render_depth,
            segmentation=False,
            normal=False,
        )
        depth_frames = None
        rgb_frames = None
        if render_depth:
            depth_frames = self._as_torch_frame(depth_out[self._scene_camera_output_indices["depth"]])
            if depth_frames.ndim == 2:
                depth_frames = depth_frames.unsqueeze(0)
        if render_rgb:
            rgb_frames = self._as_torch_frame(rgb_out[self._scene_camera_output_indices["rgb"]])
            if rgb_frames.ndim == 3:
                rgb_frames = rgb_frames.unsqueeze(0)
        return depth_frames, rgb_frames

    def _process_depth_frames(self, frames):
        depth_cfg = self._cfg.sensor.depth_camera_config
        near_clip = depth_cfg.near_clip
        far_clip = depth_cfg.far_clip
        frames = frames.clamp(near_clip, far_clip)

        height, width = frames.shape[-2:]
        top = int(getattr(depth_cfg, "crop_top", 0))
        bottom = int(getattr(depth_cfg, "crop_bottom", 0))
        left = int(getattr(depth_cfg, "crop_left", 0))
        right = int(getattr(depth_cfg, "crop_right", 0))
        end_h = height - bottom if bottom > 0 else height
        end_w = width - right if right > 0 else width
        frames = frames[:, top:end_h, left:end_w]

        target_width, target_height = self._processed_depth_resolution(depth_cfg)
        if tuple(frames.shape[-2:]) != (target_height, target_width):
            frames = F.interpolate(
                frames.unsqueeze(1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return (frames - near_clip) / max(far_clip - near_clip, 1e-6) - 0.5

    def _update_depth_images(self, latest_frame):
        if latest_frame is None:
            raise RuntimeError("Depth rendering requested without a rendered frame.")
        processed = self._process_depth_frames(latest_frame)
        env_ids = None
        if processed.shape[0] != self._num_envs:
            env_ids = self._camera_render_env_ids
            if env_ids is None:
                raise RuntimeError("Missing camera env indices for partial depth rendering.")
        if self._depth_images.shape[1] > 1:
            if env_ids is None:
                self._depth_images[:, 1:].copy_(self._depth_images[:, :-1].clone())
            else:
                self._depth_images[env_ids, 1:] = self._depth_images[env_ids, :-1].clone()
        if env_ids is None:
            self._depth_images[:, 0].copy_(processed)
        else:
            self._depth_images[env_ids, 0] = processed
        return self._depth_images

    def _update_rgb_images(self, latest_frame):
        if latest_frame is None:
            return None
        rgb_frames = self._convert_rgb_frame_to_uint8(latest_frame)
        if rgb_frames.shape[0] == self._num_envs:
            self._rgb_images.copy_(rgb_frames)
        else:
            if self._camera_render_env_ids is None:
                raise RuntimeError("Missing camera env indices for partial RGB rendering.")
            self._rgb_images.index_copy_(0, self._camera_render_env_ids, rgb_frames)
        return self._rgb_images

    def _get_robot_link_by_local_idx(self, link_idx_local):
        for link in self._robot.links:
            if link.idx - self._robot.link_start == link_idx_local:
                return link
        raise IndexError(f"Invalid robot link_idx_local for camera mount: {link_idx_local}")

    def _camera_offset_transform(self, pos_offset, euler_offset):
        pos = np.asarray(pos_offset, dtype=np.float32)
        euler = np.asarray(euler_offset, dtype=np.float32)
        sensor_quat = gu.xyz_to_quat(euler)
        sensor_rotation = gu.quat_to_R(sensor_quat)
        # DepthCameraPattern uses a robotics camera frame: forward +X, right -Y, up +Z.
        # Genesis visualizer cameras use: forward -Z, right +X, up +Y.
        # Convert the configured sensor mount into the visualizer camera basis.
        sensor_to_visualizer_camera = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        camera_rotation = sensor_rotation @ sensor_to_visualizer_camera
        return gu.trans_R_to_T(pos, camera_rotation)

    def _as_torch_frame(self, frame):
        if isinstance(frame, np.ndarray):
            if any(stride < 0 for stride in frame.strides) or not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)
            return torch.from_numpy(frame).to(self._device)
        if torch.is_tensor(frame):
            return frame.contiguous().to(self._device)
        raise TypeError(f"Unsupported Genesis camera frame type: {type(frame)}")

    def _convert_rgb_frame_to_uint8(self, frame):
        frame = frame.detach()
        if frame.dtype == torch.uint8:
            return frame
        if torch.is_floating_point(frame):
            if frame.numel() > 0 and float(frame.max().item()) <= 1.0 + 1e-6:
                frame = frame * 255.0
            return frame.clamp(0, 255).round().to(torch.uint8)
        return frame.clamp(0, 255).to(torch.uint8)

    def _prepare_depth_debug_frame(self, env_idx):
        near_clip = self._cfg.sensor.depth_camera_config.near_clip
        far_clip = self._cfg.sensor.depth_camera_config.far_clip
        clipped_depth = (self._depth_images[env_idx, 0] + 0.5) * (far_clip - near_clip) + near_clip
        clipped_depth = clipped_depth.clamp(near_clip, far_clip)
        pixel_values = (
            ((clipped_depth - near_clip) / (far_clip - near_clip)) * 255.0
        ).clamp(0, 255).cpu().numpy().astype(np.uint8)
        depth_vis = cv.applyColorMap(pixel_values, cv.COLORMAP_TURBO)
        min_depth = float(clipped_depth.min().item())
        max_depth = float(clipped_depth.max().item())
        return self._label_debug_frame(depth_vis, f"Env {env_idx}  Depth  min {min_depth:.2f}m  max {max_depth:.2f}m")

    def _prepare_rgb_debug_frame(self, env_idx):
        rgb_frame = self._rgb_images[env_idx].cpu().numpy()
        rgb_frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
        return self._label_debug_frame(rgb_frame, f"Env {env_idx}  RGB")

    def _prepare_inferred_depth_debug_frame(self, env_idx):
        depth = self._inferred_depth_images[env_idx]
        finite_mask = torch.isfinite(depth)
        if finite_mask.any():
            valid_depth = depth[finite_mask]
            min_depth = float(valid_depth.min().item())
            max_depth = float(valid_depth.max().item())
            disparity = torch.zeros_like(depth)
            disparity[finite_mask] = valid_depth.reciprocal()
            valid_disparity = disparity[finite_mask]
            lo = float(torch.quantile(valid_disparity, 0.02).item())
            hi = float(torch.quantile(valid_disparity, 0.98).item())
            if hi - lo < 1e-6:
                normalized = torch.full_like(depth, 0.5)
            else:
                normalized = ((disparity - lo) / (hi - lo)).clamp(0.0, 1.0)
            normalized = torch.where(finite_mask, normalized, torch.zeros_like(normalized))
        else:
            min_depth = 0.0
            max_depth = 0.0
            normalized = torch.zeros_like(depth)
        pixel_values = (normalized * 255.0).round().to(torch.uint8).cpu().numpy()
        depth_vis = cv.applyColorMap(pixel_values, cv.COLORMAP_TURBO)
        backend_label = (self._depth_estimator_backend_name or "inferred_depth").replace("_", " ")
        latency_suffix = ""
        if self._depth_estimator_latency_ms is not None:
            latency_suffix = f"  {self._depth_estimator_latency_ms:.1f}ms"
        return self._label_debug_frame(
            depth_vis,
            f"Env {env_idx}  {backend_label}  min {min_depth:.3f}  max {max_depth:.3f}{latency_suffix}",
        )

    def _debug_panel_builders(self):
        panel_builders = []
        if self._cfg.sensor.add_depth:
            panel_builders.append(self._prepare_depth_debug_frame)
        if getattr(self._cfg.sensor, "add_rgb", False):
            panel_builders.append(self._prepare_rgb_debug_frame)
        if getattr(self._cfg.sensor.depth_estimation, "enabled", False):
            panel_builders.append(self._prepare_inferred_depth_debug_frame)
        return panel_builders

    def _label_debug_frame(self, frame, label):
        labeled = frame.copy()
        cv.putText(
            labeled,
            label,
            (8, 18),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        return labeled

    def _pad_debug_frame(self, frame, target_height):
        if frame.shape[0] == target_height:
            return frame
        pad_bottom = target_height - frame.shape[0]
        return cv.copyMakeBorder(
            frame,
            0,
            pad_bottom,
            0,
            0,
            cv.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def _tile_debug_panels(self, panels):
        if len(panels) == 1:
            return panels[0]

        num_cols = int(np.ceil(np.sqrt(len(panels))))
        num_rows = int(np.ceil(len(panels) / num_cols))
        cell_height = max(panel.shape[0] for panel in panels)
        cell_width = max(panel.shape[1] for panel in panels)
        blank_panel = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)

        padded_panels = [
            cv.copyMakeBorder(
                panel,
                0,
                cell_height - panel.shape[0],
                0,
                cell_width - panel.shape[1],
                cv.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            for panel in panels
        ]
        while len(padded_panels) < num_rows * num_cols:
            padded_panels.append(blank_panel.copy())

        rows = []
        for row_idx in range(num_rows):
            row_start = row_idx * num_cols
            rows.append(np.concatenate(padded_panels[row_start : row_start + num_cols], axis=1))
        return np.concatenate(rows, axis=0)


    #----- Properties -----#
    @property
    def feet_contact_indices(self):
        """Returns the indices of the feet links in the contact sensors.

        Returns:
            list[int]: Indices of the feet links in the contact sensors.
        """
        return self._feet_indices
