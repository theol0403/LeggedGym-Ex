from legged_gym import SIMULATOR
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


def _num_scandot_points(terrain_cfg_cls):
    return len(terrain_cfg_cls.scandots.points_x) * len(terrain_cfg_cls.scandots.points_y)


def _critic_contact_dims(asset_cfg_cls):
    return getattr(asset_cfg_cls, "contact_state_obs_dim", len(asset_cfg_cls.contact_state_link_names) * 4)


def _actor_obs_dim(cfg_cls):
    num_actions = len(cfg_cls.asset.dof_names)
    num_goal_terms = 4
    num_gravity = 3
    num_ang_vel = 3
    num_foot_contacts = 4
    return (
        num_goal_terms
        + num_gravity
        + num_ang_vel
        + num_actions
        + num_actions
        + num_actions
        + num_foot_contacts
        + _num_scandot_points(cfg_cls.terrain)
    )


def _critic_obs_dim(cfg_cls):
    num_actions = len(cfg_cls.asset.dof_names)
    privileged_dynamics_dim = 1 + 1 + 3 + 2 + num_actions + num_actions
    return _actor_obs_dim(cfg_cls) + 3 + privileged_dynamics_dim + _critic_contact_dims(cfg_cls.asset)


class Go2ParkourTeacherCfg(Go2RoughCommonCfg):
    class env(Go2RoughCommonCfg.env):
        num_envs = 4096
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 20.0
        env_spacing = 1.0

    class terrain(Go2RoughCommonCfg.terrain):
        if SIMULATOR == "genesis":
            mesh_type = "heightfield"
        else:
            mesh_type = "trimesh"
        border_size = 5.0
        curriculum = True
        terrain_length = 16.0
        terrain_width = 6.0
        platform_size = 3.0
        num_rows = 4
        num_cols = 3
        max_init_terrain_level = 1
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]

        class scandots(Go2RoughCommonCfg.terrain.scandots):
            enable = True
            points_x = [-0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35]
            points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
            base_height_offset = 0.5
            clip_min = -1.0
            clip_max = 1.0

        class parkour(Go2RoughCommonCfg.terrain.parkour):
            enable = True
            families = ["stairs", "hurdle_block", "gap"]
            include_flat_debug = False
            force_family = None
            force_row = None
            max_waypoints = 5
            max_obstacles = 4
            max_sections = 4
            waypoint_radius = 0.45
            waypoint_dwell_steps = 3
            curriculum_progress_up_threshold = 0.9
            curriculum_progress_down_threshold = 0.35

    class init_state(Go2RoughCommonCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        roll_random_scale = 0.0
        pitch_random_scale = 0.0
        yaw_random_scale = 0.0

    class control(Go2RoughCommonCfg.control):
        stiffness = {"joint": 30.0}
        damping = {"joint": 0.75}
        action_scale = 0.25
        dt = 0.02
        decimation = 4

    class asset(Go2RoughCommonCfg.asset):
        obtain_link_contact_states = True
        contact_state_link_names = ["base", "hip", "thigh", "calf"]
        contact_state_obs_dim = 13
        penalize_contacts_on = ["hip", "thigh", "calf", "base", "Head"]
        terminate_after_contacts_on = ["base", "Head"]
        hip_joint_indices = [0, 3, 6, 9]

    class rewards(Go2RoughCommonCfg.rewards):
        only_positive_rewards = False
        foot_clearance_target = 0.09
        foot_height_offset = 0.022
        foot_clearance_tracking_sigma = 0.01
        goal_velocity_tracking_sigma = 0.25
        goal_heading_tracking_sigma = 0.5
        dynamic_motion_penalty_scale = 0.35
        progress_speed_clip = [-0.5, 1.5]

        class scales(Go2RoughCommonCfg.rewards.scales):
            termination = -50.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            tracking_goal_vel = 3.0
            tracking_goal_heading = 1.5
            progress_along_course = 2.0
            waypoint_reached_bonus = 20.0
            success_bonus = 40.0
            collision = -2.0
            feet_stumble = -1.0
            feet_edge = -2.0
            torques = -2.0e-4
            dof_power = -2.0e-4
            dof_acc = -2.0e-7
            action_rate = -0.01
            action_smoothness = -0.01
            dof_pos_limits = -2.0
            torque_limits = -0.2
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -0.5
            foot_clearance = 0.2
            hip_pos = -0.05
            feet_contact_stand_still = 0.0

    class commands(Go2RoughCommonCfg.commands):
        curriculum = False
        heading_command = False
        num_commands = 4
        goal_speed_range = [0.8, 1.0]

    class domain_rand(Go2RoughCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15.0
        max_push_vel_xy = 1.0
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = False
        randomize_joint_friction = False
        randomize_joint_damping = False

    class normalization(Go2RoughCommonCfg.normalization):
        class obs_scales(Go2RoughCommonCfg.normalization.obs_scales):
            goal_pos = 0.5
            heading = 1.0
            goal_speed = 1.0
            scandots = 1.0

    class noise(Go2RoughCommonCfg.noise):
        class noise_scales(Go2RoughCommonCfg.noise.noise_scales):
            scandots = 0.05


class Go2ParkourTeacherCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 512, 256]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        run_name = "teacher_genesis"
        experiment_name = "go2_parkour_teacher"
        num_steps_per_env = 48
        save_interval = 200
        max_iterations = 2500


Go2ParkourTeacherCfg.env.num_observations = _actor_obs_dim(Go2ParkourTeacherCfg)
Go2ParkourTeacherCfg.env.num_privileged_obs = _critic_obs_dim(Go2ParkourTeacherCfg)
