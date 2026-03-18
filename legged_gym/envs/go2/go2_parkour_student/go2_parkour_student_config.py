from legged_gym.envs.go2.go2_parkour_teacher.go2_parkour_teacher_config import (
    Go2ParkourTeacherCfg,
)
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.parkour_observation import (
    ParkourObservationSpec,
    parkour_prop_obs_dim,
    parkour_critic_obs_dim,
)


class Go2ParkourStudentCfg(Go2ParkourTeacherCfg):
    class env(Go2ParkourTeacherCfg.env):
        num_envs = 4096
        num_observations = None   # computed below (prop_dim = 53)
        num_privileged_obs = None  # computed below (num_scandots = 132)
        num_critic_obs = None      # computed below (teacher critic_dim = 232)
        num_depth_obs = None       # computed below (H * W)
        num_latent_dims = 32       # scandot_encoder output dim
        num_actions = 12
        episode_length_s = 20.0
        env_spacing = 1.0

    class terrain(Go2ParkourTeacherCfg.terrain):
        pass

    class init_state(Go2ParkourTeacherCfg.init_state):
        pass

    class control(Go2ParkourTeacherCfg.control):
        pass

    class asset(Go2ParkourTeacherCfg.asset):
        pass

    class rewards(Go2ParkourTeacherCfg.rewards):
        class scales(Go2ParkourTeacherCfg.rewards.scales):
            pass

    class commands(Go2ParkourTeacherCfg.commands):
        pass

    class domain_rand(Go2ParkourTeacherCfg.domain_rand):
        pass

    class normalization(Go2ParkourTeacherCfg.normalization):
        class obs_scales(Go2ParkourTeacherCfg.normalization.obs_scales):
            pass

    class noise(Go2ParkourTeacherCfg.noise):
        class noise_scales(Go2ParkourTeacherCfg.noise.noise_scales):
            pass

    class sensor(LeggedRobotCfg.sensor):
        add_depth = True

        class depth_camera_config(LeggedRobotCfg.sensor.depth_camera_config):
            num_sensors = 1
            num_history = 1
            near_clip = 0.1
            far_clip = 3.0
            near_plane = 0.1
            far_plane = 3.0
            resolution = (87, 58)  # (W, H)
            horizontal_fov_deg = 75
            pos = (0.3, 0.0, 0.1)
            euler = (0.0, 1.57, 0.0)  # forward-facing
            decimation = 1
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False
            link_idx_local = 0


class Go2ParkourStudentCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "ParkourStudentRunner"

    class policy(LeggedRobotCfgPPO.policy):
        scandot_encoder_hidden_dims = [128, 64, 32]
        depth_image_shape = [1, 58, 87]  # (C, H, W)
        depth_encoder_hidden_dim = 32
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 512, 256]
        teacher_checkpoint = None  # set to path of teacher .pt to init weights

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        encoder_lr = 1.0e-3
        num_encoder_epochs = 2

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticParkourStudent"
        algorithm_class_name = "PPO_ParkourStudent"
        run_name = "student_genesis"
        experiment_name = "go2_parkour_student"
        num_steps_per_env = 48
        save_interval = 200
        max_iterations = 5000


# --- Compute dimensions from config ---
_OBS_SPEC = ParkourObservationSpec.from_cfg(Go2ParkourStudentCfg)
Go2ParkourStudentCfg.env.num_observations = _OBS_SPEC.prop_dim
Go2ParkourStudentCfg.env.num_privileged_obs = _OBS_SPEC.num_scandots
Go2ParkourStudentCfg.env.num_critic_obs = _OBS_SPEC.critic_dim
_depth_res = Go2ParkourStudentCfg.sensor.depth_camera_config.resolution  # (W, H)
Go2ParkourStudentCfg.env.num_depth_obs = _depth_res[1] * _depth_res[0]  # H * W
