from legged_gym.envs.go2.go2_parkour_teacher.go2_parkour_teacher_config import (
    Go2ParkourTeacherCfg,
)
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.parkour_observation import ParkourObservationSpec


class Go2ParkourStudentCfg(Go2ParkourTeacherCfg):
    class env(Go2ParkourTeacherCfg.env):
        num_envs = 4096
        frame_stack = 10
        num_observations = None
        num_privileged_obs = None
        num_teacher_actor_obs = None
        num_history_obs = None
        num_latent_dims = 32
        num_actions = 12
        episode_length_s = 20.0
        env_spacing = 1.0
        student_depth_shape = None

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
            num_history = 2
            near_clip = 0.0
            far_clip = 2.0
            near_plane = 0.1
            far_plane = 2.0
            resolution = (106, 60)  # raw (W, H)
            processed_resolution = (87, 58)  # final (W, H)
            crop_top = 0
            crop_bottom = 2
            crop_left = 4
            crop_right = 4
            horizontal_fov_deg = 87
            pos = (0.27, 0.0, 0.03)
            euler = (0.0, 1.57, 0.0)  # forward-facing
            decimation = 5
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False
            link_idx_local = 0


class Go2ParkourStudentCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "ParkourStudentRunner"

    class policy:
        clip_actions = LeggedRobotCfg.normalization.clip_actions
        activation = "elu"
        student_depth_shape = [2, 58, 87]
        proprio_history_frames = 10
        proprio_history_hidden_dims = [128, 64]
        depth_encoder_hidden_dims = [128, 64]
        student_latent_hidden_dims = [256, 128]
        actor_hidden_dims = [512, 256, 128]

    class algorithm:
        learning_rate = 1.0e-3
        action_loss_coef = 1.0
        latent_loss_coef = 0.25
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticParkourStudent"
        algorithm_class_name = "ParkourDistillation"
        run_name = "student_genesis"
        experiment_name = "go2_parkour_student"
        num_steps_per_env = 48
        save_interval = 200
        max_iterations = 5000
        teacher_task = "go2_parkour_teacher"
        teacher_load_run = -1
        teacher_ckpt = -1


# --- Compute dimensions from config ---
_OBS_SPEC = ParkourObservationSpec.from_cfg(Go2ParkourStudentCfg)
Go2ParkourStudentCfg.env.num_observations = _OBS_SPEC.prop_dim
Go2ParkourStudentCfg.env.num_privileged_obs = _OBS_SPEC.teacher_actor_dim
Go2ParkourStudentCfg.env.num_teacher_actor_obs = _OBS_SPEC.teacher_actor_dim
Go2ParkourStudentCfg.env.num_history_obs = (
    Go2ParkourStudentCfg.env.num_observations * Go2ParkourStudentCfg.env.frame_stack
)
_student_depth_res = Go2ParkourStudentCfg.sensor.depth_camera_config.processed_resolution
Go2ParkourStudentCfg.env.student_depth_shape = [
    Go2ParkourStudentCfg.sensor.depth_camera_config.num_history,
    _student_depth_res[1],
    _student_depth_res[0],
]
