from legged_gym.envs.go2.go2_parkour_teacher.go2_parkour_teacher_config import (
    Go2ParkourTeacherCfg,
)
from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.parkour_observation import ParkourObservationSpec


class Go2ParkourStudentCfg(Go2ParkourTeacherCfg):
    class env(Go2ParkourTeacherCfg.env):
        num_envs = 48
        frame_stack = 0
        num_observations = None
        num_privileged_obs = None
        num_teacher_actor_obs = None
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
        add_rgb = False
        depth_noise_level = 0.1

        class depth_estimation(LeggedRobotCfg.sensor.depth_estimation):
            enabled = False

        class depth_camera_config(LeggedRobotCfg.sensor.depth_camera_config):
            num_sensors = 1
            num_history = 2  # buffer 2 frames; student uses previous frame
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
            # Go2 URDF front_camera_joint: xyz="0.32715 0 0.04297" rpy="0 0 0"
            pos = (0.327, 0.0, 0.043)
            euler = (0.0, 0.087, 0.0)  # 5 deg pitch down for gap visibility
            decimation = 5
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False
            link_idx_local = 0


class Go2ParkourStudentCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "ParkourStudentRunner"

    class policy:
        clip_actions = LeggedRobotCfg.normalization.clip_actions
        activation = "elu"
        student_depth_shape = [1, 58, 87]
        depth_backbone_output_dim = 32
        gru_hidden_dim = 512
        yaw_output_dim = 2
        yaw_scale = 1.5
        heading_command_indices = (4, 5)
        actor_hidden_dims = [512, 256, 128]

    class algorithm:
        learning_rate = 2.0e-3
        action_loss_coef = 1.0
        yaw_loss_coef = 1.0
        yaw_threshold = 0.6
        max_grad_norm = 1.0
        bptt_window = 24

    class runner:
        policy_class_name = "ActorCriticParkourStudent"
        algorithm_class_name = "ParkourDistillation"
        run_name = "student_genesis"
        experiment_name = "go2_parkour_student"
        sync_wandb = False
        num_steps_per_env = 120
        save_interval = 500
        max_iterations = 5000
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        teacher_task = "go2_parkour_teacher"
        teacher_load_run = -1
        teacher_ckpt = -1


# --- Depth-estimation student: uses RGB -> DA-V2 -> inferred depth ---

class Go2ParkourDepthEstStudentCfg(Go2ParkourStudentCfg):
    class env(Go2ParkourStudentCfg.env):
        pass

    class sensor(LeggedRobotCfg.sensor):
        add_depth = False
        add_rgb = True
        depth_noise_level = 0.1

        class depth_estimation(LeggedRobotCfg.sensor.depth_estimation):
            enabled = True
            model_type = "depth_anything_v2_metric_outdoor"
            model_size = "small"
            update_interval = 2

        class depth_camera_config(Go2ParkourStudentCfg.sensor.depth_camera_config):
            pass

        class rgb_camera_config(LeggedRobotCfg.sensor.rgb_camera_config):
            resolution = (106, 60)
            horizontal_fov_deg = 87
            # Go2 URDF front_camera_joint: xyz="0.32715 0 0.04297" rpy="0 0 0"
            pos = (0.327, 0.0, 0.043)
            euler = (0.0, 0.087, 0.0)  # 5 deg pitch down for gap visibility
            link_idx_local = 0
            near_plane = 0.1
            far_plane = 10.0


class Go2ParkourDepthEstStudentCfgPPO(Go2ParkourStudentCfgPPO):
    class runner(Go2ParkourStudentCfgPPO.runner):
        run_name = "depth_est_student_genesis"
        experiment_name = "go2_parkour_depth_est_student"


# --- Compute dimensions from config ---

def _finalize_student_cfg(cfg_cls):
    spec = ParkourObservationSpec.from_cfg(cfg_cls)
    cfg_cls.env.num_observations = spec.prop_dim
    cfg_cls.env.num_privileged_obs = spec.teacher_actor_dim
    cfg_cls.env.num_teacher_actor_obs = spec.teacher_actor_dim
    depth_res = cfg_cls.sensor.depth_camera_config.processed_resolution
    # Network always receives a single frame (1, H, W) regardless of buffer size.
    # The env selects which frame from the buffer to expose.
    cfg_cls.env.student_depth_shape = [1, depth_res[1], depth_res[0]]

_finalize_student_cfg(Go2ParkourStudentCfg)
_finalize_student_cfg(Go2ParkourDepthEstStudentCfg)
