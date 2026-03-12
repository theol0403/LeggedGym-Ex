from legged_gym import *
from legged_gym.envs.base.legged_robot_ee_config import LeggedRobotEECfg, LeggedRobotEECfgPPO
from legged_gym.envs.base.common_cfgs import Go2DepthSensorCfg, Go2RoughCommonCfg

class Go2EECfg( LeggedRobotEECfg ):
    class env( LeggedRobotEECfg.env ):
        num_envs = 4096
        num_single_obs = 45
        frame_stack = 20    # number of frames to stack for obs_history
        num_estimator_features = int(num_single_obs * frame_stack)
        num_estimator_labels = 24
        c_frame_stack = 5
        single_critic_obs_len = num_single_obs + 31 + 81 + 17
        num_privileged_obs = c_frame_stack * single_critic_obs_len
        # privileged_obs here is actually critic_obs
        num_actions = 12
        env_spacing = 2.0
    
    # use common terrain, init_state, control, and asset configs
    class terrain( Go2RoughCommonCfg.terrain ):
        pass
    class init_state( Go2RoughCommonCfg.init_state ):
        pass
    class control( Go2RoughCommonCfg.control ):
        pass
    class asset( Go2RoughCommonCfg.asset ):
        pass
  
    class rewards( Go2RoughCommonCfg.rewards ):
        class scales( Go2RoughCommonCfg.rewards.scales ):
            pass

    class commands( LeggedRobotEECfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotEECfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class domain_rand(LeggedRobotEECfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.
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
        joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
        joint_friction_range = [0.01, 0.02]
        joint_damping_range = [0.25, 0.3]

    class sensor(Go2DepthSensorCfg):
        pass

class Go2EECfgPPO( LeggedRobotEECfgPPO ):
    class policy( LeggedRobotEECfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        estimator_hidden_dims = [256, 128]
    class algorithm( LeggedRobotEECfgPPO.algorithm ):
        estimator_lr = 2.e-4
        num_estimator_epochs = 1
    class runner( LeggedRobotEECfgPPO.runner ):
        run_name = "ee"
        if SIMULATOR == "genesis":
            run_name += '_genesis'
        elif SIMULATOR == "isaacgym":
            run_name += '_isaacgym'
        elif SIMULATOR == "isaaclab":
            run_name += '_isaaclab'
        experiment_name = 'go2_rough'
        save_interval = 500
        max_iterations = 5000
