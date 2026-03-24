import time

from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch
from legged_gym.scripts.play_commands import PlayCommandController, resolve_command_mode, supports_manual_velocity_commands


def _copy_camera_debug_config(dst_cfg, src_cfg):
    for attr in ("resolution", "horizontal_fov_deg", "link_idx_local", "pos", "euler", "near_plane", "far_plane"):
        setattr(dst_cfg, attr, getattr(src_cfg, attr))


def override_configs(env_cfg, train_cfg, args):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
    """
    command_mode = resolve_command_mode(args)
    enable_depth_debug = bool(args.depth_debug)
    enable_inferred_depth_debug = args.depth_model_type is not None
    enable_rgb_debug = bool(args.rgb_debug or enable_inferred_depth_debug)
    sensor_debug = enable_depth_debug or enable_rgb_debug
    if sensor_debug and SIMULATOR == "isaaclab":
        raise NotImplementedError("Camera debug rendering is not implemented for Isaac Lab")
    if enable_rgb_debug and SIMULATOR != "genesis":
        raise NotImplementedError("RGB camera debug rendering is currently implemented only for Genesis")
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if enable_depth_debug:
        env_cfg.sensor.add_depth = True
    if enable_rgb_debug:
        env_cfg.sensor.add_rgb = True
        _copy_camera_debug_config(env_cfg.sensor.rgb_camera_config, env_cfg.sensor.depth_camera_config)
    env_cfg.sensor.depth_estimation.enabled = enable_inferred_depth_debug
    if args.depth_model_type is not None:
        env_cfg.sensor.depth_estimation.model_type = args.depth_model_type
    if args.depth_model_size is not None:
        env_cfg.sensor.depth_estimation.model_size = args.depth_model_size
    env_cfg.sensor.depth_estimation.update_interval = max(1, int(args.depth_update_interval))
    env_cfg.env.debug = args.debug
    env_cfg.env.debug_sensor_images = sensor_debug
    if train_cfg.runner_class_name == "CTSRunner":
        env_cfg.env.num_teacher = 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    # adjust parameters according to terrain type
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        if getattr(getattr(env_cfg.terrain, "parkour", None), "enable", False):
            env_cfg.terrain.curriculum = False
            if args.parkour_force_family is not None:
                env_cfg.terrain.parkour.force_family = args.parkour_force_family
            if args.parkour_force_row is not None:
                env_cfg.terrain.parkour.force_row = args.parkour_force_row
            if getattr(env_cfg.terrain.parkour, "force_row", None) is None:
                env_cfg.terrain.parkour.force_row = 0
            return
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.env.debug_draw_terrain_height_points = False
        
        
        # random uniform terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.random_uniform_terrain", 
        #                                   "min_height" : -0.05, "max_height": 0.05, 
        #                                   "step":0.005, "downsampled_scale" : 0.2}
        # slope
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_sloped_terrain",
        #                                   "slope": -0.4, "platform_size": 3.0}
        # stairs
        env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_stairs_terrain",
                                        "step_width": 0.31, "step_height": -0.1, "platform_size": 3.0}
        # discrete obstacles
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
        #                                   "max_height": 0.1,
        #                                   "min_size": 1.0,
        #                                   "max_size": 2.0,
        #                                   "num_rects": 20,
        #                                   "platform_size": 3.0}
        # wave terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.wave_terrain", 
        #                                   "amplitude": 0.1, "num_waves": 2}
        # stepping stones
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.stepping_stones_terrain",
        #                                   "stone_size": 1.0, "max_height": 0.1,
        #                                   "stone_distance": 0.3, "platform_size": 3.0}
        # gap terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.gap_terrain", 
        #                                   "gap_size": 0.2, "platform_size": 3.0}
        # pit terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pit_terrain", 
        #                                   "depth": 0.2, "platform_size": 3.0}
        
        
    if command_mode != "auto":
        if not supports_manual_velocity_commands(env_cfg):
            raise NotImplementedError(
                f"Manual command mode '{command_mode}' is only implemented for velocity-command tasks."
            )
        env_cfg.commands.heading_command = False

def print_debug_info(env, robot_index):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
    """
    # print debug info
    # print("base lin vel: ", env.simulator.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.simulator.base_euler[robot_index, 2].item())
    # print("base height: ", env.simulator.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.simulator.feet_pos[robot_index, :, 2].cpu().numpy())
    # print(f"ankle pitch: {env.simulator.dof_pos[robot_index, [3,7]].cpu().numpy()}")
    # print(f"actions: {env.simulator.dof_pos[robot_index].cpu().numpy()}")
    pass

def interaction_loop(env, policy, args, train_cfg, command_controller):
    """Run interaction loop between environment and policy

    Args:
        env: environment object
        policy : a policy that takes observations and outputs actions
        args: command line arguments
    """
    
    logger = Logger(env.dt) if args.log_play_stats else None
    robot_index = min(max(int(getattr(env.cfg.viewer, "ref_env", 0)), 0), env.num_envs - 1) # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
        
    # Get initial observations according to task type
    runner_class_name = train_cfg.runner_class_name
    if runner_class_name == "TSRunner":
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif runner_class_name == "ParkourStudentRunner":
        obs_buf, teacher_actor_obs, student_depth, _depth_updated = env.get_observations()
    elif runner_class_name == "EERunner":
        estimator_features, _, _ = env.get_observations()
    elif runner_class_name == "DreamWaQRunner":
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    elif runner_class_name == "CTSRunner":
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    else:
        obs = env.get_observations()
    
    frame_dt = env.dt
    next_frame_deadline = time.perf_counter()
    # interaction loop
    for i in range(10*int(env.max_episode_length)):
        if not command_controller.update(env):
            break
        
        # Non-LeggedRobot tasks can still opt into play-time camera follow here.
        if getattr(env, "follow_robot", bool(getattr(env.cfg.viewer, "follow_robot", False))) and SIMULATOR != "genesis":
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.pos, dtype=np.float32)
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.lookat, dtype=np.float32)
            env.set_viewer_camera(pos, lookat)
        
        # Step the environment according to task type
        if runner_class_name == "TSRunner":
            with torch.inference_mode():
                actions = policy(obs_buf, obs_history)
        elif runner_class_name == "ParkourStudentRunner":
            with torch.inference_mode():
                actions = policy(obs_buf, student_depth)
        elif runner_class_name == "EERunner":
            with torch.inference_mode():
                actions = policy(estimator_features.detach())
        elif runner_class_name == "DreamWaQRunner":
            with torch.inference_mode():
                actions = policy(obs_buf, obs_history)
        elif runner_class_name == "CTSRunner":
            with torch.inference_mode():
                actions = policy(obs_buf, obs_history)
        else:
            with torch.inference_mode():
                actions = policy(obs.detach())

        if runner_class_name == "TSRunner":
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif runner_class_name == "ParkourStudentRunner":
            obs_buf, teacher_actor_obs, student_depth, _depth_updated, rews, dones, infos = env.step(
                actions.detach()
            )
        elif runner_class_name == "EERunner":
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        elif runner_class_name == "DreamWaQRunner":
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        elif runner_class_name == "CTSRunner":
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        else:
            obs, _, rews, dones, infos = env.step(actions.detach())
        
        # print debug info
        print_debug_info(env, robot_index)
        
        # Update logger info
        if logger is not None and i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.simulator.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.simulator.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.simulator.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.simulator.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.simulator.link_contact_forces[robot_index, 
                                                                          env.simulator.feet_indices, 2].cpu().numpy()
                }
            )
        elif logger is not None and i==stop_state_log:
            logger.plot_states()
        if logger is not None and 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif logger is not None and i==stop_rew_log:
            logger.print_rewards()
        
        # sleep for the remainder of the frame budget to match real-time playback
        next_frame_deadline += frame_dt
        now = time.perf_counter()
        remaining = next_frame_deadline - now
        if remaining > 0:
            time.sleep(remaining)
        else:
            # If rendering or policy inference overruns the frame budget, drop the
            # wall-clock deadline instead of accumulating timing drift forever.
            next_frame_deadline = now

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg):
    """export the policy as jit script according to different task types

    Args:
        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    policy_class_name = train_cfg.runner.policy_class_name
    if policy_class_name == "ActorCriticTS":
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
    elif policy_class_name == "ActorCriticParkourStudent":
        exporter = PolicyExporterParkourStudent(alg_runner.alg.actor_critic)
    elif policy_class_name == "ActorCriticEE":
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
    elif policy_class_name == "ActorCriticDreamWaQ":
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
    exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    

def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    args.resume = args.resume or args.load_run is not None or (
        args.ckpt is not None and args.ckpt >= 0
    )
    ensure_runtime_initialized(args)
    args.command_mode = resolve_command_mode(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, train_cfg, args)
    command_controller = PlayCommandController(args, env_cfg)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.external_command_source_enabled = command_controller.requires_external_command_source
    command_controller.initialize_env_commands(env)
    # load policy
    train_cfg.runner.resume = args.resume
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++ or python)
    if args.resume:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'logs',
            train_cfg.runner.experiment_name,
            str(train_cfg.runner.load_run),
            'exported',
        )
        export_policy(ppo_runner, path, args, env_cfg, train_cfg)

    try:
        interaction_loop(env, policy, args, train_cfg, command_controller)
    finally:
        command_controller.close()
        env.close()
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
