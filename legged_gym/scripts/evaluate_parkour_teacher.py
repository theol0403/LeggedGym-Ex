import argparse
import math
from types import SimpleNamespace

import torch

from legged_gym import SIMULATOR
from legged_gym.envs import *
from legged_gym.utils import ensure_runtime_initialized, task_registry


def _build_args(cli_args):
    resume = cli_args.load_run is not None or cli_args.ckpt >= 0
    return SimpleNamespace(
        task=cli_args.task,
        headless=cli_args.headless,
        cpu=cli_args.cpu,
        num_envs=cli_args.num_envs,
        max_iterations=None,
        resume=resume,
        sync_wandb=False,
        export_onnx=False,
        debug=False,
        depth_debug=False,
        rgb_debug=False,
        depth_model_type=None,
        depth_model_size=None,
        depth_update_interval=1,
        load_run=cli_args.load_run,
        ckpt=cli_args.ckpt,
        command_mode="auto",
        command_scale=1.0,
        use_joystick=False,
        joystick_type="xbox",
        follow_robot=False,
        log_play_stats=False,
        motion_file=None,
        parkour_force_family=cli_args.parkour_force_family,
        parkour_force_row=cli_args.parkour_force_row,
    )


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(value)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="go2_parkour_teacher")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--parkour_force_family",
        type=str,
        default=None,
        choices=["stairs", "hurdle_block", "gap"],
    )
    parser.add_argument("--parkour_force_row", type=int, default=0)
    return parser.parse_args()


def evaluate(cli_args):
    if SIMULATOR != "genesis":
        raise RuntimeError("Parkour teacher evaluation only supports SIMULATOR=genesis.")

    args = _build_args(cli_args)
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(args.num_envs, env_cfg.env.num_envs)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.parkour.force_row = args.parkour_force_row
    if args.parkour_force_family is not None:
        env_cfg.terrain.parkour.force_family = args.parkour_force_family

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        obs, _ = env.reset()
        completed_episodes = 0
        success_sum = 0.0
        progress_sum = 0.0
        waypoints_sum = 0.0
        out_of_lane_sum = 0.0
        global_oob_sum = 0.0

        max_steps = cli_args.max_steps
        if max_steps is None:
            max_steps = int(math.ceil(cli_args.episodes / env.num_envs) * env.max_episode_length * 2)

        for _ in range(max_steps):
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, dones, infos = env.step(actions)
            num_resets = int(torch.sum(dones).item())
            if num_resets <= 0 or "episode" not in infos:
                continue
            episode_info = infos["episode"]
            completed_episodes += num_resets
            success_sum += _to_float(episode_info["success"]) * num_resets
            progress_sum += _to_float(episode_info["progress_ratio"]) * num_resets
            waypoints_sum += _to_float(episode_info["waypoints_reached"]) * num_resets
            out_of_lane_sum += _to_float(episode_info["out_of_lane"]) * num_resets
            global_oob_sum += _to_float(episode_info["global_out_of_bounds"]) * num_resets
            if completed_episodes >= cli_args.episodes:
                break
    finally:
        env.close()

    if completed_episodes == 0:
        raise RuntimeError("Evaluation completed without any finished episodes.")

    denom = float(completed_episodes)
    print(f"task={args.task}")
    print(f"load_run={train_cfg.runner.load_run}")
    print(f"checkpoint={train_cfg.runner.checkpoint}")
    print(f"episodes={completed_episodes}")
    print(f"family={env_cfg.terrain.parkour.force_family}")
    print(f"row={env_cfg.terrain.parkour.force_row}")
    print(f"success_rate={success_sum / denom:.4f}")
    print(f"mean_progress_ratio={progress_sum / denom:.4f}")
    print(f"mean_waypoints_reached={waypoints_sum / denom:.4f}")
    print(f"out_of_lane_rate={out_of_lane_sum / denom:.4f}")
    print(f"global_out_of_bounds_rate={global_oob_sum / denom:.4f}")


if __name__ == "__main__":
    evaluate(_parse_args())
