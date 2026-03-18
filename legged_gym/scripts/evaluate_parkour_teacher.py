import argparse
import math
from types import SimpleNamespace

import torch

from legged_gym import SIMULATOR
from legged_gym.envs import *
from legged_gym.utils import ensure_runtime_initialized, task_registry


def _build_args(cli_args):
    return SimpleNamespace(
        task=cli_args.task,
        headless=cli_args.headless,
        cpu=cli_args.cpu,
        num_envs=cli_args.num_envs,
        resume=True,
        sync_wandb=False,
        debug=False,
        follow_robot=False,
        motion_file=None,
        max_iterations=None,
        load_run=cli_args.load_run if cli_args.load_run is not None else -1,
        ckpt=cli_args.ckpt,
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
    parser.add_argument("--sweep_all", action="store_true", default=False)
    parser.add_argument(
        "--parkour_force_family",
        type=str,
        default=None,
        choices=["stairs", "hurdle_block", "gap"],
    )
    parser.add_argument("--parkour_force_row", type=int, default=0)
    return parser.parse_args()


def _evaluate_single_combo(cli_args, family, row):
    if SIMULATOR != "genesis":
        raise RuntimeError("Parkour teacher evaluation only supports SIMULATOR=genesis.")

    args = _build_args(cli_args)
    args.parkour_force_family = family
    args.parkour_force_row = row
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(args.num_envs, env_cfg.env.num_envs)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.parkour.force_row = row
    env_cfg.terrain.parkour.force_family = family

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
    return {
        "task": args.task,
        "load_run": train_cfg.runner.load_run,
        "checkpoint": train_cfg.runner.checkpoint,
        "episodes": completed_episodes,
        "family": env_cfg.terrain.parkour.force_family,
        "row": env_cfg.terrain.parkour.force_row,
        "success_rate": success_sum / denom,
        "mean_progress_ratio": progress_sum / denom,
        "mean_waypoints_reached": waypoints_sum / denom,
        "out_of_lane_rate": out_of_lane_sum / denom,
        "global_out_of_bounds_rate": global_oob_sum / denom,
    }


def _print_metrics(metrics):
    print(f"task={metrics['task']}")
    print(f"load_run={metrics['load_run']}")
    print(f"checkpoint={metrics['checkpoint']}")
    print(f"episodes={metrics['episodes']}")
    print(f"family={metrics['family']}")
    print(f"row={metrics['row']}")
    print(f"success_rate={metrics['success_rate']:.4f}")
    print(f"mean_progress_ratio={metrics['mean_progress_ratio']:.4f}")
    print(f"mean_waypoints_reached={metrics['mean_waypoints_reached']:.4f}")
    print(f"out_of_lane_rate={metrics['out_of_lane_rate']:.4f}")
    print(f"global_out_of_bounds_rate={metrics['global_out_of_bounds_rate']:.4f}")


def evaluate(cli_args):
    if cli_args.sweep_all:
        env_cfg, _ = task_registry.get_cfgs(name=cli_args.task)
        families = list(env_cfg.terrain.parkour.families)
        rows = range(env_cfg.terrain.num_rows)
        for family in families:
            for row in rows:
                metrics = _evaluate_single_combo(cli_args, family=family, row=row)
                _print_metrics(metrics)
                print("-" * 40)
        return

    metrics = _evaluate_single_combo(
        cli_args,
        family=cli_args.parkour_force_family,
        row=cli_args.parkour_force_row,
    )
    _print_metrics(metrics)


if __name__ == "__main__":
    evaluate(_parse_args())
