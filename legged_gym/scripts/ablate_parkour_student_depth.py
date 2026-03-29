#!/usr/bin/env python3
"""Headless rollout evaluation with depth ablations for parkour student policies.

Compare runs with the same checkpoint:

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/ablate_parkour_student_depth.py \\
        --task go2_parkour_depth_est_student --headless --episodes 256 \\
        --depth_ablation none

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/ablate_parkour_student_depth.py \\
        --task go2_parkour_depth_est_student --headless --episodes 256 \\
        --depth_ablation zero

If ``success_rate`` and ``mean_progress_ratio`` collapse under ``zero`` / ``noise`` but stay
high under ``none``, the policy is likely using the depth stream (not only proprioception).
``shuffle`` requires ``num_envs >= 2`` and breaks env-depth correspondence.
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch

from legged_gym import SIMULATOR
from legged_gym.envs import *
from legged_gym.utils import ensure_runtime_initialized, task_registry
from legged_gym.utils.student_depth_ablation import apply_student_depth_ablation


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(value)


def _build_args(cli, depth_ablation: str):
    return SimpleNamespace(
        task=cli.task,
        headless=cli.headless,
        cpu=cli.cpu,
        num_envs=cli.num_envs,
        resume=True,
        sync_wandb=False,
        debug=False,
        follow_robot=False,
        motion_file=None,
        max_iterations=None,
        load_run=cli.load_run if cli.load_run is not None else -1,
        ckpt=cli.ckpt,
        parkour_force_family=cli.parkour_force_family,
        parkour_force_row=cli.parkour_force_row,
        depth_ablation=depth_ablation,
    )


def _parse_experiment_flags(flag_str):
    """Parse 'key=val,key2=val2' into a dict with auto-typed values."""
    if not flag_str:
        return {}
    flags = {}
    for item in flag_str.split(","):
        k, v = item.split("=", 1)
        # Auto-type: bool, int, float, string
        if v.lower() in ("true", "false"):
            flags[k.strip()] = v.lower() == "true"
        else:
            try:
                flags[k.strip()] = int(v)
            except ValueError:
                try:
                    flags[k.strip()] = float(v)
                except ValueError:
                    flags[k.strip()] = v.strip()
    return flags


def _evaluate(cli_args, depth_ablation: str):
    if SIMULATOR != "genesis":
        raise RuntimeError("This script only supports SIMULATOR=genesis.")

    args = _build_args(cli_args, depth_ablation)
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    allowed_runners = ("ParkourStudentRunner", "ParkourScandotStudentRunner")
    if train_cfg.runner_class_name not in allowed_runners:
        raise RuntimeError(f"Task {args.task!r} must use one of {allowed_runners}, got {train_cfg.runner_class_name!r}.")

    env_cfg.env.num_envs = min(args.num_envs, env_cfg.env.num_envs)
    env_cfg.terrain.curriculum = False
    if getattr(env_cfg.terrain, "parkour", None) is not None and getattr(env_cfg.terrain.parkour, "enable", False):
        env_cfg.terrain.parkour.force_row = args.parkour_force_row
        env_cfg.terrain.parkour.force_family = args.parkour_force_family

    # Apply experiment flags so depth processing matches training
    experiment_flags = _parse_experiment_flags(getattr(cli_args, "experiment_flags", None))
    if experiment_flags:
        env_cfg.experiment_flags = experiment_flags
        for k, v in experiment_flags.items():
            print(f"[eval] experiment_flag.{k} = {v}")

    # Apply env overrides (e.g., sensor.depth_estimation.model_size=base)
    if hasattr(cli_args, "env_override") and cli_args.env_override:
        import json as _json
        for item in cli_args.env_override:
            k, v = item.split("=", 1)
            # Auto-type: JSON first (handles lists/dicts), then bool, int, float, string
            if v.startswith("[") or v.startswith("{"):
                typed_v = _json.loads(v)
            elif v.lower() in ("true", "false"):
                typed_v = v.lower() == "true"
            else:
                try:
                    typed_v = int(v)
                except ValueError:
                    try:
                        typed_v = float(v)
                    except ValueError:
                        typed_v = v
            parts = k.split(".")
            obj = env_cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], typed_v)
            print(f"[eval] env_cfg.{k} = {typed_v}")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        obs, _teacher_obs, student_depth, _depth_updated = env.reset()
        completed_episodes = 0
        success_sum = 0.0
        progress_sum = 0.0
        waypoints_sum = 0.0

        max_steps = cli_args.max_steps
        if max_steps is None:
            max_steps = int(math.ceil(cli_args.episodes / env.num_envs) * env.max_episode_length * 2)

        for _ in range(max_steps):
            depth_in = apply_student_depth_ablation(student_depth, depth_ablation)
            with torch.inference_mode():
                actions = policy(obs, depth_in)
            (
                obs,
                _teacher_obs,
                student_depth,
                _depth_updated,
                _rews,
                dones,
                infos,
            ) = env.step(actions.detach())

            num_resets = int(torch.sum(dones).item())
            if num_resets <= 0 or "episode" not in infos:
                continue
            episode_info = infos["episode"]
            completed_episodes += num_resets
            success_sum += _to_float(episode_info["success"]) * num_resets
            progress_sum += _to_float(episode_info["progress_ratio"]) * num_resets
            waypoints_sum += _to_float(episode_info["waypoints_reached"]) * num_resets
            if completed_episodes >= cli_args.episodes:
                break
    finally:
        env.close()

    if completed_episodes == 0:
        raise RuntimeError("Evaluation completed without any finished episodes.")

    denom = float(completed_episodes)
    return {
        "depth_ablation": depth_ablation,
        "task": args.task,
        "load_run": train_cfg.runner.load_run,
        "checkpoint": train_cfg.runner.checkpoint,
        "episodes": completed_episodes,
        "family": getattr(env_cfg.terrain.parkour, "force_family", None),
        "row": getattr(env_cfg.terrain.parkour, "force_row", None),
        "success_rate": success_sum / denom,
        "mean_progress_ratio": progress_sum / denom,
        "mean_waypoints_reached": waypoints_sum / denom,
    }


def _print_metrics(m):
    print(f"depth_ablation={m['depth_ablation']}")
    print(f"task={m['task']}")
    print(f"load_run={m['load_run']}")
    print(f"checkpoint={m['checkpoint']}")
    print(f"episodes={m['episodes']}")
    print(f"family={m['family']}")
    print(f"row={m['row']}")
    print(f"success_rate={m['success_rate']:.4f}")
    print(f"mean_progress_ratio={m['mean_progress_ratio']:.4f}")
    print(f"mean_waypoints_reached={m['mean_waypoints_reached']:.4f}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=str, default="go2_parkour_depth_est_student")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--num_envs", type=int, default=48)
    p.add_argument("--load_run", type=str, default=None)
    p.add_argument("--ckpt", type=int, default=-1)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument(
        "--depth_ablation",
        type=str,
        default="none",
        choices=["none", "zero", "noise", "shuffle"],
        help="Corrupt depth before policy forward (see legged_gym.utils.student_depth_ablation).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run none, zero, and noise back-to-back (shuffle too if num_envs>=2).",
    )
    p.add_argument("--parkour_force_family", type=str, default=None, choices=["stairs", "hurdle_block", "gap"])
    p.add_argument("--parkour_force_row", type=int, default=0)
    p.add_argument("--experiment_flags", type=str, default=None,
                   help="Comma-separated key=val pairs, e.g. 'depth_temporal_smooth=0.5,depth_gradient=true'")
    p.add_argument("--env_override", type=str, nargs="*", default=None,
                   help="Override env config, e.g. 'sensor.depth_estimation.model_size=base'")
    return p.parse_args()


def main():
    cli_args = _parse_args()
    if cli_args.compare:
        modes = ["none", "zero", "noise"]
        if cli_args.num_envs >= 2:
            modes.append("shuffle")
        for mode in modes:
            print("=" * 60)
            metrics = _evaluate(cli_args, mode)
            _print_metrics(metrics)
            print()
        return

    metrics = _evaluate(cli_args, cli_args.depth_ablation)
    _print_metrics(metrics)


if __name__ == "__main__":
    main()
