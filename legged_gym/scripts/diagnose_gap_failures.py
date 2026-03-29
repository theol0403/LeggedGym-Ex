#!/usr/bin/env python3
"""Diagnose gap failures: per-row success, depth signal quality, action divergence.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/diagnose_gap_failures.py \
        --headless --episodes 200

Runs the student on gap terrain at each difficulty row and collects:
- Per-row success rate and progress ratio
- Depth ablation comparison (none vs zero) to measure depth reliance
- Action divergence between student and teacher near gap edges
- Depth signal statistics (mean, std, range near gaps)
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace
from collections import defaultdict

import torch
import numpy as np

from legged_gym import SIMULATOR
from legged_gym.envs import *
from legged_gym.utils import ensure_runtime_initialized, task_registry
from legged_gym.utils.helpers import class_to_dict, get_load_path
from rsl_rl.modules import ActorCriticParkour, ActorCriticParkourStudent


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(value)


def _build_args(cli):
    return SimpleNamespace(
        task="go2_parkour_depth_est_student",
        headless=cli.headless,
        cpu=False,
        num_envs=cli.num_envs,
        resume=True,
        sync_wandb=False,
        debug=False,
        follow_robot=False,
        motion_file=None,
        max_iterations=None,
        load_run=cli.load_run if cli.load_run is not None else -1,
        ckpt=cli.ckpt,
        parkour_force_family="gap",
        parkour_force_row=0,  # will be overridden
        depth_ablation="none",
    )


def evaluate_row(cli, row, depth_ablation="none"):
    """Run student on gaps at specified row, return detailed metrics."""
    args = _build_args(cli)
    args.parkour_force_row = row
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(cli.num_envs, env_cfg.env.num_envs)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.parkour.force_row = row
    env_cfg.terrain.parkour.force_family = "gap"

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg,
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        # Also load teacher for action comparison
        teacher = ppo_runner.teacher

        obs, teacher_obs, student_depth, _depth_updated = env.reset()
        completed = 0
        success_sum = 0.0
        progress_sum = 0.0

        # Metrics collectors
        action_diffs = []  # L2 distance between student/teacher actions
        depth_stats = []   # depth image statistics per step
        near_gap_action_diffs = []  # action diff when near a gap obstacle
        far_from_gap_action_diffs = []  # action diff when far from gap

        max_steps = int(math.ceil(cli.episodes / env.num_envs) * env.max_episode_length * 2)

        for step in range(max_steps):
            # Apply depth ablation
            if depth_ablation == "zero":
                depth_in = torch.zeros_like(student_depth)
            else:
                depth_in = student_depth.clone()

            with torch.inference_mode():
                # Student action
                student_actions = policy(obs, depth_in)

                # Teacher action (for comparison)
                teacher_actions = teacher.act_inference(teacher_obs)

            # Compute action divergence
            action_l2 = torch.norm(student_actions - teacher_actions, dim=-1)
            action_diffs.append(action_l2.mean().item())

            # Collect depth statistics
            depth_mean = student_depth.mean().item()
            depth_std = student_depth.std().item()
            depth_min = student_depth.min().item()
            depth_max = student_depth.max().item()
            depth_stats.append((depth_mean, depth_std, depth_min, depth_max))

            # Check if robots are near gap obstacles using terrain_row/progress
            # Use progress_ratio to distinguish near-gap vs far
            if hasattr(env, 'progress_ratio'):
                progress = env.progress_ratio
                # Near gap = high progress (approaching next obstacle)
                near_mask = progress > 0.3
                far_mask = ~near_mask
                if near_mask.any():
                    near_gap_action_diffs.append(action_l2[near_mask].mean().item())
                if far_mask.any():
                    far_from_gap_action_diffs.append(action_l2[far_mask].mean().item())

            # Step env
            (obs, teacher_obs, student_depth, _depth_updated, _rews, dones, infos) = env.step(student_actions.detach())

            num_resets = int(torch.sum(dones).item())
            if num_resets > 0 and "episode" in infos:
                ep = infos["episode"]
                completed += num_resets
                success_sum += _to_float(ep["success"]) * num_resets
                progress_sum += _to_float(ep["progress_ratio"]) * num_resets
                if completed >= cli.episodes:
                    break

    finally:
        env.close()

    denom = max(completed, 1)
    return {
        "row": row,
        "depth_ablation": depth_ablation,
        "episodes": completed,
        "success_rate": success_sum / denom,
        "progress_ratio": progress_sum / denom,
        "mean_action_diff": np.mean(action_diffs) if action_diffs else 0,
        "std_action_diff": np.std(action_diffs) if action_diffs else 0,
        "near_gap_action_diff": np.mean(near_gap_action_diffs) if near_gap_action_diffs else 0,
        "far_from_gap_action_diff": np.mean(far_from_gap_action_diffs) if far_from_gap_action_diffs else 0,
        "depth_mean": np.mean([s[0] for s in depth_stats]) if depth_stats else 0,
        "depth_std_mean": np.mean([s[1] for s in depth_stats]) if depth_stats else 0,
        "depth_range": np.mean([s[3] - s[2] for s in depth_stats]) if depth_stats else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=48)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--episodes", type=int, default=200)
    cli = parser.parse_args()

    # Gap difficulty parameters for reference
    gap_params = {
        0: {"width": 0.24, "y_half": 0.85, "obstacles": 2},
        1: {"width": 0.32, "y_half": 0.95, "obstacles": 3},
        2: {"width": 0.42, "y_half": 1.05, "obstacles": 3},
        3: {"width": 0.50, "y_half": 1.15, "obstacles": 3},
    }

    print("=" * 80)
    print("GAP FAILURE DIAGNOSIS")
    print("=" * 80)

    all_results = []

    # Phase 1: Per-row success rates
    print("\n--- Phase 1: Per-row gap success (normal depth) ---")
    for row in range(4):
        params = gap_params[row]
        print(f"\nEvaluating Row {row}: gap_width={params['width']}m, obstacles={params['obstacles']}...")
        result = evaluate_row(cli, row, depth_ablation="none")
        all_results.append(result)
        print(f"  Success: {result['success_rate']:.1%}  Progress: {result['progress_ratio']:.3f}")
        print(f"  Action diff (student vs teacher): {result['mean_action_diff']:.4f} ± {result['std_action_diff']:.4f}")
        print(f"  Depth stats: mean={result['depth_mean']:.3f}, std={result['depth_std_mean']:.3f}, range={result['depth_range']:.3f}")

    # Phase 2: Depth ablation on hardest failing row
    print("\n\n--- Phase 2: Depth ablation comparison (Row 0 vs Row 3) ---")
    for row in [0, 3]:
        print(f"\n  Row {row} with ZERO depth:")
        result_zero = evaluate_row(cli, row, depth_ablation="zero")
        normal = [r for r in all_results if r['row'] == row][0]
        delta = normal['success_rate'] - result_zero['success_rate']
        print(f"    Success: {result_zero['success_rate']:.1%} (vs {normal['success_rate']:.1%} with depth, delta={delta:+.1%})")
        print(f"    Action diff: {result_zero['mean_action_diff']:.4f} (vs {normal['mean_action_diff']:.4f})")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Row':>3} {'Width':>6} {'Obs':>3} {'Success':>8} {'Progress':>9} {'ActionDiff':>10}")
    print("-" * 50)
    for r in all_results:
        params = gap_params[r['row']]
        print(f"{r['row']:>3} {params['width']:>6.2f}m {params['obstacles']:>3} {r['success_rate']:>8.1%} {r['progress_ratio']:>9.3f} {r['mean_action_diff']:>10.4f}")


if __name__ == "__main__":
    main()
