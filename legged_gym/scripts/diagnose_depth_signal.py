#!/usr/bin/env python3
"""Diagnose depth signal quality at gap approach points.

Runs student on gap terrain and captures:
- GT depth (from simulator) vs estimated depth (from DepthAnything)
- Scandot observations (what teacher sees)
- Correlation metrics between GT and estimated depth
- Per-pixel depth error statistics

CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/diagnose_depth_signal.py --headless
"""

import argparse
import os
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import numpy as np

from legged_gym import SIMULATOR
from legged_gym.envs import *
from legged_gym.utils import ensure_runtime_initialized, task_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--norm_mode", type=str, default=None,
                        help="Override depth normalization mode (e.g. calibrated_linear)")
    parser.add_argument("--affine_scale", type=float, default=None)
    parser.add_argument("--affine_offset", type=float, default=None)
    cli = parser.parse_args()

    args = SimpleNamespace(
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
        parkour_force_row=cli.row,
        depth_ablation="none",
    )
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(cli.num_envs, env_cfg.env.num_envs)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.parkour.force_row = cli.row
    env_cfg.terrain.parkour.force_family = "gap"

    # Apply normalization overrides via experiment_flags
    if cli.norm_mode:
        if not hasattr(env_cfg, "experiment_flags"):
            env_cfg.experiment_flags = {}
        env_cfg.experiment_flags["norm_mode"] = cli.norm_mode
        if cli.affine_scale is not None:
            env_cfg.experiment_flags["affine_scale"] = cli.affine_scale
        if cli.affine_offset is not None:
            env_cfg.experiment_flags["affine_offset"] = cli.affine_offset
        print(f"Using norm_mode={cli.norm_mode}, flags={env_cfg.experiment_flags}")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg,
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        obs, teacher_obs, student_depth, _depth_updated = env.reset()

        # Collectors
        gt_vs_est_correlations = []
        gt_vs_est_l1_errors = []
        depth_spatial_variance = []
        gt_depth_at_gap = []
        est_depth_at_gap = []
        student_depth_values = []

        # Track what the student depth looks like
        depth_histograms = []

        for step in range(cli.steps):
            with torch.inference_mode():
                actions = policy(obs, student_depth)

            # === Capture depth diagnostics before stepping ===

            # 1. Student's normalized depth (what the CNN sees)
            sd = student_depth.clone()  # (N, 1, H, W) in [-0.5, 0.5]
            student_depth_values.append({
                "mean": sd.mean().item(),
                "std": sd.std().item(),
                "min": sd.min().item(),
                "max": sd.max().item(),
                "spatial_std_per_env": sd.std(dim=(-2, -1)).mean().item(),
            })

            # 2. Correlate depth with scandot heights (teacher terrain info)
            # Scandots are height measurements at grid points; depth image captures similar info
            prop_dim = env.num_obs
            scandots = teacher_obs[:, prop_dim:]  # (N, scandot_dim)
            scandot_mean = scandots.mean(dim=1)  # per-env average scandot height
            depth_mean_per_env = sd.mean(dim=(-2, -1)).squeeze()  # per-env average depth
            # Correlation between scandot height and depth value
            if scandot_mean.numel() > 2:
                corr = torch.corrcoef(torch.stack([scandot_mean, depth_mean_per_env]))[0, 1].item()
                if not math.isnan(corr):
                    gt_vs_est_correlations.append(corr)
            # Also check: when scandots show a gap (large negative values), does depth change?
            # Gap in scandots = very low values (terrain drops away)
            gap_scandot_threshold = -0.3  # scandot values below this indicate a gap
            has_gap = (scandots.min(dim=1).values < gap_scandot_threshold)
            if has_gap.any():
                gap_depth = sd[has_gap].mean().item()
                no_gap_depth = sd[~has_gap].mean().item() if (~has_gap).any() else gap_depth
                gt_vs_est_l1_errors.append(gap_depth - no_gap_depth)  # depth diff at gap vs no gap

            # 3. Check depth spatial structure (does the depth image have gap-like features?)
            # A gap should appear as a region of high depth (far) surrounded by low depth (near ground)
            # Look at the bottom half of the image (closer to robot)
            bottom_half = sd[:, :, sd.shape[2]//2:, :]
            top_half = sd[:, :, :sd.shape[2]//2, :]
            depth_spatial_variance.append({
                "bottom_std": bottom_half.std().item(),
                "top_std": top_half.std().item(),
                "bottom_mean": bottom_half.mean().item(),
                "top_mean": top_half.mean().item(),
            })

            # === Step environment ===
            (obs, teacher_obs, student_depth, _depth_updated, _rews, dones, infos) = env.step(actions.detach())

        # === Print diagnostics ===
        print("\n" + "=" * 70)
        print(f"DEPTH SIGNAL DIAGNOSIS (Row {cli.row}, {cli.steps} steps)")
        print("=" * 70)

        # Student depth stats
        means = [s["mean"] for s in student_depth_values]
        stds = [s["std"] for s in student_depth_values]
        spatial_stds = [s["spatial_std_per_env"] for s in student_depth_values]
        print(f"\n--- Student Normalized Depth (what CNN sees) ---")
        print(f"  Mean of means: {np.mean(means):.4f}")
        print(f"  Mean of stds:  {np.mean(stds):.4f}")
        print(f"  Mean spatial std per env: {np.mean(spatial_stds):.4f}")
        print(f"  Range: [{np.mean([s['min'] for s in student_depth_values]):.3f}, {np.mean([s['max'] for s in student_depth_values]):.3f}]")

        # Spatial structure
        print(f"\n--- Depth Spatial Structure ---")
        bot_stds = [s["bottom_std"] for s in depth_spatial_variance]
        top_stds = [s["top_std"] for s in depth_spatial_variance]
        bot_means = [s["bottom_mean"] for s in depth_spatial_variance]
        top_means = [s["top_mean"] for s in depth_spatial_variance]
        print(f"  Bottom half (near ground): mean={np.mean(bot_means):.4f}, std={np.mean(bot_stds):.4f}")
        print(f"  Top half (further away):   mean={np.mean(top_means):.4f}, std={np.mean(top_stds):.4f}")
        print(f"  Bottom-Top mean diff:      {np.mean(bot_means) - np.mean(top_means):.4f}")

        # Scandot-depth correlation
        if gt_vs_est_correlations:
            print(f"\n--- Scandot-Depth Correlation ---")
            print(f"  Mean correlation (scandot height vs depth mean): {np.mean(gt_vs_est_correlations):.4f}")
            print(f"  Std correlation: {np.std(gt_vs_est_correlations):.4f}")
        if gt_vs_est_l1_errors:
            print(f"  Depth difference (gap present vs absent): {np.mean(gt_vs_est_l1_errors):.4f}")
            print(f"    (positive = depth is higher/further when gap detected by scandots)")
        else:
            print(f"\n--- Scandot-Depth: no gap detections in scandots ---")

        # Also: compare what scandots see vs depth
        print(f"\n--- Scandot (Teacher) Observations ---")
        # The teacher_obs includes scandots after proprioception
        # prop_dim is typically 53 for Go2 parkour
        prop_dim = env.num_obs  # student prop obs dim
        teacher_extra = teacher_obs.shape[1] - prop_dim
        print(f"  Teacher obs dim: {teacher_obs.shape[1]}")
        print(f"  Student obs dim (prop): {prop_dim}")
        print(f"  Scandot dim: {teacher_extra}")
        # Scandot values from last step
        scandots = teacher_obs[:, prop_dim:]
        print(f"  Scandot mean: {scandots.mean().item():.4f}")
        print(f"  Scandot std:  {scandots.std().item():.4f}")
        print(f"  Scandot range: [{scandots.min().item():.3f}, {scandots.max().item():.3f}]")

        # Simulator info
        print(f"\n--- Simulator Info ---")
        print(f"  _use_inferred_depth: {env._use_inferred_depth}")
        print(f"  _depth_render_interval: {env._depth_render_interval}")
        print(f"  depth buffer shape: {env._depth_buffer.shape}")
        print(f"  norm_mode: {env._experiment_norm_mode or 'ema (default)'}")
        if hasattr(env, '_depth_ema_lo'):
            print(f"  EMA lo: {env._depth_ema_lo.item():.4f}")
            print(f"  EMA hi: {env._depth_ema_hi.item():.4f}")
            print(f"  EMA span: {(env._depth_ema_hi - env._depth_ema_lo).item():.4f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
