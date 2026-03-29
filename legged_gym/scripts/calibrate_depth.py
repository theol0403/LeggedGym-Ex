#!/usr/bin/env python3
"""Calibrate affine mapping from DA2 raw depth to GT depth (meters).

Enables BOTH GT depth and depth estimation simultaneously, runs the env for
several hundred steps, and fits a robust affine transform:

    gt_meters = scale * da2_raw + offset

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/calibrate_depth.py --headless
"""

import argparse
from types import SimpleNamespace

import torch
import numpy as np

from legged_gym import SIMULATOR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import ensure_runtime_initialized, task_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    cli = parser.parse_args()

    args = SimpleNamespace(
        task="go2_parkour_depth_est_student",
        headless=cli.headless,
        cpu=False,
        num_envs=cli.num_envs,
        resume=False,
        sync_wandb=False,
        debug=False,
        follow_robot=False,
        motion_file=None,
        max_iterations=None,
        load_run=None,
        ckpt=-1,
        parkour_force_family=None,
        parkour_force_row=None,
        depth_ablation="none",
    )
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(cli.num_envs, env_cfg.env.num_envs)

    # Enable BOTH GT depth and depth estimation
    env_cfg.sensor.add_depth = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    try:
        # Storage for paired samples
        gt_all = []
        da2_all = []

        obs, teacher_obs, student_depth, _ = env.reset()
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

        for step in range(cli.steps):
            obs, teacher_obs, student_depth, depth_updated, rews, dones, infos = env.step(actions)

            if not depth_updated[0]:
                continue

            # GT depth: stored in simulator._depth_images, already linearly normalized
            # Layout: (N, num_history, H, W) in [-0.5, 0.5]
            gt_norm = env.simulator._depth_images[:, 0]  # (N, H, W) newest frame
            near = env.cfg.sensor.depth_camera_config.near_clip
            far = env.cfg.sensor.depth_camera_config.far_clip
            gt_meters = (gt_norm + 0.5) * (far - near) + near  # undo linear norm

            # DA2 raw: stored in simulator._inferred_depth_images (N, H, W)
            da2_raw = env.simulator._inferred_depth_images.clone()  # (N, H, W)

            # The GT depth may be a different resolution than DA2 raw if
            # processed_resolution != rgb resolution. Resize DA2 to match GT.
            if da2_raw.shape[-2:] != gt_meters.shape[-2:]:
                da2_raw = torch.nn.functional.interpolate(
                    da2_raw.unsqueeze(1),
                    size=gt_meters.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            # Flatten and filter valid pixels (GT in [0, far] range)
            gt_flat = gt_meters.reshape(-1)
            da2_flat = da2_raw.reshape(-1)
            valid = (gt_flat >= 0) & (gt_flat <= far) & torch.isfinite(da2_flat)
            gt_all.append(gt_flat[valid].cpu())
            da2_all.append(da2_flat[valid].cpu())

            if step % 100 == 0:
                print(f"  Step {step}/{cli.steps}, collected {sum(t.numel() for t in gt_all)} pairs")

        gt_cat = torch.cat(gt_all).numpy()
        da2_cat = torch.cat(da2_all).numpy()
        n = len(gt_cat)
        print(f"\nCollected {n:,} valid pixel pairs")
        print(f"  GT meters range:  [{gt_cat.min():.3f}, {gt_cat.max():.3f}]")
        print(f"  DA2 raw range:    [{da2_cat.min():.3f}, {da2_cat.max():.3f}]")

        # Fit affine: gt = scale * da2 + offset via least squares
        A = np.column_stack([da2_cat, np.ones(n)])
        result, residuals, rank, sv = np.linalg.lstsq(A, gt_cat, rcond=None)
        scale, offset = result
        print(f"\n{'='*60}")
        print(f"  CALIBRATION RESULT (least squares)")
        print(f"  gt_meters = {scale:.6f} * da2_raw + {offset:.6f}")
        print(f"{'='*60}")

        # Also fit with RANSAC-like robust estimation (median of slopes)
        # Use random subsets of 2 points to estimate slope, take median
        rng = np.random.default_rng(42)
        n_samples = min(50000, n)
        idx = rng.choice(n, size=(n_samples, 2), replace=True)
        dx = da2_cat[idx[:, 1]] - da2_cat[idx[:, 0]]
        dy = gt_cat[idx[:, 1]] - gt_cat[idx[:, 0]]
        valid_pairs = np.abs(dx) > 1e-3
        slopes = dy[valid_pairs] / dx[valid_pairs]
        robust_scale = np.median(slopes)
        robust_offset = np.median(gt_cat - robust_scale * da2_cat)
        print(f"\n  ROBUST FIT (Theil-Sen)")
        print(f"  gt_meters = {robust_scale:.6f} * da2_raw + {robust_offset:.6f}")

        # Evaluate both fits
        for name, s, o in [("Least Squares", scale, offset), ("Theil-Sen", robust_scale, robust_offset)]:
            pred = s * da2_cat + o
            pred_clamp = np.clip(pred, 0, far)
            mae = np.mean(np.abs(gt_cat - pred_clamp))
            rmse = np.sqrt(np.mean((gt_cat - pred_clamp) ** 2))
            corr = np.corrcoef(gt_cat, pred_clamp)[0, 1]
            print(f"\n  {name}: MAE={mae:.4f}m, RMSE={rmse:.4f}m, corr={corr:.4f}")

        # Gap signal analysis: how much of the [0, 2] range do gaps occupy?
        pred_ls = np.clip(scale * da2_cat + offset, 0, far)
        # Pixels near the max depth are likely gaps (terrain drops away)
        gap_threshold = far * 0.8  # 1.6m
        gap_fraction = np.mean(pred_ls > gap_threshold)
        near_fraction = np.mean(pred_ls < far * 0.3)
        print(f"\n  Gap signal analysis (LS fit):")
        print(f"    Pixels > {gap_threshold:.1f}m (gap-like):   {gap_fraction*100:.1f}%")
        print(f"    Pixels < {far*0.3:.1f}m (near ground): {near_fraction*100:.1f}%")
        print(f"    Depth range used: [{pred_ls.min():.3f}, {pred_ls.max():.3f}]m")

        print(f"\n  Use these values in experiment_flags:")
        print(f'    "affine_scale": {scale:.6f},')
        print(f'    "affine_offset": {offset:.6f},')

    finally:
        env.close()


if __name__ == "__main__":
    main()
