#!/usr/bin/env python3
"""Check whether DA2 actually produces different depth values for gap vs non-gap pixels.

Runs on gap terrain with both GT depth and DA2 enabled, collecting per-pixel
comparisons specifically at gap boundaries.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/diagnose_da2_gaps.py --headless
"""

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import numpy as np

from legged_gym import SIMULATOR
from legged_gym.envs import *  # noqa
from legged_gym.utils import ensure_runtime_initialized, task_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--row", type=int, default=0)
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
        load_run=-1,
        ckpt=-1,
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
    # Enable BOTH GT depth and DA2
    env_cfg.sensor.add_depth = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg,
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        obs, teacher_obs, student_depth, _ = env.reset()

        near = env.cfg.sensor.depth_camera_config.near_clip
        far = env.cfg.sensor.depth_camera_config.far_clip

        # Collect statistics
        gap_da2_values = []      # DA2 values where GT shows gap (depth >= 1.5m)
        ground_da2_values = []   # DA2 values where GT shows ground (depth < 1.0m)
        mid_da2_values = []      # DA2 values where GT is in between
        all_gt = []
        all_da2 = []

        for step in range(cli.steps):
            with torch.inference_mode():
                actions = policy(obs, student_depth)

            obs, teacher_obs, student_depth, depth_updated, rews, dones, infos = env.step(actions.detach())

            if not depth_updated[0]:
                continue

            # GT depth: normalized in [-0.5, 0.5], convert to meters
            gt_norm = env.simulator._depth_images[:, 0]  # (N, H, W)
            gt_meters = (gt_norm + 0.5) * (far - near) + near

            # DA2 raw
            da2_raw = env.simulator._inferred_depth_images.clone()

            # Resize DA2 to match GT if needed
            if da2_raw.shape[-2:] != gt_meters.shape[-2:]:
                da2_raw = F.interpolate(
                    da2_raw.unsqueeze(1), size=gt_meters.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).squeeze(1)

            gt_flat = gt_meters.reshape(-1).cpu().numpy()
            da2_flat = da2_raw.reshape(-1).cpu().numpy()

            # Classify pixels
            gap_mask = gt_flat >= 1.5  # GT says far → likely gap
            ground_mask = gt_flat < 1.0  # GT says near → ground
            mid_mask = (~gap_mask) & (~ground_mask)

            gap_da2_values.extend(da2_flat[gap_mask].tolist())
            ground_da2_values.extend(da2_flat[ground_mask].tolist())
            mid_da2_values.extend(da2_flat[mid_mask].tolist())

            # Sample for overall stats
            if step % 10 == 0:
                all_gt.extend(gt_flat[:1000].tolist())
                all_da2.extend(da2_flat[:1000].tolist())

        print(f"\n{'='*70}")
        print(f"DA2 GAP ANALYSIS (Row {cli.row}, {cli.steps} steps)")
        print(f"{'='*70}")

        gap_arr = np.array(gap_da2_values)
        ground_arr = np.array(ground_da2_values)
        mid_arr = np.array(mid_da2_values)

        print(f"\nPixel counts:")
        print(f"  Gap (GT >= 1.5m):    {len(gap_arr):,}")
        print(f"  Ground (GT < 1.0m):  {len(ground_arr):,}")
        print(f"  Mid (1.0-1.5m):      {len(mid_arr):,}")

        if len(gap_arr) > 0:
            print(f"\nDA2 raw values by GT depth class:")
            print(f"  Gap pixels:    mean={gap_arr.mean():.3f}, std={gap_arr.std():.3f}, "
                  f"median={np.median(gap_arr):.3f}, range=[{gap_arr.min():.3f}, {gap_arr.max():.3f}]")
        if len(ground_arr) > 0:
            print(f"  Ground pixels: mean={ground_arr.mean():.3f}, std={ground_arr.std():.3f}, "
                  f"median={np.median(ground_arr):.3f}, range=[{ground_arr.min():.3f}, {ground_arr.max():.3f}]")
        if len(mid_arr) > 0:
            print(f"  Mid pixels:    mean={mid_arr.mean():.3f}, std={mid_arr.std():.3f}, "
                  f"median={np.median(mid_arr):.3f}, range=[{mid_arr.min():.3f}, {mid_arr.max():.3f}]")

        if len(gap_arr) > 0 and len(ground_arr) > 0:
            separation = gap_arr.mean() - ground_arr.mean()
            pooled_std = np.sqrt((gap_arr.std()**2 + ground_arr.std()**2) / 2)
            d_prime = separation / pooled_std if pooled_std > 0 else 0
            print(f"\n  Gap-Ground separation:")
            print(f"    Mean difference: {separation:.3f}")
            print(f"    d-prime (effect size): {d_prime:.3f}")
            print(f"    (d'>1.0 = good separation, d'>0.5 = moderate, d'<0.2 = poor)")

        # Overall pixel correlation
        if len(all_gt) > 1000:
            gt_arr = np.array(all_gt)
            da2_arr_all = np.array(all_da2)
            corr = np.corrcoef(gt_arr, da2_arr_all)[0, 1]
            print(f"\n  Overall pixel correlation (GT vs DA2): {corr:.4f}")

        # What does calibrated linear produce?
        print(f"\n  Calibrated linear output (scale=0.081393, offset=0.994514):")
        if len(gap_arr) > 0:
            cal_gap = np.clip(0.081393 * gap_arr + 0.994514, 0, 2) / 2 - 0.5
            print(f"    Gap pixels:    mean={cal_gap.mean():.4f}, range=[{cal_gap.min():.4f}, {cal_gap.max():.4f}]")
        if len(ground_arr) > 0:
            cal_ground = np.clip(0.081393 * ground_arr + 0.994514, 0, 2) / 2 - 0.5
            print(f"    Ground pixels: mean={cal_ground.mean():.4f}, range=[{cal_ground.min():.4f}, {cal_ground.max():.4f}]")
        if len(gap_arr) > 0 and len(ground_arr) > 0:
            cal_sep = cal_gap.mean() - cal_ground.mean()
            print(f"    Normalized gap signal: {cal_sep:.4f} (out of 1.0 range)")
            print(f"    Signal as % of range: {abs(cal_sep)*100:.1f}%")

        # Compare to EMA norm output
        print(f"\n  EMA norm output (for comparison):")
        if len(gap_arr) > 0 and len(ground_arr) > 0:
            all_vals = np.concatenate([gap_arr, ground_arr, mid_arr])
            p2, p98 = np.percentile(all_vals, [2, 98])
            ema_gap = np.clip((gap_arr - p2) / max(p98 - p2, 1e-3), 0, 1) - 0.5
            ema_ground = np.clip((ground_arr - p2) / max(p98 - p2, 1e-3), 0, 1) - 0.5
            ema_sep = ema_gap.mean() - ema_ground.mean()
            print(f"    Gap pixels:    mean={ema_gap.mean():.4f}")
            print(f"    Ground pixels: mean={ema_ground.mean():.4f}")
            print(f"    Normalized gap signal: {ema_sep:.4f}")
            print(f"    Signal as % of range: {abs(ema_sep)*100:.1f}%")

    finally:
        env.close()


if __name__ == "__main__":
    main()
