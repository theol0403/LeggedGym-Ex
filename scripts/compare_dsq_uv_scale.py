#!/usr/bin/env python3
"""Compare DA2 depth signal quality (DSQ) between uv_scale=10 and uv_scale=4.

Runs a short rollout for each texture mode at each UV scale, computing Pearson
correlation between GT depth and DA2 depth.
"""
import math
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from legged_gym import SIMULATOR
from legged_gym.envs import *
from legged_gym.utils import ensure_runtime_initialized, task_registry

TEXTURE_MODES = ["checkerboard", "wood", "concrete"]
UV_SCALES = [10.0, 4.0]
NUM_STEPS = 200
NUM_ENVS = 16
TASK = "go2_parkour_depth_est_student"
LOAD_RUN = "Apr11_16-03-28_BASE3_da2_base_texture"
CKPT = 5000


def compute_dsq(uv_scale: float, texture_mode: str) -> float:
    args = SimpleNamespace(
        task=TASK, headless=True, cpu=False, num_envs=NUM_ENVS,
        resume=True, sync_wandb=False, debug=False, follow_robot=False,
        motion_file=None, max_iterations=None, load_run=LOAD_RUN,
        ckpt=CKPT, parkour_force_family="gap", parkour_force_row=1,
        depth_ablation="none",
    )
    ensure_runtime_initialized(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK)

    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.add_texture = True
    env_cfg.terrain.texture_mode = texture_mode
    env_cfg.terrain.texture_uv_scale = uv_scale
    env_cfg.sensor.depth_noise_level = 0.0
    env_cfg.sensor.depth_estimation.model_size = "base"
    if hasattr(env_cfg.terrain, "parkour") and hasattr(env_cfg.terrain.parkour, "force_row"):
        env_cfg.terrain.parkour.force_row = 1
        env_cfg.terrain.parkour.force_family = "gap"

    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    correlations = []
    try:
        runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=TASK, args=args, train_cfg=train_cfg,
        )
        policy = runner.get_inference_policy(device=env.device)
        obs, _teacher_obs, student_depth, _depth_updated = env.reset()

        for step in range(NUM_STEPS):
            with torch.inference_mode():
                actions = policy(obs, student_depth)
            obs, _teacher_obs, student_depth, _depth_updated, _rews, dones, infos = env.step(actions.detach())

            if _depth_updated[0]:
                da2_raw = env.simulator._inferred_depth_images
                if da2_raw is not None:
                    sim = env.simulator
                    rgb_cam_idx = sim._scene_camera_output_indices.get("rgb")
                    if rgb_cam_idx is not None:
                        _, depth_out, _, _ = sim._scene.render_all_cameras(
                            rgb=False, depth=True, segmentation=False, normal=False,
                        )
                        gt_raw = sim._as_torch_frame(depth_out[rgb_cam_idx])
                        if gt_raw.ndim == 2:
                            gt_raw = gt_raw.unsqueeze(0)
                        gt_depth = sim._process_depth_frames(gt_raw)
                        if da2_raw.shape[-2:] != gt_depth.shape[-2:]:
                            da2_resized = F.interpolate(
                                da2_raw.unsqueeze(1), size=gt_depth.shape[-2:],
                                mode="bilinear", align_corners=False,
                            ).squeeze(1)
                        else:
                            da2_resized = da2_raw
                        gt_flat = gt_depth.reshape(gt_depth.shape[0], -1).float()
                        da2_flat = da2_resized.reshape(da2_resized.shape[0], -1).float()
                        gt_c = gt_flat - gt_flat.mean(dim=1, keepdim=True)
                        da2_c = da2_flat - da2_flat.mean(dim=1, keepdim=True)
                        numer = (gt_c * da2_c).sum(dim=1)
                        denom = gt_c.norm(dim=1) * da2_c.norm(dim=1) + 1e-8
                        pearson = (numer / denom).mean().item()
                        if not math.isnan(pearson):
                            correlations.append(pearson)
    finally:
        env.close()

    return sum(correlations) / len(correlations) if correlations else float("nan")


def main():
    print(f"\n{'Texture':<15} {'uv=10.0':>10} {'uv=4.0':>10} {'delta':>10}")
    print("-" * 50)
    results = {}
    for texture in TEXTURE_MODES:
        row = {}
        for uv in UV_SCALES:
            print(f"  Running: {texture} @ uv={uv} ...", end=" ", flush=True)
            dsq = compute_dsq(uv, texture)
            row[uv] = dsq
            print(f"DSQ={dsq:.4f}")
        delta = row[4.0] - row[10.0]
        print(f"{texture:<15} {row[10.0]:>10.4f} {row[4.0]:>10.4f} {delta:>+10.4f}")
        results[texture] = row

    avg_10 = sum(r[10.0] for r in results.values()) / len(results)
    avg_4 = sum(r[4.0] for r in results.values()) / len(results)
    print("-" * 50)
    print(f"{'AVERAGE':<15} {avg_10:>10.4f} {avg_4:>10.4f} {avg_4 - avg_10:>+10.4f}")


if __name__ == "__main__":
    main()
