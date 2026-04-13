#!/usr/bin/env python3
"""Domain invariance evaluation for parkour student policies.

Sweeps visual perturbations (texture, lighting, color jitter) across
DA2-based, ResNet-RGB, and GT-depth students to test the hypothesis
that DA2 depth provides a domain-invariant representation.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/evaluate_domain_invariance.py \
        --headless --episodes 256

Results are printed as a table and optionally saved to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from legged_gym import SIMULATOR
from legged_gym.envs import *
from legged_gym.utils import ensure_runtime_initialized, task_registry
from legged_gym.utils.student_depth_ablation import apply_student_depth_ablation


# ── Student configurations ──────────────────────────────────────────────
# Each entry: (task_name, load_run, ckpt, env_overrides, label)
STUDENTS = {
    "DA2-base+tex": {
        "task": "go2_parkour_depth_est_student",
        "load_run": "Apr11_16-03-28_BASE3_da2_base_texture",
        "ckpt": 10000,
        "env_overrides": {
            "sensor.depth_estimation.model_size": "base",
            "sensor.depth_noise_level": 0.0,
        },
    },
}

# ── Visual perturbation conditions ───────────────────────────────────────
# Each: dict of env_overrides and experiment_flags to apply
PERTURBATIONS = {
    # Baseline: training conditions
    "baseline": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "checkerboard"},
        "experiment_flags": {},
    },
    # Texture perturbations
    "tex_none": {
        "env_overrides": {"terrain.add_texture": False},
        "experiment_flags": {},
        "description": "No texture (flat grey terrain)",
    },
    "tex_solid_red": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "solid_red"},
        "experiment_flags": {},
    },
    "tex_solid_blue": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "solid_blue"},
        "experiment_flags": {},
    },
    "tex_noise": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "noise"},
        "experiment_flags": {},
    },
    "tex_bricks": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "bricks"},
        "experiment_flags": {},
    },
    # Realistic surface textures
    "tex_concrete": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "concrete"},
        "experiment_flags": {},
    },
    "tex_wood": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "wood"},
        "experiment_flags": {},
    },
    "tex_grass": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "grass"},
        "experiment_flags": {},
    },
    "tex_stone_tiles": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "stone_tiles"},
        "experiment_flags": {},
    },
    "tex_gravel": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "gravel"},
        "experiment_flags": {},
    },
    # Lighting perturbations (image-space transforms applied before DA2)
    "light_dim": {
        "env_overrides": {},
        "experiment_flags": {"brightness_scale": 0.3},
        "description": "Dim lighting (30% brightness)",
    },
    "light_bright": {
        "env_overrides": {},
        "experiment_flags": {"brightness_scale": 2.0},
        "description": "Bright lighting (200% brightness)",
    },
    "light_low_gamma": {
        "env_overrides": {},
        "experiment_flags": {"gamma": 2.2},
        "description": "Low-contrast / washed out (gamma 2.2)",
    },
    # Color jitter (image-space)
    "jitter_brightness_mild": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "checkerboard"},
        "experiment_flags": {"brightness_jitter": 0.3},
    },
    "jitter_brightness_strong": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "checkerboard"},
        "experiment_flags": {"brightness_jitter": 0.6},
    },
    "jitter_color_mild": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "checkerboard"},
        "experiment_flags": {"color_jitter": 0.15},
    },
    "jitter_color_strong": {
        "env_overrides": {"terrain.add_texture": True, "terrain.texture_mode": "checkerboard"},
        "experiment_flags": {"color_jitter": 0.3},
    },
    # Combined perturbations
    "combined_mild": {
        "env_overrides": {
            "terrain.add_texture": True, "terrain.texture_mode": "concrete",
        },
        "experiment_flags": {"brightness_scale": 0.6, "brightness_jitter": 0.2},
    },
    "combined_extreme": {
        "env_overrides": {
            "terrain.add_texture": True, "terrain.texture_mode": "noise",
        },
        "experiment_flags": {"brightness_scale": 0.4, "gamma": 1.8, "color_jitter": 0.2},
    },
}


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().item())
    return float(value)


def _evaluate_single(
    task: str,
    load_run,
    ckpt: int,
    env_overrides: dict,
    experiment_flags: dict,
    num_envs: int,
    episodes: int,
    headless: bool,
    force_family: str | None,
    force_row: int,
    compute_depth_metric: bool = False,
):
    """Run one evaluation and return metrics dict."""
    args = SimpleNamespace(
        task=task,
        headless=headless,
        cpu=False,
        num_envs=num_envs,
        resume=True,
        sync_wandb=False,
        debug=False,
        follow_robot=False,
        motion_file=None,
        max_iterations=None,
        load_run=load_run if load_run is not None else -1,
        ckpt=ckpt,
        parkour_force_family=force_family,
        parkour_force_row=force_row,
        depth_ablation="none",
    )
    ensure_runtime_initialized(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)

    env_cfg.env.num_envs = min(num_envs, env_cfg.env.num_envs)
    env_cfg.terrain.curriculum = False
    if getattr(env_cfg.terrain, "parkour", None) is not None and getattr(env_cfg.terrain.parkour, "enable", False):
        env_cfg.terrain.parkour.force_row = force_row
        env_cfg.terrain.parkour.force_family = force_family

    # Apply env overrides
    for key, value in env_overrides.items():
        parts = key.split(".")
        obj = env_cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    # Apply experiment flags
    if experiment_flags:
        if not hasattr(env_cfg, "experiment_flags") or env_cfg.experiment_flags is None:
            env_cfg.experiment_flags = {}
        env_cfg.experiment_flags.update(experiment_flags)

    is_da2_student = (
        getattr(getattr(env_cfg, "sensor", None), "depth_estimation", None) is not None
        and getattr(env_cfg.sensor.depth_estimation, "enabled", False)
    )

    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=task, args=args, train_cfg=train_cfg,
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        obs, _teacher_obs, student_depth, _depth_updated = env.reset()
        completed_episodes = 0
        success_sum = 0.0
        progress_sum = 0.0
        frame_correlations = []
        track_dsq = compute_depth_metric and is_da2_student

        max_steps = int(math.ceil(episodes / env.num_envs) * env.max_episode_length * 2)

        for _ in range(max_steps):
            depth_in = apply_student_depth_ablation(student_depth, "none")
            with torch.inference_mode():
                actions = policy(obs, depth_in)
            obs, _teacher_obs, student_depth, _depth_updated, _rews, dones, infos = env.step(actions.detach())

            # Depth signal quality: Pearson correlation between GT and DA2 depth
            if track_dsq and _depth_updated[0]:
                da2_raw = env.simulator._inferred_depth_images  # (N, H_raw, W_raw)
                if da2_raw is not None:
                    # Render GT depth from the RGB camera (no add_depth needed)
                    sim = env.simulator
                    rgb_cam_idx = sim._scene_camera_output_indices.get("rgb")
                    if rgb_cam_idx is not None:
                        _, depth_out, _, _ = sim._scene.render_all_cameras(
                            rgb=False, depth=True, segmentation=False, normal=False,
                        )
                        gt_raw = sim._as_torch_frame(depth_out[rgb_cam_idx])
                        if gt_raw.ndim == 2:
                            gt_raw = gt_raw.unsqueeze(0)
                        gt_depth = sim._process_depth_frames(gt_raw)  # (N, H, W) normalized
                        # Resize DA2 to match GT
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
                            frame_correlations.append(pearson)

            num_resets = int(torch.sum(dones).item())
            if num_resets <= 0 or "episode" not in infos:
                continue
            episode_info = infos["episode"]
            completed_episodes += num_resets
            success_sum += _to_float(episode_info["success"]) * num_resets
            progress_sum += _to_float(episode_info["progress_ratio"]) * num_resets
            if completed_episodes >= episodes:
                break
    finally:
        env.close()

    if completed_episodes == 0:
        return {"success_rate": 0.0, "progress_ratio": 0.0, "episodes": 0}

    result = {
        "success_rate": success_sum / completed_episodes,
        "progress_ratio": progress_sum / completed_episodes,
        "episodes": completed_episodes,
    }
    if frame_correlations:
        result["depth_signal_quality"] = sum(frame_correlations) / len(frame_correlations)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--num_envs", type=int, default=48)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--force_family", type=str, default="gap",
                   choices=["stairs", "hurdle_block", "gap", None])
    p.add_argument("--force_row", type=int, default=0)
    p.add_argument("--students", type=str, nargs="*", default=None,
                   help=f"Students to evaluate (default: all). Choices: {list(STUDENTS.keys())}")
    p.add_argument("--perturbations", type=str, nargs="*", default=None,
                   help=f"Perturbations to test (default: all). Choices: {list(PERTURBATIONS.keys())}")
    p.add_argument("--output", type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument("--compute_depth_metric", action="store_true", default=False,
                   help="Compute depth signal quality (Pearson r between GT and DA2 depth)")
    cli = p.parse_args()

    student_names = cli.students or list(STUDENTS.keys())
    perturbation_names = cli.perturbations or list(PERTURBATIONS.keys())

    results = []
    total = len(student_names) * len(perturbation_names)
    idx = 0

    for student_name in student_names:
        student = STUDENTS[student_name]
        for perturb_name in perturbation_names:
            idx += 1
            perturb = PERTURBATIONS[perturb_name]
            print(f"\n{'='*70}")
            print(f"[{idx}/{total}] Student: {student_name}  |  Perturbation: {perturb_name}")
            print(f"{'='*70}")

            # Merge student env_overrides with perturbation overrides
            merged_overrides = {**student["env_overrides"], **perturb["env_overrides"]}
            merged_flags = {**perturb.get("experiment_flags", {})}

            t0 = time.time()
            try:
                metrics = _evaluate_single(
                    task=student["task"],
                    load_run=student["load_run"],
                    ckpt=student["ckpt"],
                    env_overrides=merged_overrides,
                    experiment_flags=merged_flags,
                    num_envs=cli.num_envs,
                    episodes=cli.episodes,
                    headless=cli.headless,
                    force_family=cli.force_family,
                    force_row=cli.force_row,
                    compute_depth_metric=cli.compute_depth_metric,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                metrics = {"success_rate": -1, "progress_ratio": -1, "episodes": 0, "error": str(e)}

            elapsed = time.time() - t0
            metrics["student"] = student_name
            metrics["perturbation"] = perturb_name
            metrics["elapsed_s"] = elapsed
            results.append(metrics)

            print(f"  success_rate = {metrics['success_rate']:.4f}")
            print(f"  progress_ratio = {metrics.get('progress_ratio', -1):.4f}")
            if "depth_signal_quality" in metrics:
                print(f"  depth_signal_quality = {metrics['depth_signal_quality']:.4f}")
            print(f"  episodes = {metrics['episodes']}")
            print(f"  elapsed = {elapsed:.1f}s")

    # Check if any result has DSQ
    has_dsq = any("depth_signal_quality" in r for r in results)

    # Print summary table
    print(f"\n{'='*100}")
    print("DOMAIN INVARIANCE RESULTS SUMMARY")
    print(f"Family: {cli.force_family}  Row: {cli.force_row}  Episodes: {cli.episodes}")
    print(f"{'='*100}")
    header = f"{'Perturbation':<25}"
    for sn in student_names:
        header += f" | {sn:>15}"
    if has_dsq:
        header += f" | {'DSQ':>8}"
    print(header)
    print("-" * len(header))

    for pn in perturbation_names:
        row = f"{pn:<25}"
        dsq_val = None
        for sn in student_names:
            match = [r for r in results if r["student"] == sn and r["perturbation"] == pn]
            if match:
                sr = match[0]["success_rate"]
                row += f" | {sr:>14.1%}" if sr >= 0 else f" | {'ERROR':>15}"
                if dsq_val is None and "depth_signal_quality" in match[0]:
                    dsq_val = match[0]["depth_signal_quality"]
            else:
                row += f" | {'N/A':>15}"
        if has_dsq:
            row += f" | {dsq_val:>8.3f}" if dsq_val is not None else f" | {'---':>8}"
        print(row)

    # Compute deltas from baseline
    print(f"\n{'='*100}")
    print("DELTA FROM BASELINE (pp)")
    print(f"{'='*100}")
    delta_header = f"{'Perturbation':<25}"
    for sn in student_names:
        delta_header += f" | {sn:>15}"
    print(delta_header)
    print("-" * len(delta_header))

    baseline_rates = {}
    for sn in student_names:
        match = [r for r in results if r["student"] == sn and r["perturbation"] == "baseline"]
        baseline_rates[sn] = match[0]["success_rate"] if match else 0.0

    for pn in perturbation_names:
        if pn == "baseline":
            continue
        row = f"{pn:<25}"
        for sn in student_names:
            match = [r for r in results if r["student"] == sn and r["perturbation"] == pn]
            if match and match[0]["success_rate"] >= 0:
                delta = (match[0]["success_rate"] - baseline_rates[sn]) * 100
                row += f" | {delta:>+14.1f}"
            else:
                row += f" | {'N/A':>15}"
        print(row)

    # Save to JSON
    if cli.output:
        output_data = {
            "config": {
                "force_family": cli.force_family,
                "force_row": cli.force_row,
                "episodes": cli.episodes,
                "num_envs": cli.num_envs,
            },
            "results": results,
        }
        with open(cli.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {cli.output}")


if __name__ == "__main__":
    main()
