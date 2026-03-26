#!/usr/bin/env python3
"""Save depth images from a parkour student rollout to verify gap visibility.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/visualize_student_depth.py \
        --task go2_parkour_student --headless --num_steps 200 \
        --parkour_force_family gap --parkour_force_row 1 \
        --out_dir logs/depth_viz

Saves depth PNGs every depth render step for env 0, annotated with robot
position and distance to the next waypoint.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch

from legged_gym import SIMULATOR
from legged_gym.envs import *
from legged_gym.utils import ensure_runtime_initialized, task_registry


def _build_args(cli):
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
    )


def _save_depth_png(depth_tensor, path):
    """Normalize a (H, W) depth tensor to 0-255 and save as grayscale PNG."""
    img = depth_tensor.cpu().float().numpy()
    img = ((img + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(img, mode="L").save(path)
    except ImportError:
        import cv2
        cv2.imwrite(path, img)


def main():
    cli = _parse_args()
    if SIMULATOR != "genesis":
        raise RuntimeError("This script only supports SIMULATOR=genesis.")

    args = _build_args(cli)
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(args.num_envs, env_cfg.env.num_envs)
    env_cfg.terrain.curriculum = False
    if hasattr(env_cfg.terrain, "parkour"):
        env_cfg.terrain.parkour.force_row = args.parkour_force_row
        env_cfg.terrain.parkour.force_family = args.parkour_force_family

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    out_dir = cli.out_dir
    os.makedirs(out_dir, exist_ok=True)

    obs, _teacher_obs, student_depth, _depth_updated = env.reset()
    viz_env = 0  # which env to visualize
    frame_idx = 0
    depth_frame_count = 0

    info_lines = []
    for step in range(cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs, student_depth)
        (
            obs, _teacher_obs, student_depth, depth_updated,
            _rews, dones, infos,
        ) = env.step(actions.detach())

        if depth_updated[viz_env]:
            # Get robot local position
            local_pos = env.local_base_pos[viz_env].cpu().numpy()
            wp_idx = int(env.active_waypoint_idx[viz_env].item())
            goal_dist = float(torch.norm(env.commands[viz_env, :2]).item())
            family_id = int(env.simulator.lane_family[viz_env].item())
            family_names = {0: "stairs", 1: "hurdle", 2: "gap"}
            family_str = family_names.get(family_id, f"id{family_id}")

            # Save the depth image (student sees this frame)
            depth_img = student_depth[viz_env, 0]  # (H, W)
            fname = f"depth_{depth_frame_count:04d}.png"
            _save_depth_png(depth_img, os.path.join(out_dir, fname))

            # Also save the current (newest) frame for comparison
            if hasattr(env, '_depth_buffer') and env._depth_buffer.shape[1] > 1:
                current_img = env._depth_buffer[viz_env, 0]  # newest frame
                fname_cur = f"depth_{depth_frame_count:04d}_current.png"
                _save_depth_png(current_img, os.path.join(out_dir, fname_cur))

            line = (
                f"frame={depth_frame_count:4d}  step={step:4d}  "
                f"pos=({local_pos[0]:.2f},{local_pos[1]:.2f},{local_pos[2]:.2f})  "
                f"wp={wp_idx}  goal_dist={goal_dist:.3f}  family={family_str}"
            )
            info_lines.append(line)
            print(line)
            depth_frame_count += 1

        if dones[viz_env]:
            info_lines.append(f"--- env {viz_env} reset at step {step} ---")
            print(f"--- env {viz_env} reset at step {step} ---")

    # Save annotation log
    with open(os.path.join(out_dir, "depth_log.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    print(f"\nSaved {depth_frame_count} depth frames to {out_dir}/")
    print(f"Annotation log: {out_dir}/depth_log.txt")
    env.close()


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=str, default="go2_parkour_student")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--load_run", type=str, default=None)
    p.add_argument("--ckpt", type=int, default=-1)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="logs/depth_viz")
    p.add_argument("--parkour_force_family", type=str, default="gap",
                    choices=["stairs", "hurdle_block", "gap"])
    p.add_argument("--parkour_force_row", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    main()
