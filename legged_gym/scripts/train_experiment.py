"""Experiment launcher for RGB depth-estimation student convergence experiments.

Usage:
    CUDA_VISIBLE_DEVICES=1 python legged_gym/scripts/train_experiment.py --experiment_id O1-gt-ema --headless
    CUDA_VISIBLE_DEVICES=2 python legged_gym/scripts/train_experiment.py --experiment_id O2-gt-affine --headless
    CUDA_VISIBLE_DEVICES=3 python legged_gym/scripts/train_experiment.py --experiment_id O3-rgb-interval1 --headless
"""

import argparse
import os
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import task_registry, ensure_runtime_initialized
from legged_gym.envs import *  # noqa: F401,F403  — registers all tasks

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each entry maps experiment_id -> dict with:
#   task: base registered task name
#   run_name: TensorBoard run name suffix
#   max_iterations: training length
#   env_overrides: dict of dotted config paths -> values applied to env_cfg
#   train_overrides: dict of dotted config paths -> values applied to train_cfg
#   experiment_flags: dict of special flags consumed by the env at runtime
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    # ── Baseline with GRU fix ────────────────────────────────────────────
    "baseline-fix": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "baseline_gru_fix",
        "max_iterations": 5000,
        "env_overrides": {},
        "train_overrides": {},
        "experiment_flags": {},
    },
    # ── Phase 0: Oracle experiments ──────────────────────────────────────
    "O1-gt-ema": {
        "task": "go2_parkour_student",
        "run_name": "O1_gt_ema",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {},
        "experiment_flags": {"force_ema_norm": True},
    },
    "O2-gt-affine": {
        "task": "go2_parkour_student",
        "run_name": "O2_gt_affine",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {},
        "experiment_flags": {"random_affine_depth": True},
    },
    "O3-rgb-interval1": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "O3_rgb_interval1",
        "max_iterations": 2000,
        "env_overrides": {
            "sensor.depth_estimation.update_interval": 1,
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    # ── Phase 1: Normalization sweep ─────────────────────────────────────
    "N1-affine-fit": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "N1_affine_fit",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {},
        "experiment_flags": {"norm_mode": "affine_fit"},
    },
    "N2-ema-fast": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "N2_ema_fast",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {},
        "experiment_flags": {"ema_alpha": 0.1},
    },
    "N3-ema-slow": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "N3_ema_slow",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {},
        "experiment_flags": {"ema_alpha": 0.005},
    },
    "N4-quantile-wide": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "N4_quantile_wide",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {},
        "experiment_flags": {"ema_quantiles": (0.01, 0.99)},
    },
    "N5-meanstd": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "N5_meanstd",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {},
        "experiment_flags": {"norm_mode": "meanstd"},
    },
    "N5-meanstd-nonoise": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "N5_meanstd_nonoise",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {},
        "experiment_flags": {"norm_mode": "meanstd"},
    },
    "T5-noise-zero": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "T5_noise_zero",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    # ── Phase 2: Depth model sweep ───────────────────────────────────────
    "D1-indoor": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "D1_indoor",
        "max_iterations": 2000,
        "env_overrides": {
            "sensor.depth_estimation.model_type": "depth_anything_v2_metric_indoor",
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    "D2-relative-inv": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "D2_relative_inv",
        "max_iterations": 2000,
        "env_overrides": {
            "sensor.depth_estimation.model_type": "depth_anything_v2",
        },
        "train_overrides": {},
        "experiment_flags": {"invert_relative_depth": True},
    },
    "D3-outdoor-base": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "D3_outdoor_base",
        "max_iterations": 2000,
        "env_overrides": {
            "sensor.depth_estimation.model_size": "base",
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    # ── Phase 3: Training hyperparameters ────────────────────────────────
    "T1-bptt32": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "T1_bptt32",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {
            "algorithm.bptt_window": 32,
        },
        "experiment_flags": {},
    },
    "T2-bptt48": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "T2_bptt48",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {
            "algorithm.bptt_window": 48,
        },
        "experiment_flags": {},
    },
    "T3-lr-low": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "T3_lr_low",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {
            "algorithm.learning_rate": 5e-4,
        },
        "experiment_flags": {},
    },
    "T4-lr-warmup": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "T4_lr_warmup",
        "max_iterations": 2000,
        "env_overrides": {},
        "train_overrides": {
            "algorithm.lr_warmup_iters": 200,
        },
        "experiment_flags": {},
    },
}


def _set_nested_attr(obj, dotted_key, value):
    """Set a nested attribute like 'sensor.depth_estimation.update_interval'."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _get_nested_attr(obj, dotted_key):
    """Get a nested attribute like 'sensor.depth_estimation.update_interval'."""
    parts = dotted_key.split(".")
    for part in parts:
        obj = getattr(obj, part)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Run RGB convergence experiment")
    parser.add_argument("--experiment_id", type=str, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help="Experiment identifier")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Override experiment max_iterations")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed (for replication)")
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment_id]
    task_name = exp["task"]

    # Build a minimal args namespace matching what task_registry expects
    class TrainArgs:
        task = task_name
        headless = args.headless
        cpu = False
        num_envs = args.num_envs
        max_iterations = args.max_iterations or exp["max_iterations"]
        resume = False
        sync_wandb = False
        export_onnx = False
        debug = False
        depth_debug = False
        rgb_debug = False
        depth_model_type = None
        depth_model_size = None
        depth_update_interval = 1
        depth_ablation = "none"
        load_run = None
        ckpt = -1
        command_mode = "auto"
        command_scale = 1.0
        use_joystick = False
        joystick_type = "xbox"
        follow_robot = False
        log_play_stats = False
        motion_file = None
        parkour_force_family = None
        parkour_force_row = None
        parkour_gauntlet = False
        gauntlet_obstacles_per_family = 4
        gauntlet_difficulty = 3
        teacher_task = None
        teacher_load_run = None
        teacher_ckpt = None

    train_args = TrainArgs()

    ensure_runtime_initialized(train_args)

    # Get configs BEFORE creating env so we can apply overrides
    env_cfg, train_cfg = task_registry.get_cfgs(task_name)

    # Apply env config overrides to env_cfg BEFORE env creation
    for key, value in exp.get("env_overrides", {}).items():
        old = _get_nested_attr(env_cfg, key)
        _set_nested_attr(env_cfg, key, value)
        print(f"[Experiment {args.experiment_id}] env_cfg.{key}: {old} -> {value}")

    # Apply experiment flags to env_cfg so they're available during __init__
    experiment_flags = exp.get("experiment_flags", {})
    if not hasattr(env_cfg, "experiment_flags"):
        env_cfg.experiment_flags = {}
    env_cfg.experiment_flags = experiment_flags
    for key, value in experiment_flags.items():
        print(f"[Experiment {args.experiment_id}] experiment_flag.{key} = {value}")

    # Apply train config overrides
    for key, value in exp.get("train_overrides", {}).items():
        old = _get_nested_attr(train_cfg, key)
        _set_nested_attr(train_cfg, key, value)
        print(f"[Experiment {args.experiment_id}] train_cfg.{key}: {old} -> {value}")

    # Override run name and iterations
    train_cfg.runner.run_name = exp["run_name"]
    train_cfg.runner.max_iterations = args.max_iterations or exp["max_iterations"]
    if args.seed is not None:
        train_cfg.seed = args.seed

    # Create env with the pre-modified config
    env, env_cfg = task_registry.make_env(name=task_name, args=train_args, env_cfg=env_cfg)

    # Create runner with overridden train_cfg
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=train_args, train_cfg=train_cfg
    )

    print(f"\n{'='*60}")
    print(f"  Experiment: {args.experiment_id}")
    print(f"  Task: {task_name}")
    print(f"  Run name: {train_cfg.runner.run_name}")
    print(f"  Max iterations: {train_cfg.runner.max_iterations}")
    print(f"  Flags: {experiment_flags}")
    print(f"{'='*60}\n")

    try:
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
