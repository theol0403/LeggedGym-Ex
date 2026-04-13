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
    # ── Priority 1: Curriculum & Stability ─────────────────────────────────
    "S1-curriculum-freeze": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S1_curriculum_freeze",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {},
        # Freeze curriculum for first 60000 env steps (~500 iters * 120 steps/iter)
        "experiment_flags": {"curriculum_freeze_until": 60000},
    },
    "S2-gap-only": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S2_gap_only",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "terrain.parkour.force_family": "gap",
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    "S3-slow-curriculum": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S3_slow_curriculum",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "terrain.parkour.curriculum_progress_up_threshold": 0.95,
            "terrain.parkour.curriculum_progress_down_threshold": 0.20,
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    "S4-lr-cosine-decay": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S4_lr_cosine_decay",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {},
    },
    # ── Priority 2: Checkpoint & Duration ──────────────────────────────────
    "S5-best-gap-save": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S5_best_gap_save",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    "S6-long-run": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S6_long_run",
        "max_iterations": 8000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    "S7-ema-warmup": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S7_ema_warmup",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {},
        # Warmup period in env steps (~200 iters * 120 steps/iter = 24000)
        "experiment_flags": {"ema_warmup_iters": 24000},
    },
    # ── Priority 3: Gradient Stabilization ─────────────────────────────────
    "S8-grad-accum-2x": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S8_grad_accum_2x",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "runner.num_steps_per_env": 240,
        },
        "experiment_flags": {},
    },
    "S9-lower-grad-clip": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S9_lower_grad_clip",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.max_grad_norm": 0.5,
        },
        "experiment_flags": {},
    },
    "S10-bptt16": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S10_bptt16",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.bptt_window": 16,
        },
        "experiment_flags": {},
    },
    # ── Priority 4: Depth Model Alternatives ───────────────────────────────
    "D5-metric3d-small": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "D5_metric3d_small",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "sensor.depth_estimation.model_type": "metric3d_v2",
            "sensor.depth_estimation.model_size": "small",
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    "D6-unidepth-small": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "D6_unidepth_small",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "sensor.depth_estimation.model_type": "unidepth_v2",
            "sensor.depth_estimation.model_size": "small",
        },
        "train_overrides": {},
        "experiment_flags": {},
    },
    # ── Priority 5: Combined Best ──────────────────────────────────────────
    "C1-combined": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "C1_combined",
        "max_iterations": 8000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "terrain.parkour.curriculum_progress_up_threshold": 0.95,
            "terrain.parkour.curriculum_progress_down_threshold": 0.20,
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 8000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {"curriculum_freeze_until": 60000},
    },
    # ── S15: Stronger differential LR ──────────────────────────────────
    "S15-diff-lr-strong": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S15_diff_lr_strong",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.05,
        },
        "experiment_flags": {},
    },
    "C2-combined-seed42": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "C2_combined_seed42",
        "max_iterations": 8000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "terrain.parkour.curriculum_progress_up_threshold": 0.95,
            "terrain.parkour.curriculum_progress_down_threshold": 0.20,
        },
        "train_overrides": {
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 8000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {"curriculum_freeze_until": 60000},
    },
    # ── Targeted experiments ───────────────────────────────────────────────
    "S11-freeze-actor": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S11_freeze_actor",
        "max_iterations": 3000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.freeze_actor": True,
        },
        "experiment_flags": {},
    },
    "S13-diff-lr": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S13_diff_lr",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
        },
        "experiment_flags": {},
    },
    "S14-diff-lr-cosine": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S14_diff_lr_cosine",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {},
    },
    # ── Fine-tuning from S14's model_5000 at low constant LR ──────
    "F1-finetune-lr1e4": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "F1_finetune_lr1e4",
        "max_iterations": 2000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.learning_rate": 1e-4,
            "algorithm.actor_lr_scale": 0.1,
        },
        "experiment_flags": {},
    },
    "F2-finetune-lr5e5": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "F2_finetune_lr5e5",
        "max_iterations": 2000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.learning_rate": 5e-5,
            "algorithm.actor_lr_scale": 0.1,
        },
        "experiment_flags": {},
    },
    "F3-finetune-lr1e4-frozen-actor": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "F3_finetune_lr1e4_frozen_actor",
        "max_iterations": 2000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.learning_rate": 1e-4,
            "algorithm.freeze_actor": True,
        },
        "experiment_flags": {},
    },
    # ── Calibrated depth normalization ────────────────────────────────
    "CAL1-calibrated-linear": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "CAL1_calibrated_linear",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {
            "norm_mode": "calibrated_linear",
            "affine_scale": 0.081393,
            "affine_offset": 0.994514,
        },
    },
    "CAL2-fixed-range": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "CAL2_fixed_range",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {
            "norm_mode": "fixed_range",
            "fixed_range_lo": 3.5,
            "fixed_range_hi": 12.0,
        },
    },
    # ── Texture experiments ──────────────────────────────────────────
    "TEX1-ema-texture": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "TEX1_ema_texture",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "terrain.add_texture": True,

        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {},  # default EMA normalization
    },
    # ── Architecture: depth gradient channel ──────────────────────────
    "GRAD1-sobel-ema": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "GRAD1_sobel_ema",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "env.student_depth_shape": [2, 58, 87],
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
            "policy.student_depth_shape": [2, 58, 87],
        },
        "experiment_flags": {
            "depth_gradient": True,
        },
    },
    "GRAD2-sobel-texture": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "GRAD2_sobel_texture",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "env.student_depth_shape": [2, 58, 87],
            "terrain.add_texture": True,

        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
            "policy.student_depth_shape": [2, 58, 87],
        },
        "experiment_flags": {
            "depth_gradient": True,
        },
    },
    "TEX2-fixed-range-texture": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "TEX2_fixed_range_texture",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "terrain.add_texture": True,

        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {
            "norm_mode": "fixed_range",
            "fixed_range_lo": 3.5,
            "fixed_range_hi": 12.0,
        },
    },
    # ── DA2 base model (larger, more accurate) ─────────────────────────
    "BASE1-da2-base": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "BASE1_da2_base",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "sensor.depth_estimation.model_size": "base",
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {},
    },
    # ── Temporal depth smoothing (reduce DA2 flicker) ────────────────
    "SMOOTH1-temporal-ema": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "SMOOTH1_temporal_ema",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {
            "depth_temporal_smooth": 0.5,
        },
    },
    # ── DA2 base + temporal smoothing ────────────────────────────────
    "BASE2-da2-base-smooth": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "BASE2_da2_base_smooth",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "sensor.depth_estimation.model_size": "base",
        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {
            "depth_temporal_smooth": 0.5,
        },
    },
    # ── Direct scandot prediction from depth ───────────────────────────
    "SCANDOT1-direct-prediction": {
        "task": "go2_parkour_scandot_student",
        "run_name": "SCANDOT1_direct_prediction",
        "max_iterations": 5000,
        "env_overrides": {},
        "train_overrides": {
            "algorithm.scandot_loss_coef": 1.0,
            "algorithm.action_loss_coef": 1.0,
        },
        "experiment_flags": {},
    },
    # ── DA2 base + texture (combining two biggest improvements) ─────────
    "BASE3-da2-base-texture": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "BASE3_da2_base_texture",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "sensor.depth_estimation.model_size": "base",
            "terrain.add_texture": True,

        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {},
    },
    # ── DA2 base + texture + temporal smoothing ──────────────────────────
    "BASE4-da2-base-texture-smooth": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "BASE4_da2_base_texture_smooth",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
            "sensor.depth_estimation.model_size": "base",
            "terrain.add_texture": True,

        },
        "train_overrides": {
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {
            "depth_temporal_smooth": 0.5,
        },
    },
    # ── DA2 estimated depth + scandot prediction ────────────────────────
    "SCANDOT2-da2-scandot": {
        "task": "go2_parkour_depth_est_scandot_student",
        "run_name": "SCANDOT2_da2_scandot",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.scandot_loss_coef": 1.0,
            "algorithm.action_loss_coef": 1.0,
            "algorithm.actor_lr_scale": 0.1,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
        },
        "experiment_flags": {
            "depth_temporal_smooth": 0.5,
        },
    },
    "S12-freeze-actor-cosine": {
        "task": "go2_parkour_depth_est_student",
        "run_name": "S12_freeze_actor_cosine",
        "max_iterations": 5000,
        "env_overrides": {
            "sensor.depth_noise_level": 0.0,
        },
        "train_overrides": {
            "algorithm.freeze_actor": True,
            "algorithm.lr_schedule": "cosine",
            "algorithm.lr_schedule_max_iters": 5000,
            "algorithm.lr_schedule_min_lr": 1e-4,
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
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--load_weights", type=str, default=None,
                        help="Load model weights (no optimizer/iter) for fine-tuning")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Override checkpoint save interval")
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
        try:
            old = _get_nested_attr(env_cfg, key)
        except AttributeError:
            old = "<new>"
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
        try:
            old = _get_nested_attr(train_cfg, key)
        except AttributeError:
            old = "<new>"
        _set_nested_attr(train_cfg, key, value)
        print(f"[Experiment {args.experiment_id}] train_cfg.{key}: {old} -> {value}")

    # Override run name and iterations
    train_cfg.runner.run_name = exp["run_name"]
    train_cfg.runner.max_iterations = args.max_iterations or exp["max_iterations"]
    if args.save_interval is not None:
        train_cfg.runner.save_interval = args.save_interval
    if args.seed is not None:
        train_cfg.seed = args.seed

    # Create env with the pre-modified config
    env, env_cfg = task_registry.make_env(name=task_name, args=train_args, env_cfg=env_cfg)

    # Create runner with overridden train_cfg
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=train_args, train_cfg=train_cfg
    )

    # Load weights for fine-tuning (fresh optimizer, iter=0)
    if args.load_weights:
        import torch as _torch
        print(f"Loading weights for fine-tuning: {args.load_weights}")
        ckpt = _torch.load(args.load_weights, map_location=ppo_runner.device)
        ppo_runner.alg.actor_critic.load_state_dict(ckpt["model_state_dict"])
        print("  Loaded model weights (fresh optimizer, starting from iter 0)")

    # Resume from checkpoint if specified (restores optimizer + iteration)
    resume_iter = 0
    if args.resume_path:
        import re
        print(f"Resuming from checkpoint: {args.resume_path}")
        ppo_runner.load(args.resume_path)
        match = re.search(r"model_(\d+)\.pt$", args.resume_path)
        if match:
            resume_iter = int(match.group(1))
        stored_iter = ppo_runner.current_learning_iteration
        resume_iter = max(resume_iter, stored_iter)
        ppo_runner.current_learning_iteration = resume_iter
        if hasattr(ppo_runner.alg, '_iteration'):
            ppo_runner.alg._iteration = resume_iter
            print(f"  Restored LR schedule iteration to {resume_iter}")
        print(f"  Resuming from iteration {resume_iter}")

    max_iterations = args.max_iterations or exp["max_iterations"]
    remaining_iterations = max_iterations - resume_iter

    print(f"\n{'='*60}")
    print(f"  Experiment: {args.experiment_id}")
    print(f"  Task: {task_name}")
    print(f"  Run name: {train_cfg.runner.run_name}")
    print(f"  Max iterations: {max_iterations}")
    if resume_iter > 0:
        print(f"  Resumed from iteration: {resume_iter}")
        print(f"  Remaining iterations: {remaining_iterations}")
    print(f"  Flags: {experiment_flags}")
    print(f"{'='*60}\n")

    if remaining_iterations <= 0:
        print(f"Already completed {resume_iter}/{max_iterations} iterations. Nothing to do.")
    else:
        try:
            ppo_runner.learn(
                num_learning_iterations=remaining_iterations,
                init_at_random_ep_len=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
