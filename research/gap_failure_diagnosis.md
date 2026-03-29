# Gap Failure Diagnosis: Monocular Depth Student

**Date**: 2026-03-28
**Model evaluated**: S14 (diff-lr + cosine decay), model_5000.pt
**Checkpoint path**: `logs/go2_parkour_depth_est_student/Mar27_21-12-08_S14_diff_lr_cosine/model_5000.pt`

## Executive Summary

The RGB depth-estimation student achieves only 18–55% gap success rate compared to the teacher's 98–100%. The root cause is **not** the student architecture or training procedure — it is the monocular depth estimator (DepthAnything V2), which produces depth images with **zero correlation** to the actual terrain geometry. Gap features appear as only 4.5% of the depth image's dynamic range, making them invisible to the student's CNN encoder.

A GT-depth student (using simulator raycast depth) achieves 87–98% gap success with the same architecture, confirming the bottleneck is purely in the depth signal.

## 1. Per-Row Gap Success Comparison

Evaluated on 200 episodes per row, 48 environments, curriculum disabled.

| Row | Gap Width | Obstacles | Teacher | GT-Depth Student | Est-Depth Student | Est (zero depth) |
|-----|-----------|-----------|---------|------------------|-------------------|-----------------|
| 0   | 0.24m     | 2         | 98.5%   | 97.5%            | 45.5%             | 8.0%            |
| 1   | 0.32m     | 3         | 100.0%  | 94.0%            | 55.2%             | —               |
| 2   | 0.42m     | 3         | 99.0%   | 93.0%            | 50.0%             | —               |
| 3   | 0.50m     | 3         | 98.5%   | 86.5%            | 18.5%             | 0.0%            |

**Key observations:**
- Teacher: trivially easy at all rows (98–100%)
- GT-depth student: near-teacher at rows 0–2, slight degradation at row 3 (86.5%)
- Est-depth student: 45–55% on rows 0–2, collapses to 18.5% on row 3
- Zero-depth ablation: 0–8% confirms student IS using depth signal, but it is very weak
- Non-monotonic pattern (row 1 > row 0) suggests depth quality is inconsistent across episodes

## 2. Depth Signal Analysis

Captured over 2000 steps on gap row 0 (`diagnose_depth_signal.py`).

### 2.1 Student Depth Statistics (normalized, what CNN sees)
| Metric | Value |
|--------|-------|
| Mean | -0.166 |
| Std | 0.229 |
| Spatial std per env | 0.178 |
| Range | [-0.496, 0.482] |

### 2.2 Spatial Structure
| Region | Mean | Std |
|--------|------|-----|
| Bottom half (near ground) | -0.317 | 0.145 |
| Top half (further away) | -0.016 | 0.192 |
| Bottom–Top difference | -0.301 | — |

The depth image HAS spatial structure (bottom is closer/smaller values), but this is just the distance gradient, not gap-specific features.

### 2.3 Correlation with Terrain (Scandots)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Correlation (scandot height vs depth mean) | **-0.10** | No correlation |
| Depth difference (gap present vs absent) | **-0.045** | 4.5% of range |

The scandot terrain heights (what the teacher sees) have **zero correlation** with the estimated depth values. When scandots indicate a gap is present, the depth image changes by only 0.045 units on a [-0.5, 0.5] scale.

### 2.4 EMA Normalization Parameters
| Parameter | Value |
|-----------|-------|
| EMA lo (p2 percentile) | 4.928 |
| EMA hi (p98 percentile) | 11.489 |
| EMA span | 6.561 |

DepthAnything V2 outputs raw values in ~[5, 11.5] range for simulation renders. The EMA normalization maps this 6.5m span into [-0.5, 0.5], compressing any local terrain features.

## 3. Why the Depth Estimator Fails on Gaps

1. **Visual features**: Simulation terrain has flat, textureless surfaces. Gaps are narrow (0.24–0.50m) and lack the visual cues that monocular depth estimation relies on (texture gradients, occlusion boundaries, perspective).

2. **Scale mismatch**: DepthAnything V2 is trained on natural images with typical indoor/outdoor depth ranges. A 0.24m gap at 0.5–1.5m distance from a robot camera is far outside its training distribution.

3. **EMA normalization**: The percentile-based normalization removes absolute scale and compresses the dynamic range. A gap that causes a 0.3m depth change in a 6.5m span becomes 4.5% of the normalized range — below the noise floor.

4. **Update interval**: Depth updates every 5 physics steps (~10 Hz at 50 Hz sim). The robot may pass over narrow gaps between depth frames.

## 4. What Teacher Sees vs Student Sees

### Teacher (Scandots)
- 132 height measurements in a 12×11 grid
- Range: -0.3m to +1.35m forward, ±0.75m lateral
- Updated every physics step (50 Hz)
- Direct terrain height relative to robot base (±1.0 clip)
- A gap shows as several points with values near -0.5 (terrain drops away)

### Student (Estimated Depth)
- 58×87 depth image from forward camera
- Camera at (0.327, 0, 0.043)m, 5° pitch down, 87° FOV
- Updated every 5 physics steps (~10 Hz)
- Monocular depth estimation from RGB render, EMA-normalized to [-0.5, 0.5]
- A gap shows as... almost nothing (0.045 change vs 0.23 std)

## 5. Training History

Over 20+ experiments confirmed that hyperparameter tuning cannot overcome the depth signal limitation:

| Approach | Best Gap Success | Notes |
|----------|-----------------|-------|
| Baseline (no fixes) | ~25–35% | Wild oscillation |
| Diff-lr (S13, actor×0.1) | ~48% plateau | Reduced oscillation |
| Diff-lr + cosine (S14) | ~60% | Cosine fine-tuning broke through 48% plateau |
| Fine-tuning at low LR (F1–F3) | ~60% | No further gains |
| Curriculum modifications (C1, S1–S3) | ~48% | No improvement |
| Stronger diff-lr (S15, actor×0.05) | ~48% | No improvement |
| Various normalizations (N1–N5) | <40% | Worse than EMA |
| BPTT/LR modifications (T1–T4) | <40% | Diverged |

## 6. Conclusions

1. **The student architecture works.** GT-depth student achieves 87–98% gap success.
2. **The monocular depth signal is the sole bottleneck.** Zero terrain correlation, 4.5% gap signal.
3. **Hyperparameter tuning is exhausted.** 20+ experiments confirm the ~60% ceiling.
4. **The problem is fundamentally about the depth estimator**, not training.

## 7. Recommended Next Steps

### Option A: Use simulator GT depth camera
Skip DepthAnything entirely. Render true depth from the simulator's depth camera and feed it to the student. This would immediately achieve 87–98% gap success. Limitation: does not transfer to real-world.

### Option B: Improve the depth-to-terrain mapping
- Train a lightweight network to predict scandot-like terrain heights from estimated depth
- Use GT scandots as supervision signal during training
- This auxiliary task would force the depth encoder to learn gap-relevant features

### Option C: Use RGB directly
- Skip the depth estimation step
- Feed raw RGB images to a larger CNN/ViT encoder
- Let the network learn what visual features matter for gaps
- May require more training data/compute but avoids the depth estimation bottleneck

### Option D: Hybrid approach
- Use GT depth during training (to learn gap behaviors)
- Gradually introduce estimated depth noise during training (domain randomization)
- Fine-tune on estimated depth at the end

## Appendix: Reproduction Commands

```bash
# Evaluate estimated-depth student per row
CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/ablate_parkour_student_depth.py \
    --task go2_parkour_depth_est_student --headless --episodes 200 --num_envs 48 \
    --load_run Mar27_21-12-08_S14_diff_lr_cosine --ckpt 5000 \
    --parkour_force_family gap --parkour_force_row 0 --depth_ablation none

# Evaluate GT-depth student per row
CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/ablate_parkour_student_depth.py \
    --task go2_parkour_student --headless --episodes 200 --num_envs 48 \
    --load_run Mar26_16-48-36_student_genesis --ckpt 3000 \
    --parkour_force_family gap --parkour_force_row 0 --depth_ablation none

# Evaluate teacher per row
CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/evaluate_parkour_teacher.py \
    --task go2_parkour_teacher --headless --episodes 200 --num_envs 48 \
    --parkour_force_family gap --parkour_force_row 0

# Depth signal diagnosis
CUDA_VISIBLE_DEVICES=0 .venv/bin/python legged_gym/scripts/diagnose_depth_signal.py \
    --headless --num_envs 16 --row 0 --steps 2000 \
    --load_run Mar27_21-12-08_S14_diff_lr_cosine --ckpt 5000
```
