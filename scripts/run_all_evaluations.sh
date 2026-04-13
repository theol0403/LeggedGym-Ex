#!/bin/bash
# Run all evaluations for the retrained DA2-base+texture student.
# Produces data for Table 4 (gap rows 0-3) and Table 5 (domain invariance).
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_evaluations.sh

set -euo pipefail

RUN_DIR="Apr11_16-03-28_BASE3_da2_base_texture"
CKPT=10000
MODEL_PATH="logs/go2_parkour_depth_est_student/${RUN_DIR}/model_${CKPT}.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: $MODEL_PATH not found. Training may not be complete yet."
    exit 1
fi

echo "=== Starting evaluations with $MODEL_PATH ==="

# ── Table 4: DA2-base+texture on gap rows 0-3 (baseline perturbation) ──
echo ""
echo "=== Table 4: Gap terrain, rows 0-3 ==="
for ROW in 0 1 2 3; do
    echo "--- Gap Row $ROW ---"
    .venv/bin/python legged_gym/scripts/evaluate_domain_invariance.py \
        --headless --episodes 256 --num_envs 48 \
        --students DA2-base+tex \
        --perturbations baseline \
        --force_family gap --force_row "$ROW" \
        --output "thesis/domain_invariance_gap${ROW}_table4.json" \
        2>&1 | tail -5
done

# ── Table 5: Domain invariance across 6 terrain configs ──
echo ""
echo "=== Table 5: Domain invariance ==="
for FAMILY in gap stairs hurdle_block; do
    for ROW in 0 3; do
        echo "--- $FAMILY Row $ROW ---"
        .venv/bin/python legged_gym/scripts/evaluate_domain_invariance.py \
            --headless --episodes 256 --num_envs 48 \
            --students DA2-base+tex \
            --force_family "$FAMILY" --force_row "$ROW" \
            --compute_depth_metric \
            --output "thesis/domain_invariance_${FAMILY}${ROW}.json" \
            2>&1 | tail -20
    done
done

echo ""
echo "=== All evaluations complete ==="
echo "JSON results saved in thesis/ directory"
