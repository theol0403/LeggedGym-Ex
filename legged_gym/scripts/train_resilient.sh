#!/bin/bash
# Resilient training loop: saves frequently and auto-resumes after crashes.
# Usage: CUDA_VISIBLE_DEVICES=1 bash legged_gym/scripts/train_resilient.sh CAL2-fixed-range 16

EXPERIMENT_ID=${1:?Usage: train_resilient.sh EXPERIMENT_ID [NUM_ENVS]}
NUM_ENVS=${2:-16}
SAVE_INTERVAL=100
LOG_DIR="logs"
LOGFILE="${LOG_DIR}/${EXPERIMENT_ID}_resilient.log"

echo "=== Resilient training: ${EXPERIMENT_ID}, num_envs=${NUM_ENVS}, save_interval=${SAVE_INTERVAL} ===" | tee -a "$LOGFILE"

while true; do
    # Find latest checkpoint for this experiment
    LATEST_CKPT=""
    for dir in ${LOG_DIR}/go2_parkour_depth_est_student/*_$(echo $EXPERIMENT_ID | sed 's/-/_/g; s/CAL/CAL/'); do
        if [ -d "$dir" ]; then
            for ckpt in $(ls -t "$dir"/model_*.pt 2>/dev/null); do
                LATEST_CKPT="$ckpt"
                break
            done
        fi
    done

    # Also search with the run_name format
    for dir in ${LOG_DIR}/go2_parkour_depth_est_student/*; do
        if [ -d "$dir" ]; then
            for ckpt in $(ls -t "$dir"/model_*.pt 2>/dev/null | grep -v model_0.pt); do
                # Check if this is a newer checkpoint than what we found
                ITER=$(echo "$ckpt" | grep -oP 'model_\K\d+')
                if [ -n "$ITER" ] && [ "$ITER" -gt 0 ]; then
                    CURR_ITER=0
                    if [ -n "$LATEST_CKPT" ]; then
                        CURR_ITER=$(echo "$LATEST_CKPT" | grep -oP 'model_\K\d+' || echo 0)
                    fi
                    if [ "$ITER" -gt "$CURR_ITER" ]; then
                        # Verify this dir matches our experiment
                        RUN_NAME=$(echo "$EXPERIMENT_ID" | sed 's/-/_/g')
                        if echo "$dir" | grep -qi "$RUN_NAME"; then
                            LATEST_CKPT="$ckpt"
                        fi
                    fi
                fi
            done
        fi
    done

    RESUME_ARG=""
    if [ -n "$LATEST_CKPT" ]; then
        echo "$(date): Resuming from: $LATEST_CKPT" | tee -a "$LOGFILE"
        RESUME_ARG="--resume_path $LATEST_CKPT"
    else
        echo "$(date): Starting fresh" | tee -a "$LOGFILE"
    fi

    .venv/bin/python legged_gym/scripts/train_experiment.py \
        --experiment_id "$EXPERIMENT_ID" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --save_interval "$SAVE_INTERVAL" \
        $RESUME_ARG \
        2>&1 | tee -a "$LOGFILE"

    EXIT_CODE=$?
    echo "$(date): Process exited with code $EXIT_CODE" | tee -a "$LOGFILE"

    # Check if training completed (exit 0 with "Already completed" or normal finish)
    if grep -q "Already completed\|Finished learning" "$LOGFILE" 2>/dev/null; then
        echo "$(date): Training completed!" | tee -a "$LOGFILE"
        break
    fi

    echo "$(date): Waiting 30s before restart..." | tee -a "$LOGFILE"
    sleep 30
done
