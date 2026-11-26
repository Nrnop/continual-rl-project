#!/bin/bash
# Launch 15 full-length runs in parallel (vanilla, pt, ewc x seeds 0-4), then plot.
# Each run is 3,072,000 steps with a task switch every 614,400 steps, in step_by_step mode.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PKG_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
# step_by_step (batch_size=1) updates on small MLPs are faster on CPU, 4 threads per job, than
# sharing one GPU, which is bound by kernel launch and sync overhead.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p src_continuous_control/runs/logs
mkdir -p src_continuous_control/results

AGENTS=("vanilla" "pt" "ewc")
SEEDS=(0 1 2 3 4)

echo "=== Launching 15 full-scale training runs in parallel ==="
echo "Configuration: --total-steps 3072000 --switch 614400 --step-by-step true"
echo "Log directory: src_continuous_control/runs/logs/"

PIDS=()
for agent in "${AGENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        LOG_FILE="src_continuous_control/runs/logs/${agent}_seed_${seed}.log"
        echo "[launcher] Starting ${agent} seed=${seed} -> ${LOG_FILE}"
        /venv/main/bin/python -m src_continuous_control.train \
            --agent "${agent}" \
            --seed "${seed}" \
            --total-steps 3072000 \
            --switch 614400 \
            --step-by-step true \
            --eval-interval-updates 50 \
            --save-checkpoints \
            --no-wandb \
            --no-tb > "${LOG_FILE}" 2>&1 &
        PIDS+=($!)
    done
done

echo "[launcher] All 15 jobs launched. PIDs: ${PIDS[*]}"
echo "[launcher] Waiting for all runs to complete..."

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAIL=1
done

if [ ${FAIL} -ne 0 ]; then
    echo "[launcher] ERROR: One or more training runs failed. Check logs in src_continuous_control/runs/logs/."
    exit 1
fi

echo "[launcher] All 15 training runs completed successfully!"
echo "=== Updating Visualizations ==="
/venv/main/bin/python -m src_continuous_control.plots.plot_compare --seeds 0 1 2 3 4

echo "=== Full training and plotting suite completed successfully ==="
