#!/bin/bash
# Parallel execution of single-task baseline across seeds 0..4 for vanilla, pt, and ewc agents.
# Total: 15 runs. Each run performs 3,072,000 steps with task switching disabled (--disable-task-switch) in step_by_step mode.
# Video rendering is enabled on Seed 0 for each agent (--render --render-freq 25) using EGL to capture trajectory evolution.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PKG_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export MUJOCO_GL="egl"

mkdir -p src_continuous_control/runs/logs
mkdir -p src_continuous_control/results_singletask
mkdir -p src_continuous_control/runs_singletask/videos

AGENTS=("vanilla" "pt" "ewc")
SEEDS=(0 1 2 3 4)

echo "=== Launching 15 Single-Task Baseline runs in parallel ==="
echo "Configuration: --total-steps 3072000 --disable-task-switch --step-by-step true"
echo "Rendering: Enabled on Seed 0 (--render --render-freq 25), disabled on Seeds 1..4"
echo "Log directory: src_continuous_control/runs/logs/"
echo "Results directory: src_continuous_control/results_singletask/"

PIDS=()
for agent in "${AGENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        LOG_FILE="src_continuous_control/runs/logs/singletask_${agent}_seed_${seed}.log"
        echo "[launcher] Starting ${agent} seed=${seed} -> ${LOG_FILE}"
        
        EXTRA_ARGS=""
        if [ "${seed}" -eq 0 ]; then
            EXTRA_ARGS="--render --render-freq 25"
        fi
        
        /workspace/venv_continuous/bin/python -u -m src_continuous_control.train \
            --agent "${agent}" \
            --seed "${seed}" \
            --total-steps 3072000 \
            --switch 614400 \
            --step-by-step true \
            --disable-task-switch \
            --eval-interval-updates 50 \
            --save-checkpoints \
            --results-dir "src_continuous_control/results_singletask" \
            --runs-dir "src_continuous_control/runs_singletask" \
            ${EXTRA_ARGS} \
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
    echo "[launcher] ERROR: One or more training runs failed. Check logs in src_continuous_control/runs/logs/singletask_*.log."
    exit 1
fi

echo "[launcher] All 15 single-task baseline runs completed successfully!"
