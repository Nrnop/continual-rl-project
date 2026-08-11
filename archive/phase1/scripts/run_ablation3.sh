#!/bin/bash
# PT ablation round 3 — DECAY=0 (fixes a confirmed bug), full horizon, 5 seeds.
#
# THE BUG: SplitCritic.decay_transient(d) scales every PARAMETER of the transient MLP by d, which
# does NOT scale the OUTPUT V_trans by d for a nonlinear net (measured ~33% error at d=0.5). So the
# value-preserving consolidation identity holds ONLY at d=0; at d=0.5 every consolidation injects an
# uncontrolled perturbation into the acting value, ~150x per run (1500 updates / k=10).
# This matches the evidence: critic_loss stays LOW (transient re-fits after each shock) while returns
# crater, and damage compounds from phase 3 on as the LR anneals.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     bash src_continuous_control/scripts/run_ablation3.sh
#
# Baseline is FREE: existing full-sweep PT (results/pt_ppo_seed_*).
# Writes to abl_results/<variant>/ — results/ is NEVER touched.
# 10 runs (2 variants x 5 seeds) at full horizon, 3 concurrent => ~1.8 h.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

# Concurrent runs. Each uses num_envs=8 worker processes, so budget ~8 threads per job.
# 24-thread box -> 3; 32-thread box -> 4.  Override:  MAXJOBS=4 bash scripts/run_ablation3.sh
MAXJOBS=${MAXJOBS:-3}

# name            config
JOBS=(
  "pt_decay0      abl_pt_decay0"
  "pt_decay0_fast abl_pt_decay0_fast"
)

echo "=== PT ablation round 3 START $(date) ==="
for seed in 0 1 2 3 4; do
  for j in "${JOBS[@]}"; do
    set -- $j; name=$1 config=$2
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "launch $name  seed=$seed  $(date +%H:%M:%S)"
    python -u -m src_continuous_control.train --agent pt --config "$config" \
        --seed "$seed" --no-wandb --no-tb \
        --results-dir "abl_results/${name}" --runs-dir "abl_runs/${name}" \
        > "abl_logs/${name}_seed${seed}.log" 2>&1 &
  done
done
wait
echo "=== PT ablation round 3 DONE $(date) ==="
echo
echo "REPORT per-phase MEAN return (AUC-style, lower variance than the single end-of-phase point)."
echo "Phase boundaries: 0-614400 | -1228800 | -1843200 | -2457600 | -3072000"
echo "Also report end-of-phase values and mean critic_loss, and compare to:"
echo "  full-sweep PT phase3 mean -100 (1/5 positive) | vanilla 950 (5/5) | EWC 1453 (5/5)"
