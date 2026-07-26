#!/bin/bash
# PT ablation round 4 — THE CAUSAL CONTROL: PT with consolidation disabled entirely.
#
# Root cause measured directly: consolidation transfers ~0.05% of the transient into the permanent
# while the decay wipes 100% of it, so the acting value loses ~98% of its magnitude every k=10
# updates (~150x per run), corrupting the values GAE uses. This run removes consolidation completely.
#   PT tracks vanilla  -> consolidation PROVEN to be the cause (diagnosis confirmed, story closed).
#   PT still collapses -> diagnosis incomplete; something else is also at fault.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     MAXJOBS=4 bash src_continuous_control/scripts/run_ablation4.sh
#
# 5 runs (1 variant x 5 seeds) at the full 3.07M horizon => ~1.1 h at MAXJOBS=4.
# Writes to abl_results/pt_noconsol/ — results/ is NEVER touched.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

# 24-thread box -> 3; 32-thread box -> 4.
MAXJOBS=${MAXJOBS:-3}

echo "=== PT ablation round 4 (no-consolidation control) START $(date) ==="
for seed in 0 1 2 3 4; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "launch pt_noconsol seed=$seed  $(date +%H:%M:%S)"
  python -u -m src_continuous_control.train --agent pt --config abl_pt_noconsol \
      --seed "$seed" --no-wandb --no-tb \
      --results-dir "abl_results/pt_noconsol" --runs-dir "abl_runs/pt_noconsol" \
      > "abl_logs/pt_noconsol_seed${seed}.log" 2>&1 &
done
wait
echo "=== PT ablation round 4 DONE $(date) ==="
echo
echo "REPORT per-phase MEAN return (primary metric) + end-of-phase values + seeds-positive count."
echo "Phase ends: 614400 | 1228800 | 1843200 | 2457600 | 3072000"
echo "Compare at phase 3:  PT baseline -100 (1/5)   vanilla 950 (5/5)   EWC 1453 (5/5)"
echo "KEY: does PT-without-consolidation now track VANILLA (not just beat the broken PT)?"
