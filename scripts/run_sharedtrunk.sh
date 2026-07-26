#!/bin/bash
# PT SHARED-TRUNK — the fix for the measured root cause. First FAIR test of the PT mechanism.
#
# Every earlier PT run exercised a consolidation operator that was destroying its own value function:
# with two separate MLP trunks, consolidation must make V_perm *learn* old_V_perm + V_trans by
# regression (one MLP representing the sum of two), which is lossy by construction — ~98% of the
# acting value lost per consolidation at production settings, ~150x per run.
#
# This variant uses one shared trunk + two LINEAR heads, so V = (w_P + w_T)·phi(s) and consolidation
# is exact weight arithmetic: w_P += (1-decay)·w_T ; w_T *= decay. Value drift is exactly zero
# (unit-tested). No regression, no consolidation buffer, no lr_perm.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     MAXJOBS=4 bash src_continuous_control/scripts/run_sharedtrunk.sh
#
# 5 runs (5 seeds) at the full 3.07M horizon => ~1.1 h at MAXJOBS=4.
# Writes to abl_results/pt_sharedtrunk/ — results/ is NEVER touched.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

# 24-thread box -> 3; 32-thread box -> 4.
MAXJOBS=${MAXJOBS:-3}

echo "=== PT shared-trunk START $(date) ==="
for seed in 0 1 2 3 4; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "launch pt_sharedtrunk seed=$seed  $(date +%H:%M:%S)"
  python -u -m src_continuous_control.train --agent pt --config abl_pt_sharedtrunk \
      --seed "$seed" --no-wandb --no-tb \
      --results-dir "abl_results/pt_sharedtrunk" --runs-dir "abl_runs/pt_sharedtrunk" \
      > "abl_logs/pt_sharedtrunk_seed${seed}.log" 2>&1 &
done
wait
echo "=== PT shared-trunk DONE $(date) ==="
echo
echo "REPORT per-phase MEAN return (primary) + end-of-phase values + seeds-positive per phase."
echo "Phase ends: 614400 | 1228800 | 1843200 | 2457600 | 3072000"
echo "Compare at phase 3:  broken PT -100 (1/5)   vanilla 950 (5/5)   EWC 1453 (5/5)"
echo "KEY: does PT now (a) avoid the phase-3+ collapse, and (b) reach or beat VANILLA?"
