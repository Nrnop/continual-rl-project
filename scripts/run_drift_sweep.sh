#!/bin/bash
# Faster drift settings for HalfCheetah, vanilla vs pt, 3 seeds (12 runs).
#   drift_fast      10x faster single-timescale drift (~52% change per consolidation cycle)
#   drift_twoscale  slow trend (amplitude 0.4, period 1.23M) plus a fast fluctuation
#                   (amplitude 0.2, period 30720): the permanent should hold the trend while
#                   the transient absorbs the fluctuation
# EWC is omitted: with no task boundary its Fisher is never computed and it matches vanilla.
#
# Run from the parent directory of src_continuous_control/:
#     MAXJOBS=3 bash src_continuous_control/scripts/run_drift_sweep.sh
#
# Writes to abl_results/<setting>_<agent>/. Keep TensorBoard on: with --no-wandb, adding --no-tb
# makes the logger a no-op.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

MAXJOBS=${MAXJOBS:-3}

echo "=== drift sweep START $(date) ==="
for seed in 0 1 2; do
  for CFG in drift_twoscale drift_fast; do
    for AG in vanilla pt; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
      echo "launch ${CFG}_${AG} seed=$seed  $(date +%H:%M:%S)"
      python -u -m src_continuous_control.train --agent "$AG" --config "$CFG" \
          --seed "$seed" --no-wandb \
          --results-dir "abl_results/${CFG}_${AG}" --runs-dir "abl_runs/${CFG}_${AG}" \
          > "abl_logs/${CFG}_${AG}_seed${seed}.log" 2>&1 &
    done
  done
done
wait
echo "=== drift sweep DONE $(date) ==="
echo
echo "REPORT, per setting, PT vs vanilla:"
echo "  return by 614400-step segment (mean +/- SEM over 3 seeds)"
echo "  and state plainly whether PT beats vanilla OUTSIDE the combined SEM."
echo "Reference - the SLOW drift run already done (n=5): vanilla 569/1631/1320/1772/1655,"
echo "  pt 546/1564/1300/1870/1808 (tied everywhere)."
echo "The question: does a fast component, or an explicit slow+fast split, change that?"
echo "n=3, so be explicit about noise."
