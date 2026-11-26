#!/bin/bash
# Boundary-free Lipschitz drift on HalfCheetah: vanilla, pt and ewc, 5 seeds each, full 3.07M steps.
# The reward is fixed; joint damping and ground friction drift sinusoidally (amplitude 0.5, period
# 1.23M steps, about 2.5 cycles per run). There are no boundaries, so EWC never computes a Fisher
# and should match vanilla, while PT consolidates on a timer and is unaffected.
#
# Run from the parent directory of src_continuous_control/:
#     MAXJOBS=3 bash src_continuous_control/scripts/run_drift.sh
#
# Writes to abl_results/drift_<agent>/. Keep TensorBoard on: with --no-wandb, adding --no-tb
# makes the logger a no-op.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

# 24-thread box -> 3; 32-thread box -> 4.
MAXJOBS=${MAXJOBS:-3}

echo "=== Lipschitz drift sweep START $(date) ==="
for seed in 0 1 2 3 4; do
  for AG in vanilla pt ewc; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "launch drift_${AG} seed=$seed  $(date +%H:%M:%S)"
    python -u -m src_continuous_control.train --agent "$AG" --config drift \
        --seed "$seed" --no-wandb \
        --results-dir "abl_results/drift_${AG}" --runs-dir "abl_runs/drift_${AG}" \
        > "abl_logs/drift_${AG}_seed${seed}.log" 2>&1 &
  done
done
wait
echo "=== Lipschitz drift sweep DONE $(date) ==="
echo
echo "REPORT:"
echo "  1. Return over training for all three agents (mean +/- SEM over 5 seeds). There are NO"
echo "     phases here -- instead split the run into 5 equal segments of 614400 steps so the"
echo "     numbers line up with the task-switching tables."
echo "  2. Whether EWC differs from vanilla AT ALL. Prediction: it should not -- no boundaries means"
echo "     no Fisher, so EWC reduces to vanilla. Any gap is seed noise. Confirm by checking whether"
echo "     EWC ever logs a non-zero train/ewc_penalty."
echo "  3. Whether PT beats vanilla. THIS is the question the experiment exists to answer."
echo "  4. drift/multiplier over training, to read the return curves against the physics."
echo "Task-switching reference (per-phase means): vanilla 743/468/243/375/-34 |"
echo "  ewc 743/517/705/1238/533 | pt_sharedtrunk 814/394/27/212/-176"
