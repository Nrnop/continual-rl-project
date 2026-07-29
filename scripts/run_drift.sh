#!/bin/bash
# SMOOTH LIPSCHITZ DRIFT — the setting the thesis proposal specifies, and the one experiment where a
# POSITIVE result for PT is still plausible.
#
# Every result so far used DirectionalHalfCheetah, where the REWARD flips at discrete task
# boundaries. Here the reward is fixed and the PHYSICS drift continuously (joint damping + ground
# friction, sinusoidal, amplitude 0.5, period 1.23M steps => 2.5 cycles over the run). No boundaries,
# no task index, no reset signal.
#
# THE HYPOTHESIS: EWC computes its Fisher information AT A TASK BOUNDARY. With no boundaries,
# on_task_switch never fires, no Fisher is accumulated, and EWC degenerates into vanilla. PT's
# consolidation runs on a TIMER and needs no boundary, so its mechanism survives intact. EWC was the
# strongest agent under task switching; here it is structurally handicapped and PT is not. That is
# exactly the gap the proposal identifies.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     MAXJOBS=3 bash src_continuous_control/scripts/run_drift.sh
#
# 3 agents x 5 seeds at the full 3.07M horizon. Writes to abl_results/drift_<agent>/ —
# results/ is NEVER touched. Do NOT add --no-tb (with --no-wandb it makes the logger a no-op).
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
