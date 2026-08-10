#!/bin/bash
# STAGE 12 -- HALFCHEETAH CONFIRMATION of the Stage 9/10 reduction.
#
# On DirectionalPointMass, pt_full's entire advantage over vanilla PPO turned out to be periodic
# multiplicative shrinkage of the policy toward zero: a clean dose-response in rho with the
# permanent zeroed and beta=0 (Stage 9), and full reproduction by periodic policy shrinkage added to vanilla at
# every decay factor (Stage 10, p >= 0.44). Everything else was eliminated -- the permanent
# (p=1.000), the KL anchor, actor and critic capacity, and the Adam flush (p=0.234).
#
# That claim needs one real-physics confirmation before it goes in the thesis. Four arms:
#
#   van          vanilla PPO
#   van_shrink   vanilla + policy shrink x0.5 every 8 updates   <- THE REDUCTION
#   pt           full pt_full apparatus, live permanent
#   frozen        pt_full with lr_perm=0 (shrink still active)
#
# PREDICTION: van_shrink ~= pt ~= frozen, and all three > van.
# If van_shrink falls short of pt on HalfCheetah, the reduction is point-mass-specific and the
# mechanism is doing something the shrinkage does not capture -- which would be a positive result
# for the method and should be reported as such.
#
# Parameter parity is computed, not guessed: vanilla [64,64] actor+critic = 11 085 params;
# pt_full perm [51,51] + trans [32,32] = 11 005 (0.993x). log_std is frozen on EVERY arm, since
# pt_full freezes it by C4 while vanilla learns it and that gap is large on HalfCheetah.
#
#   MAXJOBS=7 SEEDS="0 1 2 3 4 5" bash src_continuous_control/scripts/run_stage12_halfcheetah.sh
set -u

cd "$(dirname "$0")/../.." || exit 1
[ -d "src_continuous_control" ] || { echo "ERROR: run from the PARENT of src_continuous_control/ (cwd=$(pwd))"; exit 1; }

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

MAXJOBS=${MAXJOBS:-7}; SEEDS=${SEEDS:-"0 1 2 3 4 5"}
RES="stage12_results"; LOG="stage12_logs"; mkdir -p "$LOG"
if compgen -G "${RES}/*/*_scalars.pkl" > /dev/null 2>&1; then
  echo "ERROR: ${RES}/ already contains results."; exit 1
fi

echo "=== Stage 12 START $(date)  seeds=[$SEEDS] ==="
for seed in $SEEDS; do
  for A in van van_shrink pt frozen; do
    AG=$(python -c "import yaml;print(yaml.safe_load(open('src_continuous_control/configs/stage12_${A}.yaml'))['agent'])")
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    python -u -m src_continuous_control.train --agent "$AG" --config "stage12_${A}" --seed "$seed" \
      --no-wandb --no-tb --no-eval \
      --results-dir "${RES}/${A}" --runs-dir "${RES}/_runs/${A}" \
      > "${LOG}/${A}_seed${seed}.log" 2>&1 &
  done
done
wait
echo "=== Stage 12 DONE $(date) ==="
