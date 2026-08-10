#!/bin/bash
# STAGE 2 -- the centroid ladder. Does the PT mechanism stop costing, and start paying, once
# Theorem 5's fixed point E_tau[v_tau] carries real information?
#
# Stage 1 (FULL_PT.md §9) found that the live permanent is a NET COST, and that the cost
# shrinks monotonically as the task centroid becomes informative: -102 (E=0), -20 (E=+0.75),
# -3 (three-task). Stage 1's task sets, though, confounded asymmetry with difficulty.
#
# This ladder does not. Both targets stay exactly 2.50 apart and mean|target| stays 1.25 at every
# level, so switching distance and task difficulty are constant by construction; only the
# centroid slides:
#
#   L00  +1.25 / -1.25   E = 0.00     permanent's fixed point is structurally EMPTY
#   L05  +1.75 / -0.75   E = 0.50
#   L07  +2.00 / -0.50   E = 0.75
#   L10  +2.25 / -0.25   E = 1.00
#   L12  +2.50 / +0.00   E = 1.25     permanent's fixed point IS most of the task
#
# THE MEASUREMENT is not "does pt beat vanilla". It is (live - frozen) as a function of E: the
# same architecture, the same frozen sigma, the same KL anchor, differing only in whether the
# permanent learns. Prediction: that gap is negative at L00 and crosses zero as E grows.
#
#   MAXJOBS=7 SEEDS="0 1 2 3 4 5 6 7" bash scripts/run_stage2.sh
set -u

cd "$(dirname "$0")/../.." || exit 1
[ -d "src_continuous_control" ] || { echo "ERROR: run from the PARENT of src_continuous_control/ (cwd=$(pwd))"; exit 1; }

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

MAXJOBS=${MAXJOBS:-7}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
RUN_TAG=${RUN_TAG:-stage2}
RES_DIR="${RUN_TAG}_results"; LOG_DIR="${RUN_TAG}_logs"

if compgen -G "${RES_DIR}/*/*_scalars.pkl" > /dev/null 2>&1; then
  echo "ERROR: ${RES_DIR}/ already contains results. Use RUN_TAG=<something-new>."; exit 1
fi
mkdir -p "$LOG_DIR"

LEVELS="L00 L05 L07 L10 L12"
echo "=== Stage 2 START $(date)  seeds=[$SEEDS] maxjobs=$MAXJOBS ==="
for seed in $SEEDS; do
  for L in $LEVELS; do
    for SPEC in "pt_${L}:pt_full" "inert_${L}:pt_full" "van_${L}:vanilla"; do
      NAME="${SPEC%%:*}"; AGENT="${SPEC#*:}"
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 3; done
      python -u -m src_continuous_control.train --agent "$AGENT" --config "stage2_${NAME}" \
        --seed "$seed" --no-wandb --no-tb --no-eval \
        --results-dir "${RES_DIR}/${NAME}" --runs-dir "${RES_DIR}/_runs/${NAME}" \
        > "${LOG_DIR}/${NAME}_seed${seed}.log" 2>&1 &
    done
  done
done
wait
echo "=== Stage 2 DONE $(date) ==="
