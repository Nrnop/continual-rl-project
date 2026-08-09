#!/bin/bash
# STAGE 3 -- the KL-anchor (beta) sweep + the consolidation-shuffle defect test.
# All at centroid E=0 (targets +-1.25), the case matching the thesis benchmark.
set -u
cd "$(dirname "$0")/../.." || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MAXJOBS=${MAXJOBS:-7}; SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
RES="stage3_results"; LOG="stage3_logs"; mkdir -p "$LOG"
if compgen -G "${RES}/*/*_scalars.pkl" >/dev/null 2>&1; then echo "ERROR: ${RES} not empty"; exit 1; fi
for seed in $SEEDS; do
  for NAME in pt_b000 inert_b000 pt_b0001 inert_b0001 pt_b001 inert_b001 \
              pt_b01 inert_b01 pt_b1 inert_b1 pt_shuf inert_shuf; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 3; done
    python -u -m src_continuous_control.train --agent pt_full --config "stage3_${NAME}" \
      --seed "$seed" --no-wandb --no-tb --no-eval \
      --results-dir "${RES}/${NAME}" --runs-dir "${RES}/_runs/${NAME}" \
      > "${LOG}/${NAME}_seed${seed}.log" 2>&1 &
  done
done
wait; echo "=== Stage 3 DONE $(date) ==="
