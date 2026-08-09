#!/bin/bash
# STAGES 4-7, run back to back. Each closes one of the outstanding TODOs.
#
#   4  SMOOTH DRIFT      the continuity type never tested. No boundaries, so EWC never
#                        accumulates a Fisher and degenerates into vanilla by construction,
#                        while PT's timer-based consolidation is unaffected. PT's best case.
#   5  ONE-LINE CONTROL  vanilla + frozen sigma + KL-to-ZERO-prior. Tests whether the entire
#                        measured pt_full benefit reduces to a single penalty term.
#   6  FREQUENCY LADDER  a cleaner centroid control than Stage 2: both tasks byte-identical at
#                        every level, only visitation frequency changes, so per-task difficulty
#                        is exactly constant (Stage 2 could not achieve this -- see §11a).
#   7  EWC +/- log_std   is EWC's advantage weight protection, or exploration preservation?
#
#   MAXJOBS=7 SEEDS="0 1 2 3 4 5 6 7" bash scripts/run_stages_4_7.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
[ -d "src_continuous_control" ] || { echo "ERROR: run from the PARENT of src_continuous_control/ (cwd=$(pwd))"; exit 1; }

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MAXJOBS=${MAXJOBS:-7}; SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}

# stage:config-basename:agent
JOBS=""
for a in pt inert van ewc;      do JOBS="$JOBS stage4:stage4_${a}:AUTO"; done
for r in r0001 r001 r01 r1 h32 h32_r001 h64_r001; do JOBS="$JOBS stage5:stage5_van_${r}:vanilla"; done
for f in f5 f6 f7 f8; do for a in pt inert van; do JOBS="$JOBS stage6:stage6_${a}_${f}:AUTO"; done; done
for e in withstd nostd;         do JOBS="$JOBS stage7:stage7_ewc_${e}:ewc"; done

for STAGE in stage4 stage5 stage6 stage7; do
  RES="${STAGE}_results"; LOG="${STAGE}_logs"; mkdir -p "$LOG"
  if compgen -G "${RES}/*/*_scalars.pkl" > /dev/null 2>&1; then
    echo "SKIP ${STAGE}: results already present"; continue
  fi
  echo "=== ${STAGE} START $(date) ==="
  for seed in $SEEDS; do
    for J in $JOBS; do
      S="${J%%:*}"; REST="${J#*:}"; CFG="${REST%%:*}"; AGENT="${REST#*:}"
      [ "$S" = "$STAGE" ] || continue
      # AUTO: read the agent out of the config so arms cannot drift from their own file
      if [ "$AGENT" = "AUTO" ]; then
        AGENT=$(python -c "import yaml,sys;print(yaml.safe_load(open('src_continuous_control/configs/${CFG}.yaml'))['agent'])")
      fi
      NAME="${CFG#${STAGE}_}"
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 3; done
      python -u -m src_continuous_control.train --agent "$AGENT" --config "$CFG" \
        --seed "$seed" --no-wandb --no-tb --no-eval \
        --results-dir "${RES}/${NAME}" --runs-dir "${RES}/_runs/${NAME}" \
        > "${LOG}/${NAME}_seed${seed}.log" 2>&1 &
    done
  done
  wait
  echo "=== ${STAGE} DONE $(date) ==="
done
echo "=== ALL STAGES 4-7 COMPLETE $(date) ==="
