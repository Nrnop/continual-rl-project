#!/bin/bash
# TRANSMISSION TEST — does PT's critic knowledge reach the policy at all?
#
# The hypothesis (supervisor's, 2026-08-06): the continual-learning mechanism exists only on the
# critic. In DQN the value function IS the policy (argmax over Q), so decaying the transient at a
# switch returns behaviour to the task-average policy instantly, with zero gradient steps — that
# instantaneous behavioural reset is the jumpstart Theorem 6 describes. In PPO the policy is
# unchanged by any value update until gradients flow through the advantage, so PT's benefit has no
# surface to act on, while its cost (consolidation displaces V by design, keep=1) is absorbed in
# full. Cost with no compensation.
#
# Two things fall out of that, and they are what this sweep measures:
#
#   D1  PT's critic should be AT LEAST AS GOOD as vanilla's while its behaviour is worse.
#       `diag/explained_var` at collection time, in the window after each switch.
#       Better/equal critic + worse return = transmission gap, demonstrated rather than argued.
#
#   D2  The cost should be locked to the CONSOLIDATION grid (every k=60 updates), not to the task
#       grid. `diag/consol_age` tags each rollout with its position in that cycle. A dip at low
#       age, measured away from every boundary, cannot be explained by non-stationarity.
#
# Both diagnostics are always-on and consume no RNG, so these runs stay seed-comparable with the
# existing final sweep. Nothing about the agents changed — this is instrumentation only.
#
# Arms: vanilla_paper + pt_paper (the paper-faithful pair). pt_inert is included because it
# isolates D2: it pays the same consolidation/decay cost with a permanent that does not learn, so
# if the k-locked dip is real it should appear there too.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     MAXJOBS=6 SEEDS="0 1 2 3 4 5 6 7 8 9" bash src_continuous_control/scripts/run_transmission.sh
#
# Writes to ${RUN_TAG}_results/<arm>/ (default tag: trans2) — results/ is NEVER touched, and
# neither is any earlier sweep: the script aborts if the target already holds results.
# --no-eval on EVERY arm (protocol, REINVESTIGATION.md §8). Do NOT add --no-tb.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

MAXJOBS=${MAXJOBS:-6}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

# OUTPUT GOES TO A FRESH, TAGGED DIRECTORY — never on top of an earlier sweep.
#
# Two reasons, both learned the hard way. (1) The first sweep's `trans_results/` is the evidence
# behind the corrected D1 result; re-running into it would destroy the only positive finding we
# have. (2) If any seed fails, writing into a populated folder leaves a MIXTURE of pkls from two
# code versions with identical filenames, and an analysis that pools them silently is exactly the
# defect-#14 failure (two jobs launched differently, compared as if they were one).
#
# Override with RUN_TAG=... to start another clean sweep. The script REFUSES to start if the
# target already holds results.
RUN_TAG=${RUN_TAG:-trans2}
RES_DIR="${RUN_TAG}_results"; RUNS_DIR="${RUN_TAG}_runs"; LOG_DIR="${RUN_TAG}_logs"

if compgen -G "${RES_DIR}/*/*_scalars.pkl" > /dev/null 2>&1; then
  echo "ERROR: ${RES_DIR}/ already contains results."
  echo "       Refusing to overwrite or interleave. Use RUN_TAG=<something-new>, or move the"
  echo "       existing directory aside first."
  exit 1
fi
mkdir -p "$LOG_DIR"

# D3 — HOW MUCH POWER DOES THE POLICY HAVE vs HOW MUCH IS THE CRITIC EXERTING?
#
# Measured two ways, and they check each other:
#
#   correlational, free, on every arm: the advantage splits EXACTLY into reward / permanent /
#     transient parts (delta is affine in V, GAE is linear in delta), so the covariance share of
#     each is an exact attribution of the policy's update signal. Since the advantage is the
#     critic's only channel to the policy, `perm + trans` IS the critic's total influence on
#     decision-making, and `perm` alone is the permanent's.
#
#   causal, the ablation arms below: remove a component from the advantage the ACTOR trains on
#     while leaving `returns` — and so the critic's own target — untouched. Both critics still
#     learn normally; the policy simply cannot see one of them.
#
# Set ARMS_FULL=1 to include the causal arms (6 arms x 10 seeds instead of 3). The correlational
# measurement needs no extra runs at all and comes out of the base three.
#
# arm_name:agent:config
ARMS="vanilla:vanilla:vanilla_paper pt:pt:pt_paper pt_inert:pt:abl_pt_inert"
if [ "${ARMS_FULL:-0}" = "1" ]; then
  ARMS="$ARMS pt_advtrans:pt:abl_pt_advsrc_trans"     # permanent hidden from the actor
  ARMS="$ARMS pt_advnone:pt:abl_pt_advsrc_none"       # whole critic hidden from the actor
  ARMS="$ARMS van_advnone:vanilla:abl_vanilla_advsrc_none"   # same, for the baseline
fi

echo "=== Transmission sweep START $(date)  seeds=[$SEEDS] maxjobs=$MAXJOBS ==="
for seed in $SEEDS; do
  for A in $ARMS; do
    NAME="${A%%:*}"; REST="${A#*:}"; AGENT="${REST%%:*}"; CFG="${REST#*:}"
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "launch ${NAME} seed=$seed  $(date +%H:%M:%S)"
    python -u -m src_continuous_control.train --agent "$AGENT" --config "$CFG" \
        --seed "$seed" --no-wandb --no-eval \
        --results-dir "${RES_DIR}/${NAME}" --runs-dir "${RUNS_DIR}/${NAME}" \
        > "${LOG_DIR}/${NAME}_seed${seed}.log" 2>&1 &
  done
done
wait
echo "=== Transmission sweep DONE $(date) ==="
echo
ARM_NAMES=""
for A in $ARMS; do ARM_NAMES="$ARM_NAMES ${A%%:*}"; done
echo "Now run the analysis:"
echo "  python -m src_continuous_control.scripts.analyze_transmission \\"
echo "      --results-dir ${RES_DIR} --arms${ARM_NAMES} > ${RUN_TAG}_report.txt"
