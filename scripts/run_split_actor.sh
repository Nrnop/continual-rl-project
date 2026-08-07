#!/bin/bash
# SPLIT ACTOR — put the permanent/transient decomposition on the POLICY.
#
# The hypothesis (supervisor, 2026-08-06): "we need to make the policy both permanent and
# transient not just the value function."
#
# Why it is worth running. On the CRITIC the decomposition is provably invisible to behaviour:
# V_trans is fit to R - V_perm, so A_trans = A_reward - A_perm identically, the two components
# come out anti-correlated at ~-1.0, and they cancel before anything reaches the actor
# (TRANSMISSION_RESULTS.md §4). That is the mechanism behind pt vs pt_inert, p=0.597.
#
# On the ACTOR they cannot cancel that way, because mu_perm + mu_trans IS the action. Decaying
# the transient changes behaviour with ZERO gradient steps -- which is exactly how PT works in
# DQN (argmax over Q_perm + Q_trans), and exactly what a value split cannot do in an actor-critic.
#
# THE headline measurement is `probe/decay_gain`: at each boundary, evaluate, decay mu_trans,
# evaluate again, with no gradient step in between. On a split critic that is provably 0.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     MAXJOBS=6 SEEDS="0 1 2 3 4 5 6 7 8 9" bash src_continuous_control/scripts/run_split_actor.sh
#
# Writes to ${RUN_TAG}_results/<arm>/ (default tag: split). Aborts if the target already holds
# results, so no earlier sweep can be overwritten or interleaved.
# --no-eval on EVERY arm. The boundary probe uses its own env and is NOT disabled by that flag.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

MAXJOBS=${MAXJOBS:-6}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
RUN_TAG=${RUN_TAG:-split}
RES_DIR="${RUN_TAG}_results"; RUNS_DIR="${RUN_TAG}_runs"; LOG_DIR="${RUN_TAG}_logs"

if compgen -G "${RES_DIR}/*/*_scalars.pkl" > /dev/null 2>&1; then
  echo "ERROR: ${RES_DIR}/ already contains results."
  echo "       Refusing to overwrite or interleave. Use RUN_TAG=<something-new> instead."
  exit 1
fi
mkdir -p "$LOG_DIR"

# vanilla and pt are re-run rather than reused: identical flags on every arm of a comparison is
# the protocol (defect #14), and these two are the references the new arms are judged against.
#
# arm_name:agent:config
ARMS="vanilla:vanilla:vanilla_paper \
      pt:pt:pt_paper \
      pt_actor:vanilla:abl_pt_actor \
      pt_both:pt:abl_pt_both"

echo "=== Split-actor sweep START $(date)  seeds=[$SEEDS] maxjobs=$MAXJOBS ==="
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
echo "=== Split-actor sweep DONE $(date) ==="
echo
echo "CHECK THIS FIRST, before any return comparison:"
echo "  grep -l 'INERT PERMANENT POLICY' ${LOG_DIR}/pt_actor_*.log ${LOG_DIR}/pt_both_*.log"
echo "  Any hit means lr_actor_perm is untuned and the arm says nothing (defect #9)."
echo
echo "Then the headline number:"
echo "  probe/decay_gain  -- return change from decaying mu_trans with ZERO gradient steps"
echo "  diag/actor_perm_trans_corr -- if this sits near -1, the actor cancels like the critic did"
