#!/bin/bash
# ROUND 7 (2x2) — isolate the transient DECAY, and test whether stale optimiser state explains it.
#
# Round 6 killed the memorisation hypothesis: consolidation is near-exact in situ (~0.3% drift on
# fitted AND held-out states, every seed, all of training) yet PT still collapses. So the transfer
# is not the problem. What separates healthy from collapsing runs is the decay -- every decay=0.5
# run survives phase 2 (+332/+394/+134), every decay=0.0 run does not (-319/+30/-594), including one
# with completely untrained consolidation. Earlier rounds never varied decay alone, so that
# comparison was confounded (FINDINGS.md 5.5 correction).
#
# Mechanism under test: decay_transient scales the transient's PARAMETERS but leaves Adam's
# exp_avg / exp_avg_sq untouched, so the next step pushes the just-zeroed weights straight back out
# using momentum from a network that no longer exists.
#
#   decay 0.0 | 0.5   x   reset_trans_optim_on_decay false | true    x  3 seeds  = 12 runs
#
#   reset fixes decay=0    -> stale optimiser state was the mechanism; PT may be salvageable
#   only decay=0.5 matters -> decay is the lever, but via some other route
#   neither helps          -> the reset is not implicated; stop looking here
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     MAXJOBS=3 bash src_continuous_control/scripts/run_ablation7.sh
#
# ~2-2.5 h at MAXJOBS=3. Writes to abl_results/r7_<variant>/ — results/ is NEVER touched.
# Do NOT add --no-tb (with --no-wandb it makes the logger a no-op).
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

MAXJOBS=${MAXJOBS:-3}

VARIANTS="decay00_noreset decay00_reset decay05_noreset decay05_reset"

echo "=== Round 7 (decay x optimiser-state reset) START $(date) ==="
for seed in 0 1 2; do
  for V in $VARIANTS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "launch r7_${V} seed=$seed  $(date +%H:%M:%S)"
    python -u -m src_continuous_control.train --agent pt --config "abl_pt_r7_${V}" \
        --seed "$seed" --no-wandb \
        --results-dir "abl_results/r7_${V}" --runs-dir "abl_runs/r7_${V}" \
        > "abl_logs/r7_${V}_seed${seed}.log" 2>&1 &
  done
done
wait
echo "=== Round 7 DONE $(date) ==="
echo
echo "REPORT per-phase MEAN return + SEM for all four cells, laid out as the 2x2:"
echo "                      decay=0.0        decay=0.5"
echo "   no reset            ...              ...       <- decay05_noreset should match the shipped"
echo "   reset optim state   ...              ...          config (475/332/-279/-249/-396)"
echo
echo "Key questions, in order:"
echo "  1. Does decay=0.5 beat decay=0.0 with the reset OFF? (isolates decay, un-confounded)"
echo "  2. Does the reset rescue decay=0.0? (tests the stale-momentum mechanism)"
echo "  3. Does any cell beat vanilla's 743/468/243/375/-34? THAT is the question that matters."
echo "Note n=3 here: phase-2 SEM has been as high as 451 in a 3-seed run, so say when a gap is"
echo "inside noise rather than reporting it as a difference."
