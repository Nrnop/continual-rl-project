#!/bin/bash
# DIAGNOSTIC round 6 — does the consolidation regression MEMORISE its buffer?
#
# Round 5 puzzle: with Adam / lr_perm=1e-3 / 20 epochs the in-situ consolidation error fell to
# ~0.0-0.7% (an essentially perfect fit) yet PT did WORSE than every other variant and collapsed at
# the FIRST switch. Hypothesis: that metric only sees the states just trained on. The permanent net
# memorises the 20 480-state buffer and extrapolates badly onto the NEW states visited next --
# worst right after a switch, when the state distribution has just moved.
#
# This run holds out 20% of the buffer from the regression and reports the same value-drift metric
# on it (train/consolidation_error_holdout_pct):
#   fitted low + held-out low  -> hypothesis wrong
#   fitted low + held-out high -> confirmed
#
# 3 seeds only: this measures a mechanism, not a noisy return difference.
#
# IMPORTANT: do NOT pass --no-tb. With both --no-wandb and --no-tb the logger backend resolves to
# "none" and log_scalar becomes a no-op, so these metrics would never be written anywhere.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     MAXJOBS=3 bash src_continuous_control/scripts/run_ablation6.sh
#
# Writes to abl_results/pt_consol_holdout/ and abl_runs/pt_consol_holdout/ (TB event files carry
# the metrics). results/ is NEVER touched.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

# 24-thread box -> 3; 32-thread box -> 4.
MAXJOBS=${MAXJOBS:-3}

echo "=== PT ablation round 5 (trained consolidation) START $(date) ==="
for seed in 0 1 2; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "launch pt_consol_holdout seed=$seed  $(date +%H:%M:%S)"
  python -u -m src_continuous_control.train --agent pt --config abl_pt_consol_holdout \
      --seed "$seed" --no-wandb \
      --results-dir "abl_results/pt_consol_holdout" --runs-dir "abl_runs/pt_consol_holdout" \
      > "abl_logs/pt_consol_holdout_seed${seed}.log" 2>&1 &
done
wait
echo "=== PT ablation round 5 DONE $(date) ==="
echo
echo "REPORT: for each seed, the TREND over training of BOTH"
echo "  train/consolidation_error_pct           (states the regression fitted)"
echo "  train/consolidation_error_holdout_pct   (buffered states EXCLUDED from the regression)"
echo "read from the TB event files in abl_runs/pt_consol_holdout/."
echo "Key question: does a low fitted error coincide with a HIGH held-out error?"
echo "Also report per-phase mean return, but this run is a diagnostic, not a performance comparison."
