#!/bin/bash
# PT ablation round 5 — separate trunks + a PROPERLY TRAINED consolidation regression.
#
# Closes the one gap in the null result. The PT idea needs a permanent critic that both ACCUMULATES
# knowledge and is INSULATED from the fast timescale; every config run so far breaks one of those:
#   consolidation off  -> insulated but learns nothing
#   shared trunk       -> accumulates exactly, but its features move at the fast rate
#   shipped separate   -> both in principle, but the regression is never trained (0.05% transfer)
# This runs separate trunks with Adam, lr_perm 1e-3 and 20 consolidation epochs (6400 gradient steps
# per consolidation instead of 320).
#
# The run also logs train/consolidation_error_pct -- the in-situ % change in the acting value across
# each consolidation -- so it reports how well the transfer actually works, not just final return.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     MAXJOBS=4 bash src_continuous_control/scripts/run_ablation5.sh
#
# NOTE: consolidation roughly doubles the gradient work, so expect ~3x a normal run
# (~1.5-2 h each; ~3-4 h total for 5 seeds at MAXJOBS=4).
# Writes to abl_results/pt_trained_consol/ — results/ is NEVER touched.
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
for seed in 0 1 2 3 4; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "launch pt_trained_consol seed=$seed  $(date +%H:%M:%S)"
  python -u -m src_continuous_control.train --agent pt --config abl_pt_trained_consol \
      --seed "$seed" --no-wandb --no-tb \
      --results-dir "abl_results/pt_trained_consol" --runs-dir "abl_runs/pt_trained_consol" \
      > "abl_logs/pt_trained_consol_seed${seed}.log" 2>&1 &
done
wait
echo "=== PT ablation round 5 DONE $(date) ==="
echo
echo "REPORT:"
echo "  1. per-phase MEAN return + seeds-positive per phase (primary metric)"
echo "  2. train/consolidation_error_pct over training -- does the transfer actually work in situ?"
echo "     (0% = value-preserving; the offline probe predicted ~29-39% at this width)"
echo "Compare per-phase means against:"
echo "  vanilla 743/468/243/375/-34   ewc 743/517/705/1238/533"
echo "  pt(shipped) 475/332/-279/-249/-396   pt_sharedtrunk 814/394/27/212/-176"
