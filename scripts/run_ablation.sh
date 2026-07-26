#!/bin/bash
# PT ablation — isolate why PT collapses after the 2nd task switch.
# Short 3-phase (2-switch) runs, 2 seeds each: 3 PT variants.
# (No vanilla re-run: the full sweep's vanilla at step 1,843,200 is already the same-conditions
#  reference — identical phases/steps. Its P2 end ~= 950.)
#
#   RUN FROM THE PARENT of src_continuous_control/ (the folder that contains it):
#     bash src_continuous_control/scripts/run_ablation.sh
#
# Writes pkls to abl_results/<variant>/ so the full-sweep results/ (vanilla, ewc, pt) are UNTOUCHED.
# ~40 min on a 24-thread box. CPU-only + per-process single-thread (avoids oversubscription).
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

MAXJOBS=3            # concurrent runs (num_envs=8 each ~= 24 threads)
STEPS=1843200       # 3 phases / 2 switches at switch=614400

# name              config                (all PT; each writes to its own abl_results/<name>/)
JOBS=(
  "pt_baseline      abl_pt_baseline"
  "pt_noswitchdecay abl_pt_noswitchdecay"
  "pt_unfreezeperm  abl_pt_unfreezeperm"
)

echo "=== PT ablation START $(date) ==="
for seed in 0 1; do
  for j in "${JOBS[@]}"; do
    set -- $j; name=$1 config=$2
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "launch $name  seed=$seed  $(date +%H:%M:%S)"
    python -u -m src_continuous_control.train --agent pt --config "$config" \
        --seed "$seed" --total-steps "$STEPS" --no-wandb --no-tb \
        --results-dir "abl_results/${name}" --runs-dir "abl_runs/${name}" \
        > "abl_logs/${name}_seed${seed}.log" 2>&1 &
  done
done
wait
echo "=== PT ablation DONE $(date) ==="
echo "Compare phase-2 recovery across variants:"
echo "  for f in abl_logs/*.log; do echo \"== \$f ==\"; grep return= \"\$f\" | tail -3; done"
