#!/bin/bash
# PT ablation round 2 — FAST TRANSIENT at FULL horizon.
#
# Round 1 lesson: shortening total_steps changed the LR-anneal schedule (num_updates = total_steps/batch),
# which confounded the results. These runs use the FULL 3.07M / 5-phase horizon, so they are directly
# comparable to the existing full-sweep PT / vanilla / EWC results in results/.
#
# Hypothesis: PT is critic-only, so its only edge is adapting advantages FASTER after a switch — but
# the sweep used lr_trans = 3e-4 = vanilla's lr_critic, i.e. no fast timescale at all. These variants
# give the transient a real speed advantage (1e-3 and 3e-3).
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     bash src_continuous_control/scripts/run_ablation2.sh
#
# Baseline is FREE: the existing full-sweep PT data (results/pt_ppo_seed_*).
# Writes to abl_results/<variant>/ — results/ is NEVER touched.
# 10 runs (2 variants x 5 seeds) at ~32 min each, 3 concurrent => ~1.8 h.
set -u

if [ ! -d "src_continuous_control" ]; then
  echo "ERROR: run from the PARENT dir that contains src_continuous_control/  (cwd=$(pwd))"; exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p abl_logs

MAXJOBS=3            # concurrent runs (num_envs=8 each ~= 24 threads)

# name             config
JOBS=(
  "pt_fasttrans    abl_pt_fasttrans"
  "pt_fasttrans_x3 abl_pt_fasttrans_x3"
)

echo "=== PT ablation round 2 START $(date) ==="
for seed in 0 1 2 3 4; do
  for j in "${JOBS[@]}"; do
    set -- $j; name=$1 config=$2
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    echo "launch $name  seed=$seed  $(date +%H:%M:%S)"
    python -u -m src_continuous_control.train --agent pt --config "$config" \
        --seed "$seed" --no-wandb --no-tb \
        --results-dir "abl_results/${name}" --runs-dir "abl_runs/${name}" \
        > "abl_logs/${name}_seed${seed}.log" 2>&1 &
  done
done
wait
echo "=== PT ablation round 2 DONE $(date) ==="
echo
echo "Report end-of-phase returns per run (phases end at 614400/1228800/1843200/2457600/3072000):"
echo "  for f in abl_logs/pt_fasttrans*.log; do echo \"== \$f ==\"; grep -E 'step (6[01][0-9]{4}|12[23][0-9]{4}|18[34][0-9]{4}|24[56][0-9]{4}|30[67][0-9]{4})/' \"\$f\" | tail -1; done"
echo "Compare against existing full-sweep PT (P2 mean -100, 1/5 seeds positive) and vanilla (P2 mean 950, 5/5 positive)."
