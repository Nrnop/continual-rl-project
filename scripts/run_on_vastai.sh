#!/bin/bash
# ONE-SHOT REMOTE RUNNER -- the outstanding HalfCheetah work, sized for a many-core box.
#
# WHAT IT RUNS: stage 14, the HalfCheetah comparison at a normal exploration level
# (log_std_init = -1.0, sigma ~ 0.37). Four arms x SEEDS:
#
#   van          vanilla PPO
#   van_shrink   vanilla + policy shrink x0.5 every 8 updates   <- THE REDUCTION
#   pt           full pt_full apparatus, live permanent
#   frozen        pt_full with lr_perm=0 (shrink still active)
#
# Locally this is ~290 min at 7-way parallelism. On 24 threads it is ~90 min, which is why it
# moved to the box; everything else in the study is point-mass and finished locally.
#
#   RUN FROM THE PARENT of src_continuous_control/:
#     bash src_continuous_control/scripts/run_on_vastai.sh
#
#   Optional:  MAXJOBS=23  SEEDS="0 1 2 3 4 5 6 7"  bash .../run_on_vastai.sh
set -u

cd "$(dirname "$0")/../.." || exit 1
[ -d "src_continuous_control" ] || { echo "ERROR: run from the PARENT of src_continuous_control/ (cwd=$(pwd))"; exit 1; }

# CPU-bound: the wall clock is MuJoCo physics, the nets are tiny. Force CPU and pin each process
# to one thread -- without the pinning, concurrent runs thrash and throughput collapses
# (VASTAI_SETUP.md measured ~40 sps unpinned against ~1600 pinned).
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NPROC=$(nproc 2>/dev/null || echo 8)
MAXJOBS=${MAXJOBS:-$((NPROC - 1))}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}
RES="stage14_results"; LOG="stage14_logs"; mkdir -p "$LOG"

if compgen -G "${RES}/*/*_scalars.pkl" > /dev/null 2>&1; then
  echo "ERROR: ${RES}/ already has results. Move it aside first."; exit 1
fi

echo "=== host: $(hostname)  cores: ${NPROC}  maxjobs: ${MAXJOBS}  seeds: [${SEEDS}] ==="

# PRE-FLIGHT: assert the realised sigma matches across arms BEFORE burning hours on the sweep.
# A config key that one agent reads and another ignores is invisible in the returns and already
# invalidated one 24-run sweep (FULL_PT.md §22).
echo "--- pre-flight: realised log_std per arm (all four MUST agree) ---"
python - <<'PY' || { echo "PRE-FLIGHT FAILED"; exit 1; }
import sys, yaml, torch
sys.path.insert(0, ".")
from src_continuous_control.agents import AGENTS
vals = {}
for arm in ["van", "van_shrink", "pt", "inert"]:
    cfg = yaml.safe_load(open(f"src_continuous_control/configs/stage14_{arm}.yaml"))
    agent = AGENTS[cfg["agent"]](17, 6, cfg, torch.device("cpu"))
    vals[arm] = round(float(agent.actor.log_std.detach().mean()), 6)
    print(f"    {arm:12s} agent={cfg['agent']:8s} log_std={vals[arm]:+.4f}")
if len(set(vals.values())) != 1:
    print(f"    MISMATCH: {vals} -- the comparison would be confounded by exploration. ABORT.")
    raise SystemExit(1)
print("    OK: all arms share one exploration level.")
PY

echo "=== Stage 14 START $(date) ==="
for seed in $SEEDS; do
  for A in van van_shrink pt frozen; do
    AG=$(python -c "import yaml;print(yaml.safe_load(open('src_continuous_control/configs/stage14_${A}.yaml'))['agent'])")
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    python -u -m src_continuous_control.train --agent "$AG" --config "stage14_${A}" --seed "$seed" \
      --no-wandb --no-tb --no-eval \
      --results-dir "${RES}/${A}" --runs-dir "${RES}/_runs/${A}" \
      > "${LOG}/${A}_seed${seed}.log" 2>&1 &
  done
done
wait
echo "=== Stage 14 DONE $(date) ==="

echo
echo "Bring the results home (small -- pkl scalars only):"
echo "  tar czf stage14_out.tgz ${RES} ${LOG}"
echo "Then locally:  python src_continuous_control/scripts/analyze_all_stages.py"
