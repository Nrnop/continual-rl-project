import os
import subprocess
import sys
import time
from multiprocessing import Pool

def run_seed(seed):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    
    cmd = [
        sys.executable, "-m", "src_continuous_control.train",
        "--agent", "vanilla",
        "--seed", str(seed),
        "--no-wandb", "--no-tb"
    ]
    print(f"[runner] Starting Vanilla PPO seed {seed}...")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[runner] ERROR on seed {seed} after {dt:.1f}s:\n{res.stderr}")
    else:
        print(f"[runner] Completed Vanilla PPO seed {seed} in {dt:.1f}s")
    return seed, res.returncode

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    print(f"[runner] Launching Vanilla PPO 5-seed baseline across seeds: {seeds}")
    t0 = time.time()
    with Pool(len(seeds)) as pool:
        results = pool.map(run_seed, seeds)
    
    print(f"[runner] All runs finished in {time.time() - t0:.1f}s. Results: {results}")
    
    # Run plotting
    print("[runner] Regenerating comparative plots...")
    plot_cmd = [sys.executable, "-m", "src_continuous_control.plots.plot_compare", "--seeds", "0", "1", "2", "3", "4"]
    res_plot = subprocess.run(plot_cmd, capture_output=True, text=True)
    if res_plot.returncode != 0:
        print(f"[runner] ERROR plotting:\n{res_plot.stderr}")
    else:
        print(f"[runner] Plots generated successfully!\n{res_plot.stdout}")
