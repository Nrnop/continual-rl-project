import os
import subprocess
import sys
import time
from multiprocessing import Pool

def run_job(args):
    agent_name, seed = args
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    
    cmd = [
        sys.executable, "-m", "src_continuous_control.train",
        "--agent", agent_name,
        "--seed", str(seed),
        "--total-steps", "3072000",
        "--switch", "614400",
        "--step-by-step", "true",
        "--eval-interval-updates", "50",
        "--save-checkpoints",
        "--no-wandb", "--no-tb"
    ]
    print(f"[runner] Starting {agent_name.upper()} PPO seed {seed}...")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[runner] ERROR on {agent_name.upper()} seed {seed} after {dt:.1f}s:\n{res.stderr}")
    else:
        print(f"[runner] Completed {agent_name.upper()} seed {seed} in {dt:.1f}s")
    return agent_name, seed, res.returncode, dt

if __name__ == "__main__":
    agents = ["vanilla", "ewc", "pt"]
    seeds = [0, 1, 2, 3, 4]
    
    print(f"[runner] Launching full 5-seed training + offline zero-momentum evaluation for agents: {agents}")
    total_t0 = time.time()
    
    all_results = []
    for agent in agents:
        print(f"\n=======================================================")
        print(f"[runner] Starting agent batch: {agent.upper()}")
        print(f"=======================================================")
        batch_args = [(agent, s) for s in seeds]
        t0 = time.time()
        with Pool(len(seeds)) as pool:
            results = pool.map(run_job, batch_args)
        all_results.extend(results)
        print(f"[runner] Batch {agent.upper()} finished in {time.time() - t0:.1f}s")
    
    print(f"\n=======================================================")
    print(f"[runner] All 15 runs completed in {time.time() - total_t0:.1f}s.")
    for ag, s, code, dt in all_results:
        status = "SUCCESS" if code == 0 else f"FAILED (code {code})"
        print(f"  - {ag.upper():7s} seed {s}: {status} in {dt:.1f}s")
    print(f"=======================================================\n")
    
    # Regenerate final figure suite
    print("[runner] Regenerating final comparative figure suite...")
    plot_cmd = [sys.executable, "-m", "src_continuous_control.plots.plot_compare", "--seeds", "0", "1", "2", "3", "4"]
    res_plot = subprocess.run(plot_cmd, capture_output=True, text=True)
    if res_plot.returncode != 0:
        print(f"[runner] ERROR plotting:\n{res_plot.stderr}")
    else:
        print(f"[runner] All figures generated successfully!\n{res_plot.stdout}")
