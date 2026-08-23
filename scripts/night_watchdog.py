"""Keep the machine busy for a whole unattended night: restart the queue launcher if it dies.

    python -m src_continuous_control.scripts.night_watchdog --plan night_0817 \
        --results-dir src_continuous_control/results --jobs 7

THE GAP THIS FILLS. `queue_runs.py` is a single Python process holding the whole plan in memory. It
is robust to a RUN dying — a crashed run frees a slot and the next one fills it — but not to
ITSELF dying. If the launcher is killed at 2am (console closed, session reaped, OOM, Windows
deciding to be Windows) the runs already in flight finish normally and then the box sits idle
until morning. On a 12-hour queue that is most of the night thrown away, and nothing in the logs
says so; they just stop.

`watch_sweep.py` does not cover this. It is a failure DETECTOR — it exits non-zero when a run looks
broken so a human can react. This is the opposite job: notice that nothing is running, and act.

WHY IT CANNOT DOUBLE-LAUNCH A RUN. The dangerous move would be relaunching while orphaned training
processes are still alive: `queue_runs` treats a run as complete only when its returns.pkl reaches
full length, so a run that is midway through is "incomplete" and would be started a SECOND time,
with two processes writing one result file. This watchdog therefore relaunches only when the
machine is COMPLETELY idle — zero training processes and no launcher. Orphans are left to finish
first, which costs a little idle time and buys correctness.

It exits 0 when the plan is finished, so it is also a clean "tell me when the night is done".
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

from .queue_runs import PLANS, _complete, _running


def _launcher_alive():
    """Is a queue_runs process running? Anyone's — this watchdog does not own it."""
    try:
        if os.name == "nt":
            ps = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                  "Where-Object { $_.CommandLine -like '*queue_runs*' } | "
                  "Measure-Object | ForEach-Object { $_.Count }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True, timeout=60).stdout
            return int(out.strip() or 0) > 0
        out = subprocess.run(["pgrep", "-fc", "queue_runs"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0) > 0
    except Exception:
        # Fail SAFE: if we cannot tell, assume it is alive rather than starting a second one.
        return True


def _remaining(plan, results_dir, seeds, min_steps):
    todo = 0
    for entry in PLANS[plan]:
        arm, agent = entry[0], entry[1]
        for s in (entry[3] if len(entry) > 3 else seeds):
            if not _complete(results_dir, arm, agent, s, min_steps):
                todo += 1
    return todo


def log(msg):
    print(f"[watchdog {datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Restart the queue launcher if it dies")
    p.add_argument("--plan", default="night_0817", choices=list(PLANS))
    p.add_argument("--results-dir", default="src_continuous_control/results")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=7)
    p.add_argument("--min-steps", type=int, default=3_000_000)
    p.add_argument("--poll", type=int, default=300)
    p.add_argument("--max-restarts", type=int, default=10,
                   help="give up after this many, so a config that crashes instantly cannot "
                        "spin all night relaunching itself")
    args = p.parse_args()

    restarts = 0
    log(f"watching plan '{args.plan}', {_remaining(args.plan, args.results_dir, args.seeds, args.min_steps)} runs to go")

    while True:
        todo = _remaining(args.plan, args.results_dir, args.seeds, args.min_steps)
        if todo == 0:
            log("plan complete — nothing left to run. Exiting 0.")
            return 0

        alive_launcher = _launcher_alive()
        alive_runs = _running()

        if alive_launcher:
            log(f"ok - launcher up, {alive_runs} run(s) training, {todo} to go")
        elif alive_runs:
            # Launcher gone but its children are still working. Let them finish; relaunching now
            # would start a second copy of whatever is midway.
            log(f"launcher GONE but {alive_runs} run(s) still training — waiting for idle "
                f"before restarting ({todo} to go)")
        else:
            if restarts >= args.max_restarts:
                log(f"launcher gone, machine idle, {todo} runs left — but already restarted "
                    f"{restarts} times. Refusing to loop. Exiting 1.")
                return 1
            restarts += 1
            log(f"launcher gone and machine IDLE with {todo} runs left — restarting "
                f"(restart {restarts}/{args.max_restarts})")
            logfile = os.path.join(args.results_dir, f"{args.plan}.log")
            with open(logfile, "a") as f:
                f.write(f"\n=== watchdog restart {restarts} at {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            with open(logfile, "a") as f:
                subprocess.Popen(
                    [sys.executable, "-u", "-m", "src_continuous_control.scripts.queue_runs",
                     "--plan", args.plan, "--results-dir", args.results_dir,
                     "--jobs", str(args.jobs), "--min-steps", str(args.min_steps)],
                    stdout=f, stderr=subprocess.STDOUT)
            time.sleep(30)      # let it claim its slots before the next poll counts processes

        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
