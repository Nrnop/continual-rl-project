"""Watch a running sweep and exit non-zero as soon as a run looks broken.

    python -m src_continuous_control.scripts.watch_sweep --runs-dir src_continuous_control/runs/hard

Exits 0 when every run has finished. Checks:
  crashed     a traceback in a log, or no training process alive before all runs finished
  stalled     CPU time per process stopped advancing, compared only across PIDs alive at both
              polls (log freshness is unreliable because stdout to a file is block-buffered)
  not-finite  NaN/inf in a reported loss or return
  ewc-inert   ewc still identical to vanilla after the first boundary (before it they must agree,
              since no Fisher has been accumulated)
  pt-silent   a `pt` run past its first boundary with no probe/decay_gain line

Log-derived progress lags unless training runs with `python -u`.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

# Matched against a whole log, so MULTILINE makes `^` mean start of line.
STEP_RE = re.compile(r"^\[train\] step (\d+)/(\d+)\s+return=(\S+).*?critic_loss=(\S+)",
                     re.IGNORECASE | re.MULTILINE)
BAD_NUMBER = re.compile(r"\b(nan|inf|-inf)\b", re.IGNORECASE)


def _read(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _progress(text):
    """Return (last step seen, total steps, done) from a training log."""
    steps, total = [], 0
    for m in STEP_RE.finditer(text):
        steps.append(int(m.group(1)))
        total = int(m.group(2))
    return (steps[-1] if steps else 0), total, "[train] Done." in text


def _process_cpu():
    """Return {pid: cpu_seconds} for training processes only (matched on the command line).

    Per PID, because the summed CPU of the live set drops whenever runs finish and are replaced.
    The watchdog itself is excluded: it sleeps and would look stalled.
    """
    try:
        if os.name == "nt":
            ps = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                  "Where-Object { $_.CommandLine -like '*src_continuous_control.train*' } | "
                  "ForEach-Object { \"$($_.ProcessId) $($_.UserModeTime + $_.KernelModeTime)\" }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True, timeout=60).stdout
        else:
            out = subprocess.run(["ps", "-eo", "pid=,time=,args="], capture_output=True,
                                 text=True, timeout=60).stdout
            rows = {}
            for r in out.splitlines():
                if "src_continuous_control.train" not in r:
                    continue
                pid, clock = r.split()[0], r.split()[1]
                parts = clock.split(":")
                rows[int(pid)] = sum(float(v) * 60 ** i for i, v in enumerate(reversed(parts)))
            return rows
        cpu = {}
        for row in out.splitlines():
            bits = row.split()
            if len(bits) == 2 and all(b.isdigit() for b in bits):
                cpu[int(bits[0])] = int(bits[1]) / 1e7        # 100-ns units -> seconds
        return cpu
    except Exception:
        return {}          # unknown; the other checks still apply


def _returns_by_step(text):
    return {int(m.group(1)): m.group(3) for m in STEP_RE.finditer(text)}


def _check_ewc_diverges(logs, first_switch_step):
    """Past the first boundary, EWC must stop being bit-identical to vanilla."""
    out = []
    for van in [p for p in logs if os.path.basename(p).startswith("vanilla_seed_")]:
        seed = os.path.basename(van)[:-4].split("_")[-1]
        ewc = os.path.join(os.path.dirname(van), f"ewc_seed_{seed}.log")
        if not os.path.exists(ewc):
            continue
        a, b = _returns_by_step(_read(van)), _returns_by_step(_read(ewc))
        shared = [s for s in sorted(set(a) & set(b)) if s > first_switch_step * 1.5]
        if len(shared) >= 3 and all(a[s] == b[s] for s in shared[-3:]):
            out.append(f"ewc_seed_{seed}: IDENTICAL TO VANILLA at steps "
                       f"{[f'{s:,}' for s in shared[-3:]]} - past the first boundary the Fisher "
                       f"penalty should have moved it; EWC is a no-op in this run")
    return out


def _check_logs(logs, first_switch_step):
    problems, statuses, finished = [], {}, 0
    for path in logs:
        name = os.path.basename(path)[:-4]
        text = _read(path)
        step, total, done = _progress(text)
        statuses[name] = (step, total, done)
        finished += bool(done)

        if "Traceback (most recent call last)" in text:
            problems.append(f"{name}: CRASHED - traceback in the log")
            continue
        last_line = text.strip().rsplit("\n", 1)[-1] if text.strip() else ""
        if BAD_NUMBER.search(last_line):
            problems.append(f"{name}: NOT FINITE - {last_line[:110]}")
        if name.startswith("pt") and step > first_switch_step * 1.2 \
                and "probe/decay_gain" not in text:
            problems.append(f"{name}: PT DIAGNOSTICS SILENT - no probe/decay_gain after a "
                            f"boundary; the arm cannot be audited")
    problems += _check_ewc_diverges(logs, first_switch_step)
    return problems, statuses, finished


def main():
    p = argparse.ArgumentParser(description="Watch a running sweep and fail fast")
    p.add_argument("--runs-dir", type=str, default="src_continuous_control/runs")
    p.add_argument("--interval", type=int, default=900, help="seconds between polls")
    p.add_argument("--expected-runs", type=int, default=20)
    p.add_argument("--switch", type=int, default=614400, help="steps per task")
    p.add_argument("--report", type=str, default=None, help="append each poll's status here")
    args = p.parse_args()

    log_dir = os.path.join(args.runs_dir, "logs")
    previous_cpu, poll = {}, 0
    print(f"[watch] polling {log_dir} every {args.interval}s", flush=True)

    while True:
        poll += 1
        logs = sorted(glob.glob(os.path.join(log_dir, "*.log")))
        problems, statuses, finished = _check_logs(logs, args.switch)
        cpu = _process_cpu()
        alive, cpu_seconds = len(cpu), sum(cpu.values())

        # Compare only PIDs alive at both polls. Require 5% of one core each (real usage ~95%).
        shared = set(cpu) & set(previous_cpu)
        if shared:
            gained = sum(cpu[pid] - previous_cpu[pid] for pid in shared)
            if gained < 0.05 * args.interval * len(shared):
                problems.append(
                    f"STALLED - {len(shared)} long-lived processes gained only {gained:.0f}s CPU "
                    f"across {args.interval}s of wall clock; alive but not working")
        previous_cpu = cpu

        pct = "  ".join(f"{n} {s / max(t, 1) * 100:4.1f}%"
                        for n, (s, t, d) in sorted(statuses.items()))
        line = (f"[watch] poll {poll:3d}  {finished}/{args.expected_runs} done  "
                f"{alive} procs  {cpu_seconds / 60:.0f} CPU-min\n"
                f"         {pct}   (log %ages lag: stdout is block-buffered)")
        print(line, flush=True)
        if args.report:
            with open(args.report, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {line}\n")

        # Check completion first: a finished sweep has no CPU growth and would read as stalled.
        if finished >= args.expected_runs:
            print("[watch] every run finished cleanly.", flush=True)
            return 0
        if problems:
            print("\n[watch] *** PROBLEM DETECTED - stopping so it can be fixed now ***",
                  flush=True)
            for prob in problems:
                print(f"  - {prob}", flush=True)
            return 1
        if logs and alive == 0:
            print(f"[watch] *** no python processes alive but only {finished}/"
                  f"{args.expected_runs} runs finished - the sweep died ***", flush=True)
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
