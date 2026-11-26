"""Recover the policy's exploration width sigma(t) from each finished run and report collapse.

Vanilla and EWC runs log `train/entropy` but not `log_std_mean`. For a diagonal Gaussian with
state-independent log_std, H = (d/2)*log(2*pi*e) + sum_i log sigma_i, so the mean log-sigma is
(H - (d/2)*log(2*pi*e)) / d. Only the mean over action dimensions is recoverable.

`--verify` compares the recovery with `pt`'s logged `log_std_mean`. Expect about 1e-2 in log
space: entropy is averaged over the PPO epochs of an update, log_std_mean is read at its end.

Arms with a frozen log_std must come out flat; a frozen arm that moves is flagged, since it
means the freeze is not applied.

Usage, from the parent directory of the package:

    python -m src_continuous_control.scripts.check_sigma_collapse
    python -m src_continuous_control.scripts.check_sigma_collapse --verify
    python -m src_continuous_control.scripts.check_sigma_collapse --plot
"""
import argparse
import math
import os
import pickle
import re

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

# HalfCheetah-v5. Task labels are appended to the observation, so the action dim is always 6.
ACT_DIM = 6
# (d/2) * log(2*pi*e): the constant term of a d-dimensional diagonal Gaussian's entropy.
ENTROPY_CONST = 0.5 * math.log(2.0 * math.pi * math.e) * ACT_DIM

SWITCH = 614400          # env steps per task
N_TASKS = 5

# (label, directory, is_sigma_frozen). Frozen arms are a control: their recovered sigma must be flat.
ARMS = [
    ("physics vanilla",   "clean/vanilla",        False),
    ("physics pt",        "clean/pt",             False),
    ("physics pt-wide",   "clean/pt_double_actor",         False),
    ("physics ewc",       "clean/ewc_fixed",      False),
    ("physics pt-frozen", "clean/pt_frozen",      False),
    ("flip vanilla",      "rewardflip/vanilla",   False),
    ("flip pt",           "rewardflip/pt",        False),
    ("flip ewc",          "rewardflip/ewc",       False),
    ("s037 vanilla",      "s14reset/van",         True),
    ("s037 pt",           "s14reset/pt",          True),
    ("s037 ewc",          "s14reset/ewc",         True),
]


def sigma_from_entropy(entropy):
    """Invert H -> mean sigma. Exact for a diagonal Gaussian with state-independent log_std."""
    return np.exp((entropy - ENTROPY_CONST) / ACT_DIM)


def load_runs(directory):
    """Yield (seed, steps, sigma, logged_log_std_or_None) for every finished run in `directory`."""
    path = os.path.join(RESULTS, directory)
    if not os.path.isdir(path):
        return
    for fname in sorted(os.listdir(path)):
        m = re.match(r"(.+)_seed_(\d+)_scalars\.pkl$", fname)
        if not m:
            continue
        with open(os.path.join(path, fname), "rb") as f:
            data = pickle.load(f)
        if "train/entropy" not in data:
            continue
        arr = data["train/entropy"]
        steps, entropy = arr[:, 0], arr[:, 1]
        logged = data.get("train/log_std_mean")
        yield int(m.group(2)), steps, sigma_from_entropy(entropy), logged


def sigma_at_boundaries(steps, sigma):
    """Sigma just before each task boundary, plus the final value."""
    out = []
    for i in range(1, N_TASKS):
        idx = np.searchsorted(steps, i * SWITCH) - 1
        out.append(sigma[max(idx, 0)])
    out.append(sigma[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the recovery against pt's directly logged log_std_mean")
    ap.add_argument("--plot", action="store_true", help="write sigma_collapse.png")
    args = ap.parse_args()

    print(f"sigma recovered from entropy; act_dim={ACT_DIM}, boundaries every {SWITCH:,} steps\n")
    header = f"{'arm':<18}{'n':>3}  {'start':>7}" + "".join(
        f"{f'b{i}':>8}" for i in range(1, N_TASKS)) + f"{'end':>8}   {'drop':>6}"
    print(header)
    print("-" * len(header))

    curves, worst_verify = {}, 0.0
    for label, directory, frozen in ARMS:
        runs = list(load_runs(directory))
        if not runs:
            print(f"{label:<18}  -  (no runs found in results/{directory})")
            continue

        starts, at_bounds, series = [], [], []
        for _seed, steps, sigma, logged in runs:
            starts.append(sigma[0])
            at_bounds.append(sigma_at_boundaries(steps, sigma))
            series.append((steps, sigma))
            if logged is not None and len(logged) == len(sigma):
                worst_verify = max(worst_verify,
                                   float(np.max(np.abs(np.log(sigma) - logged[:, 1]))))

        start = float(np.median(starts))
        med = np.median(np.asarray(at_bounds), axis=0)
        drop = med[-1] / start if start > 0 else float("nan")
        flag = ""
        if frozen and abs(drop - 1.0) > 0.01:
            flag = "  <-- FROZEN ARM MOVED; the freeze is not working"
        elif not frozen and drop < 0.25:
            flag = "  <-- collapsed"
        row = (f"{label:<18}{len(runs):>3}  {start:>7.3f}"
               + "".join(f"{v:>8.3f}" for v in med) + f"   {drop:>5.2f}x{flag}")
        print(row)
        curves[label] = series

    print("\n'b1'..'b4' are sigma just before each task boundary; 'drop' is end / start.")
    if args.verify:
        print(f"\nrecovery check vs pt's logged log_std_mean: max abs error = {worst_verify:.2e}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, prefix, title in ((axes[0], "physics ", "Physics benchmark (sigma learned)"),
                                  (axes[1], "flip ", "Reward flips (sigma learned)")):
            for label, series in curves.items():
                if not label.startswith(prefix):
                    continue
                steps = series[0][0]
                stack = np.stack([np.interp(steps, s, v) for s, v in series])
                ax.plot(steps, np.median(stack, axis=0), label=label[len(prefix):], lw=1.6)
            for i in range(1, N_TASKS):
                ax.axvline(i * SWITCH, color="0.7", lw=0.8, ls="--", zorder=0)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("env step")
            ax.legend(fontsize=8, frameon=False)
        axes[0].set_ylabel(r"$\sigma$ (median over seeds)")
        axes[0].set_yscale("log")
        fig.tight_layout()
        out = os.path.join(HERE, "plots", "figures_phase2", "sigma_collapse.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
