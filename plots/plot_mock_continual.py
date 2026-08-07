"""Plot the mock continual-tasks demo (DirectionalPointMass, 3 discrete task switches).

Example:
    python -m src_continuous_control.plots.plot_mock_continual \
        --results-dir src_continuous_control/results/mock_continual_demo \
        --seeds 0 1 2 --switch 15000 --total-steps 60000
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"vanilla": "#e76f51", "pt_full": "#2a9d8f"}
LABELS = {"vanilla": "Vanilla PPO", "pt_full": "PT-full PPO"}


def _load_returns(results_dir, agent, seeds, suffix="returns"):
    curves = []
    for seed in seeds:
        path = os.path.join(results_dir, f"{agent}_ppo_seed_{seed}_{suffix}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            values = np.asarray(pickle.load(handle), dtype=np.float64)
        if values.ndim == 2 and values.shape[1] >= 2:
            curves.append((values[:, 0], values[:, 1]))
        elif values.ndim == 1:
            curves.append((np.arange(len(values), dtype=np.float64), values))
    return curves


def _aggregate(curves, points=300):
    """Interpolate curves to a common x-grid and return mean/std, or None."""
    if not curves:
        return None
    start = max(float(x[0]) for x, _ in curves if len(x))
    end = min(float(x[-1]) for x, _ in curves if len(x))
    if end <= start:
        return None
    grid = np.linspace(start, end, min(points, max(len(y) for _, y in curves)))
    values = np.stack([np.interp(grid, x, y) for x, y in curves])
    return grid, values.mean(axis=0), values.std(axis=0)


def _plot_band(ax, aggregate, label, color):
    if aggregate is None:
        return
    x, mean, std = aggregate
    ax.plot(x, mean, color=color, label=label, linewidth=2.0)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


def main():
    p = argparse.ArgumentParser(description="Plot the mock continual-tasks demo")
    p.add_argument("--results-dir", type=str,
                    default="src_continuous_control/results/mock_continual_demo")
    p.add_argument("--agents", type=str, nargs="+", default=["vanilla", "pt_full"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--switch", type=int, default=15000, help="env steps between task switches")
    p.add_argument("--total-steps", type=int, default=60000)
    p.add_argument("--out-dir", type=str, default="src_continuous_control/plots/figures")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig, (train_ax, eval_ax) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    n_switches = args.total_steps // args.switch
    switch_steps = [args.switch * i for i in range(1, n_switches)]

    for agent in args.agents:
        curves = _load_returns(args.results_dir, agent, args.seeds)
        _plot_band(train_ax, _aggregate(curves), LABELS.get(agent, agent),
                   COLORS.get(agent, "#264653"))
        eval_curves = _load_returns(args.results_dir, agent, args.seeds, suffix="eval_returns")
        _plot_band(eval_ax, _aggregate(eval_curves), LABELS.get(agent, agent),
                   COLORS.get(agent, "#264653"))

    for ax, title in ((train_ax, "Training return (EMA)"), (eval_ax, "Zero-momentum eval return")):
        for s in switch_steps:
            ax.axvline(s, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.25)
    train_ax.set_ylabel("return")
    train_ax.legend(frameon=False)
    fig.suptitle("Mock continual demo: DirectionalPointMass, 3 task switches "
                 f"(seeds={args.seeds})")
    fig.tight_layout()

    out_path = os.path.join(args.out_dir, "mock_continual_demo.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
