"""Plot the end-to-end simple point-drift benchmark.

Example:
    python -m src_continuous_control.plots.plot_simple_drift \
        --results-dir results/simple_drift_demo --agents vanilla pt_full --seeds 0
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"vanilla": "#e76f51", "pt_full": "#2a9d8f", "pt": "#457b9d", "ewc": "#6a4c93"}
LABELS = {"vanilla": "Vanilla PPO", "pt_full": "PT-full PPO", "pt": "Legacy PT", "ewc": "PPO + EWC"}


def _load_returns(results_dir, agent, seeds):
    curves = []
    for seed in seeds:
        path = os.path.join(results_dir, f"{agent}_ppo_seed_{seed}_returns.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            values = np.asarray(pickle.load(handle), dtype=np.float64)
        if values.ndim == 2 and values.shape[1] >= 2:
            curves.append((values[:, 0], values[:, 1]))
        elif values.ndim == 1:
            curves.append((np.arange(len(values), dtype=np.float64), values))
    return curves


def _load_scalar(results_dir, agent, seeds, name):
    curves = []
    for seed in seeds:
        path = os.path.join(results_dir, f"{agent}_ppo_seed_{seed}_scalars.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if name not in payload:
            continue
        values = np.asarray(payload[name], dtype=np.float64)
        if values.ndim == 2 and values.shape[1] >= 2:
            curves.append((values[:, 0], values[:, 1]))
    return curves


def _aggregate(curves, points=300):
    """Interpolate curves to a common x-grid and return mean/std or None."""
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
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)


def make_plot(results_dir, agents, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    return_ax, drift_ax, energy_ax, kl_ax = axes.flat

    loaded = {}
    for agent in agents:
        loaded[agent] = _load_returns(results_dir, agent, seeds)
        _plot_band(return_ax, _aggregate(loaded[agent]), LABELS.get(agent, agent),
                   COLORS.get(agent, "#264653"))
    return_ax.set_title("Learning curve")
    return_ax.set_ylabel("EMA return")
    return_ax.legend(frameon=False)
    return_ax.grid(alpha=0.25)

    drift = _load_scalar(results_dir, agents[0], seeds, "drift/multiplier") if agents else []
    _plot_band(drift_ax, _aggregate(drift), "drift multiplier", "#264653")
    drift_ax.set_title("Environment drift")
    drift_ax.set_ylabel("drag multiplier")
    drift_ax.grid(alpha=0.25)

    for agent in agents:
        if agent != "pt_full":
            continue
        perm = _load_scalar(results_dir, agent, seeds, "train/value_perm_l2")
        trans = _load_scalar(results_dir, agent, seeds, "train/value_trans_l2")
        perm_agg = _aggregate(perm)
        trans_agg = _aggregate(trans)
        if perm_agg is not None:
            energy = (perm_agg[0], trans_agg[1] / np.maximum(perm_agg[1], 1e-8),
                      np.zeros_like(perm_agg[1])) if trans_agg is not None else None
            _plot_band(energy_ax, energy, "||V_T|| / ||V_P||", COLORS[agent])
    energy_ax.set_title("PT energy balance")
    energy_ax.set_ylabel("transient / permanent")
    energy_ax.grid(alpha=0.25)

    for agent in agents:
        if agent == "pt_full":
            kl = _load_scalar(results_dir, agent, seeds, "train/kl_prior")
            _plot_band(kl_ax, _aggregate(kl), "KL to permanent prior", COLORS[agent])
            absorption = _load_scalar(results_dir, agent, seeds, "train/consol/absorbed_frac")
            _plot_band(kl_ax, _aggregate(absorption), "critic absorbed fraction", "#f4a261")
    kl_ax.set_title("Policy regularization and consolidation")
    kl_ax.set_ylabel("diagnostic value")
    kl_ax.grid(alpha=0.25)

    for ax in axes[1]:
        ax.set_xlabel("environment steps")
    fig.suptitle("Simple drifting point mass: PT-PPO end-to-end diagnostics", x=0.02,
                 ha="left", fontsize=14)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = os.path.join(out_dir, f"simple_drift_learning.{extension}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/simple_drift_demo")
    parser.add_argument("--out-dir", default="plots/figures")
    parser.add_argument("--agents", nargs="+", default=["vanilla", "pt_full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    paths = make_plot(args.results_dir, args.agents, args.seeds, args.out_dir)
    for path in paths:
        print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()
