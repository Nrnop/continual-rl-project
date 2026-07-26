"""Plotting script for Single-Task Baseline runs (live logs + completed pkls).

Usage:
    python -m src_continuous_control.plots.plot_singletask_live --out-dir src_continuous_control/plots/figures_singletask
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np


def _load_or_parse_seed(agent_name, seed, results_dir="src_continuous_control/results_singletask", logs_dir="src_continuous_control/runs/logs"):
    # 1. Try loading completed return pkl if present
    pkl_file = os.path.join(results_dir, f"{agent_name}_ppo_seed_{seed}_returns.pkl")
    if os.path.exists(pkl_file):
        try:
            with open(pkl_file, "rb") as f:
                arr = np.asarray(pickle.load(f), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 0], arr[:, 1]
            else:
                x = np.arange(len(arr), dtype=np.float32) * 2048
                return x, arr
        except Exception as e:
            print(f"[plot_singletask] Error reading pkl {pkl_file}: {e}")

    # 2. Parse live training log file
    log_file = os.path.join(logs_dir, f"singletask_{agent_name}_seed_{seed}.log")
    if not os.path.exists(log_file):
        print(f"[plot_singletask] Neither pkl nor log file found for {agent_name} seed {seed}")
        return None, None

    pattern = re.compile(r"\[train\] step (\d+)/\d+\s+return=([-\d\.]+)")
    steps = []
    returns = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    steps.append(float(match.group(1)))
                    returns.append(float(match.group(2)))
    except Exception as e:
        print(f"[plot_singletask] Error reading log {log_file}: {e}")

    if steps and returns:
        return np.array(steps, dtype=np.float32), np.array(returns, dtype=np.float32)
    return None, None


def _load_or_parse_critic_loss(agent_name, seed, logs_dir="src_continuous_control/runs/logs"):
    log_file = os.path.join(logs_dir, f"singletask_{agent_name}_seed_{seed}.log")
    if not os.path.exists(log_file):
        return None, None
    pattern = re.compile(r"\[train\] step (\d+)/\d+.*critic_loss=([-+]?\d*\.\d+|\d+)")
    steps, losses = [], []
    try:
        with open(log_file, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    steps.append(float(match.group(1)))
                    losses.append(float(match.group(2)))
    except Exception as e:
        print(f"[plot_singletask] Error reading log {log_file}: {e}")
    if steps and losses:
        return np.array(steps, dtype=np.float32), np.array(losses, dtype=np.float32)
    return None, None


def _smooth(arr, window=15):
    if len(arr) < window or window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-dir", type=str, default="src_continuous_control/results_singletask")
    parser.add_argument("--logs-dir", type=str, default="src_continuous_control/runs/logs")
    parser.add_argument("--out-dir", type=str, default="src_continuous_control/plots/figures_singletask")
    parser.add_argument("--switch", type=int, default=614400)
    parser.add_argument("--smooth", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    agent_data = {}
    for label, ag_key, color in [
        ("PT-PPO (Single-Task)", "pt", "#2196F3"),
        ("Vanilla PPO (Single-Task)", "vanilla", "#FF7043"),
        ("Online EWC (Single-Task)", "ewc", "#66BB6A"),
    ]:
        curves = []
        x_curves = []
        for s in args.seeds:
            x, y = _load_or_parse_seed(ag_key, s, args.results_dir, args.logs_dir)
            if x is not None and y is not None and len(y) > 0:
                curves.append(y)
                x_curves.append(x)
        if curves:
            agent_data[label] = (curves, x_curves, color)

    if not agent_data:
        print("[plot_singletask] No single-task data found.")
        sys.exit(1)

    # ---- Figure 1: Overlaid Single-Task Return Curves ----
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    max_step = 0
    for label, (curves, x_curves, color) in agent_data.items():
        # Interpolate curves onto a common step grid up to the min or common max step
        min_max_step = min(x[-1] for x in x_curves)
        common_x = x_curves[0][x_curves[0] <= min_max_step]
        
        interp_y = []
        for x, y in zip(x_curves, curves):
            iy = np.interp(common_x, x, y)
            interp_y.append(iy)
        mat = np.stack(interp_y)
        mean = mat.mean(axis=0)
        if mat.shape[0] > 1:
            se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
            ci = 1.96 * se
        else:
            ci = np.zeros_like(mean)
        
        sm_mean = _smooth(mean, args.smooth)
        sm_low = _smooth(mean - ci, args.smooth)
        sm_high = _smooth(mean + ci, args.smooth)
        
        ls = "--" if "EWC" in label else "-"
        lw = 2.2 if "EWC" in label else 1.8
        ax1.plot(common_x, sm_mean, label=f"{label} (n={len(curves)})", color=color, linewidth=lw, linestyle=ls)
        ax1.fill_between(common_x, sm_low, sm_high, color=color, alpha=0.2)
        if len(common_x) > 0:
            max_step = max(max_step, float(common_x[-1]))

    ax1.set_xlabel("Environment Time Steps")
    ax1.set_ylabel("Average Return (EMA)")
    ax1.set_title("Continual HalfCheetah: Single-Task Baseline Progression across 5 Seeds")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, f"singletask_return_curves.{ext}")
        fig1.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[plot_singletask] Saved {path}")
    plt.close(fig1)

    # ---- Figure 2: Per-Seed Trajectories (3 Subplots) ----
    fig2, axes2 = plt.subplots(1, len(agent_data), figsize=(5 * len(agent_data), 4.5), sharey=True)
    if len(agent_data) == 1:
        axes2 = [axes2]
    
    for ax, (label, (curves, x_curves, color)) in zip(axes2, agent_data.items()):
        for idx, (x, y) in enumerate(zip(x_curves, curves)):
            ax.plot(x, _smooth(y, args.smooth), alpha=0.4, linewidth=1.0, label=f"Seed {args.seeds[idx]}")
        
        min_max_step = min(x[-1] for x in x_curves)
        common_x = x_curves[0][x_curves[0] <= min_max_step]
        mat = np.stack([np.interp(common_x, x, y) for x, y in zip(x_curves, curves)])
        mean = _smooth(mat.mean(axis=0), args.smooth)
        ax.plot(common_x, mean, color=color, linewidth=2.2, label="Mean")
            
        ax.set_xlabel("Time Steps")
        ax.set_title(label.replace(" (Single-Task)", ""))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    
    axes2[0].set_ylabel("Average Return")
    fig2.suptitle("Single-Task Baseline: Per-Seed Trajectories", fontsize=13)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, f"singletask_per_seed_curves.{ext}")
        fig2.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[plot_singletask] Saved {path}")
    plt.close(fig2)

    # ---- Figure 3: Current Average Return Bar Chart (Latest 100k steps) ----
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    bar_labels = []
    bar_means = []
    bar_stds = []
    bar_colors = []
    
    for label, (curves, x_curves, color) in agent_data.items():
        seed_latest_perf = []
        for x, y in zip(x_curves, curves):
            if len(x) == 0:
                continue
            max_x = x[-1]
            mask = x >= max(0, max_x - 100000)
            if np.any(mask):
                seed_latest_perf.append(y[mask].mean())
            else:
                seed_latest_perf.append(y[-1])
        if seed_latest_perf:
            bar_labels.append(label.replace(" (Single-Task)", ""))
            bar_means.append(np.mean(seed_latest_perf))
            bar_stds.append(np.std(seed_latest_perf))
            bar_colors.append(color)

    if bar_labels:
        x_pos = np.arange(len(bar_labels))
        ax3.bar(x_pos, bar_means, yerr=bar_stds, color=bar_colors, alpha=0.8, capsize=5, width=0.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(bar_labels)
        ax3.set_ylabel("Mean Return (Latest ~100k Steps)")
        ax3.set_title("Single-Task Performance across 5 Seeds")
        ax3.grid(True, axis="y", alpha=0.3)
        fig3.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(args.out_dir, f"singletask_current_bar.{ext}")
            fig3.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[plot_singletask] Saved {path}")
        plt.close(fig3)

    # ---- Figure 4: Physical Velocity Curve (from completed pkl files) ----
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    has_vel = False
    max_vel_step = 0
    for label, ag_key, color in [
        ("PT-PPO (Single-Task)", "pt", "#2196F3"),
        ("Vanilla PPO (Single-Task)", "vanilla", "#FF7043"),
        ("Online EWC (Single-Task)", "ewc", "#66BB6A"),
    ]:
        vel_curves = []
        for s in args.seeds:
            fname = os.path.join(args.results_dir, f"{ag_key}_ppo_seed_{s}_velocities.pkl")
            if os.path.exists(fname):
                try:
                    with open(fname, "rb") as f:
                        arr = np.asarray(pickle.load(f), dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        vel_curves.append((arr[:, 0], arr[:, 1]))
                    else:
                        vel_curves.append((np.arange(len(arr), dtype=np.float32) * 2048, arr))
                except Exception as e:
                    print(f"[plot_singletask] Error reading velocity pkl {fname}: {e}")
        if not vel_curves:
            continue
        min_len = min(len(c[1]) for c in vel_curves)
        x_vals = vel_curves[0][0][:min_len]
        y_mat = np.stack([c[1][:min_len] for c in vel_curves])
        mean_y = y_mat.mean(axis=0)
        if y_mat.shape[0] > 1:
            ci_y = 1.96 * y_mat.std(axis=0, ddof=1) / np.sqrt(y_mat.shape[0])
        else:
            ci_y = np.zeros_like(mean_y)
        sm_mean = _smooth(mean_y, min(args.smooth, len(mean_y)))
        sm_low = _smooth(mean_y - ci_y, min(args.smooth, len(mean_y)))
        sm_high = _smooth(mean_y + ci_y, min(args.smooth, len(mean_y)))
        ls = "--" if "EWC" in label else "-"
        lw = 2.2 if "EWC" in label else 1.8
        ax4.plot(x_vals, sm_mean, label=label, color=color, linewidth=lw, linestyle=ls)
        ax4.fill_between(x_vals, sm_low, sm_high, color=color, alpha=0.2)
        if len(x_vals) > 0:
            max_vel_step = max(max_vel_step, float(x_vals[-1]))
        has_vel = True

    if has_vel:
        ax4.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
        ax4.set_xlabel("Environment Time Steps")
        ax4.set_ylabel("Mean X-Velocity (m/s)")
        ax4.set_title("Continual HalfCheetah: Single-Task Physical Forward Velocity")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(args.out_dir, f"singletask_velocity_curves.{ext}")
            fig4.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[plot_singletask] Saved {path}")
        plt.close(fig4)
    else:
        plt.close(fig4)
        print("[plot_singletask] Notice: No single-task velocity curves found yet. "
              "Physical velocities are tracked in RAM during execution and saved to "
              f"{args.results_dir}/<agent>_seed_<s>_velocities.pkl once each run completes.")

    # ---- Figure 5: Zero-Momentum Offline Evaluation Curves (`_eval_returns.pkl`) ----
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    has_offline = False
    max_eval_step = 0
    for label, ag_key, color in [
        ("PT-PPO (Single-Task)", "pt", "#2196F3"),
        ("Vanilla PPO (Single-Task)", "vanilla", "#FF7043"),
        ("Online EWC (Single-Task)", "ewc", "#66BB6A"),
    ]:
        eval_curves = []
        for s in args.seeds:
            fname = os.path.join(args.results_dir, f"{ag_key}_ppo_seed_{s}_eval_returns.pkl")
            if os.path.exists(fname):
                try:
                    with open(fname, "rb") as f:
                        arr = np.asarray(pickle.load(f), dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        eval_curves.append((arr[:, 0], arr[:, 1]))
                    else:
                        eval_curves.append((np.arange(len(arr), dtype=np.float32) * 2048, arr))
                except Exception as e:
                    print(f"[plot_singletask] Error reading eval pkl {fname}: {e}")
        if not eval_curves:
            continue
        min_len = min(len(c[1]) for c in eval_curves)
        x_vals = eval_curves[0][0][:min_len]
        y_mat = np.stack([c[1][:min_len] for c in eval_curves])
        mean_y = y_mat.mean(axis=0)
        if y_mat.shape[0] > 1:
            ci_y = 1.96 * y_mat.std(axis=0, ddof=1) / np.sqrt(y_mat.shape[0])
        else:
            ci_y = np.zeros_like(mean_y)
        sm_mean = _smooth(mean_y, min(args.smooth, len(mean_y)))
        sm_low = _smooth(mean_y - ci_y, min(args.smooth, len(mean_y)))
        sm_high = _smooth(mean_y + ci_y, min(args.smooth, len(mean_y)))
        ls = "--" if "EWC" in label else "-"
        lw = 2.2 if "EWC" in label else 1.8
        ax5.plot(x_vals, sm_mean, label=label, color=color, linewidth=lw, marker="o", markersize=3, linestyle=ls)
        ax5.fill_between(x_vals, sm_low, sm_high, color=color, alpha=0.2)
        if len(x_vals) > 0:
            max_eval_step = max(max_eval_step, float(x_vals[-1]))
        has_offline = True

    if has_offline:
        ax5.set_xlabel("Environment Time Steps")
        ax5.set_ylabel("Zero-Momentum Evaluation Return")
        ax5.set_title("Continual HalfCheetah: Single-Task Offline Evaluation Progression")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        fig5.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(args.out_dir, f"singletask_offline_curves.{ext}")
            fig5.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[plot_singletask] Saved {path}")
        plt.close(fig5)
    else:
        plt.close(fig5)
        print("[plot_singletask] Notice: No single-task offline evaluation curves found yet. "
              f"They are saved to {args.results_dir}/<agent>_seed_<s>_eval_returns.pkl upon completion.")

    # ---- Figure 6: Value Estimation TD Error (`critic_loss`) Curves ----
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    has_td = False
    max_td_step = 0
    for label, ag_key, color in [
        ("PT-PPO (Single-Task)", "pt", "#2196F3"),
        ("Vanilla PPO (Single-Task)", "vanilla", "#FF7043"),
        ("Online EWC (Single-Task)", "ewc", "#66BB6A"),
    ]:
        td_curves = []
        x_td_curves = []
        for s in args.seeds:
            x, y = _load_or_parse_critic_loss(ag_key, s, args.logs_dir)
            if x is not None and y is not None and len(y) > 0:
                td_curves.append(y)
                x_td_curves.append(x)
        if not td_curves:
            continue
        min_max_step = min(x[-1] for x in x_td_curves)
        common_x = x_td_curves[0][x_td_curves[0] <= min_max_step]
        interp_y = [np.interp(common_x, x, y) for x, y in zip(x_td_curves, td_curves)]
        y_mat = np.stack(interp_y)
        mean_y = y_mat.mean(axis=0)
        if y_mat.shape[0] > 1:
            ci_y = 1.96 * y_mat.std(axis=0, ddof=1) / np.sqrt(y_mat.shape[0])
        else:
            ci_y = np.zeros_like(mean_y)
        sm_mean = _smooth(mean_y, min(args.smooth, len(mean_y)))
        sm_low = _smooth(mean_y - ci_y, min(args.smooth, len(mean_y)))
        sm_high = _smooth(mean_y + ci_y, min(args.smooth, len(mean_y)))
        ls = "--" if "EWC" in label else "-"
        lw = 2.2 if "EWC" in label else 1.8
        ax6.plot(common_x, sm_mean, label=label, color=color, linewidth=lw, linestyle=ls)
        ax6.fill_between(common_x, sm_low, sm_high, color=color, alpha=0.2)
        if len(common_x) > 0:
            max_td_step = max(max_td_step, float(common_x[-1]))
        has_td = True

    if has_td:
        ax6.set_xlabel("Environment Time Steps")
        ax6.set_ylabel("Mean Squared TD Error (Critic Loss)")
        ax6.set_title("Continual HalfCheetah: Single-Task Value Estimation TD Error Progression")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        fig6.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(args.out_dir, f"singletask_td_error_curves.{ext}")
            fig6.savefig(path, dpi=200, bbox_inches="tight")
            print(f"[plot_singletask] Saved {path}")
        plt.close(fig6)
    else:
        plt.close(fig6)
        print("[plot_singletask] Notice: No TD error logs found.")


if __name__ == "__main__":
    main()
