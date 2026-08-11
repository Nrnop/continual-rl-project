"""Compare PT-PPO vs Vanilla PPO vs Online EWC return curves across seeds.

Usage:
    python -m src_continuous_control.plots.plot_compare --seeds 0 1 2 3 4

Produces:
    1. Overlaid mean ± CI return curves with task-boundary markers.
    2. Boundary-drop bar chart (PT vs Vanilla vs EWC).
    3. Recovery-time bar chart.
    4. Standardized offline adaptation curve (line plot).
    5. Asymptotic vs. online cumulative return chart.
    6. Physical velocity curve.

Outputs saved under src_continuous_control/plots/figures_phase2/.

NOTE: the old default was plots/figures/, which is now archive/phase1/plots/figures/ --
the reward-switch figure set. Phase 2 output goes to its own directory so the two sets
cannot be confused; mixing them is exactly how stale figures survived once before
(see archive/phase1/plots/README.md).
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np


def _load_returns(results_dir, agent_name, seeds, n_steps=2048):
    """Load per-seed return curves.  Returns (y_curves, x_curves).

    Handles both old format (1-D float array, x inferred from index × n_steps)
    and new format (2-D array with [step, value] columns).
    """
    y_curves = []
    x_curves = []
    for s in seeds:
        fname = os.path.join(results_dir, f"{agent_name}_ppo_seed_{s}_returns.pkl")
        if not os.path.exists(fname):
            print(f"[plot] WARNING: {fname} not found — skipping seed {s}")
            continue
        with open(fname, "rb") as f:
            arr = np.asarray(pickle.load(f), dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            x_curves.append(arr[:, 0])
            y_curves.append(arr[:, 1])
        else:
            x_curves.append(np.arange(len(arr), dtype=np.float32) * n_steps)
            y_curves.append(arr)
    return y_curves, x_curves


def _load_eval_curves(results_dir, agent_name, seeds, n_steps=2048):
    """Load zero-momentum offline evaluation curves.  Returns list of (x, y)."""
    curves = []
    for s in seeds:
        fname = os.path.join(results_dir, f"{agent_name}_ppo_seed_{s}_eval_returns.pkl")
        if not os.path.exists(fname):
            continue
        with open(fname, "rb") as f:
            arr = np.asarray(pickle.load(f), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                curves.append((arr[:, 0], arr[:, 1]))
            elif arr.ndim == 1:
                curves.append((np.arange(len(arr), dtype=np.float32) * n_steps, arr))
    return curves


def _load_velocity_curves(results_dir, agent_name, seeds, n_steps=2048):
    """Load physical velocity curves.  Returns list of (x, y)."""
    curves = []
    for s in seeds:
        fname = os.path.join(results_dir, f"{agent_name}_ppo_seed_{s}_velocities.pkl")
        if not os.path.exists(fname):
            continue
        with open(fname, "rb") as f:
            arr = np.asarray(pickle.load(f), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                curves.append((arr[:, 0], arr[:, 1]))
            elif arr.ndim == 1:
                curves.append((np.arange(len(arr), dtype=np.float32) * n_steps, arr))
    return curves


def _load_or_parse_critic_loss(agent_name, seed, logs_dir="src_continuous_control/runs/logs"):
    candidate_dirs = [logs_dir, "src_continuous_control/runs/logs"]
    log_file = None
    for d in candidate_dirs:
        cand = os.path.join(d, f"{agent_name}_seed_{seed}.log")
        if os.path.exists(cand):
            log_file = cand
            break
    if not log_file:
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
        print(f"[plot] Error reading log {log_file}: {e}")
    if steps and losses:
        return np.array(steps, dtype=np.float32), np.array(losses, dtype=np.float32)
    return None, None


def _mean_ci(curves):
    """Compute mean and 95 % CI from a list of equal-length arrays."""
    if not curves:
        return None, None, None
    min_len = min(len(c) for c in curves)
    mat = np.stack([c[:min_len] for c in curves])
    mean = mat.mean(axis=0)
    if mat.shape[0] > 1:
        se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ci = 1.96 * se
    else:
        ci = np.zeros_like(mean)
    return mean, ci, min_len


def _smooth(arr, window=50):
    """Simple moving average for visual clarity."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-dir", type=str, default="src_continuous_control/results")
    parser.add_argument("--logs-dir", type=str, default="src_continuous_control/runs/logs")
    parser.add_argument("--switch", type=int, default=614400,
                        help="Switch interval in env steps (for drawing boundary lines)")
    parser.add_argument("--total-steps", type=int, default=3072000)
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout length (used for old-format x-axis inference)")
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="src_continuous_control/plots/figures_phase2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Number of data points (rollout intervals) per task phase
    updates_per_switch = max(1, args.switch // args.n_steps)

    # Load return curves
    pt_curves, pt_x = _load_returns(args.results_dir, "pt", args.seeds, args.n_steps)
    van_curves, van_x = _load_returns(args.results_dir, "vanilla", args.seeds, args.n_steps)
    ewc_curves, ewc_x = _load_returns(args.results_dir, "ewc", args.seeds, args.n_steps)

    if not pt_curves and not van_curves and not ewc_curves:
        print("[plot] No results found. Run training first.")
        sys.exit(1)

    # ---- Figure 1: overlaid return curves ----
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, curves, x_arrs, color in [
        ("PT-PPO", pt_curves, pt_x, "#2196F3"),
        ("Vanilla PPO", van_curves, van_x, "#FF7043"),
        ("Online EWC", ewc_curves, ewc_x, "#66BB6A"),
    ]:
        mean, ci, n = _mean_ci(curves)
        if mean is None:
            continue
        x = x_arrs[0][:n] if x_arrs else np.arange(n, dtype=np.float32) * args.n_steps
        sm = _smooth(mean, args.smooth)
        ax.plot(x, sm, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, _smooth(mean - ci, args.smooth),
                        _smooth(mean + ci, args.smooth),
                        color=color, alpha=0.2)

    # Task-boundary lines
    all_x_arrs = [x for xa in [pt_x, van_x, ewc_x] for x in xa]
    max_step = max((float(x[-1]) for x in all_x_arrs), default=0) if all_x_arrs else 0
    for bx in range(args.switch, int(max_step) + 1, args.switch):
        ax.axvline(bx, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Environment Time Steps")
    ax.set_ylabel("Average Return (EMA)")
    ax.set_title("Continual HalfCheetah: PT-PPO vs Vanilla PPO vs Online EWC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path1 = os.path.join(args.out_dir, f"return_curves.{ext}")
        fig.savefig(path1, dpi=200, bbox_inches="tight")
        print(f"[plot] Saved {path1}")
    plt.close(fig)

    # ---- Figure 2: Relative boundary-drop bar chart ----
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    drops = {}
    window = max(1, updates_per_switch // 10)
    for label, curves, color in [
        ("PT-PPO", pt_curves, "#2196F3"),
        ("Vanilla PPO", van_curves, "#FF7043"),
        ("Online EWC", ewc_curves, "#66BB6A"),
    ]:
        mean, ci, n = _mean_ci(curves)
        if mean is None:
            continue
        boundary_drops = []
        for bx in range(updates_per_switch, n, updates_per_switch):
            pre = mean[max(0, bx - window):bx].mean()
            post = mean[bx:min(n, bx + window)].mean()
            # Relative return drop: (pre - post) / abs(pre)
            rel_drop = (pre - post) / (np.abs(pre) + 1e-8)
            boundary_drops.append(rel_drop)
        if boundary_drops:
            drops[label] = (np.mean(boundary_drops), np.std(boundary_drops), color)

    if drops:
        labels = list(drops.keys())
        means = [drops[l][0] * 100 for l in labels]  # Convert to percentage
        stds = [drops[l][1] * 100 for l in labels]
        colors = [drops[l][2] for l in labels]
        bars = ax2.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax2.set_ylabel("Mean Relative Return Drop (%)")
        ax2.set_title("Relative Task-Switch Return Drop")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            path2 = os.path.join(args.out_dir, f"boundary_drop.{ext}")
            fig2.savefig(path2, dpi=200, bbox_inches="tight")
            print(f"[plot] Saved {path2}")
        plt.close(fig2)

    # ---- Figure 3: Recovery Time bar chart ----
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    recovery_times = {}
    for label, curves, color in [
        ("PT-PPO", pt_curves, "#2196F3"),
        ("Vanilla PPO", van_curves, "#FF7043"),
        ("Online EWC", ewc_curves, "#66BB6A"),
    ]:
        mean, ci, n = _mean_ci(curves)
        if mean is None:
            continue

        # Calculate Task 1 peak (95th percentile of smoothed Task 1 performance)
        t1_smooth = _smooth(mean[:updates_per_switch], args.smooth)
        t1_peak = np.percentile(t1_smooth, 95)
        target = 0.90 * t1_peak

        recoveries = []
        # Look at switch 2 (Task 2 -> Task 1)
        bx = 2 * updates_per_switch
        if bx < n:
            post_switch_perf = _smooth(mean[bx:min(n, bx + updates_per_switch)], args.smooth)
            # Find the first index where performance >= target
            reached = np.where(post_switch_perf >= target)[0]
            if len(reached) > 0:
                rec_time = reached[0]
            else:
                rec_time = len(post_switch_perf)  # Failed to recover within the phase
            recoveries.append(rec_time)

        if recoveries:
            recovery_times[label] = (np.mean(recoveries), 0.0, color)

    if recovery_times:
        labels = list(recovery_times.keys())
        means = [recovery_times[l][0] for l in labels]
        colors = [recovery_times[l][2] for l in labels]
        bars = ax3.bar(labels, means, color=colors, alpha=0.8)
        ax3.set_ylabel("Rollout Intervals to 90 % Task 1 Peak")
        ax3.set_title("Recovery Time (Switch 2)")
        ax3.grid(True, axis="y", alpha=0.3)
        fig3.tight_layout()
        for ext in ("pdf", "png"):
            path3 = os.path.join(args.out_dir, f"recovery_time.{ext}")
            fig3.savefig(path3, dpi=200, bbox_inches="tight")
            print(f"[plot] Saved {path3}")
        plt.close(fig3)

    # ---- Figure 4: Standardized Offline Adaptation Curve (Line Plot) ----
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    has_offline = False
    max_eval_step = 0
    for label, ag_prefix, color in [
        ("PT-PPO", "pt", "#2196F3"),
        ("Vanilla PPO", "vanilla", "#FF7043"),
        ("Online EWC", "ewc", "#66BB6A"),
    ]:
        eval_curves = _load_eval_curves(args.results_dir, ag_prefix, args.seeds, args.n_steps)
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
        ax4.plot(x_vals, sm_mean, label=label, color=color, linewidth=1.5,
                 marker="o", markersize=3)
        ax4.fill_between(x_vals, sm_low, sm_high, color=color, alpha=0.2)
        if len(x_vals) > 0:
            max_eval_step = max(max_eval_step, float(x_vals[-1]))
        has_offline = True

    if has_offline:
        for bx in range(args.switch, int(max_eval_step) + 1, args.switch):
            ax4.axvline(bx, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        ax4.set_xlabel("Environment Time Steps")
        ax4.set_ylabel("Zero-Momentum Evaluation Return")
        ax4.set_title("Continual HalfCheetah: Standardized Zero-Momentum Offline Adaptation")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        for ext in ("pdf", "png"):
            path4 = os.path.join(args.out_dir, f"offline_curves.{ext}")
            fig4.savefig(path4, dpi=200, bbox_inches="tight")
            print(f"[plot] Saved {path4}")
        plt.close(fig4)
    else:
        plt.close(fig4)
        print("[plot] Notice: No offline evaluation curves found. "
              "Skipping Figure 4 (offline_curves).")

    # ---- Figure 5: Asymptotic vs. Online Cumulative Return Chart ----
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    agents_data = []
    agent_names_map = {"PT-PPO": "pt", "Vanilla PPO": "vanilla", "Online EWC": "ewc"}
    for label, curves, color in [
        ("PT-PPO", pt_curves, "#2196F3"),
        ("Vanilla PPO", van_curves, "#FF7043"),
        ("Online EWC", ewc_curves, "#66BB6A"),
    ]:
        if not curves:
            continue
        ag_prefix = agent_names_map[label]
        cum_list = []
        asymp_list = []
        for idx, c in enumerate(curves):
            seed = args.seeds[idx] if idx < len(args.seeds) else idx
            p2_start = updates_per_switch
            p2_end = min(len(c), 2 * updates_per_switch)
            if p2_end > p2_start:
                cum_list.append(np.mean(c[p2_start:p2_end]))
                ep_file = os.path.join(
                    args.results_dir,
                    f"{ag_prefix}_ppo_seed_{seed}_ep_returns.pkl")
                if os.path.exists(ep_file):
                    with open(ep_file, "rb") as ef:
                        ep_arr = np.asarray(pickle.load(ef), dtype=np.float32)
                    asymp_list.append(np.mean(ep_arr[-75:]))
                else:
                    asymp_list.append(
                        np.mean(c[max(p2_start, p2_end - 75):p2_end]))
        if cum_list and asymp_list:
            agents_data.append((
                label, np.mean(cum_list), np.std(cum_list),
                np.mean(asymp_list), np.std(asymp_list), color))

    if agents_data:
        x_pos = np.arange(len(agents_data))
        width = 0.35
        labels = [d[0] for d in agents_data]
        cum_means = [d[1] for d in agents_data]
        cum_stds = [d[2] for d in agents_data]
        asymp_means = [d[3] for d in agents_data]
        asymp_stds = [d[4] for d in agents_data]
        colors = [d[5] for d in agents_data]

        ax5.bar(x_pos - width / 2, cum_means, width, yerr=cum_stds,
                label="Overall Phase Cumulative Return", color=colors,
                alpha=0.5, edgecolor=colors, hatch="//", capsize=5)
        ax5.bar(x_pos + width / 2, asymp_means, width, yerr=asymp_stds,
                label="Isolated Asymptotic Return (Last 75)", color=colors,
                alpha=0.9, edgecolor="black", capsize=5)

        ax5.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(labels)
        ax5.set_ylabel("Average Return (Phase 2)")
        ax5.set_title("Phase 2: Overall Cumulative vs. Isolated Asymptotic Return")
        ax5.legend()
        ax5.grid(True, axis="y", alpha=0.3)
        fig5.tight_layout()
        for ext in ("pdf", "png"):
            path5 = os.path.join(args.out_dir, f"asymptotic_bar.{ext}")
            fig5.savefig(path5, dpi=200, bbox_inches="tight")
            print(f"[plot] Saved {path5}")
        plt.close(fig5)

    # ---- Figure 6: Physical Velocity Curve ----
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    has_vel = False
    max_vel_step = 0
    for label, ag_prefix, color in [
        ("PT-PPO", "pt", "#2196F3"),
        ("Vanilla PPO", "vanilla", "#FF7043"),
        ("Online EWC", "ewc", "#66BB6A"),
    ]:
        vel_curves = _load_velocity_curves(
            args.results_dir, ag_prefix, args.seeds, args.n_steps)
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
        ax6.plot(x_vals, sm_mean, label=label, color=color, linewidth=1.5)
        ax6.fill_between(x_vals, sm_low, sm_high, color=color, alpha=0.2)
        if len(x_vals) > 0:
            max_vel_step = max(max_vel_step, float(x_vals[-1]))
        has_vel = True

    if has_vel:
        for bx in range(args.switch, int(max_vel_step) + 1, args.switch):
            ax6.axvline(bx, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        ax6.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
        ax6.set_xlabel("Environment Time Steps")
        ax6.set_ylabel("Mean X-Velocity")
        ax6.set_title("Continual HalfCheetah: Physical Velocity")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        fig6.tight_layout()
        for ext in ("pdf", "png"):
            path6 = os.path.join(args.out_dir, f"velocity_curves.{ext}")
            fig6.savefig(path6, dpi=200, bbox_inches="tight")
            print(f"[plot] Saved {path6}")
        plt.close(fig6)
    else:
        plt.close(fig6)
        print("[plot] Notice: No velocity curves found. "
              "Skipping Figure 6 (velocity_curves).")

    # ---- Figure 7: Value Estimation TD Error (`critic_loss`) Curves ----
    fig7, ax7 = plt.subplots(figsize=(12, 5))
    has_td = False
    max_td_step = 0
    for label, ag_prefix, color in [
        ("PT-PPO", "pt", "#2196F3"),
        ("Vanilla PPO", "vanilla", "#FF7043"),
        ("Online EWC", "ewc", "#66BB6A"),
    ]:
        td_curves = []
        x_td_curves = []
        for s in args.seeds:
            x, y = _load_or_parse_critic_loss(ag_prefix, s, args.logs_dir)
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
        ax7.plot(common_x, sm_mean, label=label, color=color, linewidth=1.5)
        ax7.fill_between(common_x, sm_low, sm_high, color=color, alpha=0.2)
        if len(common_x) > 0:
            max_td_step = max(max_td_step, float(common_x[-1]))
        has_td = True

    if has_td:
        for bx in range(args.switch, int(max_td_step) + 1, args.switch):
            ax7.axvline(bx, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        ax7.set_xlabel("Environment Time Steps")
        ax7.set_ylabel("Mean Squared TD Error (Critic Loss)")
        ax7.set_title("Continual HalfCheetah: Value Estimation TD Error Progression across Tasks")
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        fig7.tight_layout()
        for ext in ("pdf", "png"):
            path7 = os.path.join(args.out_dir, f"td_error_curves.{ext}")
            fig7.savefig(path7, dpi=200, bbox_inches="tight")
            print(f"[plot] Saved {path7}")
        plt.close(fig7)
    else:
        plt.close(fig7)
        print("[plot] Notice: No TD error curves found.")


if __name__ == "__main__":
    main()
