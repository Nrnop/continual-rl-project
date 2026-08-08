"""Analyze the completed real HalfCheetah PT-PPO and vanilla PPO runs."""
import csv
import json
import os
import pickle

import numpy as np


SWITCH = 614400
TOTAL_STEPS = 3072000
PHASES = 5
FINAL_WINDOW_UPDATES = 100

RUNS = [
    {"method": "PT-A", "label": "PT-PPO lr_perm=1e-4", "directory": "src_continuous_control/results/real_directional_demo", "prefix": "pt_full", "seed": 0},
    {"method": "PT-A", "label": "PT-PPO lr_perm=1e-4", "directory": "src_continuous_control/results/real_confirmation_lr1e4_k16", "prefix": "pt_full", "seed": 1},
    {"method": "PT-A", "label": "PT-PPO lr_perm=1e-4", "directory": "src_continuous_control/results/real_confirmation_lr1e4_k16", "prefix": "pt_full", "seed": 2},
    {"method": "PT-B", "label": "PT-PPO lr_perm=3e-4", "directory": "src_continuous_control/results/real_confirmation_lr3e4_k16", "prefix": "pt_full", "seed": 0},
    {"method": "PT-B", "label": "PT-PPO lr_perm=3e-4", "directory": "src_continuous_control/results/real_confirmation_lr3e4_k16", "prefix": "pt_full", "seed": 1},
    {"method": "PT-B", "label": "PT-PPO lr_perm=3e-4", "directory": "src_continuous_control/results/real_confirmation_lr3e4_k16", "prefix": "pt_full", "seed": 2},
    {"method": "Vanilla", "label": "Vanilla PPO", "directory": "src_continuous_control/results/real_confirmation_vanilla", "prefix": "vanilla", "seed": 0},
    {"method": "Vanilla", "label": "Vanilla PPO", "directory": "src_continuous_control/results/real_confirmation_vanilla", "prefix": "vanilla", "seed": 1},
    {"method": "Vanilla", "label": "Vanilla PPO", "directory": "src_continuous_control/results/real_confirmation_vanilla", "prefix": "vanilla", "seed": 2},
]

COLORS = {"PT-A": "#1769aa", "PT-B": "#16805c", "Vanilla": "#c75146"}
METHOD_ORDER = ["PT-A", "PT-B", "Vanilla"]


def _read_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _path(run, suffix):
    return os.path.join(
        run["directory"],
        f'{run["prefix"]}_ppo_seed_{run["seed"]}_{suffix}.pkl',
    )


def _load_run(run):
    returns = np.asarray(_read_pickle(_path(run, "returns")), dtype=np.float64)
    scalars = _read_pickle(_path(run, "scalars"))
    records_path = _path(run, "consolidation_records")
    records = _read_pickle(records_path) if os.path.exists(records_path) else []
    if returns.ndim != 2 or returns.shape[1] < 2:
        raise ValueError(f'Unexpected return format: {_path(run, "returns")}')
    if int(returns[-1, 0]) != TOTAL_STEPS:
        raise ValueError(f'Unexpected final step in {_path(run, "returns")}')
    return returns[:, 0], returns[:, 1], scalars, records


def _last_scalar(scalars, name):
    values = np.asarray(scalars.get(name, []), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return None
    return float(values[-1, 1])


def _scalar_values(scalars, name):
    values = np.asarray(scalars.get(name, []), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    return values[:, 1]


def _record_values(records, name):
    return np.asarray(
        [record[name] for record in records if record.get(name) is not None],
        dtype=np.float64,
    )


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _phase_means(x, y):
    means = []
    for phase in range(PHASES):
        start = phase * SWITCH
        end = min((phase + 1) * SWITCH, TOTAL_STEPS)
        mask = (x > start) & (x <= end)
        means.append(float(y[mask].mean()))
    return means


def _smooth(values, width=20):
    if width <= 1 or len(values) < width:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _curve_summary(curves):
    length = min(len(values) for _, values in curves)
    x = curves[0][0][:length]
    matrix = np.stack([values[:length] for _, values in curves])
    mean = matrix.mean(axis=0)
    if len(matrix) > 1:
        ci = 1.96 * matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
    else:
        ci = np.zeros_like(mean)
    return x, mean, ci


def _load_curve(run, suffix):
    path = _path(run, suffix)
    if not os.path.exists(path):
        return None
    arr = np.asarray(_read_pickle(path), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, 0], arr[:, 1]


def _method_curves(data, method, suffix="returns"):
    curves = []
    for item in data:
        if item["run"]["method"] != method:
            continue
        curve = _load_curve(item["run"], suffix)
        if curve is not None:
            curves.append(curve)
    return curves


def _write_metrics(data, out_dir):
    fields = [
        "method", "seed", "overall_return_mean", "final_phase_return_mean",
        "boundary_drop", "jumpstart_mean", "retention_perm", "retention_full",
        "retention_perm_init", "retention_zero", "absorbed_frac_mean",
        "actor_absorbed_frac_mean", "consolidations", "perm_drift_from_init",
    ]
    rows = []
    for item in data:
        row = {"method": item["run"]["method"], "seed": item["run"]["seed"]}
        row.update(item["metrics"])
        rows.append(row)
    with open(os.path.join(out_dir, "run_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    with open(os.path.join(out_dir, "run_metrics.json"), "w") as handle:
        json.dump(rows, handle, indent=2)


def _aggregate(data):
    summary = {}
    for method in METHOD_ORDER:
        items = [item for item in data if item["run"]["method"] == method]
        metric_names = list(items[0]["metrics"].keys())
        method_summary = {"n": len(items)}
        for name in metric_names:
            values = [item["metrics"][name] for item in items]
            if name.startswith("phase_"):
                for phase in range(PHASES):
                    phase_values = [value[phase] for value in values]
                    mean, std = _mean_std(phase_values)
                    method_summary[f"{name}_{phase}_mean"] = mean
                    method_summary[f"{name}_{phase}_std"] = std
            else:
                mean, std = _mean_std(values)
                method_summary[f"{name}_mean"] = mean
                method_summary[f"{name}_std"] = std
        summary[method] = method_summary
    return summary


def _write_report(data, summary, out_dir):
    lines = [
        "# Final HalfCheetah Comparison",
        "",
        "All runs use MuJoCo HalfCheetah-v5, 8 vector environments, 3,072,000 aggregate environment steps, and four task switches at 614,400-step intervals.",
        "The task sequence is +1 -> -1 -> +1 -> -1 -> +1; only the directional velocity reward changes.",
        "",
        "## Results",
        "",
        "| Method | Seeds | Boundary drop | 20-update jumpstart | Final-phase return | Retention MSE perm | Retention MSE full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = summary[method]
        lines.append(
            f'| {method} | {row["n"]} | '
            f'{row["boundary_drop_mean"]:.1f} +/- {row["boundary_drop_std"]:.1f} | '
            f'{row["jumpstart_mean_mean"]:.1f} +/- {row["jumpstart_mean_std"]:.1f} | '
            f'{row["final_phase_return_mean_mean"]:.1f} +/- {row["final_phase_return_mean_std"]:.1f} | '
            f'{row["retention_perm_mean"]:.2f} +/- {row["retention_perm_std"]:.2f} | '
            f'{row["retention_full_mean"]:.2f} +/- {row["retention_full_std"]:.2f} |'
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Boundary drop is the mean pre-switch EMA return minus the trough in the five-update post-switch window; lower is better.",
        "- Jumpstart is the mean EMA return over the first 20 PPO updates after each switch; higher is better.",
        "- Retention MSE is scored against the saved converged value of the inactive task. The perm_init and zero controls are included in run_metrics.csv because low error from an inert permanent component is not sufficient evidence of retention.",
        "- The comparison is directional task switching in real HalfCheetah, not smooth physics drift.",
        "",
    ])
    pt_a = summary["PT-A"]
    pt_b = summary["PT-B"]
    vanilla = summary["Vanilla"]
    lines.extend([
        "## Findings",
        "",
        f'- Vanilla is the strongest method on this benchmark: final-phase return is {vanilla["final_phase_return_mean_mean"]:.1f} +/- {vanilla["final_phase_return_mean_std"]:.1f}, versus {pt_a["final_phase_return_mean_mean"]:.1f} +/- {pt_a["final_phase_return_mean_std"]:.1f} for PT-A and {pt_b["final_phase_return_mean_mean"]:.1f} +/- {pt_b["final_phase_return_mean_std"]:.1f} for PT-B.',
        f'- PT has a modest boundary-drop advantage: PT-A is {pt_a["boundary_drop_mean"]:.1f} versus {vanilla["boundary_drop_mean"]:.1f} for vanilla; PT-B is {pt_b["boundary_drop_mean"]:.1f}. This local benefit is not enough to offset the lower return.',
        f'- PT does not show a jumpstart advantage here. The 20-update signed-return measure is {pt_a["jumpstart_mean_mean"]:.1f} for PT-A and {pt_b["jumpstart_mean_mean"]:.1f} for PT-B, versus {vanilla["jumpstart_mean_mean"]:.1f} for vanilla.',
        f'- The PT mechanism is active rather than inert: each PT run performs {pt_a["consolidations_mean"]:.0f} consolidation cycles, with mean critic absorption {pt_a["absorbed_frac_mean_mean"]:.3f} for PT-A and {pt_b["absorbed_frac_mean_mean"]:.3f} for PT-B.',
        f'- The retention probe does not support the PT retention claim on this task. PT-A permanent MSE ({pt_a["retention_perm_mean"]:.2f}) and PT-B permanent MSE ({pt_b["retention_perm_mean"]:.2f}) are both above their respective initial-permanent and zero controls; vanilla is below those controls.',
        "- The most defensible conclusion is that the implementation performs consolidation, but this symmetric directional HalfCheetah task does not produce a performance win for PT-PPO. The task design, signed-return metric, and only three seeds limit generalization beyond this benchmark.",
        "",
    ])
    with open(os.path.join(out_dir, "final_report.md"), "w") as handle:
        handle.write("\n".join(lines))


def _plot(data, summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for method in METHOD_ORDER:
        curves = _method_curves(data, method)
        x, mean, ci = _curve_summary(curves)
        color = COLORS[method]
        axes[0, 0].plot(x, _smooth(mean), color=color, linewidth=2, label=f"{method} (n={len(curves)})")
        axes[0, 0].fill_between(x, _smooth(mean - ci), _smooth(mean + ci), color=color, alpha=0.16)
    for boundary in range(SWITCH, TOTAL_STEPS, SWITCH):
        axes[0, 0].axvline(boundary, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0, 0].set_title("Training return, mean +/- 95% CI")
    axes[0, 0].set_xlabel("Environment steps")
    axes[0, 0].set_ylabel("EMA return")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.25)

    phase_x = np.arange(1, PHASES + 1)
    for method in METHOD_ORDER:
        row = summary[method]
        means = [row[f"phase_means_{phase}_mean"] for phase in range(PHASES)]
        errors = [row[f"phase_means_{phase}_std"] for phase in range(PHASES)]
        axes[0, 1].errorbar(phase_x, means, yerr=errors, color=COLORS[method], marker="o", linewidth=2, capsize=4, label=method)
    axes[0, 1].set_title("Return by task phase")
    axes[0, 1].set_xlabel("Phase")
    axes[0, 1].set_ylabel("Mean EMA return")
    axes[0, 1].set_xticks(phase_x)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)

    metric_specs = [("boundary_drop", "Mean boundary drop"), ("jumpstart_mean", "Mean 20-update jumpstart")]
    positions = np.arange(len(METHOD_ORDER))
    width = 0.72
    for axis, (name, title) in zip((axes[1, 0], axes[1, 1]), metric_specs):
        means = [summary[method][f"{name}_mean"] for method in METHOD_ORDER]
        errors = [summary[method][f"{name}_std"] for method in METHOD_ORDER]
        axis.bar(positions, means, yerr=errors, color=[COLORS[method] for method in METHOD_ORDER], width=width, alpha=0.82, capsize=5)
        for index, method in enumerate(METHOD_ORDER):
            values = [item["metrics"][name] for item in data if item["run"]["method"] == method]
            jitter = np.linspace(-0.12, 0.12, len(values)) if values else []
            axis.scatter(np.asarray(index) + jitter, values, color="black", s=28, zorder=3)
        axis.set_title(title)
        axis.set_xticks(positions, METHOD_ORDER)
        axis.grid(True, axis="y", alpha=0.25)
    axes[1, 0].set_ylabel("Return units")
    axes[1, 1].set_ylabel("Return units")
    fig.suptitle("Final real HalfCheetah comparison", fontsize=15)
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"final_comparison.{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for method in METHOD_ORDER:
        curves = _method_curves(data, method, "eval_returns")
        if not curves:
            continue
        x, mean, ci = _curve_summary(curves)
        axes[0].plot(x, _smooth(mean, 3), color=COLORS[method], linewidth=2, label=f"{method} (n={len(curves)})")
        axes[0].fill_between(x, _smooth(mean - ci, 3), _smooth(mean + ci, 3), color=COLORS[method], alpha=0.16)
    for boundary in range(SWITCH, TOTAL_STEPS, SWITCH):
        axes[0].axvline(boundary, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_title("Offline zero-momentum evaluation")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("True return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    names = ["retention_perm", "retention_full", "retention_perm_init", "retention_zero"]
    labels = ["perm", "full", "perm init", "zero"]
    x_pos = np.arange(len(METHOD_ORDER))
    bar_width = 0.18
    for offset, (name, label) in enumerate(zip(names, labels)):
        means = [summary[method][f"{name}_mean"] for method in METHOD_ORDER]
        errors = [summary[method][f"{name}_std"] for method in METHOD_ORDER]
        axes[1].bar(x_pos + (offset - 1.5) * bar_width, means, bar_width, yerr=errors, capsize=3, label=label, alpha=0.82)
    axes[1].set_title("Retention diagnostics and controls")
    axes[1].set_xticks(x_pos, METHOD_ORDER)
    axes[1].set_ylabel("MSE against inactive-task reference")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.25)
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"evaluation_retention.{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    pt_methods = ["PT-A", "PT-B"]
    positions = np.arange(len(pt_methods))
    width = 0.34
    for offset, (name, label) in enumerate((
            ("absorbed_frac", "critic absorbed fraction"),
            ("actor_absorbed_frac", "actor absorbed fraction"))):
        means = [summary[method][f"{name}_mean_mean"] for method in pt_methods]
        errors = [summary[method][f"{name}_mean_std"] for method in pt_methods]
        axis.bar(positions + (offset - 0.5) * width, means, width, yerr=errors,
                 capsize=4, label=label, alpha=0.82)
    axis.set_title("PT consolidation absorption diagnostics")
    axis.set_xticks(positions, pt_methods)
    axis.set_ylabel("Absorbed fraction")
    axis.set_ylim(0, 1.08)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"consolidation_diagnostics.{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--out-dir", default="plots/figures/final_comparison")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = []
    for run in RUNS:
        x, y, scalars, records = _load_run(run)
        phase_means = _phase_means(x, y)
        absorbed = _record_values(records, "absorbed_frac")
        actor_absorbed = _record_values(records, "actor_absorbed_frac")
        metrics = {
            "overall_return_mean": float(y.mean()),
            "final_phase_return_mean": float(y[-FINAL_WINDOW_UPDATES:].mean()),
            "phase_means": phase_means,
            "boundary_drop": _last_scalar(scalars, "boundary/mean_drop"),
            "jumpstart_mean": _last_scalar(scalars, "boundary/mean_jumpstart"),
            "retention_perm": _last_scalar(scalars, "retention/mse_perm"),
            "retention_full": _last_scalar(scalars, "retention/mse_full"),
            "retention_perm_init": _last_scalar(scalars, "retention/mse_perm_init"),
            "retention_zero": _last_scalar(scalars, "retention/mse_zero"),
            "absorbed_frac_mean": float(absorbed.mean()) if len(absorbed) else None,
            "actor_absorbed_frac_mean": float(actor_absorbed.mean()) if len(actor_absorbed) else None,
            "consolidations": len(records),
            "perm_drift_from_init": _last_scalar(scalars, "perm/drift_from_init"),
        }
        data.append({"run": run, "metrics": metrics})
    _write_metrics(data, args.out_dir)
    summary = _aggregate(data)
    _write_report(data, summary, args.out_dir)
    _plot(data, summary, args.out_dir)
    print(os.path.join(args.out_dir, "final_report.md"))
    for method in METHOD_ORDER:
        row = summary[method]
        print(f'{method}: n={row["n"]} drop={row["boundary_drop_mean"]:.2f}+/-{row["boundary_drop_std"]:.2f} jump={row["jumpstart_mean_mean"]:.2f}+/-{row["jumpstart_mean_std"]:.2f} final={row["final_phase_return_mean_mean"]:.2f}+/-{row["final_phase_return_mean_std"]:.2f}')


if __name__ == "__main__":
    main()