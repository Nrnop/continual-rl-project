"""Render cartpole and reacher at the three task multipliers (x1.0, x1.6, x0.6).

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_physics_change_figure

Only these two environments are drawn because their scaled quantity is geometric (pole length,
arm segment length). The other environments scale mass, damping or friction, which do not change
how a frame looks.
"""
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

SURFACE, INK, SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
AXIS = "#c3c2b7"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_thesis")

# task index -> multiplier, from task_multipliers = (1.0, 1.6, 0.6, 1.6, 0.6)
TASKS = [(0, "nominal, x1.0"), (1, "x1.6"), (2, "x0.6")]

# (environment, what the multiplier scales, the joint angles every frame is drawn at)
ROWS = [
    ("cartpole-swingup", "pole length and mass", (0.0, 0.55)),
    ("reacher-easy",     "arm segment length and mass", (0.9, -1.5)),
]


def frames(env_name, pose):
    """One frame per multiplier, all at the same joint angles.

    The pose is written directly instead of stepping the env, so the scaled quantity is the only
    difference between frames.
    """
    from src_continuous_control.envs.dm_control_drift import DmControlDrift
    out = []
    for task, label in TASKS:
        env = DmControlDrift(env_name=env_name)
        env.set_task(task)
        env.reset(seed=0)
        inner = env
        for _ in range(8):
            if hasattr(inner, "physics"):
                break
            inner = getattr(inner, "env", None) or getattr(inner, "_env", None) or inner
        with inner.physics.reset_context():
            inner.physics.data.qpos[:len(pose)] = pose
            inner.physics.data.qvel[:] = 0.0
        inner.physics.forward()
        out.append((label, inner.physics.render(height=480, width=640, camera_id=0)))
        env.close()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for env_name, scaled, pose in ROWS:
        print(f"  rendering {env_name} at three multipliers")
        rows.append((env_name, scaled, frames(env_name, pose)))

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.8, 3.3 * len(rows)))
    fig.patch.set_facecolor(SURFACE)
    for r, (env_name, scaled, fr) in enumerate(rows):
        for c, (label, img) in enumerate(fr):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(AXIS); sp.set_linewidth(0.8)
            if r == 0:
                ax.set_title(label, color=INK, fontsize=10.5, pad=6)
            if c == 0:
                ax.set_ylabel(f"{env_name}\n{scaled}", color=SECONDARY, fontsize=9.5,
                              labelpad=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig5_physics_change.{ext}")
        fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=SURFACE)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
