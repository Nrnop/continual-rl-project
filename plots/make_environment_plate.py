"""One rendered frame per environment (seven), as a plate and as single images.

    # from the parent directory of the package
    python -m src_continuous_control.plots.make_environment_plate

Each frame is taken after a fixed number of steps from a fixed seed, so the output is reproducible.
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

# palette from make_phase2_figures.py
SURFACE, INK, SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_thesis")

# (label, kind, spec, steps to take before the frame is grabbed)
ENVS = [
    ("HalfCheetah",        "gym", "HalfCheetah-v5",          120),
    ("cartpole-swingup",   "dmc", ("cartpole", "swingup"),    60),
    ("reacher-easy",       "dmc", ("reacher", "easy"),        40),
    ("ball_in_cup-catch",  "dmc", ("ball_in_cup", "catch"),    5),
    # few steps for the walkers, so they are still upright under random torques
    ("walker-stand",       "dmc", ("walker", "stand"),         4),
    ("walker-walk",        "dmc", ("walker", "walk"),          4),
    ("cheetah-run",        "dmc", ("cheetah", "run"),         80),
]


def frame_gym(env_id, steps):
    import gymnasium as gym
    env = gym.make(env_id, render_mode="rgb_array",
                   width=1010, height=760)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        env.step(rng.uniform(-0.3, 0.3, size=env.action_space.shape).astype(np.float32))
    img = env.render()
    env.close()
    return img


def frame_dmc(domain, task, steps):
    from dm_control import suite
    env = suite.load(domain, task, task_kwargs={"random": 0})
    # the default offscreen buffer is 640x480; raise it before the first render creates it
    env.physics.model.vis.global_.offwidth = 1280
    env.physics.model.vis.global_.offheight = 960
    spec = env.action_spec()
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(steps):
        env.step(rng.uniform(spec.minimum, spec.maximum).astype(np.float64))
    return env.physics.render(height=960, width=1280, camera_id=0)



def crop(img, margin=0.22, extra=0.0):
    """Crop to the robot, keeping the frame's aspect ratio.

    Bodies render warm (orange or tan) against a cool background, so the subject is the pixels whose
    red channel exceeds blue by a margin. Reacher's red target is included by the same rule.
    """
    a = img.astype(np.int16)
    h, w, _ = a.shape
    mask = (a[:, :, 0] - a[:, :, 2]) > 38
    if mask.sum() < 80:
        return img
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    m2 = margin + extra
    my, mx = int((y1 - y0) * m2) + 6, int((x1 - x0) * m2) + 6
    y0, y1, x0, x1 = y0 - my, y1 + my, x0 - mx, x1 + mx
    want = w / h
    ph, pw = max(y1 - y0, 1), max(x1 - x0, 1)
    if pw / ph < want:
        need = int(ph * want); cx = (x0 + x1) // 2
        x0, x1 = cx - need // 2, cx + need // 2
    else:
        need = int(pw / want); cy = (y0 + y1) // 2
        y0, y1 = cy - need // 2, cy + need // 2
    # slide back inside the frame rather than clipping, so the aspect ratio is preserved
    if x0 < 0: x1, x0 = x1 - x0, 0
    if y0 < 0: y1, y0 = y1 - y0, 0
    if x1 > w: x0, x1 = x0 - (x1 - w), w
    if y1 > h: y0, y1 = y0 - (y1 - h), h
    return img[max(0, y0):y1, max(0, x0):x1, :]


def single_frames(frames):
    """One image per environment, for the section that introduces it."""
    import matplotlib.pyplot as plt
    for label, img in frames:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        fig.patch.set_facecolor(SURFACE)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7"); sp.set_linewidth(0.8)
        fig.tight_layout()
        slug = label.replace("-", "_").replace(" ", "_").lower()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"fig5_env_{slug}.{ext}"),
                        dpi=190, bbox_inches="tight", facecolor=SURFACE)
        print(f"  wrote fig5_env_{slug}.png")
        plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = []
    for label, kind, spec, steps in ENVS:
        img = frame_gym(spec, steps) if kind == "gym" else frame_dmc(spec[0], spec[1], steps)
        # ball_in_cup gets a loose crop: the string below the cup is part of the task
        frames.append((label, crop(img, extra=0.9) if "ball_in_cup" in label else crop(img)))
        print(f"  rendered {label:<20} {img.shape}")

    single_frames(frames)

    fig, axes = plt.subplots(2, 4, figsize=(12.6, 5.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (label, img) in zip(axes.ravel(), frames):
        ax.imshow(img)
        ax.set_title(label, color=INK, fontsize=10.5, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
            sp.set_linewidth(0.8)
    # the eighth cell holds a note on return ceilings
    blank = axes.ravel()[len(frames)]
    blank.axis("off")
    blank.text(0.5, 0.55,
               "Six of the seven share\na reward in [0, 1] per step\nover exactly 1000 steps,\n"
               "so their ceiling is 1000.\n\nHalfCheetah's is unbounded.",
               ha="center", va="center", color=SECONDARY, fontsize=9.5, linespacing=1.7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig5_2_environments.{ext}")
        fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=SURFACE)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
