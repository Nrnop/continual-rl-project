---
name: plot-results
description: Generate the comparison figures for this continuous-control PT project (return curves, boundary-drop, recovery-time, offline-adaptation, velocity, critic-loss) from the per-seed result pkls. Use whenever asked to plot, compare agents, or produce thesis figures. Wraps plots/plot_compare.py and points to dataviz for styling.
---

# Plotting & comparing results

Figures come from `plots/plot_compare.py`, which reads the per-seed `results/*.pkl` written by
training and writes PNG/PDF to `plots/figures/`.

## Golden rules

1. **Run from the PARENT directory** (the one containing `src_continuous_control/`) — the script's
   default `--results-dir`/`--out-dir` are relative to that (`src_continuous_control/results`,
   `src_continuous_control/plots/figures`).
2. **`--switch`, `--total-steps`, `--n-steps` must match the runs** being plotted, or the
   task-boundary markers and x-axis will be wrong. Defaults match `continual_fast`/`pt_fixed`
   (`switch=614400`, `total_steps=3072000`, `n_steps=2048`).
3. Results must exist first — for all three agents (`vanilla`, `pt`, `ewc`) across the seeds you
   pass. Missing agents/seeds are skipped, not errored. If none exist, run training first
   (see the `run-experiment` skill).

## Command

```bash
cd <PARENT of src_continuous_control>   # Windows: "e:/update-single task + videos"  ·  Linux: clone dir
python -m src_continuous_control.plots.plot_compare --seeds 0 1 2 3 4
# override if the runs used different settings:
#   --switch <n> --total-steps <n> --n-steps <n> --smooth <k> --out-dir <path>
```

## Figures produced (in `plots/figures/`)

- `return_curves` — overlaid mean ± CI episodic return, with task-boundary lines.
- `boundary_drop` — relative return drop at each switch (lower = better stability).
- `recovery_time` — steps to recover after a switch (lower = faster adaptation).
- `offline_curves` — standardized zero-momentum offline eval.
- `asymptotic_bar` — asymptotic vs online cumulative return.
- (velocity and `critic_loss` diagnostic curves).

The thesis claim lives in `boundary_drop` + `recovery_time`: PT should show a smaller boundary drop
and faster recovery than vanilla/EWC.

## Styling for the thesis

Before writing or editing any chart code, colors, or layout, **load the `dataviz` skill first** —
it defines the palette and mark/axis/legend conventions so every figure reads as one system. Apply
it consistently across all figures (same agent→color mapping everywhere: e.g. PT, Vanilla, EWC).
Keep the true (un-normalized) return on the y-axis and label boundaries clearly.
