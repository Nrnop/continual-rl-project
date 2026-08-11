# Phase 1 figures and their generators

Two figure sets, both from the **reward-switch** benchmark that Phase 2 replaces.

| directory | what it is |
|---|---|
| `figures/` | the project's canonical figure set — return curves, boundary drop, phase means, consolidation diagnostics, retention. Post-fix (see `figures/PROVENANCE.md`). |
| `figures_pt_full/` | the 14 figures of the Phase 1 reduction study, walked through in `../docs/figures_full_pt_guide.md`. |

The generators that produced them are in this directory (`make_reinvestigation_figures.py`,
`make_pt_full_figures.py`, and the consolidation plotters). They read raw per-seed result
pickles from `results/` directories that are **not tracked**, so regenerating requires re-running
the sweeps. The committed images are the surviving record.

## Deleted, deliberately

`figures_before_fixes/` held a parallel set produced *before* fifteen defects were fixed --
including the alpha_P tuning, the `decay_mode` fix and the theta_P initialisation fix. In those
runs the permanent component was effectively inert, but the figures still looked plausible and
carried nothing to say so.

`figures/PROVENANCE.md` records that keeping two folders side by side is precisely how the stale
set survived and was mistaken for real results. It was deleted rather than archived for that
reason; `git log` still has it if it is ever genuinely needed.
