# Phase 1 archive

Superseded by `PHASE2_INSTRUCTIONS.md`. Kept because parts of it are still citable.

`FULL_PT.md` is the full record of the Phase 1 study on the reward-switch benchmark
(~979 runs). It was archived at the supervisors' request: its headline framing rests on a
comparison against a non-standard baseline — plain PPO with periodic multiplicative
shrinkage of the policy's output layer — which is a heuristic without theory behind it.

## What that decision does and does not invalidate

The shrinkage arm was an *additional* arm, not a change to any other agent. Every result
that compares `pt_full` against vanilla PPO, or against its own frozen-permanent control,
was measured without it and stands on its own:

- `pt_full` beats vanilla PPO on HalfCheetah (760 vs 50, p = 0.015).
- Zeroing the permanent network changes nothing (p = 1.000).
- The KL anchor explains none of the gain (beta = 0 gives the same result).
- No setting of the permanent's learning rate beats vanilla (all p <= 0.001), and the
  curve is worst at intermediate rates -- a stale-anchor effect.
- The advantage components cancel in the actor-critic: corr(A_perm, A_trans) ~= -1.0.
- Under boundary-free drift, EWC degenerates into vanilla exactly (p = 1.000).
- The implementation passes all eight constraints of the specification; one defect found
  and measured benign (p = 0.878).

What is contested is the *interpretation* -- "the gain reduces to periodic shrinkage" --
because that claim is only as strong as the baseline it rests on.

## Also here

Figures for the above remain committed at `plots/figures_pt_full/`, with a plain-language
walkthrough in `figures_full_pt_guide.md`. The generator, `plots/make_pt_full_figures.py`,
is archived alongside, so the figures can still be regenerated from the raw result pickles.
