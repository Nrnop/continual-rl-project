# Final HalfCheetah Comparison

All runs use MuJoCo HalfCheetah-v5, 8 vector environments, 3,072,000 aggregate environment steps, and four task switches at 614,400-step intervals.
The task sequence is +1 -> -1 -> +1 -> -1 -> +1; only the directional velocity reward changes.

## Results

| Method | Seeds | Boundary drop | 20-update jumpstart | Final-phase return | Retention MSE perm | Retention MSE full |
|---|---:|---:|---:|---:|---:|---:|
| PT-A | 3 | 234.5 +/- 9.8 | 14.8 +/- 68.4 | -490.8 +/- 39.8 | 41.84 +/- 11.74 | 42.08 +/- 11.51 |
| PT-B | 3 | 257.8 +/- 37.8 | 78.3 +/- 99.5 | -407.9 +/- 255.2 | 22.79 +/- 10.30 | 22.90 +/- 10.49 |
| Vanilla | 3 | 291.9 +/- 65.2 | 1088.8 +/- 334.6 | 832.6 +/- 234.0 | 3.02 +/- 2.02 | 3.02 +/- 2.02 |

## Interpretation

- Boundary drop is the mean pre-switch EMA return minus the trough in the five-update post-switch window; lower is better.
- Jumpstart is the mean EMA return over the first 20 PPO updates after each switch; higher is better.
- Retention MSE is scored against the saved converged value of the inactive task. The perm_init and zero controls are included in run_metrics.csv because low error from an inert permanent component is not sufficient evidence of retention.
- The comparison is directional task switching in real HalfCheetah, not smooth physics drift.

## Findings

- Vanilla is the strongest method on this benchmark: final-phase return is 832.6 +/- 234.0, versus -490.8 +/- 39.8 for PT-A and -407.9 +/- 255.2 for PT-B.
- PT has a modest boundary-drop advantage: PT-A is 234.5 versus 291.9 for vanilla; PT-B is 257.8. This local benefit is not enough to offset the lower return.
- PT does not show a jumpstart advantage here. The 20-update signed-return measure is 14.8 for PT-A and 78.3 for PT-B, versus 1088.8 for vanilla.
- The PT mechanism is active rather than inert: each PT run performs 94 consolidation cycles, with mean critic absorption 0.864 for PT-A and 0.967 for PT-B.
- The retention probe does not support the PT retention claim on this task. PT-A permanent MSE (41.84) and PT-B permanent MSE (22.79) are both above their respective initial-permanent and zero controls; vanilla is below those controls.
- The most defensible conclusion is that the implementation performs consolidation, but this symmetric directional HalfCheetah task does not produce a performance win for PT-PPO. The task design, signed-return metric, and only three seeds limit generalization beyond this benchmark.
