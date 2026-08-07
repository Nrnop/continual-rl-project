# Does the critic's knowledge reach the policy?

Two sweeps, 6 arms × 10 seeds × 3.07M steps, `--no-eval` on every arm, medians and
Mann-Whitney throughout. Answers two questions the supervisor raised on 2026-08-06.

---

## 1. PT's critic is fine. Its behaviour is not.

Measured in the 20 updates after each task switch.

| | vanilla | pt | p |
|---|---|---|---|
| explained variance (pooled) | 0.721 | **0.769** | 0.406 |
| absolute value error | 0.120 | 0.124 | 0.821 |
| **return** | **1420.9** | **928.7** | **0.013** |

PT's critic is statistically indistinguishable from vanilla's — nominally better. Its
behaviour is significantly worse. The deficit is not a value-learning failure.

> **Correction.** An earlier version of this table reported PT's critic as significantly
> *worse* (p=0.007). That came from averaging a per-update ratio `1 − Var(A)/Var(R)` whose
> denominator collapses on a degraded policy: 31% of PT's post-switch updates had a negative
> value against 11% of vanilla's. Pooling the variances and forming the ratio once removes it.
> Any statistic that averages a ratio should be treated as suspect.

---

## 2. The critic matters a lot on this benchmark

Removing the critic from the policy's advantage entirely (`van_advnone`):

| | vanilla | no critic | change |
|---|---|---|---|
| post-switch return | 1420.9 | 532.5 | **−63%** (p=0.0005) |
| whole-run return | 929.3 | 135.6 | **−85%** |

So "the critic can't influence the policy in PPO" is false. The channel is load-bearing.
Whatever limits PT, it is not that a critic-side mechanism is irrelevant here.

---

## 3. How much of the policy's update signal is the critic?

The advantage is the critic's only channel to the policy, and it splits **exactly**:

```
A  =  A_reward  +  A_perm  +  A_trans
```

(δ is affine in V and GAE is a linear filter over δ, so there is no remainder.)
Covariance shares, post-switch:

| arm | reward | perm | trans | **critic total** |
|---|---|---|---|---|
| vanilla | 0.665 | 0.335 | 0.000 | **0.335** |
| pt | 0.471 | 0.177 | 0.353 | **0.529** |

---

## 4. The finding: the transient cancels the permanent

The shares above hide the important number. Each component's variance, relative to the
variance of the advantage that actually survives:

| arm | Var(A_perm)/Var(A) | Var(A_trans)/Var(A) | corr(A_perm, A_trans) |
|---|---|---|---|
| vanilla | 0.88 | 0.00 | — |
| **pt** | **4.85** | **11.26** | **≈ −1.0** |
| **pt_inert** | **11.39** | **20.71** | **≈ −1.0** |

The permanent injects a term ~5× larger than the final advantage, and the transient
subtracts almost exactly that term.

**This is structural, not incidental.** The transient is trained so that `V_P + V_T ≈ R`,
which gives `δ_T = δ_R − δ_P` identically, hence `A_trans = A_reward − A_perm`. The transient
*is defined* as the thing that cancels the permanent.

Three consequences:

1. **It explains `pt` vs `pt_inert`, p = 0.597** — the number nothing in this project has
   moved. A working permanent and a dead one look identical to the actor because the transient
   absorbs the difference either way.
2. **`pt_inert` shows the most extreme cancellation** (11.4 / 20.7), exactly as
   REINVESTIGATION.md §6a predicts: a frozen random `V_P` is the hardest thing to cancel.
3. **For the policy, only `V = V_P + V_T` exists.** The decomposition is value-preserving, so
   it is invisible at the point of use. It can only matter through its *dynamics* —
   consolidation and decay changing V discontinuously. In DQN that changes behaviour instantly
   (argmax over Q). In PPO it reaches behaviour only by retraining, and the transient's job is
   to undo it.

---

## 5. What did not work

**D2 (is there a per-consolidation cost?) is unresolved.** Twice the vanilla control — which
never consolidates — reproduced the effect at the same sign and a larger magnitude than any PT
arm (+60.6 against PT's +42.9). First attempt binned a 0.99 EMA, whose slope any binning
recovers; the corrected event study on raw per-rollout returns failed the same control. The
question needs a different instrument, not a third estimator.

**`pt_advtrans` is not a valid ablation.** Feeding the actor `A − A_perm` hands it an advantage
baselined on `V_T` alone — but `V_T` is a residual, not a value function, so this injects an
uncancelled term ~11× the advantage's variance rather than removing 17.7% of a signal. Return
collapses to −204.2, worse than removing the whole critic (98.0). The `trans_only` and
`perm_only` settings of `actor_advantage_source` should not be used. `none` is valid (pure
reward-to-go, unbiased), which is why `pt_advnone` and `van_advnone` behave sensibly.

---

## 6. Reproduce

```bash
cd "<parent of src_continuous_control>"
ARMS_FULL=1 MAXJOBS=<cores-1> SEEDS="0 1 2 3 4 5 6 7 8 9" \
  bash src_continuous_control/scripts/run_transmission.sh
python -m src_continuous_control.scripts.analyze_transmission \
  --results-dir trans2_results \
  --arms vanilla pt pt_inert pt_advtrans pt_advnone van_advnone
```

`pytest src_continuous_control/tests -q` → 71 passed.
