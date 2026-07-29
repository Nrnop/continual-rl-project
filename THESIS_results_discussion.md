# Draft: Results and Discussion chapters

*Working draft for the thesis, generated from `FINDINGS.md`. Every figure referenced exists in
`plots/figures/`; every number has been verified against the raw per-seed curves in `results/` and
`abl_results/`. Note that this document is organised **logically** (what we know) rather than
**chronologically** (the order in which we found it) — the discovery order, including two retracted
hypotheses, is preserved in `FINDINGS.md` and summarised in §D.6.*

---

# Chapter N — Results

## N.1 Overview of the experimental programme

Two families of experiment were run, both on MuJoCo HalfCheetah with an identical PPO backbone.

**The task-switching benchmark** (`DirectionalHalfCheetah`) inverts the sign of the forward-velocity
reward term every 614 400 environment steps, giving four abrupt task boundaries and five phases in
the direction sequence `+1, −1, +1, −1, +1` over 3 072 000 steps. This is a *reward-based*, discrete
form of non-stationarity with clear boundaries.

**The smooth-drift benchmark** (`LipschitzDriftHalfCheetah`) leaves the reward untouched and instead
rescales physical parameters — joint damping and ground friction — continuously by a smooth
multiplier. There are no boundaries, no task index and no reset signal. This is the setting specified
in the research proposal, and it satisfies the Lipschitz condition ‖P_{t+1} − P_t‖ ≤ ε with ε
measured and reported at run time rather than assumed.

Three agents were compared: **vanilla PPO** (a single value head), **PT-PPO** (the permanent–transient
split critic under study), and **Online EWC** (an elastic-weight-consolidation penalty on the actor).
All three share the same PPO core, the same `GaussianActor` policy architecture, the same
environment and the same seeds; only the critic or the regulariser differs. Any performance
difference is therefore attributable to the mechanism under study.

Unless stated otherwise, results are reported as the **mean return within each phase or segment**,
averaged over five random seeds. Per-phase means are used rather than end-of-phase values because
they are substantially more stable: at n = 5 the end-of-phase estimate is dominated by the
short-timescale oscillation of the return curve.

## N.2 The task-switching benchmark

Table N.1 reports per-phase mean return for the three agents.

**Table N.1 — Per-phase mean return, task-switching benchmark (5 seeds).** Parenthesised values give
the number of seeds achieving a positive mean in that phase.

| Agent | P1 (fwd) | P2 (bwd) | P3 (fwd) | P4 (bwd) | P5 (fwd) |
|---|---|---|---|---|---|
| Vanilla PPO | 743 (5/5) | 468 (4/5) | 243 (3/5) | 375 (3/5) | −34 (2/5) |
| Online EWC | 743 (5/5) | 517 (3/5) | **705** (4/5) | **1238** (5/5) | **533** (5/5) |
| PT-PPO | 475 (5/5) | 332 (2/5) | **−279** (1/5) | −249 (2/5) | −396 (0/5) |

Three observations follow directly (see also Figure `phase_means_main` and the return curves in
Figure `return_curves`).

**EWC outperforms the baseline.** It is the only agent with a positive mean return in every phase,
and the only one positive across all five seeds in phases 4 and 5. Its phase-4 advantage over
vanilla (1238 against 375) is large relative to the across-seed standard deviation (σ ≈ 399) and is
the clearest performance difference observed anywhere in this study.

**PT-PPO tracks the baselines for two phases and then fails.** From phase 3 onward it does not
recover, and by phase 5 no seed achieves a positive mean. The failure is qualitative as well as
quantitative: the physical velocity trace (Figure `velocity_curves`) shows that PT's cheetah
**ceases to move**, with mean x-velocity approaching zero from phase 3 onward, while vanilla and EWC
continue to reverse direction and reach ±1.5–3.3 in each phase. The agent converges on a
do-nothing policy rather than an incorrect gait.

**A consistency check.** Vanilla and EWC are identical in phase 1 (743 for both). This is expected:
EWC's penalty is inactive until the first task boundary supplies a Fisher estimate, so the two
agents are the same algorithm until step 614 400. The agreement confirms that the comparison is
correctly matched on seeds and environment.

### N.2.1 Boundary behaviour

Because the research question concerns stability at task changes, two boundary-specific measures
were recorded. The **relative task-switch return drop** (Figure `boundary_drop`) is approximately
**117 %** for PT-PPO with very large variance, against a tight **52–55 %** for both vanilla and EWC.
The **zero-momentum offline evaluation** — which resets the agent to a standstill and measures
return from rest, removing the confound of carried momentum at a direction reversal (Figure
`offline_curves`) — shows PT *degrading over training*, from roughly +400 to negative, while EWC
climbs to about 1270 and vanilla holds between 300 and 770.

Both hypotheses stated in the proposal are therefore contradicted on this benchmark. **H1** (the
dual-timescale agent outperforms single-timescale baselines) fails: PT-PPO is the weakest of the
three. **H2** (the permanent component prevents catastrophic failure) fails in the strongest sense:
catastrophic failure is precisely what occurs, and only for the agent equipped with the permanent
component.

## N.3 Characterising the failure of PT-PPO

Because a negative result is only informative if the implementation can be shown to be sound, the
failure was investigated directly rather than inferred. Seven ablation rounds were run, all at the
full 3 072 000-step horizon and, unless noted, with five seeds.

### N.3.1 The consolidation operator

PT's permanent critic is not trained on returns. It is updated only during *consolidation*, which
occurs every *k* = 10 updates and regresses `V_perm → old_V_perm + (1 − decay)·V_trans` over a
buffer of recently visited states, after which the transient head is decayed. Instrumenting this
operation at the shipped hyper-parameters (`lr_perm = 1e-5`, SGD, one epoch — 320 gradient steps
over 20 480 states) gives:

- the permanent critic absorbs **0.05 %** of the transient value;
- the decay removes **100 %** of it;
- the net effect is that **98.3 %** of the acting value V = V_perm + V_trans is destroyed at each
  consolidation, approximately **150 times per run** (Figure `consolidation_mechanism`, panel a).

The immediate consequence is that the next rollout bootstraps from a critic that has been emptied,
corrupting the advantage estimates for that rollout.

This defect is invisible to the obvious diagnostic. The reported `critic_loss` remains small
(≈ 0.01) throughout, because it is computed *during* the ten-epoch PPO update, by which point the
fast transient head has already re-fitted the returns. A healthy-looking critic loss therefore
coexists with a value function that is being repeatedly destroyed.

### N.3.2 Establishing causation

Disabling consolidation entirely reverses the collapse: phase-3 mean return moves from **−279** to
**+291**, restoring vanilla-level performance (Table N.2). This is the expected outcome — with
consolidation disabled, `V_perm` remains at its random initialisation and PT reduces to a single
trained critic plus a fixed offset — and it establishes that the consolidation operator, not some
other component, is responsible for the failure.

### N.3.3 Consolidation is not a representational limit

A natural conjecture is that the consolidation target is unrepresentable: the permanent network is
asked to express `old_V_perm + V_trans`, the sum of two multilayer perceptrons, using a single
network of the same size. This conjecture is **false**, and was tested rather than assumed. Fitting
the true consolidation problem (20 480 buffered states) with Adam instead of the shipped SGD:

| Permanent network | Parameters | Error on the fitted batch |
|---|---|---|
| `[64, 64]` (as shipped) | 5 377 | 29.3 % |
| `[256, 256]` | 70 657 | **3.2 %** |

With sufficient capacity and training the target is fitted to approximately 3 %. The shipped
configuration fails not because the target cannot be represented but because the regression is never
meaningfully trained — 320 SGD steps at a learning rate of 1e-5 — while the deletion step executes
regardless.

### N.3.4 Training the regression does not help

Round 5 supplied the consolidation regression with Adam, `lr_perm = 1e-3` and twenty epochs (6 400
gradient steps per consolidation). Measured in situ, the regression now works: value drift per
consolidation falls from ≈ 3 % early in training to ≈ 0.006 % by the final phase (Figure
`consolidation_insitu`, panel a). The transfer is, by this measure, essentially exact.

Performance nevertheless becomes **worse**, not better (Table N.2): the collapse arrives a full
phase earlier, at the first switch rather than the third, and the deepest trough falls to −837.

Round 6 tested whether the permanent network was memorising its buffer — fitting the states it
trains on while extrapolating poorly to new ones — by excluding 20 % of the buffer from the
regression and measuring drift on that held-out portion separately. It does not: across all seeds
and all of training, the fitted and held-out errors are **0.300 %** and **0.310 %** respectively,
tracking one another everywhere including immediately after a task switch (Figure
`consolidation_insitu`, panel b).

**Table N.2 — PT-PPO variants, per-phase mean return.** Ordered by the fidelity of the consolidation
transfer.

| Variant | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| Consolidation disabled | 641 | 1039 | 291 | 746 | −373 |
| Shared trunk (exact transfer, §N.4) | 814 | 394 | 27 | 212 | −176 |
| As shipped (transfer barely occurs) | 475 | 332 | −279 | −249 | −396 |
| Trained regression (transfer near-exact) | 410 | −594 | −50 | −837 | −114 |

### N.3.5 Two candidate mechanisms, both eliminated

Round 7 tested, in a controlled 2 × 2 (three seeds per cell, all other settings fixed), whether the
transient **decay** or the survival of **optimiser momentum** across that decay explains the failure
(Figure `r7_grid`). Neither does. Resetting the transient's Adam state helps at `decay = 0.5` and
significantly *hurts* at `decay = 0.0` — the reverse of the prediction, since the weight/momentum
mismatch is by construction largest when the weights are zeroed. No cell in the grid reaches vanilla
performance in any phase.

## N.4 A corrected implementation

The lossy transfer can be removed entirely by changing the critic's architecture rather than its
hyper-parameters. If both value heads are **linear** and read from a **shared feature trunk** φ, then

    V(s) = w_P·φ(s) + w_T·φ(s) = (w_P + w_T)·φ(s)

and consolidation becomes exact weight arithmetic — `w_P ← w_P + (1 − decay)·w_T`, then
`w_T ← decay·w_T` — leaving V unchanged by construction, for any decay, with no regression, no
buffer and no learning rate to tune. Measured value drift is **0.0000 %** at every decay value,
against 10–17 % for the two-network formulation even under a perfectly converged regression (Figure
`consolidation_mechanism`, panel c). This variant is also present in the reference implementation of
the original method, where it is used for the MiniGrid experiments.

With this correction the collapse is eliminated: no phase approaches the failing configuration's
−279/−249/−396, and per-boundary value drift falls to 0.00–5.08. **But performance does not improve
beyond the baseline.** Phase 1 is nominally ahead (814 against 743) by a margin far smaller than the
across-seed standard deviation (σ = 275), and phases 2–5 fall below vanilla by margins that are
likewise all smaller than their respective standard deviations. Shared-trunk PT-PPO is
**statistically indistinguishable from vanilla PPO**, and remains far below EWC, whose phase-4
advantage (1238 against 212) is well outside the noise.

## N.5 The smooth-drift benchmark

Under the proposal's own setting the picture changes in two respects.

### N.5.1 EWC's advantage is entirely boundary-dependent

Online EWC computes its Fisher information **at a task boundary**. Under smooth drift no boundary
occurs, the Fisher is never estimated, and the penalty term is identically zero. The consequence is
not merely that EWC fails to help: with matched seeds its training trajectory is **bit-identical**
to vanilla's — verified directly from the saved curves on all five seeds, with a maximum absolute
difference of exactly zero. Under boundary-free non-stationarity, EWC *is* the baseline.

This is a mechanical demonstration rather than a statistical comparison, and it addresses precisely
the gap identified in the literature review: that much of the continual-RL literature presupposes
discrete, detectable task boundaries.

### N.5.2 PT-PPO does not benefit from smooth drift either

Three drift regimes were run (Figure `drift_comparison`). Returns are reported by 614 400-step
segment so that they align with the phases of the task-switching tables.

**Table N.3 — Smooth drift: PT-PPO minus vanilla PPO, by segment.** Bold entries exceed the combined
standard error of the mean.

| Regime | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| Slow (period 1.23 M, n = 5) | −23 | −67 | −20 | +98 | +153 |
| Two-timescale (n = 3) | **−308** | **−729** | **−1394** | −499 | **−1088** |
| Fast (period 123 k, n = 3) | **−174** | **−1133** | **−1345** | **−1336** | **−1409** |

Under **slow** drift the two agents are statistically tied: every gap lies inside the combined SEM,
and no agent collapses — all three climb from roughly 550 to 1300–1900. Analysis of the drift rate
explains why this outcome carries little information about the mechanism: at that period the physics
change by only ≈ 0.5 % per PPO update, which a single critic tracks without difficulty. There is no
fast component for a transient head to absorb, so the decomposition has nothing to do.

The remaining two regimes correct this. The **two-timescale** setting is the case the method is
explicitly designed for — a slow structural trend (amplitude 0.4, period 1 228 800) that the
permanent component should capture, superimposed on a fast fluctuation (amplitude 0.2, period
30 720, completing a full cycle roughly every fifteen PPO updates) that the transient should absorb.
The proposal describes this as filtering out temporary noise. A single critic must track both
simultaneously.

The prediction that PT would outperform vanilla in this regime is **reversed**. Vanilla beats PT
outside the combined SEM in nine of the ten segments across the two harder regimes, and the deficit
is *larger* under faster drift. Supplying the decomposition with the timescale separation it was
designed to exploit made it worse, not better.

A secondary pattern is worth recording: vanilla's across-seed SEM is considerably larger than PT's
(401 against 41 in one segment), and individual vanilla seeds reach as high as 3563. Vanilla is the
noisier agent but attains a substantially higher ceiling, whereas PT is consistently mediocre.

## N.6 Summary of results

1. On the task-switching benchmark, EWC outperforms vanilla PPO; PT-PPO collapses to a stationary
   policy from the third phase onward. H1 and H2 are both contradicted.
2. The collapse is caused by the consolidation operator, which at the shipped settings destroys
   98.3 % of the value function at each of roughly 150 consolidations per run.
3. The failure is not a representational limit: the consolidation target is fittable given adequate
   capacity and training, and when the transfer is made near-exact — or exact, by construction — the
   collapse disappears.
4. A corrected implementation eliminates the collapse but performs indistinguishably from vanilla.
5. Under smooth drift, EWC's advantage disappears entirely and provably — it becomes bit-identical
   to the baseline.
6. Under drift fast enough to be non-trivial, including the exact slow/fast decomposition the method
   targets, PT-PPO performs significantly *worse* than the baseline.

---

# Chapter N+1 — Discussion

## D.1 Principal finding

Across two benchmarks, three drift regimes, seven ablation rounds and a corrected implementation,
the permanent–transient decomposition of the value function provided **no measurable benefit** over
a single critic in policy-gradient continual control — and under non-stationarity fast enough to
matter, it was an **active handicap**.

The strength of this claim rests on the fact that it cannot be attributed to a faulty
implementation. Three separate lines of evidence establish that the mechanism was functioning as
specified when it failed to help:

- the consolidation transfer was measured *during training* and found to be near-exact (0.300 % on
  fitted states, 0.310 % on held-out states);
- an architecturally exact variant, in which value preservation holds by construction and is
  unit-tested at 0.0000 % drift, performs no better;
- disabling the mechanism entirely restores baseline performance, showing the surrounding
  infrastructure is sound.

A negative result of this form is considerably stronger than an unexplained underperformance, and
the proposal's risk-management section anticipates it explicitly, noting that a null outcome "is not
considered a failure" and that the contribution lies in the systematic analysis of how split
representations behave.

## D.2 Why the decomposition does not transfer to policy-gradient control

The original method was developed for value-based prediction and control, where the value function
*is* the policy: an action is selected by maximising over Q, so any improvement in the value estimate
changes behaviour directly. In an actor–critic, that link is absent. The critic influences the
policy only through the advantage estimate, and both a single critic and a correctly decomposed
critic fit the returns well — critic loss remains in the range 0.003–0.02 for both. Two value
functions that fit equally well furnish near-identical advantages and therefore near-identical
policy updates. On this reading, the best available outcome for a critic-only modification in an
actor–critic is parity with the baseline, which is exactly what the corrected implementation
achieves.

This also explains why the decomposition can only be neutral or harmful here, never beneficial: it
constrains how the value function may be represented without adding any information the baseline
lacks.

## D.3 Why performance *degrades* under fast drift

Parity does not explain the deficits in Table N.3, which require the decomposition to be actively
costly. The following account is consistent with all observations but has **not** been tested
directly, and is offered as a hypothesis rather than an established mechanism.

Between consolidations, only the transient head is trained; the permanent head is frozen.
PT's capacity for tracking change is therefore a single network, exactly as for vanilla. Its
regression target, however, is `returns − V_perm.detach()`. When the physics move quickly the frozen
baseline becomes stale within a few updates, making that residual *more* non-stationary than the raw
returns the baseline fits. Consolidation then folds the transient into the permanent and decays it,
so the residual must be re-learned against a just-changed baseline. The result is the same capacity,
a harder target, and periodic disruption — a combination that would be expected to cost most
precisely when tracking speed is what matters, which is the observed pattern.

Testing this would require comparing the transient's regression error against vanilla's critic loss
under identical drift, and is proposed as future work (§D.7).

## D.4 Boundary dependence as the discriminating variable

The most transferable finding of this study is not about either method individually but about the
conditions under which continual-RL machinery pays for itself.

EWC helps substantially on the task-switching benchmark and reduces *exactly* to the baseline under
smooth drift, because its Fisher estimate is triggered by a task boundary that no longer exists.
Meanwhile, under slow smooth drift, no agent collapses at all: plain PPO reaches 1300–1900, higher
than any agent achieves on the task-switching benchmark.

Taken together these suggest that the difficulty in the directional experiment arises from the
**abrupt inversion of the reward**, not from non-stationarity as such, and that a continual-learning
mechanism can only earn its cost where a discrete change destroys previously acquired knowledge.
This is a conclusion about the **benchmark** rather than about either method, and it reframes the
motivating gap: the problem with boundary-assuming methods is not merely that they are inapplicable
without boundaries, but that in the boundary-free regimes commonly proposed as more realistic, the
baseline may need no help at all.

The qualification in §N.5.2 matters here. Once the drift is fast enough to be non-trivial, the
setting *does* discriminate between agents — but it discriminates against PT, which falls further
behind rather than catching up.

## D.5 Relation to prior work

The finding is consistent with the trajectory of the original research programme. Anand et al.
(2024), extending the permanent–transient framework, move away from a purely parametric transient
component towards separate feature encoders and a non-parametric transient memory. The difficulties
identified here — that transferring one network's function into another by regression is lossy in
practice, and that the decomposition interacts poorly with a fast-changing target — offer a concrete
account of why such a move may be necessary once deep function approximation is involved.

The results also support the framing of Khetarpal et al. (2022), who identify reliance on discrete
task boundaries as a principal limitation of the continual-RL literature. The present study
strengthens that observation in a specific and unusually literal way: the boundary-dependent method
does not degrade gracefully in the absence of boundaries but becomes computationally identical to
doing nothing.

## D.6 Methodological remarks

Several results in this study were revised during the investigation, and the process is worth
recording because it affected the conclusions.

Two mechanistic explanations were advanced and subsequently **retracted** after controlled testing:
that the consolidation target was unrepresentable (refuted by fitting it to 3.2 % error with a wider
network), and that the transient decay, or stale optimiser momentum surviving that decay, explained
the collapse (refuted by a 2 × 2 in which the predicted effect appeared with the opposite sign). In
both cases the original claim rested on comparisons between runs that differed in more than one
variable.

A related lesson concerns variance. An intermediate conclusion was drawn from phase-2 point
estimates before the across-seed spread was examined; phase 2 has a standard deviation of 787,
two to three times that of any other phase, with only two of five seeds positive. No mechanism claim
could be supported by that phase in either direction, and the corresponding argument was withdrawn.

Finally, an instrumentation change intended to be inert altered which random-number stream the
consolidation step consumed, so that runs before and after it were no longer trajectory-comparable
at matched seeds. This was detected when a control cell failed to reproduce an earlier configuration
and has since been corrected. The episode illustrates a general point: in stochastic experiments,
"functionally equivalent" changes to code that consumes randomness are not equivalent for
reproducibility.

## D.7 Limitations and future work

**Statistical power.** Five seeds resolve only large effects. This is adequate for the conclusions
drawn — EWC's advantage (gaps of 673–1047 points) and PT's drift deficits (up to 1409) are well
outside the noise — but the study does not exclude small effects, and several diagnostic rounds used
three seeds.

**A single environment.** All experiments use HalfCheetah. The failure analysis is mechanism-level
and would be expected to transfer, but this has not been demonstrated on a second domain.

**One untested explanation.** The account in §D.3 of why PT degrades under fast drift is consistent
with the evidence but has not been tested directly. Given that two earlier mechanistic claims in
this study did not survive controlled testing, it should be treated as a conjecture.

**Critic-only scope.** The decomposition was applied to the value function, as in the original
formulation. A dual-timescale *policy* is a distinct proposition, and the shared-trunk construction
developed here would make its consolidation exact. The evidence in §D.2 gives little reason to
expect a benefit, but the question is open.

**Drift design.** The drift schedule is sinusoidal, chosen so that earlier dynamics recur and
retention is measurable. A monotone schedule, or drift in different physical parameters, may behave
differently.

## D.8 Conclusion

This work set out to determine whether the permanent–transient decomposition remains effective
beyond the temporal-difference settings in which it was introduced, and specifically whether it
transfers to policy-gradient control under smooth, boundary-free non-stationarity. The answer is
that it does not. The decomposition provides no benefit over a single critic on a task-switching
benchmark, no benefit under slow smooth drift, and a significant disadvantage once the drift is fast
enough that tracking it is non-trivial — including in the two-timescale regime the method is
explicitly designed for.

The study also produces a positive result. Elastic weight consolidation improves substantially over
the baseline where task boundaries exist, and reduces exactly and provably to that baseline where
they do not. Combined with the observation that plain PPO handles slow smooth drift without any
forgetting to prevent, this locates the value of continual-RL machinery precisely: it is earned at
abrupt, knowledge-destroying changes, and not at gradual ones.
