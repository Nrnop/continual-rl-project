# Permanent–Transient Representations for Continual RL in Continuous Control
## Progress report and findings

**Author:** Nura Nabipour (BIHE) · **Supervisors:** Mr. Shahrad Mohamadzadeh, Mr. Ehsan Es'haghi
**Codebase:** `src_continuous_control` · **Status of this document:** all experiments complete.

---

## 1. Executive summary

We ported the Permanent–Transient (PT) value decomposition of Anand & Precup (2023) from value-based
prediction to a **PPO actor–critic on MuJoCo HalfCheetah** under continual non-stationarity, and
benchmarked it against a vanilla PPO baseline and an Online-EWC baseline (3 agents × 5 seeds ×
3.07 M steps).

Eight results:

1. **PT fails in this setting** — it collapses to a do-nothing standstill policy from the third task
   phase onward (phase-3 mean return **−279** vs vanilla **+243**).
2. **We identified the exact cause:** the consolidation operator destroys most of the value function
   every *k* updates (~150× per run). Two defects compound. The transfer is a regression — the
   permanent critic must *learn* `old_V_perm + V_trans` — and at `lr_perm = 1e-5` with one epoch
   (320 SGD steps) it does not descend at all: measured over a full run, the loss at the last
   gradient step of a cycle equals the loss at the first (**1.0×** reduction). Meanwhile the decay
   removes far more than intended: at the configured `decay = 0.5` the transient's output norm falls
   to **16.6 %**, not 50 %, because scaling an MLP's *parameters* does not scale its *output*
   proportionally. Both are confirmed in situ over 3.07 M steps and 3 seeds (§6.1). See §6.3 for
   what is and is not a fundamental limit here.
3. **We proved causation**: disabling consolidation entirely reverses the collapse
   (phase-3 mean **−279 → +291**).
4. **We built and validated a fix** — a shared trunk with linear heads makes consolidation
   mathematically exact (0.0000 % value drift, confirmed in production by near-zero boundary drift).
   **The collapse is eliminated.** But the repaired mechanism performs **statistically
   indistinguishably from vanilla PPO** (every phase difference smaller than that phase's own
   standard deviation).
5. **EWC is the positive result** — it beats both, decisively and outside noise (phase-4 mean
   **1238** vs vanilla **375** and PT **212**; gap ≈ 1026 against σ ≈ 399).

6. **Consolidation fidelity is not the problem, and no finer mechanism was found** (§5.5–5.7).
   Training the regression properly makes PT no better; a held-out measurement shows the transfer is
   near-exact in situ (~0.3 % value drift on *both* fitted and unseen states) while PT still
   collapses. Two candidate mechanisms — the transient decay, and stale optimiser momentum surviving
   that decay — were then tested in a controlled 2×2 and **both failed**, the second producing the
   opposite of its prediction. Phase 2, on which an earlier draft based a decay argument, turns out
   to have sd 787 across seeds (2–3× every other phase) and cannot support such an argument at all;
   that claim is retracted.

7. **Under smooth Lipschitz drift — the proposal's own setting — EWC's advantage vanishes
   entirely** (§8). With a fixed reward and continuously drifting physics, **EWC becomes
   byte-identical to vanilla**, because with no task boundary its Fisher matrix is never computed.
   No agent collapses; under *slow* drift all three are statistically tied.
8. **Under drift fast enough to matter, PT is actively worse than the baseline** (§8.1). Given the
   slow-trend-plus-fast-fluctuation decomposition it was explicitly designed to exploit, vanilla
   beats PT **outside the combined SEM in 9 of 10 segments**, and the deficit *grows* with more
   fast-timescale content. The prediction that PT would win here was not merely unconfirmed but
   reversed — the strongest single piece of evidence in the study against the mechanism.

**The central finding.** Value decomposition of the critic confers **no benefit** over a single
critic in policy-gradient continual control, and under non-stationarity fast enough to matter it is
an active handicap. This is established in its strongest form: the mechanism can be shown to be
functioning *correctly* (consolidation is near-exact in situ) when it fails to help, so the result
cannot be dismissed as an implementation defect. It holds under discrete task switching, under slow
smooth drift, and — most tellingly — under the explicit slow/fast decomposition the method was
designed for.

**The positive finding.** Regularisation-based continual learning (EWC) is substantially superior —
**but only where discrete task boundaries exist.** Under boundary-free drift it has no mechanism
left and reduces exactly to the baseline. Taken together with the observation that *nothing*
collapses under smooth drift, the study's sharpest conclusion is about the **benchmark rather than
the method**: continual-RL machinery pays for itself only when an abrupt change destroys prior
knowledge. Smooth dynamics drift of this magnitude does not, so there is nothing for it to prevent.

**Note on scope.** The proposal specifies smooth, Lipschitz-continuous *dynamics* drift; the main
experiments use **discrete directional task-switching** (the forward-velocity reward term flips sign
every 614 400 steps). Both are now implemented and reported — the task-switching study in §4–§7 and
the drift study in §8 — so the thesis covers the proposed setting as well as the one that turned out
to be the discriminating one.

---

## 2. Experimental setup

| Item | Value |
|---|---|
| Environments | **`DirectionalHalfCheetah`** — the *reward's* forward-velocity term flips sign at discrete boundaries (§4–§7); **`LipschitzDriftHalfCheetah`** — the reward is fixed and the *physics* drift continuously, with no boundaries (§8) |
| Non-stationarity (switching) | 4 switches / 5 phases, direction sequence `+1, −1, +1, −1, +1`, switch interval 614 400 steps |
| Non-stationarity (drift) | joint damping + ground friction rescaled by a smooth multiplier; three regimes — slow (period 1.23 M), fast (period 123 k) and two-timescale (slow trend + fast fluctuation). Measured Lipschitz bound reported at run time |
| Horizon | 3 072 000 env steps per run |
| Agents | `vanilla` (single critic), `pt` (split critic), `ewc` (online diagonal-Fisher penalty) |
| Seeds | 0–4 (5 seeds per agent) |
| PPO recipe | 8 async envs × 256 steps (batch 2048), 10 epochs, minibatch 64, γ=0.99, λ=0.95, clip 0.2, LR anneal, obs+reward normalisation |
| Networks | `[64, 64]` MLPs; **all three agents share an identical `GaussianActor`** |
| Compute | vast.ai CPU instances (workload is MuJoCo-bound; GPU gives no benefit) |

All three agents share the PPO core, the actor architecture, the environment, and the seeds. **Only
the critic (or the regulariser) differs**, so any performance difference is attributable to the
mechanism under study.

---

## 3. Infrastructure corrections (prerequisite work)

Before any comparison was meaningful, three classes of defect had to be fixed. These affected
**all** agents, including the baselines.

### 3.1 PPO returned negative reward on HalfCheetah
The environment was constructed without normalisation, and the default update path was a per-step
online update rather than batch PPO. HalfCheetah is a standard case where PPO only reaches its
benchmark with the full recipe. Corrections applied:

- running-mean/std **observation normalisation** (clipped ±10)
- discounted-return **reward normalisation** (clipped ±10)
- **vectorised batch PPO** with GAE over 8 parallel envs
- **Adam ε = 1e-5** and **linear LR annealing**
- episodic returns logged from `RecordEpisodeStatistics` *before* normalisation, so reported returns
  are the true, un-normalised values
- evaluation uses the *training* normaliser's frozen statistics (avoids a distribution shift)

**Validation:** single-task runs (no switching, 500 k steps, 2 seeds) now reach **+800 – 1150**
return with a small, stable critic loss — the expected HalfCheetah learning curve.

### 3.2 Test suite repaired
The unit tests had not been updated after an earlier vectorisation refactor. Nine tests were failing
against the current code. One was a **genuine production bug**: `_online_step_update` still passed a
1-D observation to a `get_value` that expected a batch. Fixed, and the suite updated to the
vectorised buffer signature. **15/15 passing** at that point; the suite has since grown to
**31 tests** as each new mechanism and environment was added.

### 3.3 PT algorithm corrections
Three defects in the PT agent were fixed before the main sweep:

1. **Value inflation.** Consolidation regressed `V_perm → old_V_perm + V_trans` and *then* decayed the
   transient, which inflated the acting value by `decay·V_trans` every cycle. Corrected to
   `old_V_perm + (1−decay)·V_trans`, restoring the intended value-preserving identity.
   *(Later found to be insufficient — see §6.1.)*
2. **`consolidate_on_switch` was ignored.** The boundary handler only decayed the transient without
   absorbing it first. Corrected to consolidate-then-decay.
   *(Later found to be **inert** — see §5.1.)*
3. **PT made critic-only.** The agent used a `SplitActor` (`mean = perm_mean + trans_mean`) that had
   **no consolidation mechanism**: `perm_mean` sat at its random initialisation (LR 1e-5) while
   `trans_mean` did all the learning and was decayed at every boundary — partially wiping the policy
   each switch. This also contradicted the paper (PT is defined on the value function) and broke the
   apples-to-apples comparison. Removed; PT now uses the same `GaussianActor` as the baselines.

---

## 4. Main results (3 agents × 5 seeds × 3.07 M steps)

**Per-phase mean return** (primary metric — lower variance than the end-of-phase value), with the
number of seeds achieving a positive mean:

| Agent | P1 (+) | P2 (−) | P3 (+) | P4 (−) | P5 (+) |
|---|---|---|---|---|---|
| **vanilla** | 743 (5/5) | 468 (4/5) | 243 (3/5) | 375 (3/5) | −34 (2/5) |
| **EWC** | 743 (5/5) | 517 (3/5) | **705** (4/5) | **1238** (5/5) | **533** (5/5) |
| **PT** | 475 (5/5) | 332 (2/5) | **−279** (1/5) | −249 (2/5) | −396 (0/5) |

End-of-phase returns show the same ordering (vanilla 1524/1350/950/1425/962; EWC
1524/1545/1453/2739/1785; PT 1124/1456/−100/−21/−346).

![Figure 4.1 — Per-phase mean return, three agents, 5 seeds. Error bars are SEM.](plots/figures/phase_means_main.png)

*Figure 4.1 — Per-phase mean return, three agents, 5 seeds. Error bars are SEM.*

![Figure 4.2 — Return over training with task boundaries marked. PT (blue) never recovers after the second switch.](plots/figures/return_curves.png)

*Figure 4.2 — Return over training with task boundaries marked. PT (blue) never recovers after the second switch.*

![Figure 4.3 — Mean x-velocity. PT's cheetah stops moving from phase 3 onward, while vanilla and EWC keep reversing direction each phase.](plots/figures/velocity_curves.png)

*Figure 4.3 — Mean x-velocity. PT's cheetah stops moving from phase 3 onward, while vanilla and EWC keep reversing direction each phase.*


> **Data source.** All per-phase means are computed from `results/*_returns.pkl`, which records one
> point per PPO update (2 048 env steps → 1 500 points per run). The exported CSVs in
> `numeric_logs_csv/` are a **10× subsample** of the same runs (one point per 20 480 steps, first
> sample at 20 480) and therefore give phase means differing by ~2–3 %: they omit the earliest — and
> lowest — points of each phase window, biasing the means away from zero. The full-resolution `.pkl`
> values are used throughout this report and in the figures. End-of-phase values are identical in
> both, since the phase boundaries fall on both sampling grids.

**Observations**

- **EWC is the clear winner** — the only agent with a positive mean in every phase, and the only one
  positive across all 5 seeds in phases 4 and 5.
- **PT tracks the baselines for two phases, then collapses** and never recovers. The velocity trace
  confirms the failure mode physically: PT's cheetah **stops moving** (mean x-velocity ≈ 0) from
  phase 3 onward, while vanilla and EWC keep swinging to ±1.5–3.3 to chase each direction.
- **Boundary metrics** (the thesis's stability measure): PT's mean relative task-switch return drop
  is **~117 %** with very large variance, versus a tight **52–55 %** for vanilla and EWC. **H1 and H2
  are refuted** — as implemented PT is *worse* at boundaries, and §7 shows that repairing the
  mechanism raises it only to parity with vanilla, never above it. The dual-timescale agent does not
  outperform the single-timescale baseline (H1), and the permanent component does not prevent
  degradation (H2); the regularisation baseline does both.
- **Offline zero-momentum evaluation** (momentum-free adaptation probe): PT *degrades over training*
  (+400 → negative); EWC climbs to ~1270; vanilla holds ~300–770.

![Figure 4.4 — Relative return drop at a task switch. PT's ~117% with very large variance against a tight 52–55% for the baselines.](plots/figures/boundary_drop.png)

*Figure 4.4 — Relative return drop at a task switch. PT's ~117% with very large variance against a tight 52–55% for the baselines.*

![Figure 4.5 — Zero-momentum offline evaluation, which removes the carried-momentum confound at a direction reversal.](plots/figures/offline_curves.png)

*Figure 4.5 — Zero-momentum offline evaluation, which removes the carried-momentum confound at a direction reversal.*

![Figure 4.6 — Rollouts required to recover 90% of the previous phase's peak.](plots/figures/recovery_time.png)

*Figure 4.6 — Rollouts required to recover 90% of the previous phase's peak.*

![Figure 4.7 — Asymptotic versus online cumulative return.](plots/figures/asymptotic_bar.png)

*Figure 4.7 — Asymptotic versus online cumulative return.*

![Figure 4.8 — Critic loss. Note that PT's stays small (~0.01) throughout, which is why the failure evaded this diagnostic (§6.2).](plots/figures/td_error_curves.png)

*Figure 4.8 — Critic loss. Note that PT's stays small (~0.01) throughout, which is why the failure evaded this diagnostic (§6.2).*

- **Sanity check passed:** vanilla and EWC are identical in phase 1 (743) — correct, since EWC's
  penalty is inactive until the first task switch.
- **Critic loss stayed low (~0.01) for PT throughout.** This is important: the failure is **not**
  value divergence, and this metric actively *concealed* the fault for three ablation rounds (§6.2).

---

## 5. Ablation programme

Four rounds, all at 5 seeds and the full 3.07 M horizon unless noted.

### 5.1 Round 1 — permanent LR and switch-time decay *(inconclusive; two methodological traps found)*

| Variant | Phase-3 result (end-of-phase, per seed) |
|---|---|
| baseline | −225 / +1501 (2 seeds) |
| no-switch-decay | **identical to baseline, bit-for-bit** |
| unfreeze-perm (`lr_perm` 1e-5 → 1e-3) | +811 / +260 |

*(This round predates the switch to per-phase means as the primary metric; its figures are
end-of-phase values on a 2-seed, shortened-horizon run, and are reported here only to document the
two methodological traps below. They are not comparable to the per-phase means used elsewhere.)*

Two traps discovered, both important for future work:

- **The "no-switch-decay" arm tested nothing.** `k=10` divides `updates_per_switch` (614 400/2048 =
  300) exactly, so the periodic consolidation always lands on the boundary and empties the
  consolidation buffer; the boundary consolidation then hits `len(buffer)==0` and **returns
  immediately**. Corollary: **fix 3.3.2 above is inert in every configuration run so far.** Testing
  switch-time behaviour requires a `k` that does not divide 300 evenly (e.g. 7).
- **Shortening `total_steps` invalidates the comparison.** `num_updates = total_steps/batch` drives
  the LR anneal, so a 3-phase run anneals the LR to zero exactly where phase-3 recovery is measured.
  Evidence: seed 1 gave **−365** in the full sweep but **+1501** in the shortened run. **All later
  ablations use the full horizon.**

Re-analysing round 1 across *all* phases, `unfreeze-perm` was in fact **worse on the mean at every
phase** (P1 903 vs 1112, P2 −498 vs 171, P3 535 vs 638); it "won" only a 2-seed sign count.

### 5.2 Round 2 — faster transient (adaptation speed)

Motivated by the observation that `lr_trans` (3e-4) equalled vanilla's `lr_critic` (3e-4) — meaning
PT had **no fast timescale at all** relative to the baseline.

Per-phase mean return (seeds with a positive mean in brackets):

| Variant | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| PT baseline | 475 (5/5) | 332 (2/5) | −279 (1/5) | −249 (2/5) | −396 (0/5) |
| `lr_trans = 1e-3` | 691 (5/5) | 134 (3/5) | **−192** (1/5) | **−434** (0/5) | −238 (0/5) |
| `lr_trans = 3e-3` | 795 (5/5) | −215 (1/5) | **−54** (2/5) | **−553** (0/5) | −53 (2/5) |

Phase 3 improves over the baseline (−279 → −192 → −54) but **never becomes positive**, and both
variants are **worse than the baseline at phase 4** (−434 and −553 vs −249). Adaptation *speed* is
not the lever.

### 5.3 Round 3 — `decay = 0` (after finding the decay bug)

| Variant | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| `decay=0` | 394 | −319 | −0.2 | −348 | −213 |
| `decay=0`, `lr_trans=1e-3` | 488 | 30 | 145 | −625 | 72 |

No fix. Phase 4 still collapses. (Apparent phase-3/5 gains in the second variant were driven by a
single oscillating seed.)

### 5.4 Round 4 — consolidation disabled entirely *(the causal control)*

| Agent | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| PT (as implemented) | 475 | 332 | **−279** | −249 | −396 |
| **PT, consolidation off** | 641 | 1039 | **+291** | 746 | −373 |
| vanilla (reference) | 743 | 468 | 243 | 375 | −34 |

**This is the decisive result.** Removing consolidation reverses the collapse and returns PT to
vanilla-level performance — exactly as predicted, because with consolidation disabled `V_perm` is a
frozen random network and PT degenerates into "vanilla plus a fixed offset".


![Figure 5.1 — PT variants ordered by how much consolidation regression actually runs. Removing the regression entirely (violet, red) performs best; training it well (yellow) performs worst.](plots/figures/phase_means_ablation.png)

*Figure 5.1 — PT variants ordered by how much consolidation regression actually runs. Removing the regression entirely (violet, red) performs best; training it well (yellow) performs worst.*

*Interpretation discipline:* differences between `PT-no-consolidation` and vanilla in phases 2–4 sit
**within seed noise** (σ = 350–680, n = 5). The defensible claim is **"removing consolidation restores
vanilla-level performance"**, not "PT beats vanilla". The phase-3 *reversal* (−279 → +291, consistent
across 4/5 seeds) is large relative to that noise and is treated as real.

*Phase-5 decline is generic, not PT-specific:* vanilla (375 → −34) and EWC (1238 → 533) decline in
phase 5 as well, consistent with the near-fully-annealed learning rate late in training.

### 5.5 Round 5 — separate trunks with a *properly trained* consolidation regression

Rounds 1–4 and §7 each break one half of the PT idea: the permanent critic must both **accumulate**
knowledge and stay **insulated** from the fast timescale. Consolidation-off is insulated but learns
nothing; the shared trunk accumulates exactly but its features move at the fast rate; the shipped
separate-trunk config has both properties in principle but never actually trains its regression
(0.05 % transfer). This round runs the missing configuration: **separate trunks, Adam,
`lr_perm = 1e-3`, `consolidation_epochs = 20`** (6 400 gradient steps per consolidation instead of
320), `decay = 0`, full horizon, 5 seeds.

| Variant | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| shared trunk (no regression at all) | 814 | 394 | 27 | 212 | −176 |
| shipped (regression barely runs) | 475 | 332 | −279 | −249 | −396 |
| **trained consolidation (this round)** | **410** | **−594** | **−50** | **−837** | **−114** |

SEMs 26 / 47 / 106 / 162 / 109; seeds-positive 5/5, 0/5, 1/5, 0/5, 1/5.

**Training the regression made PT worse, not better** — and decisively so (the phase-2 gap alone is
~900 points against a 47-point SEM). The collapse also arrives **one phase earlier**: this variant is
already deeply negative at P2, the very first switch, whereas the shipped version survives to P3.

This closes off "the consolidation was never trained" as the explanation for the original collapse.

> **Correction.** An earlier version of this section read the variants as an ordering by *how much
> consolidation regression happens* and concluded "more regression, worse performance". **That
> comparison was confounded.** The shipped config and this round differ in **two** ways — the amount
> of regression training *and* the decay — and the collapse tracks the **decay**:
>
> | variant | decay | consolidation training | P2 |
> |---|---|---|---|
> | shipped | 0.5 | barely runs | **+332** |
> | shared trunk | 0.5 | none (exact) | **+394** |
> | fast transient | 0.5 | barely runs | **+134** |
> | `decay=0` (§5.3) | **0.0** | barely runs | **−319** |
> | `decay=0` + fast transient | **0.0** | barely runs | **+30** |
> | trained consolidation (this round) | **0.0** | fits well | **−594** |
>
> Every `decay = 0.5` run is positive at phase 2 and every `decay = 0.0` run is not — and crucially
> `decay=0` already collapses at P2 with *untrained* consolidation. So the amount of regression is
> not what separates them.
>
> **⚠ This grouping is itself retracted — see §5.7.** A controlled 2×2 varying decay alone found the
> opposite ordering at phase 2, and the shipped run's phase-2 values have sd 787 across seeds
> (2–3× every other phase, only 2/5 seeds positive), so phase-2 point estimates cannot support a
> mechanism claim in either direction. What both readings share, and what does survive, is the
> narrower statement above: the amount of consolidation training is not what distinguishes these
> runs.

**The in-situ measurement, and the puzzle it creates.** This round logs
`train/consolidation_error_pct`: the % change in the acting value across each consolidation,
measured on the states being consolidated. It falls to essentially zero:

| training stage | consolidation error (fitted states) |
|---|---|
| first cycle | ~2.5–3.0 % |
| mid-training | ~0.3–0.7 % |
| final phase | **~0.00–0.07 %** |

So the regression *does* fit, and the transfer *is* value-preserving — **on the states it trained
on**. Yet performance is the worst of any variant. Panel (a) of
`plots/figures/consolidation_insitu.png` plots this directly from the run's own logs: the drift
falls from ~3 % to ~0.006 % over training, on every seed.

**Leading explanation at the time (since falsified — see §5.6).** The metric above is
in-distribution only. With Adam, `lr_perm = 1e-3` and 6 400 gradient steps on a `[64,64]` net over a
20 480-state buffer, the permanent net plausibly **memorises the buffer** and extrapolates badly onto
the *new* states the policy visits immediately afterwards — worst right after a switch, when the
state distribution has just moved, which matches the collapse arriving one phase earlier. Holding
out 20 % of the buffer from the regression and measuring drift on it separately reproduces exactly
this signature offline:

| consolidation epochs | error on fitted states | error on held-out states |
|---|---|---|
| 1 | 89.5 % | 89.6 % |
| 20 | 41.3 % | 55.8 % |
| 60 | **24.2 %** | **53.4 %** |

More training improves the fitted states while held-out error barely moves — the gap widens from
~0 to ~29 points. `configs/abl_pt_consol_holdout.yaml` runs this measurement *during training* to
confirm it in situ (3 seeds; it measures a mechanism, not a return difference). Until that lands,
this explanation is **suggestive rather than established** — the round-5 result itself (more
regression ⇒ worse performance) is what stands on the evidence.

### 5.6 Round 6 — the memorisation hypothesis is falsified, and the decay comes into focus

§5.5 proposed that the permanent net *memorises* the consolidation buffer: near-perfect on the
states it fits, badly wrong on the new states visited next. Round 6 tested it by excluding 20 % of
the buffer from the regression and measuring value drift on that held-out portion separately
(3 seeds, full horizon).

| seed | mean error, fitted states | mean error, held-out states | max gap |
|---|---|---|---|
| 0 | 0.31 % | 0.32 % | +0.25 % |
| 1 | 0.28 % | 0.29 % | +0.22 % |
| 2 | 0.30 % | 0.31 % | +0.24 % |

**The two track each other almost exactly, everywhere in training, including immediately after a
switch, and both converge to ~0.00 % by the end.** No gap ever opens — panel (b) of
`plots/figures/consolidation_insitu.png` shows the two curves lying on top of one another for the
whole run. Measured over every logged consolidation: fitted 0.300 %, held-out 0.310 %. **The memorisation hypothesis
is wrong**, and the offline probe's prediction (a gap widening with training effort) does not
reproduce in situ.

**An important limitation of this test, however.** The held-out portion is a *random* 20 % of the
same 20 480-state buffer, so both halves are drawn from the identical recent on-policy distribution.
It therefore establishes that the permanent net **interpolates within its own recent state
distribution** rather than memorising it — but it does *not* test extrapolation to the states of the
**next** rollout, which are temporally later and, right after a switch, drawn from a shifted
distribution. That stricter question remains open (§10h).


![Figure 5.2 — In-situ consolidation quality from the runs' own logs. (a) Round 5: the regression converges to a near-exact transfer. (b) Round 6: held-out error tracks fitted error everywhere, falsifying the memorisation hypothesis.](plots/figures/consolidation_insitu.png)

*Figure 5.2 — In-situ consolidation quality from the runs' own logs. (a) Round 5: the regression converges to a near-exact transfer. (b) Round 6: held-out error tracks fitted error everywhere, falsifying the memorisation hypothesis.*

**What this leaves.** Consolidation demonstrably works: the transfer is near-exact, on fitted and
unseen states alike, throughout training — and PT collapses anyway. The failure is therefore *not*
in the transfer. Combined with the decay grouping in §5.5, attention moves to what happens
**immediately after** consolidation.

**Leading hypothesis (untested).** `decay_transient` scales the transient's *parameters*
(`p.data.mul_(decay)`) but leaves the **optimiser state untouched** — Adam's `exp_avg` and
`exp_avg_sq` for those parameters survive the reset. So after `θ_T ← 0` the next optimiser step
displaces the zeroed weights by roughly `lr · exp_avg / √exp_avg_sq`, driven by momentum from a
network that no longer exists. Consolidation makes `V_perm` and `V_trans` mutually consistent, and
one update later that consistency is destroyed. This predicts exactly the observed pattern: `decay = 0`
maximises the weight/state mismatch and collapses; `decay = 0.5` halves it and survives.

### 5.7 Round 7 — both hypotheses fail, and phase 2 turns out to be uninterpretable

Round 7 ran the 2×2 — decay (0.0 vs 0.5) × resetting the transient's optimiser state on decay
(yes/no), 3 seeds per cell, everything else held fixed. Per-phase mean return ± SEM:

| | decay = 0.0 | decay = 0.5 |
|---|---|---|
| **no reset** | 369±2 / −164±73 / −69±71 / −369±37 / −299±4 | 485±10 / −567±94 / 243±52 / −619±171 / 40±62 |
| **reset optimiser state** | 34±44 / −504±64 / −232±130 / −632±92 / −204±125 | 446±13 / −161±34 / 321±92 / −827±168 / 196±33 |


![Figure 5.3 — Round 7: transient decay × resetting the transient's optimiser state. No cell reaches the vanilla reference (dashed). The reset helps at decay 0.5 and hurts at decay 0.0 — the reverse of the prediction.](plots/figures/r7_grid.png)

*Figure 5.3 — Round 7: transient decay × resetting the transient's optimiser state. No cell reaches the vanilla reference (dashed). The reset helps at decay 0.5 and hurts at decay 0.0 — the reverse of the prediction.*


**Both hypotheses are disconfirmed, and one runs backwards.**

1. **The stale-momentum mechanism is wrong.** It predicts the reset should help *most* at
   `decay = 0`, where zeroed weights and full-scale momentum are maximally mismatched. The opposite
   happens: at `decay = 0` the reset makes P1, P2 and P4 significantly *worse* (P2: −164 → −504,
   gap ~340 against a combined SEM of ~137), while it helps only at `decay = 0.5`
   (P2: −567 → −161). That is the reverse of the prediction — a positive disconfirmation, not a
   null.
2. **The decay grouping of §5.5 does not survive a controlled test.** In this un-confounded
   comparison `decay = 0.0` is *better* than `decay = 0.5` at phase 2 (−164 vs −567), the very
   phase the grouping was built on.
3. **Nothing beats vanilla** (743/468/243/375/−34) in any cell, in any phase.

**Why §5.5's grouping was unsound — phase 2 cannot carry an argument.** The shipped PT run's own
phase-2 values across its five seeds are **+800, +1479, −210, −15, −393**: mean +332 but **sd 787**,
with only **2/5 seeds positive**. That standard deviation is 2–3× every other phase (P1 71, P3 338,
P4 312, P5 271), and the positive mean rests on two outlier seeds. Round 7's cell mean of −567 sits
~1.1 sd below it — comfortably inside noise. **The §5.5 decay grouping is therefore retracted**: it
read a difference between chaotic phase-2 point estimates as a mechanism.

**A reproducibility defect found in the process (now fixed).** The holdout instrumentation added for
round 6 was intended to be inert at `consolidation_holdout_frac = 0`, but it changed which RNG
stream consolidation consumes (`np.random.permutation` + `torch.randperm` in place of
`np.random.shuffle`). That shifts every downstream random draw, so same-seed runs before and after
it are **not** trajectory-comparable, and round 7's cells cannot be compared seed-by-seed against
the main sweep. The default path now reproduces the original RNG consumption exactly, and the
diagnostic path is explicitly documented as not seed-comparable. Round 7's *internal* comparisons
(all cells, same code, same session) are unaffected and remain valid.

**What survives.** PT's collapse in phases 3–5 is robust — those phases have sd 271–338 with means
clearly negative across every variant tested. What does *not* survive is any fine-grained causal
story pinned to phase 2. Neither decay nor optimiser state explains PT's failure.

---

## 6. Root cause

### 6.1 Two defects in the consolidation operator

**(a) The decay does not do what the algorithm assumes.** `decay_transient(d)` multiplies every
*parameter* of the transient network by `d`. For a nonlinear network (tanh + biases) this does **not**
scale the *output* `V_trans` by `d`:

| decay | measured error vs `decay · V_trans` |
|---|---|
| 0.00 | 0 % (exact) |
| 0.25 | ~66 % |
| 0.50 | ~33 % |
| 0.75 | ~13 % |

So the value-preserving identity holds **only at `decay = 0`**; at the configured `decay = 0.5` every
consolidation injected an uncontrolled perturbation into the acting value.

**(b) The transfer never happens, but the deletion always does.** Measured at the exact production
settings (`lr_perm = 1e-5`, SGD, 1 epoch over 20 480 states = 320 gradient steps):

```
transient magnitude the permanent must absorb : 2.8107
permanent movement actually achieved          : 0.0015   ->  0.05 % of the required transfer
transient remaining after decay               : 0.0000   ->  100 % deleted

net effect on the acting value V = V_perm + V_trans : 98.3 % destroyed
```

**The value function is annihilated every `k = 10` updates — roughly 150 times per run.** The next
rollout then collects bootstrap values from a gutted critic and feeds them to GAE, corrupting the
advantages for that entire rollout.


![Figure 6.1 — (a) One consolidation at the shipped settings: 0.1% of the transient absorbed, 100% deleted. (b) The target is fittable given capacity and training, but held-out error does not improve. (c) Parameter scaling is not output scaling — exact for any decay only with linear heads.](plots/figures/consolidation_mechanism.png)

*Figure 6.1 — **Offline measurement** of the real consolidation code on **synthetic (iid Gaussian)
probe states**, not on states the agent visited. (a) One consolidation at the shipped settings: 0.1 %
of the transient absorbed, 100 % deleted. (b) The target is fittable given capacity and training, but
held-out error does not improve. (c) Parameter scaling is not output scaling — exact for any decay
only with linear heads. The synthetic inputs make this a controlled probe rather than a measurement
of training; the in-situ counterparts on real rollout states are Figure 6.2 and §5.6, which agree
with (a) and (c) and are the numbers quoted in the executive summary.*

#### In-situ confirmation of both defects

The two defects above were originally measured offline, on synthetic probe states. They have since
been confirmed **during real training**, over the full 3 072 000-step horizon with three seeds, by
recording the consolidation regression's loss and the transient's magnitude either side of the decay
at every consolidation.

| Measured over a full run | Shipped config (`decay = 0.5`, SGD, 1 epoch) | Trained regression (`decay = 0`, Adam, 20 epochs) |
|---|---|---|
| ‖V_trans‖₂ ratio, after ÷ before decay | **0.166** | 0.0000 (exact) |
| …expected if the output scaled by `decay` | 0.5 | 0.0 |
| Within-cycle loss reduction (first ÷ last gradient step) | **1.0×** | **2 996×** |

**Defect (a), the decay.** With `decay = 0.5` the transient's output norm falls to **16.6 %** of its
prior value, not 50 % — consistent across seeds (0.167 / 0.168 / 0.163). Scaling an MLP's
*parameters* by one half shrinks its *output* by far more than half, so the decay removes roughly
83 % of the transient where the algorithm assumes it removes 50 %. This is the in-situ counterpart
of the offline measurement in §6.1(a). The `decay = 0` run confirms the boundary case: the ratio is
exactly 0, because zeroing every parameter does zero the output.

**Defect (b), the regression.** In the shipped configuration the loss at the *last* gradient step of
a consolidation equals the loss at the *first* — a reduction of **1.0×**, i.e. none at all. Panel (a)
of Figure 6.2 shows the three loss traces lying exactly on top of one another for the whole run: 320
SGD steps at `lr_perm = 1e-5` accomplish nothing measurable. The same measurement with Adam,
`lr_perm = 1e-3` and twenty epochs gives a **2 996×** reduction within each cycle, so the flat curve
is a property of the shipped hyper-parameters rather than of the optimisation problem.

Together these confirm, on the real system rather than on probes, that the shipped consolidation
**deletes far more of the transient than intended while transferring essentially none of it**.

![Figure 6.2 — Consolidation internals under the shipped configuration.](plots/figures/consolidation_internals_shipped.png)

*Figure 6.2 — Consolidation internals, shipped configuration (`decay = 0.5`), 3 seeds. (a) The
regression loss at the first, mean and last gradient step of each cycle are indistinguishable — the
regression never descends. (b) The transient's mean value over the batch, showing the sawtooth as it
re-accumulates between consolidations and resets at each task boundary. (c) Its L2 norm before and
after the decay: the gap is far larger than the factor of two the configured decay implies.*

### 6.2 Why this evaded detection for three rounds

`critic_loss` is computed **during** the 10-epoch PPO update, by which point the fast transient has
already re-fitted the returns. The metric therefore looks healthy (~0.01, comparable to vanilla)
while the damage is happening. This is why the failure looked mysterious, and it explains why every
earlier hypothesis failed: `lr_perm = 1e-3` still transfers only ~5 %; a faster transient merely
re-fits sooner; and `decay = 0` vs `0.5` is irrelevant when the value dies either way.

### 6.3 How far the regression can be pushed — and what actually limits it

An earlier draft of this report claimed the consolidation target was **not representable** — that a
single MLP cannot express `old_V_perm + V_trans`, the sum of two MLPs, because the function class is
not closed under addition. **That claim was wrong, and this section corrects it.** The objection that
prompted the check was straightforward: on a *finite* batch a sufficiently large network should
simply overfit the target, so any observed error may be an optimisation or capacity artefact rather
than a representational limit. That is what the measurement shows.

Setup: the true production consolidation problem — 20 480 buffered states, target `old_V_perm(s) +
V_trans(s)` with both components `[64,64]` MLPs — fitted with Adam (lr 1e-3, batch 256) rather than
the shipped SGD at 1e-5, and evaluated both on the fitted batch and on held-out states.

| permanent net | params | epochs | error on fitted batch | error on held-out states |
|---|---|---|---|---|
| `[64,64]` (production) | 5 377 | 50 | 35.6 % | 39.2 % |
| `[64,64]` | 5 377 | 200 | 29.3 % | 38.9 % |
| `[256,256]` | 70 657 | 50 | 23.8 % | 38.4 % |
| **`[256,256]`** | 70 657 | 200 | **3.2 %** | 44.2 % |
| `[512,512]` | 267 265 | 200 | 4.3 % | 42.5 % |

**Three conclusions:**

1. **The target is fittable.** With enough capacity and training the batch error reaches ~3 %. There
   is no representational barrier; the earlier "sum of two MLPs" argument does not hold.
2. **The shipped configuration is nowhere near that.** `lr_perm = 1e-5`, SGD, one epoch (320
   gradient steps) transfers **0.05 %**. The production failure is overwhelmingly a matter of the
   consolidation regression never being trained — while the decay deletes the transient regardless.
   This alone accounts for the collapse, and it is the finding the ablations confirm (§5.4).
3. **What does not improve is generalisation.** Held-out error floors at **≈ 38–40 %** across every
   capacity and budget tried, and past a point more fitting makes it *worse* (`[256,256]`: batch
   23.8 % → 3.2 % while held-out 38.4 % → 44.2 %). This is the operationally relevant quantity:
   consolidation fits the states of the last *k* rollouts, but the resulting value function is then
   used to bootstrap on the **new** states of the next rollout.

**Caveat.** The probe states here are iid Gaussian in 17 dimensions, which is a harder
generalisation problem than genuine on-policy states, which concentrate on a low-dimensional
manifold. The held-out floor should therefore be read as indicative of a generalisation gap, not as
a calibrated estimate of it. Measuring the same quantity on real rollout states is a worthwhile
follow-up (§10g).

**What this means for the mechanism.** Consolidation-by-regression is not impossible, but it is
expensive (thousands of gradient steps every *k* updates to approach a good fit), it did not happen
at all in the shipped configuration, and even when performed well it leaves a substantial error on
the states that matter next. The shared-trunk formulation in §7 sidesteps the entire question: the
transfer becomes exact weight arithmetic, with no regression, no training budget and no
generalisation gap — which is why it is the right construction regardless of how this section
resolves. Notably, Anand et al. (2024, ICLR under review, ref. [4] of the proposal) also move away
from a naive parametric transient, towards separate feature encoders and non-parametric transient
memory.

---

## 7. The fix: shared trunk with linear heads — and its result

If both heads read from a **shared feature trunk** φ and are **linear**, then

```
V = w_P·φ(s) + w_T·φ(s) = (w_P + w_T)·φ(s)
```

and the sum of two linear functions on the same features is linear on those features. Consolidation
becomes **exact weight arithmetic** — `w_P += (1−decay)·w_T`, then `w_T *= decay` — leaving `V`
unchanged. No regression, no consolidation buffer, no `lr_perm`; and decaying a *linear* head scales
its output exactly, so defect 6.1(a) also disappears.

**Measured consolidation error:**

| decay | two separate MLPs (best case) | shared trunk + linear heads |
|---|---|---|
| 0.25 | 17.1 % | **0.0000 %** |
| 0.50 | 16.3 % | **0.0000 %** |
| 0.75 | 10.6 % | **0.0000 %** |

This is **not an invention** — it is the shared-trunk two-head variant that the reference
implementation already uses for its MiniGrid experiments. Our analysis shows *why* that variant is
the correct choice under deep function approximation.

**Implementation status:** `SharedTrunkSplitCritic` added; selected by `critic_arch: shared_trunk`
(default remains the original, so all prior behaviour is unchanged); permanent head zero-initialised
so it starts empty and accumulates only through consolidation; four unit tests assert that
consolidation preserves the acting value exactly for every decay value (suite total: **31 tests**).

**Known trade-off (should appear in the thesis discussion).** Because the trunk is shared, the
permanent head's *weights* are frozen between consolidations but its *output* still moves as the
features are learned. The variant therefore trades **permanent insulation** for **exact
consolidation**; the two-trunk variant makes the opposite trade. Whether exact-but-shared preserves
the PT benefit is exactly what this experiment measures.

### 7.1 Result (5 seeds, full horizon)

**Per-phase mean return, all variants:**

| Phase | vanilla | EWC | PT (broken) | PT (no consolidation) | **PT (shared trunk)** |
|---|---|---|---|---|---|
| 1 | 743 | 743 | 475 | 641 | **814** |
| 2 | 468 | 517 | 332 | 1039 | **394** |
| 3 | 243 | **705** | −279 | 291 | **27** |
| 4 | 375 | **1238** | −249 | 746 | **212** |
| 5 | −34 | **533** | −396 | −373 | **−176** |

Per-phase σ for the shared-trunk run: 275 / 231 / 374 / 399 / 382. Critic loss 0.003–0.022
throughout. Per-boundary value drift: 0.00 / 5.08 / 3.87 / 0.00 / 2.60.

**Three conclusions, in decreasing order of confidence:**

1. **The fix works as designed.** No phase approaches the broken agent's −307 / −248 / −402. Boundary
   value drift is near zero and of the same small order as the no-consolidation control — confirming
   *in production*, not merely in unit tests, that consolidation no longer perturbs the acting value.
   The catastrophic failure mode is eliminated.

2. **PT does not beat vanilla.** Phase 1 is nominally ahead (814 vs 743) but the 71-point gap is
   negligible against σ = 275. Phases 2–5 all fall below vanilla (394 vs 468; 27 vs 243; 212 vs 375;
   −176 vs −34), and **every one of those gaps is smaller than that phase's own standard deviation**.
   The correct reading is that shared-trunk PT is **statistically indistinguishable from vanilla**.

3. **EWC is clearly superior, and this gap is not noise.** At phase 3, EWC's 705 vs PT's 27 is a
   ~678-point gap against σ = 374 (beyond one σ); at phase 4, 1238 vs 212 is a ~1026-point gap
   against σ = 399 — unambiguously outside seed noise.

**An observation we flag rather than claim.** Shared-trunk PT underperforms the *naive*
no-consolidation control in phases 2 (394 vs 1039) and 4 (212 vs 746) — i.e. even lossless
consolidation may add drag relative to not consolidating at all. However, the control's own variance
in those phases is very large (σ = 576 and 684), and a two-sample comparison gives roughly *t* ≈ 2.3
at phase 2 and *t* ≈ 1.5 at phase 4. Across five phases and several variants this does not survive
as a reliable effect at n = 5. It is recorded as a hypothesis, **not** a finding, and we do not
recommend spending further compute on it (§10j).

### 7.2 What this establishes

Round 4 showed PT's machinery was *harmful*. This run shows that once the machinery is **provably
correct**, it is merely *inert*: it neither collapses nor helps. That distinction is what converts a
weak claim ("our implementation underperformed") into a strong one:

> **Value decomposition of the critic provides no measurable benefit over a single critic in
> policy-gradient continual control, even when the consolidation operator is mathematically exact.**

The mechanism cannot be dismissed as broken — we removed the defect, verified the repair numerically
and in unit tests, and the benefit still did not materialise. A plausible explanation is structural:
PT is a **critic-only** method, and in an actor–critic the critic influences the policy solely through
the advantage estimate. Both a single critic and a correctly-decomposed critic fit the returns well
(critic loss ≈ 0.003–0.02 for both), so they furnish near-identical advantages and therefore
near-identical policies. In the value-based control setting where PT was introduced, the value
function *is* the policy, so improving it necessarily improves behaviour; that link is absent here.

---

## 8. The smooth-drift experiment (the proposal's setting)

All results above use `DirectionalHalfCheetah`, where the **reward** flips sign at discrete task
boundaries. The proposal specifies the opposite: a **fixed reward** with the **physics** drifting
smoothly and no boundaries at all. `LipschitzDriftHalfCheetah` implements it — joint damping and
ground friction rescaled every step by a sinusoid (amplitude 0.5, period 1 228 800 steps ⇒ 2.5
cycles over the run), measured Lipschitz bound 2.557 × 10⁻⁶ per env step. Verified in the logs: the
multiplier spans exactly [0.5, 1.5], crossing 1.0 at every half-period.

This was run because it is the one setting where a **positive** result for PT was still plausible:

> **EWC computes its Fisher information at a task boundary.** With no boundaries, `on_task_switch`
> never fires, no Fisher accumulates, and EWC has no mechanism left. PT's consolidation runs on a
> **timer** and is unaffected.

**Return by 614 400-step segment (mean ± SEM, 5 seeds, segments chosen to line up with the phases
of the task-switching tables):**

| Agent | Seg 1 | Seg 2 | Seg 3 | Seg 4 | Seg 5 |
|---|---|---|---|---|---|
| vanilla | 569 ± 28 | 1631 ± 34 | 1320 ± 51 | 1772 ± 58 | 1655 ± 169 |
| EWC | *byte-identical to vanilla* | | | | |
| PT (shared trunk) | 546 ± 34 | 1564 ± 78 | 1300 ± 109 | 1870 ± 80 | 1808 ± 272 |

**Three findings.**

1. **EWC's advantage is entirely boundary-dependent — confirmed exactly.** `ewc_penalty` is exactly
   0.0 at all 1 500 logged points on every seed, and with matched seeds the training trajectories
   are **byte-identical** to vanilla's — re-verified directly from the saved curves: 5/5 seeds
   identical, max \|difference\| exactly 0. Under boundary-free drift EWC does not merely fail to help:
   it *mechanically is* vanilla. This is the study's cleanest positive result, and it lands
   precisely on the gap the proposal identifies — that the continual-RL literature assumes discrete
   task boundaries.
2. **PT does not beat vanilla here either.** Every segment gap is well inside the combined SEM (the
   largest, +153 in segment 5, against ~317). It does not collapse either — this config uses the
   shared-trunk critic with `decay = 0.5`, the combination §5.5 identified as safe — so the run is
   consistent with, and further confirms, the earlier finding that a well-behaved PT tracks vanilla
   rather than beating it.
3. **Nobody collapses under smooth drift.** All three agents climb from ~550 to ~1300–1900 and
   simply track the drift cycle (segment 3 dips near the multiplier's 0.5 trough — harder physics,
   lower return, identically for every agent). Compare this with the directional task, where PT
   collapses to negative return and even vanilla only reaches ~743/468/243/375/−34.

**The last point reframes the whole study.** Smooth dynamics drift *of this magnitude* is simply
*not hard* for plain PPO: there is no catastrophic forgetting for a continual-learning method to
prevent, so none of the machinery has anything to do. The difficulty in the directional experiment
comes from the **abrupt inversion of the reward**, not from non-stationarity as such. A continual-RL
method can only pay for itself where a discrete boundary destroys prior knowledge — and that is
exactly the regime in which EWC helps and PT does not.

*(Qualified by §8.1: at faster drift the setting does become discriminating — but it discriminates
**against** PT, which falls clearly behind vanilla rather than catching up.)*

**Important limitation — this setting could not have shown a PT advantage.** At period 1 228 800
the multiplier moves by only **0.5 % per PPO update** and ~5 % per consolidation cycle. The critic
gets 10 epochs at `lr = 3e-4` per update, so it tracks that trivially: there is **no fast component
for a transient to absorb**, and therefore nothing for a permanent/transient split to do. The tie
observed here is what the decomposition *predicts* in a single-timescale, slowly-drifting world —
it is not evidence against the mechanism.

Two further settings address this directly:

| setting | change per PPO update | change per consolidation cycle |
|---|---|---|
| `drift.yaml` (run above) | 0.5 % | 5 % |
| `drift_fast.yaml` (period ÷10) | 5 % | 52 % |
| `drift_twoscale.yaml` | slow 0.4 % + a **full fast cycle every ~15 updates** | — |

`drift_twoscale` is the sharpest of the three, and the regime the proposal actually describes: a
slow structural trend (amplitude 0.4, period 1 228 800) the **permanent** part should capture, plus
a fast fluctuation (amplitude 0.2, period 30 720) the **transient** should absorb — "filtering out
temporary noise". A single critic must chase both at once. **We predicted PT > vanilla there.**

### 8.1 Result: PT does not merely tie under harder drift — it loses

Both settings, 3 seeds each, return by 614 400-step segment (mean ± SEM):

**`drift_twoscale`** (slow trend + fast fluctuation — the regime PT is designed for):

| | Seg 1 | Seg 2 | Seg 3 | Seg 4 | Seg 5 |
|---|---|---|---|---|---|
| vanilla | 870 ± 107 | 2317 ± 401 | 2745 ± 282 | 2255 ± 550 | 2550 ± 470 |
| PT | 562 ± 41 | 1588 ± 41 | 1351 ± 72 | 1756 ± 46 | 1462 ± 92 |
| **PT − vanilla** | **−308** | **−729** | **−1394** | −499 *(inside)* | **−1088** |

**`drift_fast`** (10× faster single-timescale drift):

| | Seg 1 | Seg 2 | Seg 3 | Seg 4 | Seg 5 |
|---|---|---|---|---|---|
| vanilla | 750 ± 137 | 2570 ± 295 | 3119 ± 336 | 3377 ± 318 | 3601 ± 389 |
| PT | 576 ± 28 | 1437 ± 95 | 1774 ± 303 | 2040 ± 296 | 2192 ± 315 |
| **PT − vanilla** | **−174** | **−1133** | **−1345** | **−1336** | **−1409** |


![Figure 8.1 — PT versus vanilla across all three drift regimes. Bold numbers mark gaps exceeding the combined SEM: none under slow drift, large negative gaps in both harder regimes.](plots/figures/drift_comparison.png)

*Figure 8.1 — PT versus vanilla across all three drift regimes. Bold numbers mark gaps exceeding the combined SEM: none under slow drift, large negative gaps in both harder regimes.*

**Vanilla beats PT outside the combined SEM in 9 of the 10 segments.** The prediction is not merely
unconfirmed — it is **reversed**. And the direction is systematic: under *slow* drift the two were
tied (all gaps inside noise); adding real fast-timescale content turns that tie into a clear PT
deficit, and the deficit is *larger* in `drift_fast` than in `drift_twoscale`. Giving the
decomposition the timescale separation it was designed to exploit made it **worse**, not better.

**Plausible explanation — offered as a hypothesis, not a demonstrated mechanism.** (Two earlier
mechanistic claims in this study were retracted after controlled testing; this one has *not* been
tested and should not be reported as established.) Under PT, only the transient trains between
consolidations — `V_perm` is frozen — so PT's effective capacity for tracking change is a single
MLP, exactly as for vanilla. But its regression target is `returns − V_perm.detach()`, and when the
physics move quickly that frozen baseline goes stale fast, making the target *more* non-stationary
than the raw returns vanilla fits. Every *k* updates, consolidation then folds the transient into
the permanent and decays it, so the transient must re-learn its residual against a
just-changed baseline. Same capacity, harder target, periodic disruption: PT is structurally
handicapped precisely when tracking speed is what matters.

**A secondary observation.** Vanilla's across-seed SEM is much larger than PT's (e.g. `drift_twoscale`
segment 2: 401 vs 41), and one vanilla seed reached 3563. Vanilla is noisier but attains a far higher
ceiling; PT is consistently mediocre. That is consistent with the decomposition acting as a
constraint on the value function rather than as an aid.

**This closes the search.** PT was given the setting the proposal specifies, then the harder version
of it, then the exact slow/fast decomposition it was designed for. It tied in the first and lost in
the other two.

---

## 9. Status against the proposal's deliverables

| Deliverable | Status |
|---|---|
| Python implementation of the non-stationary environments + dual-timescale agent | **Complete** — 3 agents, 2 environments (directional switching and Lipschitz drift), layered config system, 31 unit tests |
| Experimental results, plots, evaluation metrics vs baselines | **Complete** — 13 figures, 15 per-seed CSVs, a 3-agent × 5-seed benchmark, 7 ablation rounds and a 3-regime drift study |
| Final thesis document | Results and Discussion drafted; Methodology outstanding |
| Presentation slides | Not started |

The proposal's Risk Management section anticipates this outcome explicitly: *"there is a risk of null
results… **This outcome is not considered a failure.** The main contribution of the work is the
systematic analysis of how split representations behave… Even if no performance improvement is
observed, the results will still provide valuable insight into the limits of parameter-based
plasticity."*

The present work meets that bar and goes beyond it. Rather than a bare null result it delivers: a
**measured failure mechanism** (§6.1), a **causal control** isolating it (§5.4), an **architecturally
exact reimplementation** that removes the failure (§7), a **positive disconfirmation** under the
regime the method targets (§8.1), and a **clean positive result** about the baseline it was compared
against (§8, EWC's boundary dependence). Two candidate explanations were advanced and retracted after
controlled testing (§5.7, §6.3), which is recorded rather than hidden.

---

## 10. Open issues and next steps

**The experimental programme is closed.** Seven ablation rounds, an architecturally exact
reimplementation and a three-regime drift study have converged on a consistent answer. Every
candidate mechanism proposed during the investigation has been tested; none survived. Further
hyper-parameter search would amount to fishing for a favourable seed draw.

Items (a) and (b) of an earlier version of this section — implement the drift environment, and test
whether EWC's advantage is boundary-dependent — are **both now done and reported in §8 and §8.1**.
What remains is writing, plus a small number of genuinely open questions.

### Remaining work

**a) Methodology chapter.** The only substantial writing gap. Everything needed exists: both
environments, the three agents, the validated PPO recipe and the config system.

**b) Presentation slides.** Not started.

### Genuinely open questions (none affect the conclusions)

**c) The §8.1 explanation is untested.** The account of *why* PT degrades under fast drift — that its
transient must fit `returns − V_perm.detach()` against a baseline that goes stale quickly, giving it
the same capacity as vanilla's critic but a more non-stationary target — is consistent with all
observations but has not been tested directly. Given that two earlier mechanistic claims here were
retracted after controlled testing (§5.7, §6.3), it should be reported as a hypothesis. A controlled
test would compare the transient's regression error against vanilla's critic loss under identical
drift.

**d) Temporal generalisation of the consolidation transfer (§5.6).** The held-out measurement used a
random 20 % split of the *same* buffer, so it establishes that the permanent network interpolates
within its recent state distribution. It does not test extrapolation to the *next* rollout's states,
which are temporally later and, after a switch, drawn from a shifted distribution. Fitting the
regression on one rollout's buffer and evaluating on the following rollout's states would close this;
it needs a checkpoint and two rollouts, not a training run.

**e) Statistical power.** With n = 5 and per-phase σ of 230–790, only large effects are detectable.
That is why the EWC advantage (gaps of 673–1047) and PT's drift deficits (up to 1409) are reportable
while PT-vs-vanilla differences on the switching benchmark are not. **Phase 2 in particular has σ =
787, two to three times any other phase**, and no mechanism claim can be supported by it in either
direction (§5.7). Any future fine-grained comparison should use 10 seeds.

**f) Single environment.** All experiments use HalfCheetah. The failure analysis is mechanism-level
and would be expected to transfer, but this has not been demonstrated on a second domain.

**g) Drift design choices.** The schedule is sinusoidal so that earlier dynamics recur and retention
is measurable. A monotone ramp, or drift in different physical parameters (mass and armature are
implemented but unused), may behave differently.

### Explicitly not recommended

**h) Further PT hyper-parameter tuning.** Every mechanism has been eliminated; additional search
would be seed-fishing.

**i) A dual-timescale actor.** The shared-trunk construction (§7) would make an actor split exact, so
the implementation obstacle is gone. But §7.2 gives little reason to expect a benefit — the
decomposition itself, not its implementation, is what fails to help — and §8.1 shows it becoming
actively harmful as the tracking demand rises. Only worth attempting if the thesis explicitly
reframes toward a *policy*-decomposition research question.

**j) The "exact consolidation still adds drag" observation (§7.1),** which does not survive
multiple-comparison scrutiny at n = 5.

**k) Switch-time consolidation in isolation (§5.1).** Would require `k = 7` so that consolidation
does not coincide with the boundary. Low value now that consolidation is known not to be the
discriminating variable.

### A reproducibility note carried forward

The instrumentation added in §5.6 changed which RNG stream the consolidation step consumed, so runs
before and after it are not trajectory-comparable at matched seeds (§5.7). This is fixed, and the
diagnostic path is documented as non-comparable. The general lesson: in stochastic experiments,
"functionally equivalent" edits to code that consumes randomness are not equivalent for
reproducibility.

---

## 11. Reproducibility

**In the repository:**

| Item | Location |
|---|---|
| Source, configs, tests | `agents/`, `envs/`, `models/`, `utils/`, `configs/`, `tests/` |
| Figures (13) | `plots/figures/` |
| Figure and experiment scripts | `plots/plot_*.py`, `plots/make_consolidation_figure.py`, `scripts/run_*.sh` |
| Run instructions | `README.md` |

**Produced by running the experiments** (excluded from version control — regenerable, and several
hundred MB in total):

| Item | Written to |
|---|---|
| Per-seed return / eval / velocity curves | `results/*.pkl` |
| Ablation and drift runs | `abl_results/<variant>/`, with TensorBoard scalars in `abl_runs/<variant>/` and console output in `abl_logs/` |
| Exported numeric logs | `numeric_logs_csv/` |


All runs are seeded and reproducible. The environment is CPU-bound (MuJoCo physics with small
networks), so a GPU provides no benefit; throughput scales with core count and process-level
parallelism.
