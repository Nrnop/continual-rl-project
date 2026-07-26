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

Five results:

1. **PT fails in this setting** — it collapses to a do-nothing standstill policy from the third task
   phase onward (phase-3 mean return **−279** vs vanilla **+243**).
2. **We identified the exact cause**, and it is structural rather than a tuning problem: the
   consolidation operator destroys ~98 % of the value function every *k* updates (~150× per run).
   The permanent critic must *learn* `old_V_perm + V_trans` by regression — i.e. represent the **sum
   of two neural networks with a single network** — and that function class is not closed under
   addition. In the original paper's tabular/linear setting that sum *is* exactly representable,
   which is precisely why PT works there and not here.
3. **We proved causation**: disabling consolidation entirely reverses the collapse
   (phase-3 mean **−279 → +291**).
4. **We built and validated a fix** — a shared trunk with linear heads makes consolidation
   mathematically exact (0.0000 % value drift, confirmed in production by near-zero boundary drift).
   **The collapse is eliminated.** But the repaired mechanism performs **statistically
   indistinguishably from vanilla PPO** (every phase difference smaller than that phase's own
   standard deviation).
5. **EWC is the positive result** — it beats both, decisively and outside noise (phase-4 mean
   **1238** vs vanilla **375** and PT **212**; gap ≈ 1026 against σ ≈ 399).

**The central finding.** Value decomposition of the critic confers **no benefit** over a single
critic in policy-gradient continual control — and this conclusion is now established in its strongest
form, because we can demonstrate that the mechanism was functioning *correctly* when it failed to
help, rather than merely reporting that an implementation underperformed. Regularisation-based
continual learning (EWC) is substantially superior in this setting.

**Note on scope.** The proposal specifies smooth, Lipschitz-continuous *dynamics* drift. The
implemented environment is **discrete directional task-switching** (the forward-velocity reward term
flips sign every 614 400 steps). This mismatch is discussed in §9 and must be resolved — either by
implementing the drift wrapper or by adjusting the thesis framing.

---

## 2. Experimental setup

| Item | Value |
|---|---|
| Environment | `DirectionalHalfCheetah` (HalfCheetah-v5); reward's forward-velocity term flips sign |
| Non-stationarity | 4 switches / 5 phases, direction sequence `+1, −1, +1, −1, +1` |
| Switch interval | 614 400 env steps · total 3 072 000 steps |
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
vectorised buffer signature. **15/15 passing** at that point (now 19/19, see §7).

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

*Interpretation discipline:* differences between `PT-no-consolidation` and vanilla in phases 2–4 sit
**within seed noise** (σ = 350–680, n = 5). The defensible claim is **"removing consolidation restores
vanilla-level performance"**, not "PT beats vanilla". The phase-3 *reversal* (−279 → +291, consistent
across 4/5 seeds) is large relative to that noise and is treated as real.

*Phase-5 decline is generic, not PT-specific:* vanilla (375 → −34) and EWC (1238 → 533) decline in
phase 5 as well, consistent with the near-fully-annealed learning rate late in training.

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

### 6.2 Why this evaded detection for three rounds

`critic_loss` is computed **during** the 10-epoch PPO update, by which point the fast transient has
already re-fitted the returns. The metric therefore looks healthy (~0.01, comparable to vanilla)
while the damage is happening. This is why the failure looked mysterious, and it explains why every
earlier hypothesis failed: `lr_perm = 1e-3` still transfers only ~5 %; a faster transient merely
re-fits sooner; and `decay = 0` vs `0.5` is irrelevant when the value dies either way.

### 6.3 Why it is structural, not a tuning problem

Consolidation asks a single MLP to represent `old_V_perm + V_trans` — **the sum of two MLPs**. That
function class is not closed under addition, so the target is not exactly representable and the
regression is lossy *by construction*. Increasing effort does not remove it:

- 50 consolidation epochs (16 000 gradient steps): error floors at **~33 %**
- widening the permanent network: **35.2 %** `[64,64]` → **22.3 %** `[256,256]` → **17.2 %** `[512,512]`
  — reducible, but never zero, with sharply diminishing returns

**This also explains why the original method works.** In the paper's tabular/linear setting, the sum
of two linear functions *is* linear — exactly representable — so consolidation is exact there. The
operation only becomes lossy under deep function approximation.

This is consistent with the direction taken by Anand et al. (2024, ICLR under review, ref. [4] of the
proposal), who extend PT with *separate feature encoders and non-parametric transient memory* —
i.e. the original authors also moved away from the naive parametric transient.

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
so it starts empty and accumulates only through consolidation; **19/19 unit tests pass**, four of
which assert that consolidation preserves the acting value exactly for every decay value.

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
recommend spending further compute on it (§9f).

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

## 8. Status against the proposal's deliverables

| Deliverable | Status |
|---|---|
| Python implementation of non-stationary environments + dual-timescale agent | **Complete** (3 agents, config system, 19 unit tests) |
| Experimental results, plots, evaluation metrics vs baselines | **Complete** — 6 figures, 15 per-seed CSVs, 5-seed benchmark + 4 ablation rounds |
| Final thesis document | In progress |
| Presentation slides | Pending |

The proposal's Risk Management section anticipates this outcome explicitly: *"there is a risk of null
results… **This outcome is not considered a failure.** The main contribution of the work is the
systematic analysis of how split representations behave… Even if no performance improvement is
observed, the results will still provide valuable insight into the limits of parameter-based
plasticity."* The present work exceeds that bar: it delivers not merely a null result but a
**measured mechanism, a mathematical explanation, a causal control, and a candidate fix.**

---

## 9. Open issues and next steps

**The PT investigation is closed.** Four ablation rounds plus a validated fix have converged on a
consistent answer, and further hyper-parameter search would amount to fishing for a favourable seed
draw. Remaining effort is better spent on (a) and (b).

**a) Scope mismatch with the proposal title — requires a decision.**
The title promises *smooth environment non-stationarity*, but the implementation uses discrete
directional switching. Two options: implement the Lipschitz drift wrapper, or adjust the title and
framing to "boundary-based directional non-stationarity". *Recommendation:* implement the drift
wrapper (estimated 1–2 days of work plus one ~3 h sweep), because it also enables (b).

**b) The most valuable remaining experiment.**
**EWC requires task boundaries to compute its Fisher information.** Under smooth drift there are no
boundaries — so the study's strongest performer may lose its advantage precisely where the proposal's
identified gap lies ("most existing methods assume discrete task boundaries"). Given that EWC is
currently the headline positive result, testing whether that advantage is *boundary-dependent* is the
single most informative experiment left, and it directly addresses the motivating research gap.
Prediction to test: EWC degrades markedly under boundary-free drift, while vanilla and PT are
comparatively unaffected (PT's behaviour is already effectively vanilla's).

**c) Dual-timescale actor — not recommended.**
Extending PT to the policy suffers from the *identical* representability problem (a policy network
cannot represent the sum of two policy networks), and policy errors are more immediately damaging
than value errors. The shared-trunk construction would make an actor split exact, but §7.2 gives
little reason to expect a benefit: the decomposition itself, not its implementation, is what fails to
help. Only worth attempting if the thesis explicitly reframes toward a *policy*-decomposition
research question.

**d) Statistical power.** With n = 5 and per-phase σ of 230–680, only large effects are detectable —
which is precisely why the EWC advantage (gaps of 673–1047 points) is reportable while the
PT-vs-vanilla differences are not. Per-phase means (used throughout) are markedly more stable than
end-of-phase values. Any future fine-grained comparison should use 10 seeds.

**e) Untested minor lever.** The switch-time consolidation behaviour has never been properly isolated
(see §5.1); doing so requires `k = 7` so that consolidation does not coincide with the boundary.
Low expected value now that consolidation is known to be inert overall.

**f) Explicitly not recommended.** Further PT hyper-parameter tuning; and the "exact consolidation
still adds drag" observation (§7.1), which does not survive multiple-comparison scrutiny at n = 5.

---

## 10. Reproducibility

| Item | Location |
|---|---|
| Source, configs, tests | `src_continuous_control/` (git; training artefacts excluded) |
| Main-sweep figures (6) | `plots/figures/` |
| Per-seed numeric logs (15 CSVs) | `numeric_logs_csv/` |
| Full-resolution curves | `results/*.pkl` |
| Ablation outputs | `abl_results/<variant>/`, `abl_logs/` |
| Run instructions | `VASTAI_SETUP.md`, `CLAUDE.md`, `README.md` |

All runs are seeded and reproducible. The environment is CPU-bound (MuJoCo physics with small
networks), so a GPU provides no benefit; throughput scales with core count and process-level
parallelism.
