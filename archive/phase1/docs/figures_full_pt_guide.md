# Figure guide — what every plot shows, in plain words

A walkthrough of all 14 figures in `plots/figures_pt_full/`. For each one: what you are looking
at, what it says, how to read it critically, and what to be careful about. A reading aid rather
than a thesis document — the citable record is `FULL_PT.md`.

Regenerate everything with:

```bash
python -m src_continuous_control.plots.make_pt_full_figures
```

Every number is recomputed from the raw per-seed result files, so nothing here is transcribed by
hand and no figure can drift out of step with the text.

**The agents that appear everywhere:**

- **vanilla PPO** — the plain baseline.
- **`pt_full` (live permanent)** — the full method, working normally.
- **`pt_full` (frozen permanent)** — the same agent with the permanent network frozen so it never
  learns. This is the control that isolates the mechanism: if the permanent–transient idea is what
  helps, live should beat frozen.
- **PPO + shrinkage** — plain PPO with only one step copied out of the method: periodically
  multiply the policy's output layer by 0.5.

**One habit to carry through all of these:** read the *dots*, not the bar. A median computed from
six widely scattered runs is not the same kind of number as a median from six tight ones, even
when they print identically.

---

## 1. The headline

### `reduction_halfcheetah`

![](plots/figures_pt_full/reduction_halfcheetah.png)

**What you're looking at.** Four rows, one per agent. Every dot is one training run (6 seeds); the
thick vertical bar is the median. Further right is better.

**What it says.** `pt_full` scores 760 and PPO-plus-shrinkage scores 711 — statistically the same
(p = 0.485). Both crush plain PPO's 50. So the method works, but everything it achieves is
achieved by the shrinkage alone.

#### Read this one carefully — the medians mislead

**`pt_full` has the highest median on this figure.** Anyone looking at it will notice, and it is
the single weakest piece of evidence in the study. The actual per-seed numbers, middle two in bold:

| arm | six seeds | spread |
|---|---|---:|
| `pt_full` live | 230 · 365 · **734 · 786** · 931 · 1253 | **1023** |
| PPO + shrinkage | 703 · 705 · **708 · 713** · 734 · 752 | **49** |
| `pt_full` frozen | 542 · 544 · **551 · 552** · 566 · 588 | 46 |

Three things follow.

**1. The 760 rests on two adjacent seeds inside a range of 230–1253.** Drop each arm's luckiest run
and `pt_full` falls to 734 while the shrink-only arm barely moves, to 708. A ranking one seed can
rearrange is not a ranking.

**2. The arms overlap rather than separate.** Two of six `pt_full` seeds (230, 365) land *below
every single* shrink-only seed; three land above all of them. That is what p = 0.485 describes.

**3. The comparison that actually tests the mechanism is not on the leaderboard.** It is **live vs
frozen** — the same agent with the permanent switched off — which is **+209, p = 0.394.** Also not
significant.

**The honest statement about this figure alone is "not demonstrated", not "proven absent."** With
six seeds and a spread of 1023 it cannot detect a +209 effect. If this were the only experiment,
the fair conclusion would be *we cannot tell.*

What settles it is that the *other* experiments reach significance and point the other way:

| | mechanism (live − frozen) | p |
|---|---:|---:|
| this figure (k = 8, HalfCheetah) | +209 | 0.394 — can't tell |
| the published k = 16 (point-mass) | **−39.5** | **0.001** |
| the published config (HalfCheetah, next figure) | shrink-only beats the method by **760** | **0.041** |
| permanent's learning rate swept | every setting below vanilla | ≤ 0.001 |

**If you present this figure, say all of this before you are asked** — the medians are the first
thing a reader's eye lands on.

**The second finding, in the same picture.** Look at how spread out the dots are. Vanilla and the
live-permanent arm scatter across hundreds of points; the two shrinking arms are tight clusters.
Shrinkage makes the agent **16× more consistent** across seeds (787 vs 49). It doesn't only raise
the score, it makes the result reliable — and that reliability is itself why the shrink-only arm's
median can be trusted while `pt_full`'s cannot.

---

### `published_config_halfcheetah`

![](plots/figures_pt_full/published_config_halfcheetah.png)

**What you're looking at.** The same comparison run at the exact configuration published in
`PT_full.md` — k = 16, `consolidation_epochs` = 3, [256,256] and [64,64] networks — rather than at
our settings. σ frozen and verified identical across arms before launch.

**What it says.** At those settings the shrink-only control doesn't merely match the method, it
**significantly beats it**: 1275 vs 515, **p = 0.041**. And with **1/14 the parameters** — 11,085
against 153,684.

**Why this figure is much stronger than the previous one.** Look at the separation:

| arm | six seeds | spread |
|---|---|---:|
| PPO + shrinkage | 1192 · 1236 · **1272 · 1279** · 1347 · 1396 | 204 |
| `pt_full`, published config | 346 · 475 · **511 · 519** · 1057 · 1391 | 1046 |
| vanilla PPO | −411 · −362 · **−48 · 148** · 337 · 375 | 787 |

**Five of six shrink-only seeds beat five of six `pt_full` seeds.** Only `pt_full`'s single best
run (1391) reaches the shrink-only cluster at all. That is a real separation, not a median
artifact — which is exactly what the previous figure lacked.

**Why it matters.** This is the answer to "you tested your settings, not mine." The conclusion is
*stronger* at the published configuration, not weaker.

**One caveat to state.** `pt_full`'s bimodal spread (three seeds near 500, one at 1391) means the
method is not merely worse here — it is **unstable**. Two different behaviours are being averaged.
Worth saying out loud rather than letting someone find it.

---

## 2. How we found the cause

### `dose_response`

![](plots/figures_pt_full/dose_response.png)

**What you're looking at.** The x-axis is how hard the policy gets shrunk — **right = more
shrinking**. Two lines: the full PT-PPO apparatus, and plain PPO with only the shrink added.

**What it says.** Two findings in one picture:

| decay factor | full apparatus | shrink-only |
|---|---:|---:|
| 1.00 (no shrink) | **32.7** | — |
| 0.90 | 58.5 | 64.8 |
| 0.75 | 82.8 | 81.2 |
| 0.50 | 92.1 | 93.9 |
| 0.25 (hardest) | **104.4** | — |

*vanilla reference: 38.8*

**1. Clean monotone dose–response.** Shrink harder → score higher, every step, no exceptions. That
is what identifies the shrinkage as the active ingredient rather than a correlate.

**2. The two lines lie on top of each other** — within 6 points at every shared setting. The full
apparatus and the bare shrink behave identically.

**The detail that makes this decisive.** In *both* series the permanent network is zeroed and the
KL term is off. **None of the PT machinery is present in either line.** And at the far left, where
nothing is shrunk, the agent falls to **32.7 — below vanilla's 38.8**. Remove the shrink and the
apparatus is worth nothing.

**How to read it critically.** This is the strongest single figure in the study, because it is a
*dose–response* rather than a two-point comparison. A confound would have to scale smoothly with
the shrink factor across five levels in two independent implementations to fake this.

---

### `beta_sweep`

![](plots/figures_pt_full/beta_sweep.png)

**What you're looking at.** The strength of the KL penalty (β) on the x-axis, from **zero** up to
1.0. Two lines: frozen permanent and live permanent.

**What it says.**

| β | frozen | live | mechanism (live − frozen) | p |
|---|---:|---:|---:|---:|
| **0.0** | **93.7** | 77.7 | −16.0 | 0.010 |
| 0.001 | 94.0 | 61.6 | −32.5 | 0.002 |
| 0.01 | 93.4 | 69.2 | −24.3 | 0.005 |
| 0.1 | 94.6 | 69.3 | −25.4 | 0.000 |
| 1.0 | 71.8 | 40.1 | −31.7 | 0.000 |

*vanilla reference: 38.8*

**1. The anchor explains none of the gain.** The frozen arm is flat at 93–95 across four orders of
magnitude and beats vanilla by the same +55 **even at β = 0**, where the penalty is switched off
completely.

**2. The more important reading, easy to miss:** the live line sits **below** the frozen line at
*every* β, and **every one of those five differences is significant** (p ≤ 0.010). The mechanism is
not merely unhelpful here — it is a consistent, reproducible cost across the whole range.

**What to be careful about.** β = 1.0 makes both arms worse (71.8, 40.1). Don't read that as "the
anchor matters after all" — it is just an over-strong regulariser crushing the policy. The
informative region is β ∈ [0, 0.1], where the anchor is doing its intended job and changes nothing.

---

### `lr_perm_sweep`

![](plots/figures_pt_full/lr_perm_sweep.png)

**What you're looking at.** How fast the permanent network is allowed to learn, from frozen (left)
to fast (right), with the shrinkage held **exactly** fixed in every arm. Run under steady
one-directional drift — the regime most favourable to the permanent.

**What it says.** Every single point sits **below** the vanilla line (126.1). There is no learning
rate at which the permanent produces a net benefit — all p ≤ 0.001.

| `lr_perm` | 0 (frozen) | 3e−5 | 1e−4 | 3e−4 | 1e−3 |
|---|---:|---:|---:|---:|---:|
| score | 107.5 | 79.7 | **44.1** | 112.7 | 115.7 |

**The shape is the finding, not the level.** The curve **sags in the middle**. A permanent learning
*slowly* (44.1) is far worse than one frozen (107.5) *or* one learning fast (115.7).

**Why:** a **stale anchor**. Frozen, the permanent is a stable reference the policy can be pulled
toward cheaply. Fast, it tracks the drift and stays roughly current. In between it does neither —
it lags, so the KL term drags the policy toward where the task *used to be*. Under monotone drift
that is exactly the wrong direction.

**Why this experiment is the clean one.** `lr_perm` is **independent of ρ**, so the shrinkage is
byte-identical in all five arms. This is the separation that ρ could not give us (see
`decoupling`), and it is the last axis on which the mechanism could have shown a benefit.

---

## 3. The mechanism, tested directly

### `mechanism_by_regime`

![](plots/figures_pt_full/mechanism_by_regime.png)

**What you're looking at.** One bar per regime. Each bar is **live minus frozen** — the mechanism's
own contribution with everything else identical. Left of zero = it hurts. Grey = not significant.

**What it says.** Negative in every regime that reaches significance. Positive in exactly one —
linear monotone drift — which the `decoupling` and `lr_perm_sweep` figures then close off.

**Read the units carefully.** Effects are in **standard deviations across seeds (Cohen's d)**, not
percentages. This is not cosmetic: percent-of-baseline was tried first and produced **−987%** on
one regime purely because the frozen arm's score sat near zero. Any ratio with a small denominator
explodes. If someone asks why the axis isn't in percent, that is the answer.

**What to be careful about.** A grey (non-significant) bar means *we cannot tell*, not *no effect*
— the same caution that applies to the HalfCheetah headline figure. The argument rests on the
coloured bars.

---

### `decoupling`

![](plots/figures_pt_full/decoupling.png)

**What you're looking at.** Under monotone drift the permanent helps (+8.2, p = 0.010) but the
shrinkage hurts (−18.6) — so can we keep one and drop the other? Three settings, each showing
frozen beside live.

**What it says.** No. Weakening the shrinkage weakens the permanent's benefit in lockstep, because
one parameter (ρ) controls both. And forcing them apart outright — the rightmost pair — **breaks
the agent**: live drops to **−82.5** against vanilla's **126.1**.

**Why it breaks, in one line.** Consolidation is a *transfer*: the permanent gains ρ·μ_T and the
transient must lose it, so the total is unchanged. Decoupling makes it a *copy* — the permanent
gains but the transient keeps everything — so the composed policy grows by ρ·μ_T every cycle. Over
44 cycles it explodes.

**The sanity check that makes this trustworthy.** The rightmost **frozen** bar is an agent with
*neither* a learning permanent *nor* shrinkage. It lands at **123.1** against vanilla's **126.1** —
right where it should. If that control had drifted, the whole comparison would have been suspect.
Point at it if someone doubts the −82.5.

---

## 4. Is the mechanism even running?

### `consolidation_internals`

![](plots/figures_pt_full/consolidation_internals.png)

**What you're looking at.** The method's own internal telemetry over the run. Panel (a): how much
of the transient the permanent actually absorbs on the **critic** — 1.0 means it absorbed exactly
what it was asked to. Panel (b): the Robbins–Monro learning-rate schedule. Panel (c): the same
absorption question on the **policy**.

**What it says.** The live arm absorbs **0.6–1.0** throughout on both networks; the frozen control
is flat at **exactly zero**. The mechanism is genuinely running, and the control is genuinely off.

**Why this figure exists, and why it matters more than it looks.** Without it, "the mechanism
doesn't help" could just mean "the mechanism never ran." That is not hypothetical — **it happened
to us.** An earlier control was mislabelled as "mechanism off" while the shrinkage kept running the
whole time, and it invalidated a week of conclusions. This is the plot that would have caught it.

**A note on the log axis.** The frozen arm is exactly 0, which cannot be drawn on a log scale, so
it is stated in text on the figure rather than shown as a line. Don't read its absence as missing
data.

---

### `consolidation_loss_curves`

![](plots/figures_pt_full/consolidation_loss_curves.png)

**What you're looking at.** Four consolidation cycles sampled across training. Each panel is the
internal regression's error as it fits, left to right *within* that cycle.

**What it says.** Every curve descends. Consolidation is not silently failing to fit its target.

**What it is for.** Together with the previous figure this closes off "the implementation is
broken" as an explanation. One says the transfer happens; this says the fit succeeds. Combined with
the eight-constraint audit (all pass), the negative result cannot be dismissed as a bug.

**What it does *not* show.** A descending loss means the permanent learned *its target*. It says
nothing about whether that target is *useful* — and the study's finding is precisely that it isn't.
Don't let anyone use this figure as evidence the mechanism works.

---

## 5. Standard training curves

### `return_curves`

![](plots/figures_pt_full/return_curves.png)

**What you're looking at.** Score over the whole run for the four HalfCheetah agents. The vertical
lines are the four moments the task flips direction.

**What it says.** Watch what happens at each vertical line. The per-phase medians make it explicit:

| arm | phase 1 | 2 | 3 | 4 | 5 | peak |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | **1632** | 591 | 275 | 396 | **48** | 3206 |
| PPO + shrinkage | 471 | 917 | 694 | 934 | **711** | 1154 |
| `pt_full` live | 1509 | 346 | 495 | 837 | 757 | 3156 |
| `pt_full` frozen | 299 | 767 | 537 | 796 | 551 | 980 |

**This is the whole trade in one table.** Vanilla starts highest (1632) and *decays with every
switch* — 1632 → 591 → 275 → … → 48. The shrinking arm starts lowest (471) and **climbs**, ending
at 711. It never reaches vanilla's opening height, and it doesn't need to.

**The peak column is the price.** Vanilla peaks at 3206, the shrinking arm at 1154 — barely a
third. You are buying recoverability with peak performance.

**Note also:** `pt_full` live has vanilla's peak (3156) and vanilla's phase-1 score (1509), because
early on its permanent is still near zero and it behaves like vanilla. The shrinkage only starts
dominating later.

---

### `phase_means_main`

![](plots/figures_pt_full/phase_means_main.png)

**What you're looking at.** The same runs as bars — one per task phase per agent. The +1/−1 under
each group is which direction was rewarded.

**What it says.** The same trend as the curves, easier to read off: the gap opens after the first
switch and widens. Use this one in a slide; use `return_curves` when someone wants the detail.

**What to be careful about.** Phase 1 flatters vanilla and flatters `pt_full`. If you show only
phase 1, the conclusion inverts. Always show all five, and say the direction of travel out loud.

---

### `boundary_drop`

![](plots/figures_pt_full/boundary_drop.png)

**What you're looking at.** How far the score collapses immediately after a task flip. **Shorter
bars are better** — less disruption.

**What it says.** The project's conventional stability measure, included so this study can be
compared against the earlier ones on the same terms.

**A caution worth stating before someone else does.** This metric partly tracks how high the agent
was *before* the flip — an agent scoring badly cannot drop far. Vanilla's late-run bars look
respectable partly because it has already collapsed to 48 and has nothing left to lose. **Never
present this figure without `return_curves` next to it.**

---

## 6. Two figures about our own mistakes

### `benchmark_saturation`

![](plots/figures_pt_full/benchmark_saturation.png)

**What you're looking at.** How much each benchmark's score actually moves once the agent has
learned. If a benchmark barely moves, it cannot tell two methods apart.

**What it says.** The original smooth-drift environment swings only **3%** — every agent sat at
96–99% of the maximum possible score. We had measured a "significant benefit" on it (+5.6,
p = 0.000). Once the benchmark was fixed so the goal moves too (27%, then 80%), that benefit
**reversed sign**.

**The lesson, and it generalises.** A tiny p-value on a saturated benchmark measures the benchmark,
not the method. **Check the dynamic range before believing a small significant difference.** This
is one of three claims the study retracted about its own earlier results.

---

### `sigma_collapse`

![](plots/figures_pt_full/sigma_collapse.png)

**What you're looking at.** How much randomness each agent keeps in its actions over training.
Higher = still exploring; lower = always doing the same thing.

**What it says.** The agents' exploration levels **diverge over training**, and the final scores
rank in exactly the same order as the exploration levels. EWC's penalty happens to cover the
exploration parameter, so it freezes exploration **as a side effect** — nothing to do with the
weight protection it is supposed to provide.

**Why this figure exists.** It is a **warning about a confound**, not a result. Any comparison
between these agents has to hold exploration fixed, or it measures the exploration schedule rather
than the method.

**And it bit us.** A config key that one agent read and another ignored gave `pt_full` σ = 0.368
against vanilla's σ = 1.000 in one sweep, producing a spectacular fake win of **+1277, p = 0.002**.
It was caught only because the vanilla numbers were identical to a previous run to four significant
figures. **Every HalfCheetah comparison in this study now freezes σ and asserts on the realised
value before launch**, not on the config file.

---

## The one-paragraph version

`pt_full` beats plain PPO. Figures 1–2 show that plain PPO plus one copied step — periodically
shrinking the policy — matches it at our settings and significantly beats it at the published
configuration. Figures 3–5 show where that came from: a clean dose–response in the shrink strength,
with the KL anchor and the permanent's learning rate both ruled out. Figures 6–7 test the
permanent–transient mechanism on its own and find it a cost nearly everywhere, and inseparable from
the shrinkage where it helps. Figures 8–9 confirm the mechanism was genuinely running throughout,
so this is not an implementation failure. Figures 10–12 give the conventional training views.
Figures 13–14 record two places our own measurements misled us before the controls caught them.

**And the caveat to carry into every conversation:** the medians on figure 1 favour `pt_full`. That
is real, it is not significant, and the reason is visible in the dots. Say it before you are asked.
