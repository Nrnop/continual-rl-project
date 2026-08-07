"""Analysis for the transmission hypothesis: does PT's critic knowledge reach the policy?

Reads the per-seed `*_scalars.pkl` written by `utils/logger.save_scalars` and answers two
questions that the return curves alone cannot.

D1 — CRITIC QUALITY vs BEHAVIOUR
    In DQN the value function *is* the policy, so a better critic is better behaviour by
    construction. In an actor-critic the critic reaches behaviour only through the advantage, so
    the two can come apart. If PT's `diag/explained_var` is at least as good as vanilla's in the
    window after each switch while PT's return is worse, the deficit is a transmission problem,
    not a value-learning problem. That is a positive finding, not another null: it says the
    mechanism works and the architecture cannot cash it.

    Read the three outcomes as:
      PT critic BETTER/EQUAL + PT return WORSE  -> transmission gap confirmed
      PT critic WORSE                           -> the critic is the problem; hypothesis fails
      both equal                                -> the mechanism is simply not doing anything

D2 — IS THE COST LOCKED TO THE CONSOLIDATION GRID?
    Consolidation's target is `old_V_perm + V_trans` (keep = 1), which is deliberately not
    value-preserving: right after it, V = old_P + T + decay*T, an overshoot the fast transient is
    meant to correct. In DQN that displacement buys the instantaneous jumpstart. In PPO nothing is
    bought, so it should show up as a pure cost — a dip in return and a spike in advantage
    magnitude at low `diag/consol_age`, on the k=60 grid, AWAY from any task boundary.

    Boundary-adjacent updates are excluded, so a dip here cannot be non-stationarity. Vanilla is
    binned against the same synthetic k-grid as a control: it has no consolidation, so any
    structure appearing there is an artifact of the binning and invalidates the PT reading.

Usage (from the PARENT of src_continuous_control/):
    python -m src_continuous_control.scripts.analyze_transmission \
        --results-dir trans_results --arms vanilla pt pt_inert
"""
import argparse
import glob
import math
import os
import pickle
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------
def load_seed_scalars(results_dir, arm):
    """{seed: {series_name: (steps, values)}} for one arm.

    `save_scalars` writes {name: array of (step, value) pairs}, so each series carries its own
    step axis — series logged at different cadences (eval, consolidation) stay aligned correctly.
    """
    pattern = os.path.join(results_dir, arm, "*_scalars.pkl")
    out = {}
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        seed = None
        for part in base.split("_"):
            if part.isdigit():
                seed = int(part)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        series = {}
        for name, arr in payload.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] == 2:
                series[name] = (a[:, 0], a[:, 1])
        out[seed if seed is not None else len(out)] = series
    return out


def series(seed_data, name):
    """(steps, values) for one series, or (None, None) if the run never logged it."""
    got = seed_data.get(name)
    if got is None:
        return None, None
    return got


# ----------------------------------------------------------------------------------
# Stats — implemented here rather than imported
# ----------------------------------------------------------------------------------
# scipy is NOT in requirements_continuous.txt and is not installed in the training venv, so
# importing it would make this script silently print `nan` for every p-value on the box that
# actually runs the sweeps. Both tests below are exact for the sample sizes this project uses
# (n <= 10 per arm) and match scipy's exact methods.
#
# Rank statistics throughout, not t-tests: REINVESTIGATION.md §5 records that per-seed return on
# this benchmark is not Gaussian and that five separate criteria in this project ended up
# selecting on seed noise.
def _mw_null_counts(n, m):
    """Counts of the exact Mann-Whitney U null distribution over u = 0 .. n*m.

    The number of arrangements giving U = u is the number of partitions of u into at most n parts
    each of size at most m (the Gaussian binomial coefficient), built by the standard recurrence
    C(n,m)[u] = C(n-1,m)[u-m] + C(n,m-1)[u].
    """
    prev_row = [np.array([1.0])] * (m + 1)             # C(0, j) = [1] for every j
    for i in range(1, n + 1):
        row = [np.array([1.0])]                        # C(i, 0) = [1]
        for j in range(1, m + 1):
            a = prev_row[j]                            # C(i-1, j), shifted right by j
            b = row[j - 1]                             # C(i, j-1)
            size = max(len(a) + j, len(b))
            out = np.zeros(size)
            out[j:j + len(a)] += a
            out[:len(b)] += b
            row.append(out)
        prev_row = row
    return prev_row[m]


def mw(a, b, exact=False):
    """Two-sided Mann-Whitney U: (U for sample `a`, p).

    DEFAULT IS THE NORMAL APPROXIMATION WITHOUT A CONTINUITY CORRECTION, deliberately: that is
    the method behind every p-value already published in REINVESTIGATION.md, and it reproduces
    them to the digit at n=10 (U=5 -> 0.001, U=9 -> 0.002, U=28 -> 0.096, U=43 -> 0.597,
    U=29 -> 0.112, U=47 -> 0.821). Using the exact test here instead would put p=0.63 in a new
    report where every existing table says 0.597 for the same data — a discrepancy that reads as
    a contradiction rather than as a different estimator.

    `exact=True` runs the exact permutation test (no ties, n*m <= 400). It is the more accurate
    test and is worth quoting when a p sits near a decision threshold; it is simply not the
    project's default, so mixing the two silently would be worse than either.
    """
    a = np.array([x for x in a if np.isfinite(x)], dtype=float)
    b = np.array([x for x in b if np.isfinite(x)], dtype=float)
    n, m = len(a), len(b)
    if n < 2 or m < 2:
        return float("nan"), float("nan")

    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sortedv = allv[order]
    i = 0
    tie_term = 0.0
    while i < len(sortedv):                            # midranks for ties
        j = i
        while j + 1 < len(sortedv) and sortedv[j + 1] == sortedv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1

    u_a = float(ranks[:n].sum() - n * (n + 1) / 2.0)

    if exact and tie_term == 0.0 and n * m <= 400:
        counts = _mw_null_counts(n, m)
        total = counts.sum()
        u_small = min(u_a, n * m - u_a)
        p = 2.0 * counts[:int(round(u_small)) + 1].sum() / total
        return u_a, float(min(p, 1.0))

    mu = n * m / 2.0
    sigma_sq = n * m * (n + m + 1) / 12.0
    if tie_term:
        N = n + m
        sigma_sq = (n * m / 12.0) * ((N + 1) - tie_term / (N * (N - 1)))
    if sigma_sq <= 0:
        return u_a, float("nan")
    z = abs(u_a - mu) / np.sqrt(sigma_sq)
    p = math.erfc(z / np.sqrt(2.0))                    # = 2 * (1 - Phi(z))
    return u_a, float(min(p, 1.0))


def wilcoxon(x, y):
    """Two-sided exact Wilcoxon signed-rank p for paired samples; zero differences dropped.

    Used for the D2 within-arm contrast, where the same seed contributes both bins — the pairing
    removes between-seed variance, which on this benchmark dominates everything else.
    """
    d = np.array([xx - yy for xx, yy in zip(x, y)
                  if np.isfinite(xx) and np.isfinite(yy)], dtype=float)
    d = d[d != 0.0]
    n = len(d)
    if n < 3:
        return float("nan")

    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(n)
    absd = np.abs(d)[order]
    i = 0
    ties = False
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        ties |= j > i
        i = j + 1
    w_plus = float(ranks[d > 0].sum())

    if not ties and n <= 25:
        # Exact: count subsets of {1..n} summing to each value (signed-rank null).
        counts = np.zeros(n * (n + 1) // 2 + 1)
        counts[0] = 1.0
        for r in range(1, n + 1):
            counts[r:] += counts[:-r].copy()
        total = counts.sum()
        w_small = min(w_plus, n * (n + 1) / 2.0 - w_plus)
        return float(min(2.0 * counts[:int(round(w_small)) + 1].sum() / total, 1.0))

    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma <= 0:
        return float("nan")
    z = (abs(w_plus - mu) - 0.5) / sigma
    return float(min(2.0 * 0.5 * math.erfc(z / np.sqrt(2.0)), 1.0))


def fmt(vals):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return "   n/a  "
    return f"{np.median(vals):8.4f}"


# ----------------------------------------------------------------------------------
# D1 — critic quality vs behaviour in the post-switch window
# ----------------------------------------------------------------------------------
def d1(arms_data, switch, n_switches, window_updates, steps_per_update):
    """Per-seed medians of explained_var / value_rmse / return inside each post-switch window."""
    print("=" * 88)
    print("D1 - CRITIC QUALITY vs BEHAVIOUR, in the window after each switch")
    print("=" * 88)
    print(f"window = {window_updates} updates (~{window_updates * steps_per_update} env steps) "
          f"after each of {n_switches} switches\n")

    # EXPLAINED VARIANCE IS POOLED, NOT AVERAGED, AND THIS IS NOT A DETAIL.
    #
    # `diag/explained_var` is a per-update ratio 1 - Var(A)/Var(R). Its denominator is the return
    # variance *within a single rollout*, which on a collapsed policy can be near zero — so the
    # per-update value can go arbitrarily negative and the MEAN of it is dominated by that tail.
    # On the first sweep, 31% of PT's post-switch updates had EV < 0 against 11% of vanilla's,
    # and averaging turned two statistically indistinguishable critics (pooled 0.769 vs 0.721,
    # p=0.41) into "PT's critic is significantly worse, p=0.007". The wrong sign, with a p-value
    # attached.
    #
    # Pooling aggregates the variances over the whole window and forms the ratio ONCE:
    #     EV = 1 - sum(Var(A)) / sum(Var(R))
    # which is the standard definition over a set of batches and is not hostage to the smallest
    # denominator in it. `diag/adv_std` and `diag/return_std` are logged precisely so this can be
    # reconstructed. The mean is still reported alongside, labelled, so the two never get
    # confused again.
    per_arm = {}
    for arm, seeds in arms_data.items():
        acc = defaultdict(list)                       # metric -> per-seed value
        for _seed, sd in sorted(seeds.items()):
            win_hi = window_updates * steps_per_update

            def _window(st):
                m = np.zeros(len(st), dtype=bool)
                for b in range(1, n_switches + 1):
                    m |= (st >= b * switch) & (st < b * switch + win_hi)
                return m

            st_a, adv_sd = series(sd, "diag/adv_std")
            st_r, ret_sd = series(sd, "diag/return_std")
            if st_a is not None and st_r is not None:
                m = _window(st_a)
                denom = np.nansum(ret_sd[m] ** 2)
                if m.any() and denom > 1e-12:
                    acc["ev"].append(float(1.0 - np.nansum(adv_sd[m] ** 2) / denom))
                    acc["err"].append(float(np.nanmean(adv_sd[m])))

            for metric, tag in (("diag/explained_var", "ev_mean"),
                                ("train/avg_return", "ret")):
                st, vv = series(sd, metric)
                if st is None:
                    continue
                m = _window(st)
                if m.any():
                    acc[tag].append(float(np.nanmean(vv[m])))
        per_arm[arm] = acc

    print(f"{'arm':<12} {'EV (pooled)':>12} {'abs error':>10} {'return':>12} "
          f"{'[EV mean]':>10}   n")
    for arm, acc in per_arm.items():
        n = max((len(v) for v in acc.values()), default=0)
        print(f"{arm:<12} {fmt(acc['ev']):>12} {fmt(acc['err']):>10} "
              f"{fmt(acc['ret']):>12} {fmt(acc['ev_mean']):>10}   {n}")
    print("  EV (pooled) and abs error are the critic-quality measures. [EV mean] is the")
    print("  average of the per-update ratio and is shown ONLY to make the difference visible;")
    print("  it is unstable when return variance is small and must not be tested on.")

    base = "vanilla"
    if base in per_arm:
        print()
        for arm, acc in per_arm.items():
            if arm == base:
                continue
            for tag, label in (("ev", "EV (pooled)"), ("err", "abs error"), ("ret", "return")):
                u, p = mw(acc[tag], per_arm[base][tag])
                print(f"  {arm:<10} vs {base:<9} {label:<12} U={u:6.1f}  p={p:.4f}")

        print("\n  VERDICT:")
        for arm, acc in per_arm.items():
            if arm == base or not acc["ev"] or not acc["ret"]:
                continue
            ev_d = np.median(acc["ev"]) - np.median(per_arm[base]["ev"])
            ret_d = np.median(acc["ret"]) - np.median(per_arm[base]["ret"])
            _, ev_p = mw(acc["ev"], per_arm[base]["ev"])
            _, ret_p = mw(acc["ret"], per_arm[base]["ret"])
            worse_ret = ret_d < 0 and np.isfinite(ret_p) and ret_p < 0.05
            # "not worse" means NOT SIGNIFICANTLY WORSE, which is the claim the hypothesis needs.
            # A point estimate below vanilla's with p=0.4 is not evidence of a worse critic.
            critic_ok = ev_d >= 0 or (np.isfinite(ev_p) and ev_p > 0.05)
            if worse_ret and critic_ok:
                verdict = "TRANSMISSION GAP - critic not worse, behaviour worse"
            elif ev_d < 0 and np.isfinite(ev_p) and ev_p < 0.05:
                verdict = "critic IS worse - transmission hypothesis fails here"
            else:
                verdict = "inconclusive - no significant separation on either axis"
            print(f"    {arm:<10} dEV={ev_d:+.4f} (p={ev_p:.3f})  "
                  f"dRET={ret_d:+8.1f} (p={ret_p:.3f})  ->  {verdict}")
    print()


# ----------------------------------------------------------------------------------
# D2 — is the cost locked to the consolidation grid?
# ----------------------------------------------------------------------------------
def d2(arms_data, switch, n_switches, steps_per_update, k_fallback, early, guard_updates):
    """Event study around each consolidation: the W updates after it minus the W before it.

    THIS REPLACES AN EARLIER EARLY/LATE BINNING THAT DID NOT WORK, and the reason it failed is
    worth keeping. That version binned `train/avg_return` by position in the k-cycle. Two
    problems, discovered when the vanilla control — which never consolidates — showed the LARGEST
    effect of the three arms (+193.9 against PT's +127.8, p=0.002 for both):

      1. `train/avg_return` is a 0.99 EMA. An EMA of a rising series rises monotonically, so any
         binning of it recovers the local slope and nothing else. Subtracting the per-cycle mean
         removes the level but NOT the within-cycle slope, which is precisely what puts low ages
         below the cycle mean and high ages above it.
      2. Vanilla trains fastest here, so it has the steepest slope, so it showed the biggest
         "consolidation effect" — in an agent with no consolidation.

    An adjacent-window event study on the same EMA still failed (+42.4 vanilla vs +24.3 PT),
    which is what established that the metric, not the binning, was the problem. So D2 now
    requires `diag/raw_return`: the mean of the episodes that actually finished in that rollout,
    unsmoothed. Comparing the W updates immediately after a consolidation with the W immediately
    before cancels any trend to first order, and on an unsmoothed signal a real one-off
    displacement has somewhere to show up.

    Vanilla is still binned against a synthetic grid of the same period, and it is still the
    control: if it shows the same effect, the result is void regardless of what PT shows.
    """
    print("=" * 88)
    print("D2 - IS THERE A COST LOCKED TO EACH CONSOLIDATION?")
    print("=" * 88)
    print(f"event study: mean raw return over the {early} updates AFTER each consolidation,")
    print(f"  minus the {early} updates immediately BEFORE it (adjacent, so trend cancels)")
    print(f"events within {guard_updates} updates of a task switch are EXCLUDED")
    print("vanilla uses a synthetic grid of the same period and is the CONTROL")
    print()

    # EVERY seed must carry the series, not merely one of them. `any()` here would let a folder
    # holding a MIXTURE of old and new pkls — identical filenames, different code versions —
    # pass the check and get pooled silently. The runner now refuses to write into a populated
    # directory for the same reason; this is the second line of defence.
    raw_missing = []
    for a, seeds in arms_data.items():
        bad = sorted(s for s, sd in seeds.items() if "diag/raw_return" not in sd)
        if bad:
            raw_missing.append(f"{a} (seeds {', '.join(map(str, bad))} of {len(seeds)})")
    if raw_missing:
        print(f"  CANNOT BE COMPUTED: no `diag/raw_return` in {'; '.join(raw_missing)}.")
        print("  Those runs predate it. D2 is NOT being reported on `train/avg_return` instead:")
        print("  that is the EMA whose slope produced a false positive in the vanilla control")
        print("  on the first sweep. Re-run with the current train.py.")
        print()
        return

    print(f"{'arm':<12} {'k':>4} {'events':>7} {'after-before':>13} {'p':>8}")
    for arm, seeds in arms_data.items():
        per_seed, ks, n_ev = [], [], 0
        for _seed, sd in sorted(seeds.items()):
            st, raw = series(sd, "diag/raw_return")
            if st is None:
                continue
            st_a, age = series(sd, "diag/consol_age")
            _, kk = series(sd, "diag/consol_k")
            if st_a is None:
                k = k_fallback
                age = (np.arange(len(st)) % k).astype(np.float64)
            else:
                k = int(kk[0]) if kk is not None and len(kk) else k_fallback
                idx = np.searchsorted(st_a, st)
                idx = np.clip(idx, 0, len(age) - 1)
                age = age[idx]
            ks.append(k)

            # A consolidation is where the age counter resets.
            events = np.where(np.r_[False, age[1:] <= age[:-1]])[0]
            guard = guard_updates * steps_per_update
            diffs = []
            for i in events:
                if i < early or i + early > len(raw):
                    continue
                if any(abs(st[i] - b * switch) <= guard for b in range(1, n_switches + 1)):
                    continue
                before, after = raw[i - early:i], raw[i:i + early]
                if np.isnan(before).all() or np.isnan(after).all():
                    continue                            # no episode finished in either window
                diffs.append(float(np.nanmean(after) - np.nanmean(before)))
            if diffs:
                per_seed.append(float(np.mean(diffs)))
                n_ev += len(diffs)

        if not per_seed:
            print(f"{arm:<12} {'-':>4} {'no events':>7}")
            continue
        p = wilcoxon(per_seed, [0.0] * len(per_seed))
        print(f"{arm:<12} {int(np.median(ks)):>4} {n_ev:>7} {np.median(per_seed):+13.2f} "
              f"{p:8.4f}")

    print()
    print("  How to read this:")
    print("    vanilla ~ 0 and PT/pt_inert negative -> a real per-consolidation cost.")
    print("    vanilla shows the same sign and size -> VOID, whatever PT shows. Not negotiable:")
    print("      the control already produced one false positive on this exact question.")
    print("    nobody shows anything -> consolidation costs nothing measurable per cycle.")
    print()


# ----------------------------------------------------------------------------------
# D3 — how much of the policy's update signal is the critic, and which part of it?
# ----------------------------------------------------------------------------------
def d3(arms_data, switch, n_switches, window_updates, steps_per_update):
    """Covariance shares of the advantage, whole-run and in the post-switch window.

    `share_perm + share_trans` is the critic's total influence on decision-making, because the
    advantage is the critic's only channel to the policy. `1 - adv_corr_nocritic` says the same
    thing scale-free: how much the DIRECTION of the policy update owes to having a critic at all.

    The post-switch column is the one the theory speaks to. Theorem 6's jumpstart is a claim about
    behaviour immediately after a switch, so if the permanent's share is negligible exactly there,
    the jumpstart has no route to the policy no matter how good the permanent is.
    """
    print("=" * 88)
    print("D3 - HOW MUCH OF THE POLICY'S UPDATE SIGNAL IS THE CRITIC?")
    print("=" * 88)
    print("covariance shares of Var(A). they sum to 1 exactly on EVERY update; the medians")
    print("  below are medians of per-seed means, so they need not sum to 1 - that is not a bug.")
    print("for VANILLA, 'perm' is the whole critic (its transient is identically zero).")
    print("on an abl_*_advsrc_* arm these describe the FULL advantage, not the reduced one the")
    print("  actor was trained on - the shares say what the critic offered, the arm says what")
    print("  happened when it was withheld.")
    print(f"post-switch window = {window_updates} updates after each of {n_switches} switches\n")

    hdr = (f"{'arm':<12} {'scope':<12} {'reward':>8} {'perm':>8} {'trans':>8} "
           f"{'critic':>8}   {'corr_noCritic':>13} {'corr_noPerm':>12}")
    print(hdr)
    for arm, seeds in arms_data.items():
        for scope in ("whole-run", "post-switch"):
            acc = defaultdict(list)
            for _seed, sd in sorted(seeds.items()):
                for tag in ("adv_share_reward", "adv_share_perm", "adv_share_trans",
                            "adv_corr_nocritic", "adv_corr_noperm"):
                    st, vv = series(sd, f"diag/{tag}")
                    if st is None:
                        continue
                    if scope == "whole-run":
                        sel = np.ones(len(st), dtype=bool)
                    else:
                        sel = np.zeros(len(st), dtype=bool)
                        hi = window_updates * steps_per_update
                        for b in range(1, n_switches + 1):
                            sel |= (st >= b * switch) & (st < b * switch + hi)
                    if sel.any():
                        acc[tag].append(float(np.nanmean(vv[sel])))
            if not acc:
                continue
            r = np.median(acc["adv_share_reward"]) if acc["adv_share_reward"] else np.nan
            p = np.median(acc["adv_share_perm"]) if acc["adv_share_perm"] else np.nan
            t = np.median(acc["adv_share_trans"]) if acc["adv_share_trans"] else np.nan
            cn = np.median(acc["adv_corr_nocritic"]) if acc["adv_corr_nocritic"] else np.nan
            cp = np.median(acc["adv_corr_noperm"]) if acc["adv_corr_noperm"] else np.nan
            print(f"{arm:<12} {scope:<12} {r:8.4f} {p:8.4f} {t:8.4f} {p + t:8.4f}   "
                  f"{cn:13.4f} {cp:12.4f}")

    print("\n  How to read this:")
    print("    'critic' is the share of the policy's update signal that comes from the critic")
    print("      at all. Compare it against VANILLA's - that is the ceiling any critic-side")
    print("      mechanism can play for on this task.")
    print("    'perm' is the permanent's ENTIRE influence on decision-making. Note it carries a")
    print("      temporal difference of V_perm, not its level, and advantage normalisation then")
    print("      deletes any constant offset - the permanent is attenuated twice before it")
    print("      reaches the actor.")
    print("    corr_noPerm ~ 1.000 means deleting the permanent from the advantage does not")
    print("      change the direction of the policy update. Paired with the causal arm")
    print("      abl_pt_advsrc_trans (which does exactly that and retrains), that is the")
    print("      quantitative answer to 'how much power does the permanent have'.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="trans_results")
    ap.add_argument("--arms", nargs="+", default=["vanilla", "pt", "pt_inert"])
    ap.add_argument("--switch", type=int, default=614400,
                    help="env steps per task phase (default matches the 3.07M / 5-phase runs)")
    ap.add_argument("--n-switches", type=int, default=4)
    ap.add_argument("--window-updates", type=int, default=20,
                    help="D1 post-switch window, in PPO updates (matches JumpstartTracker)")
    ap.add_argument("--steps-per-update", type=int, default=2048,
                    help="n_steps * num_envs (256 * 8 for the standard config)")
    ap.add_argument("--k", type=int, default=60,
                    help="consolidation period; only used for arms that do not log diag/consol_k")
    ap.add_argument("--early", type=int, default=5,
                    help="D2: consol_age strictly below this counts as 'just consolidated'")
    ap.add_argument("--guard-updates", type=int, default=25,
                    help="D2: updates excluded on each side of every task switch")
    args = ap.parse_args()

    arms_data = {}
    for arm in args.arms:
        data = load_seed_scalars(args.results_dir, arm)
        if not data:
            print(f"[warn] no *_scalars.pkl under {os.path.join(args.results_dir, arm)}")
            continue
        arms_data[arm] = data

    if not arms_data:
        raise SystemExit("no data found — check --results-dir and --arms")

    print(f"loaded: " + ", ".join(f"{a} (n={len(d)})" for a, d in arms_data.items()))
    have_diag = any("diag/explained_var" in sd
                    for d in arms_data.values() for sd in d.values())
    if not have_diag:
        raise SystemExit(
            "these runs predate the diag/ instrumentation - no diag/explained_var series found.\n"
            "Re-run with the updated agents/ppo_base.py, agents/ppo_pt.py and train.py.")
    print()

    d1(arms_data, args.switch, args.n_switches, args.window_updates, args.steps_per_update)
    d2(arms_data, args.switch, args.n_switches, args.steps_per_update, args.k,
       args.early, args.guard_updates)
    d3(arms_data, args.switch, args.n_switches, args.window_updates, args.steps_per_update)


if __name__ == "__main__":
    main()
