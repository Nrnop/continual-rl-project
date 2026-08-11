"""Analysis for Stages 3-7. One entry point, one statistical convention.

Statistics: exact two-sided Mann-Whitney from the full permutation null (scipy is not installed
in this venv, and at n=8 vs 8 the exact enumeration is both cheap and better than a normal
approximation). Medians throughout, never means -- return distributions here are heavy-tailed
and the project has twice been misled by averaging (see TRANSMISSION_RESULTS.md §1).
"""
import glob
import itertools
import pickle
import sys

import numpy as np

ROOT = r"e:/update-single task + videos"
PHASE, NPHASE = 40_000, 9


def mw(a, b):
    """Exact two-sided Mann-Whitney p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pool = np.concatenate([a, b])
    r = pool.argsort().argsort().astype(float) + 1
    for v in np.unique(pool):
        m = pool == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    obs = r[:len(a)].sum()
    null = np.array([r[list(c)].sum() for c in itertools.combinations(range(len(pool)), len(a))])
    return float((np.abs(null - null.mean()) >= abs(obs - null.mean()) - 1e-9).mean())


def series(stage, cell, key):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/{stage}_results/{cell}/*_scalars.pkl")):
        d = pickle.load(open(f, "rb"))
        if key in d:
            out.append(d[key])
    return out


def final_return(stage, cell, frac=1.0 / NPHASE):
    """Mean return over the final phase (or final `frac` of the run for boundary-free drift)."""
    vals = []
    for s in series(stage, cell, "train/avg_return"):
        st, v = s[:, 0], s[:, 1]
        cut = st.max() * (1.0 - frac)
        vals.append(float(v[st >= cut].mean()))
    return np.array(vals)


def overall_return(stage, cell):
    return np.array([float(s[:, 1].mean()) for s in series(stage, cell, "train/avg_return")])


def scalar(stage, cell, key, how="last"):
    out = []
    for s in series(stage, cell, key):
        out.append(float(s[-1, 1] if how == "last" else np.median(s[:, 1])))
    return np.array(out) if out else np.array([np.nan])


def line(label, arr, ref=None, width=26):
    txt = f"  {label:<{width}}{np.median(arr):>10.1f}  (n={len(arr)})"
    if ref is not None:
        txt += f"   vs ref {np.median(arr) - np.median(ref):+9.1f}  p={mw(arr, ref):.3f}"
    print(txt)


def stage3():
    print("\n" + "=" * 100)
    print("STAGE 3 -- KL-ANCHOR (beta) SWEEP at centroid E=0.  Is the anchor the active ingredient?")
    print("=" * 100)
    ref = final_return("stage2", "van_L00")          # vanilla, sigma frozen, no anchor
    print(f"  reference: vanilla (sigma frozen, no anchor) = {np.median(ref):.1f}\n")
    print(f"  {'beta':<10}{'FROZEN perm':>12}{'p vs van':>10}{'LIVE perm':>12}{'p vs van':>10}"
          f"{'  LIVE-FROZEN':>13}{'p':>8}")
    for lab, b in [("b000", 0.0), ("b0001", 0.001), ("b001", 0.01), ("b01", 0.1), ("b1", 1.0)]:
        i, p = final_return("stage3", f"inert_{lab}"), final_return("stage3", f"pt_{lab}")
        if not len(i) or not len(p):
            continue
        print(f"  {b:<10}{np.median(i):>12.1f}{mw(i, ref):>10.3f}{np.median(p):>12.1f}"
              f"{mw(p, ref):>10.3f}{np.median(p) - np.median(i):>+13.1f}{mw(p, i):>8.3f}")
    print("\n  consolidation_shuffle (audit defect) at beta=0.01:")
    for arm in ["pt", "inert"]:
        base, shuf = final_return("stage3", f"{arm}_b001"), final_return("stage3", f"{arm}_shuf")
        if len(shuf):
            line(f"{arm}: shuffled", shuf, base)


def stage4():
    print("\n" + "=" * 100)
    print("STAGE 4 -- SMOOTH DRIFT (no task boundaries). EWC should degenerate to vanilla here.")
    print("=" * 100)
    print("  Drift is sinusoidal, period 80k over 360k steps. Reporting BOTH the final-window")
    print("  return (last 20% ~= 0.9 of a cycle) and the whole-run mean, since under continuous")
    print("  drift there is no 'converged' phase and either alone can mislead.\n")
    ref_f, ref_o = final_return("stage4", "van", frac=0.2), overall_return("stage4", "van")
    print(f"  {'arm':<24}{'final 20%':>11}{'p':>8}{'whole run':>12}{'p':>8}")
    for arm, desc in [("van", "vanilla (sigma frozen)"), ("ewc", "EWC (degenerate here)"),
                      ("inert", "pt_full FROZEN perm"), ("pt", "pt_full LIVE perm")]:
        f, o = final_return("stage4", arm, frac=0.2), overall_return("stage4", arm)
        if not len(f):
            continue
        pf = "--" if arm == "van" else f"{mw(f, ref_f):.3f}"
        po = "--" if arm == "van" else f"{mw(o, ref_o):.3f}"
        print(f"  {desc:<24}{np.median(f):>11.1f}{pf:>8}{np.median(o):>12.1f}{po:>8}")
    i, p = final_return("stage4", "inert", frac=0.2), final_return("stage4", "pt", frac=0.2)
    if len(i) and len(p):
        print(f"\n  THE PT MECHANISM (live - frozen): {np.median(p) - np.median(i):+.1f}  p={mw(p, i):.3f}")
    e, v = overall_return("stage4", "ewc"), ref_o
    if len(e):
        print(f"  EWC vs vanilla (should be ~0 -- no boundaries, so no Fisher is ever accumulated):"
              f" {np.median(e) - np.median(v):+.1f}  p={mw(e, v):.3f}")


def stage5():
    print("\n" + "=" * 100)
    print("STAGE 5 -- THE ONE-LINE REGULARISER. vanilla + frozen sigma + KL-to-zero-prior.")
    print("            Does it reproduce pt_full's frozen-permanent arm?")
    print("=" * 100)
    target = final_return("stage3", "inert_b001")
    van = final_return("stage2", "van_L00")
    print(f"  target  pt_full FROZEN (beta=0.01) = {np.median(target):.1f}")
    print(f"  floor   vanilla (sigma frozen)    = {np.median(van):.1f}\n")
    arms = [("r0001", "actor[64,64] mu_l2=0.001"), ("r001", "actor[64,64] mu_l2=0.01"),
            ("r01", "actor[64,64] mu_l2=0.1"), ("r1", "actor[64,64] mu_l2=1.0"),
            ("h32", "actor[32,32] mu_l2=0     <- CAPACITY control"),
            ("h32_r001", "actor[32,32] mu_l2=0.01  <- capacity + anchor"),
            ("h64_r001", "actor[64,64] mu_l2=0.01  (dup of r001, sanity)")]
    for lab, desc in arms:
        a = final_return("stage5", f"van_{lab}")
        if len(a):
            print(f"  {desc:<42}{np.median(a):>9.1f}   vs pt_full frozen-permanent "
                  f"{np.median(a) - np.median(target):+8.1f} p={mw(a, target):.3f}"
                  f"   vs vanilla {np.median(a) - np.median(van):+8.1f} p={mw(a, van):.3f}")


def stage6():
    print("\n" + "=" * 100)
    print("STAGE 6 -- FREQUENCY LADDER. Both tasks identical at every level; only how often each")
    print("            is visited changes, so per-task difficulty is exactly constant.")
    print("  CAVEAT: the number of SWITCHES falls with frequency skew (f5=8, f6=5, f7=4, f8=2),")
    print("  so higher levels are easier in a way Stage 2 was not. You cannot vary E_tau over a")
    print("  two-task set without changing either the tasks or the visitation pattern. Stage 2")
    print("  and Stage 6 therefore carry DIFFERENT confounds -- the conclusion is only safe where")
    print("  they agree, which is what makes running both worthwhile.")
    print("=" * 100)
    print(f"  {'level':<7}{'fwd:bwd':>9}{'E[target]':>11}{'van':>9}{'FROZEN':>9}{'LIVE':>9}"
          f"{'  LIVE-FROZEN':>13}{'p':>8}")
    gaps, Es = [], []
    for lab, ratio, E in [("f5", "5:4", 0.222), ("f6", "6:3", 0.667),
                          ("f7", "7:2", 1.111), ("f8", "8:1", 1.556)]:
        v = final_return("stage6", f"van_{lab}")
        i = final_return("stage6", f"inert_{lab}")
        p = final_return("stage6", f"pt_{lab}")
        if not len(i) or not len(p):
            continue
        g = np.median(p) - np.median(i)
        gaps.append(g); Es.append(E)
        print(f"  {lab:<7}{ratio:>9}{E:>11.3f}{np.median(v):>9.1f}{np.median(i):>9.1f}"
              f"{np.median(p):>9.1f}{g:>+13.1f}{mw(p, i):>8.3f}")
    if len(gaps) >= 3:
        E, g = np.array(Es), np.array(gaps)
        rx, ry = E.argsort().argsort() + 1.0, g.argsort().argsort() + 1.0
        rho = np.corrcoef(rx, ry)[0, 1]
        null = [np.corrcoef(rx, ry[list(pm)])[0, 1] for pm in itertools.permutations(range(len(g)))]
        pv = float(np.mean(np.abs(np.array(null)) >= abs(rho) - 1e-12))
        print(f"\n  TREND of (live-frozen) against centroid: Spearman rho={rho:+.3f}, exact p={pv:.4f}")


def stage7():
    print("\n" + "=" * 100)
    print("STAGE 7 -- EWC: weight protection, or exploration preservation?")
    print("=" * 100)
    with_, no = final_return("stage7", "ewc_withstd"), final_return("stage7", "ewc_nostd")
    van = final_return("stage2", "van_L00")
    if len(with_) and len(no):
        line("EWC (log_std IN Fisher)", with_)
        line("EWC (log_std EXCLUDED)", no, with_)
        print(f"\n  final entropy: with={np.median(scalar('stage7','ewc_withstd','train/entropy')):.3f}"
              f"   without={np.median(scalar('stage7','ewc_nostd','train/entropy')):.3f}")
        print(f"  (Gaussian 1-D entropy = log_std + 1.4189; vanilla-frozen reference return "
              f"{np.median(van):.1f})")


if __name__ == "__main__":
    want = sys.argv[1:] or ["3", "4", "5", "6", "7"]
    for s in want:
        try:
            {"3": stage3, "4": stage4, "5": stage5, "6": stage6, "7": stage7}[s]()
        except Exception as e:                       # a stage that has not run yet
            print(f"\n[stage {s}] unavailable: {type(e).__name__}: {e}")
