"""Stage-1 analysis: horizon ladder x task-set asymmetry on DirectionalPointMass.

Reports, per cell, the numbers that decide whether a PT arm is interpretable BEFORE the
numbers that decide whether it won:

  actor_absorbed_frac  -- normalised by rho*transient, so ~1.0 means "absorbed the rho it was
                          asked to". Far below 1 = the consolidation regression is not
                          converging and the cell says nothing (defect #9's signature).
  trans/perm energy    -- ||mu_T|| / ||mu_P||. If this is >> 1 the permanent is decorative and
                          the transient is carrying the whole policy, whatever rho claims.
  retention vs controls-- mse_perm must beat BOTH mse_perm_init and mse_zero. On a
                          sign-symmetric task pair an inert permanent scores well by accident.
"""
import glob
import pickle

import itertools

import numpy as np


def mannwhitney_exact(a, b):
    """Exact two-sided Mann-Whitney p from the permutation null (n=5 vs 5 -> 252 splits).

    scipy is not installed in this venv; at these sample sizes the exact enumeration is both
    cheap and better than a normal approximation.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    pooled = np.concatenate([a, b])
    ranks = pooled.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(pooled):
        m = pooled == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_a = len(a)
    obs = ranks[:n_a].sum()
    idx = range(len(pooled))
    null = np.array([ranks[list(c)].sum() for c in itertools.combinations(idx, n_a)])
    centre = null.mean()
    return float((np.abs(null - centre) >= abs(obs - centre) - 1e-9).mean())

RES = r"e:/update-single task + videos/stage1_results"
PHASE, NPHASE = 40_000, 9
TASKSETS = ["sym", "asym", "three"]
PT_CELLS = ["rho50_k8", "rho15_k8", "rho05_k8", "rho15_k30"]
HORIZON = {"rho50_k8": 0.4, "rho15_k8": 1.4, "rho05_k8": 4.1, "rho15_k30": 5.1}


def load(cell):
    out = []
    for f in sorted(glob.glob(f"{RES}/{cell}/*_scalars.pkl")):
        out.append(pickle.load(open(f, "rb")))
    return out


def last(d, key, default=np.nan):
    v = d.get(key)
    return float(v[-1, 1]) if v is not None and len(v) else default


def med(d, key, default=np.nan):
    v = d.get(key)
    return float(np.median(v[:, 1])) if v is not None and len(v) else default


def phase_returns(d):
    r = d["train/avg_return"]
    st, val = r[:, 0], r[:, 1]
    out = []
    for p in range(NPHASE):
        m = (st >= p * PHASE) & (st < (p + 1) * PHASE)
        out.append(float(val[m].mean()) if m.sum() else np.nan)
    return np.array(out)


def summarize(cell):
    ds = load(cell)
    if not ds:
        return None
    ph = np.stack([phase_returns(d) for d in ds])
    return dict(
        n=len(ds),
        final=np.array([p[-1] for p in ph]),
        overall=np.array([np.nanmean(p) for p in ph]),
        phases=np.nanmedian(ph, axis=0),
        drop=np.array([last(d, "boundary/mean_drop") for d in ds]),
        jump=np.array([last(d, "boundary/mean_jumpstart") for d in ds]),
        ret_perm=np.array([last(d, "retention/mse_perm") for d in ds]),
        ret_init=np.array([last(d, "retention/mse_perm_init") for d in ds]),
        ret_zero=np.array([last(d, "retention/mse_zero") for d in ds]),
        a_abs=np.array([med(d, "train/consol/actor_absorbed_frac") for d in ds]),
        c_abs=np.array([med(d, "train/consol/absorbed_frac") for d in ds]),
        energy=np.array([med(d, "train/policy_trans_l2") / max(med(d, "train/policy_perm_l2"), 1e-9)
                         for d in ds]),
        dpi=np.array([med(d, "consol/delta_pi") for d in ds]),
        logstd=np.array([last(d, "train/log_std_mean") for d in ds]),
    )


def p(a, b):
    try:
        return mannwhitney_exact(a, b)
    except Exception:
        return np.nan


print("=" * 108)
print("STAGE 1 -- DirectionalPointMass, 9 phases / 8 switches, 5 seeds/cell, medians")
print("=" * 108)

for ts in TASKSETS:
    van = summarize(f"van_{ts}")
    print(f"\n### task set '{ts}'"
          f"{'  (E_tau[target]=0, permanent structurally empty)' if ts == 'sym' else ''}")
    print(f"{'cell':<12}{'horiz':>6}{'final ret':>11}{'vs van p':>10}{'overall':>10}"
          f"{'drop':>8}{'jump':>9}{'a_absorb':>10}{'T/P energy':>12}{'retention':>22}")
    print(f"{'vanilla':<12}{'--':>6}{np.median(van['final']):>11.1f}{'--':>10}"
          f"{np.median(van['overall']):>10.1f}{np.median(van['drop']):>8.1f}"
          f"{np.median(van['jump']):>9.1f}{'--':>10}{'--':>12}"
          f"{np.median(van['ret_perm']):>8.3f} (ctl {np.median(van['ret_zero']):.3f})")
    for c in PT_CELLS:
        s = summarize(f"pt_{c}_{ts}")
        if s is None:
            continue
        beats = "OK " if np.median(s["ret_perm"]) < min(np.median(s["ret_init"]),
                                                        np.median(s["ret_zero"])) else "no "
        print(f"{c:<12}{HORIZON[c]:>5.1f}x{np.median(s['final']):>11.1f}"
              f"{p(s['final'], van['final']):>10.3f}{np.median(s['overall']):>10.1f}"
              f"{np.median(s['drop']):>8.1f}{np.median(s['jump']):>9.1f}"
              f"{np.median(s['a_abs']):>10.3f}{np.median(s['energy']):>12.2f}"
              f"{np.median(s['ret_perm']):>8.3f} {beats}(ctl "
              f"{min(np.median(s['ret_init']), np.median(s['ret_zero'])):.3f})")

print("\n" + "=" * 108)
print("PER-PHASE RETURN (median), tasks cycle in order; watch for specialisation to one task")
print("=" * 108)
for ts in TASKSETS:
    print(f"\n'{ts}':")
    v = summarize(f"van_{ts}")
    print(f"  {'vanilla':<12}" + "".join(f"{x:>8.0f}" for x in v["phases"]))
    for c in PT_CELLS:
        s = summarize(f"pt_{c}_{ts}")
        print(f"  {c:<12}" + "".join(f"{x:>8.0f}" for x in s["phases"]))

print("\n" + "=" * 108)
print("EXPLORATION (final log_std, shared across arms by C4 -- frozen in pt_full, learned in vanilla)")
print("=" * 108)
for ts in TASKSETS:
    v = summarize(f"van_{ts}")
    row = f"  {ts:<7} vanilla {np.median(v['logstd']):>7.3f}   "
    for c in PT_CELLS:
        s = summarize(f"pt_{c}_{ts}")
        row += f"{c}={np.median(s['logstd']):.3f}  "
    print(row)
