"""Print the summary tables for the cartpole-swingup runs.

    python -m src_continuous_control.scripts.report_cartpole [results_dir]

Reuses `report_tables.load` and `report_tables.perm_p` so every report uses the same test.

  avg       median across seeds of each seed's mean return over the whole run
  final     the same over the last 20% of the run
  % of max  the above divided by 1000 (dm_control reward is in [0, 1] over exactly 1000 steps)
  p         exact two-sided Mann-Whitney by full enumeration (floor 1.08e-5 at 10 vs 10)
  FWT/BWT   read from each run's transfer matrix pickle (utils/metrics.py)
  drop      recomputed from the return curve: (pre - min over the next half phase) / pre, mean
            over the four boundaries, median over seeds

The logged `boundary/mean_drop` is not used: it is an absolute drop over 10,240 steps (1.7% of a
phase) of a 0.99-EMA, so it misses troughs that take ~200k steps to form. BWT and drop improve when
an arm learns less, so they are printed next to the return columns.
"""
import glob
import pickle
import sys

import numpy as np

from .report_tables import load, perm_p

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CEILING = 1000.0
# Default is the frozen-sigma study; pass a directory (e.g. results/cartpole_learned) for another.
ROOT = sys.argv[1] if len(sys.argv) > 1 else "src_continuous_control/results/cartpole"

# arm label -> results glob. Main arms share one directory (told apart by prefix); the ablation
# has its own subdirectory.
ARMS = [
    ("vanilla", f"{ROOT}/vanilla_ppo_seed_*_returns.pkl"),
    ("EWC", f"{ROOT}/ewc_ppo_seed_*_returns.pkl"),
    ("PT", f"{ROOT}/pt_ppo_seed_*_returns.pkl"),
    ("PT frozen", f"{ROOT}/ablation_frozen/pt_ppo_seed_*_returns.pkl"),
]


def _scalar_last(pattern, key):
    """Last logged value of `key` for every run matching `pattern`."""
    out = []
    for path in sorted(glob.glob(pattern)):
        try:
            hist = pickle.load(open(path, "rb"))
        except Exception:
            continue
        series = hist.get(key)
        if series is not None and len(series):
            out.append(float(np.asarray(series, dtype=float)[-1, 1]))
    return np.array(out)


def _transfer(pattern, key):
    out = []
    for path in sorted(glob.glob(pattern)):
        try:
            rec = pickle.load(open(path, "rb"))
        except Exception:
            continue
        val = rec.get(key)
        if val is not None and np.isfinite(val):
            out.append(float(val))
    return np.array(out)


SWITCH = 614400          # env steps per task, from configs/default.yaml
DROP_WINDOW = SWITCH // 2


def _boundary_drop(pattern):
    """Per seed, the mean fractional drop across the four boundaries."""
    out = []
    for path in sorted(glob.glob(pattern)):
        if "ep_returns" in path or "eval_returns" in path:
            continue
        try:
            a = np.asarray(pickle.load(open(path, "rb")), dtype=float)
        except Exception:
            continue
        if a.ndim != 2 or a[-1, 0] < 3_000_000:
            continue
        steps, rets = a[:, 0], a[:, 1]
        fracs = []
        for b in range(1, 5):
            bx = b * SWITCH
            pre_sel = steps < bx
            post_sel = (steps >= bx) & (steps < bx + DROP_WINDOW)
            if not pre_sel.any() or not post_sel.any():
                continue
            pre = rets[pre_sel][-1]
            if pre <= 1e-9:                     # undefined as a fraction
                continue
            fracs.append((pre - rets[post_sel].min()) / pre)
        if fracs:
            out.append(float(np.mean(fracs)))
    return np.array(out)


def _med(a):
    return float(np.median(a)) if len(a) else float("nan")


def _fmt(v, pct=False):
    if not np.isfinite(v):
        return "-"
    return f"{v:.0f} ({100 * v / CEILING:.0f}%)" if pct else f"{v:.0f}"


def main():
    data = {name: load(pat) for name, pat in ARMS}

    print("## Return\n")
    print("| arm | seeds | whole-run | % of 1000 | final 20% | % of 1000 |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, _ in ARMS:
        d = data[name]
        if not len(d):
            print(f"| {name} | 0 | - | - | - | - |")
            continue
        a, f = _med(d[:, 0]), _med(d[:, 1])
        print(f"| {name} | {len(d)} | {a:.0f} | {100 * a / CEILING:.0f}% "
              f"| {f:.0f} | {100 * f / CEILING:.0f}% |")

    print("\n## Significance (exact two-sided Mann-Whitney, full enumeration)\n")
    print("| comparison | whole-run Δ | p | final-20% Δ | p |")
    print("|---|---:|---:|---:|---:|")
    for a, b in (("PT", "vanilla"), ("PT", "EWC"), ("EWC", "vanilla"),
                 ("PT", "PT frozen")):
        x, y = data[a], data[b]
        if not len(x) or not len(y):
            continue
        da = _med(x[:, 0]) - _med(y[:, 0])
        df = _med(x[:, 1]) - _med(y[:, 1])
        pa = perm_p(x[:, 0], y[:, 0])
        pf = perm_p(x[:, 1], y[:, 1])
        print(f"| {a} − {b} | {da:+.0f} | {pa:.4f} | {df:+.0f} | {pf:.4f} |")

    print("\n## Transfer and boundary behaviour\n")
    print("**Read these beside the return columns above, never alone** - every retention-flavoured "
          "metric improves when an arm simply learns less.\n")
    print("| arm | FWT | BWT | boundary drop |")
    print("|---|---:|---:|---:|")
    for name, pat in ARMS:
        tpat = pat.replace("_returns.pkl", "_transfer_matrix.pkl")
        fwt, bwt = _transfer(tpat, "fwt"), _transfer(tpat, "bwt")
        drop = _boundary_drop(pat)
        if not len(fwt) and not len(bwt) and not len(drop):
            continue
        print(f"| {name} | {_fmt(_med(fwt))} | {_fmt(_med(bwt))} | "
              f"{('%.0f%%' % (100 * _med(drop))) if len(drop) else '-'} |")

    for a, b in (("PT", "vanilla"), ("PT", "EWC")):
        pa = ARMS[[n for n, _ in ARMS].index(a)][1]
        pb = ARMS[[n for n, _ in ARMS].index(b)][1]
        x, y = _boundary_drop(pa), _boundary_drop(pb)
        if len(x) and len(y):
            print(f"\n  DROP {a} − {b} = {100 * (_med(x) - _med(y)):+.0f} pp  "
                  f"p = {perm_p(x, y):.4f}")

    for a, b in (("PT", "vanilla"), ("PT", "EWC")):
        pa = ARMS[[n for n, _ in ARMS].index(a)][1].replace("_returns.pkl", "_transfer_matrix.pkl")
        pb = ARMS[[n for n, _ in ARMS].index(b)][1].replace("_returns.pkl", "_transfer_matrix.pkl")
        for key in ("fwt", "bwt"):
            x, y = _transfer(pa, key), _transfer(pb, key)
            if len(x) and len(y):
                print(f"\n  {key.upper()}  {a} − {b} = {_med(x) - _med(y):+.0f}  "
                      f"p = {perm_p(x, y):.4f}")

    # Below 0.01 absorbed fraction, the permanent is inert and the PT arm is uninformative.
    absorbed = _scalar_last(f"{ROOT}/pt_ppo_seed_*_scalars.pkl", "train/consol/actor_absorbed_frac")
    frozen = _scalar_last(f"{ROOT}/ablation_frozen/pt_ppo_seed_*_scalars.pkl",
                          "train/consol/actor_absorbed_frac")
    print("\n## Mechanism check\n")
    if len(absorbed):
        print(f"  PT        actor_absorbed_frac  median {_med(absorbed):.4f}  "
              f"min {absorbed.min():.4f}  (must exceed 0.01 or the permanent is inert)")
    if len(frozen):
        print(f"  PT frozen actor_absorbed_frac  median {_med(frozen):.4f}  "
              f"max {frozen.max():.4f}  (must be ~0 or the control was not actually off)")
    corr = _scalar_last(f"{ROOT}/pt_ppo_seed_*_scalars.pkl", "diag/actor_perm_trans_corr")
    if len(corr):
        print(f"  PT        perm/trans correlation median {_med(corr):+.3f}  "
              f"(near -1 would mean the components cancel)")


if __name__ == "__main__":
    sys.exit(main())
