"""Print the advantage split on cartpole-swingup (thesis Table 4.1).

Reads runs made with `--log-adv-split` and prints, per arm, the median over seeds of
Var(A_P)/Var(A), Var(A_T)/Var(A), Var(A_R)/Var(A) and corr(A_P, A_T) in two windows:
  after-switch  the 20 updates after each task boundary
  whole run     every update after the first consolidation (the transient is zero before it,
                so the correlation is undefined there)
Also prints the max split residual and flags it if the split is not exact.

    python -m src_continuous_control.scripts.report_adv_split   # from the package's parent dir
"""
import glob
import os
import pickle

import numpy as np

ROOT = "src_continuous_control/results/cartpole_controls"
ARMS = ["pt", "pt_frozen", "pt_frozen_nodecay"]
KEYS = {"var_P": "train/adv_var_perm_ratio", "var_T": "train/adv_var_trans_ratio",
        "var_R": "train/adv_var_reward_ratio", "corr": "train/adv_corr_perm_trans"}
SWITCH, BATCH, AFTER = 614400, 2048, 20


def _series(d, key):
    a = np.asarray(d[key], dtype=float)
    return a[:, 0], a[:, 1]


def summarise(arm):
    rows = {"after-switch": {k: [] for k in KEYS}, "whole run": {k: [] for k in KEYS}}
    residual = 0.0
    files = sorted(glob.glob(os.path.join(ROOT, arm, "*_scalars.pkl")))
    for f in files:
        d = pickle.load(open(f, "rb"))
        steps, res = _series(d, "train/adv_split_residual")
        residual = max(residual, float(np.nanmax(res)))
        after = np.zeros_like(steps, dtype=bool)
        for b in range(1, 5):
            lo = b * SWITCH
            after |= (steps > lo) & (steps <= lo + AFTER * BATCH)
        whole = steps > 10 * BATCH
        for name, key in KEYS.items():
            _, v = _series(d, key)
            rows["after-switch"][name].append(np.nanmedian(v[after]))
            rows["whole run"][name].append(np.nanmedian(v[whole]))
    return len(files), residual, rows


def main():
    for arm in ARMS:
        n, residual, rows = summarise(arm)
        if n == 0:
            print(f"{arm}: no runs yet")
            continue
        print(f"\n{arm}: {n} seeds, max split residual {residual:.2e}"
              f"{'  <-- SPLIT NOT EXACT' if residual > 1e-3 else ''}")
        for window, r in rows.items():
            med = {k: np.median(v) for k, v in r.items()}
            rng = {k: (np.min(v), np.max(v)) for k, v in r.items()}
            print(f"  {window:12s} Var(A_P)/Var(A) {med['var_P']:6.2f}  Var(A_T)/Var(A) {med['var_T']:6.2f}"
                  f"  Var(A_R)/Var(A) {med['var_R']:6.2f}  corr {med['corr']:+.3f}"
                  f"  (corr per seed {rng['corr'][0]:+.2f} to {rng['corr'][1]:+.2f})")


if __name__ == "__main__":
    main()
