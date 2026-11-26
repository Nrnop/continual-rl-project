"""Print the return of the decay-vs-split control runs on cartpole-swingup.

Whole-run and final-fifth median return per arm, and exact permutation tests between pairs:
  vanilla            results/cartpole                  (no split, no decay)
  pt_frozen          results/cartpole/ablation_frozen  (split + output decay, nothing absorbed)
  pt                 results/cartpole                  (the full method)
  pt_frozen_nodecay  split, no decay
  van_small          vanilla at the transient's [30, 30]
  van_outdecay       vanilla with the same output-layer decay, no split
The control runs must come from the same machine as the stored runs they are compared with.

    python -m src_continuous_control.scripts.report_cartpole_controls --seeds 0 1 2 3 4
"""
import argparse
import os
import pickle

import numpy as np

from .report_tables import perm_p

STORED = "src_continuous_control/results/cartpole"
NEW = "src_continuous_control/results/cartpole_controls"
ARMS = {"vanilla": (STORED, "vanilla"),
        "pt_frozen": (f"{STORED}/ablation_frozen", "pt"),
        "pt": (STORED, "pt"),
        "pt_frozen_nodecay": (f"{NEW}/pt_frozen_nodecay", "pt"),
        "van_small": (f"{NEW}/van_small", "vanilla"),
        "van_outdecay": (f"{NEW}/van_outdecay", "vanilla")}


def load(arm, seeds):
    d, agent = ARMS[arm]
    whole, final = [], []
    for s in seeds:
        p = os.path.join(d, f"{agent}_ppo_seed_{s}_returns.pkl")
        if not os.path.exists(p):
            continue
        r = np.asarray(pickle.load(open(p, "rb")))[:, 1]
        whole.append(r.mean())
        final.append(r[int(0.8 * len(r)):].mean())
    return np.array(whole), np.array(final)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = p.parse_args()
    data = {a: load(a, args.seeds) for a in ARMS}
    print(f"seeds {args.seeds}\n{'arm':20s} {'n':>2s} {'whole-run':>10s} {'final fifth':>12s}")
    for a, (w, f) in data.items():
        print(f"{a:20s} {len(w):2d} {np.median(w) if len(w) else float('nan'):10.1f} "
              f"{np.median(f) if len(f) else float('nan'):12.1f}")
    pairs = [("pt_frozen_nodecay", "pt_frozen", "does removing the halving rescue the frozen arm?"),
             ("pt_frozen_nodecay", "vanilla", "is the split itself (no halving) a cost?"),
             ("van_outdecay", "vanilla", "is the halving alone (no split) a cost?"),
             ("van_outdecay", "pt_frozen", "halving with vs without the split"),
             ("van_small", "vanilla", "is a [30, 30] network alone a cost?"),
             ("van_small", "pt_frozen", "small network vs frozen-permanent arm")]
    print()
    for a, b, q in pairs:
        if len(data[a][0]) < 2 or len(data[b][0]) < 2:
            print(f"{a} vs {b}: not enough runs yet")
            continue
        for i, name in ((0, "whole"), (1, "final")):
            x, y = data[a][i], data[b][i]
            print(f"{a:18s} - {b:10s} {name:5s} {np.median(x) - np.median(y):+7.1f}  "
                  f"p = {perm_p(x, y):.3f}   ({q})" if i == 0 else
                  f"{'':18s}   {'':10s} {name:5s} {np.median(x) - np.median(y):+7.1f}  "
                  f"p = {perm_p(x, y):.3f}")


if __name__ == "__main__":
    main()
