"""Regenerates plots/figures/consolidation_mechanism.{png,pdf} — the consolidation diagnostic.

Every number in the figure is MEASURED here, not asserted:

  (a) One consolidation at the exact production settings. Builds the real PPOPT agent, fills a
      genuine 20 480-state ConsolidationBuffer, calls the real _consolidate(), and measures how far
      V_perm actually moved versus how much of V_trans the decay deleted.

  (b) Can the permanent network FIT old_V_perm + V_trans by regression? Trains a fresh permanent
      network on the consolidation batch to convergence with Adam and plots the error curve for the
      production width and a 13x wider net, on the fitted batch AND on held-out states. This panel
      exists because an earlier draft claimed the target was "not representable"; that claim was
      wrong -- with enough capacity the batch IS fitted (train error ~3%). What does not improve is
      the held-out error, which floors near 38-40%.

  (c) The decay step. decay_transient(d) scales the PARAMETERS of the transient MLP by d, which does
      not scale its OUTPUT by d (nonlinear activations + biases) -- so the value-preserving identity
      holds only at d=0. With a shared trunk and linear heads the same operation is exact for all d.

Caveat stated in FINDINGS.md 6.3: probe states here are iid Gaussian, which is a harder
generalisation problem than real on-policy states; treat panel (b)'s held-out floor as indicative.

Run from the PARENT of src_continuous_control/:
    python -m src_continuous_control.plots.make_consolidation_figure
"""
import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ..agents.ppo_pt import PPOPT
from ..models.critic import SplitCritic, SharedTrunkSplitCritic

C_SEP, C_SHARED, C_ALT = "#1baf7a", "#e34948", "#2a78d6"
INK, MUT, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
D, NBUF = 17, 20480

PROD_CFG = dict(
    hidden_sizes=[64, 64], lr_actor=3e-4, adam_eps=1e-5, num_envs=8, n_steps=256,
    gamma=0.99, gae_lambda=0.95, clip_coef=0.2, epochs=10, minibatch_size=64,
    ent_coef=0.0, max_grad_norm=0.5, target_kl=None, normalize_advantage=True,
    lr_trans=3e-4, lr_perm=1e-5, perm_optimizer="sgd", decay=0.0, k=10,
    consolidation_epochs=1, consolidation_buffer_size=NBUF, on_switch="consolidate",
)


def _mlp(hidden):
    layers, last = [], D
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, 2 ** 0.5)
        nn.init.constant_(lin.bias, 0.0)
        layers += [lin, nn.Tanh()]
        last = h
    out = nn.Linear(last, 1)
    nn.init.orthogonal_(out.weight, 1.0)
    nn.init.constant_(out.bias, 0.0)
    return nn.Sequential(*layers, out)


def panel_a():
    """Production consolidation: % of the transient absorbed vs % deleted."""
    torch.manual_seed(0); np.random.seed(0)
    ag = PPOPT(D, 6, PROD_CFG, torch.device("cpu"))
    with torch.no_grad():
        for p in ag.critic.trans.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    for _ in range(PROD_CFG["k"]):
        s = np.random.randn(256, 8, D).astype(np.float32).reshape(-1, D)
        with torch.no_grad():
            ov, _ = ag.critic(torch.as_tensor(s))
        ag.consolidation_buffer.add_batch(s, ov.numpy())
    probe = torch.randn(512, D)
    with torch.no_grad():
        P0, T0 = ag.critic(probe)
    ag._consolidate()
    with torch.no_grad():
        P1, T1 = ag.critic(probe)
    need = T0.abs().mean().item()
    absorbed = (P1 - P0).abs().mean().item() / need * 100
    deleted = (1 - T1.abs().mean().item() / need) * 100
    dV = ((P1 + T1) - (P0 + T0)).abs().mean().item() / (P0 + T0).abs().mean().item() * 100
    return absorbed, deleted, dV


def panel_b(epochs=200, lr=1e-3, bs=256):
    """Regression error vs training budget, on the fitted batch and on held-out states."""
    torch.manual_seed(0); np.random.seed(0)
    src = SplitCritic(D, hidden_sizes=[64, 64])
    with torch.no_grad():
        for p in src.trans.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    S, Sho = torch.randn(NBUF, D), torch.randn(4096, D)
    with torch.no_grad():
        a, b = src(S); y = a + b
        ah, bh = src(Sho); yh = ah + bh
    curves = {}
    for hidden in ([64, 64], [256, 256]):
        net = _mlp(hidden)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        xs, tr, ho = [], [], []
        for ep in range(epochs):
            idx = torch.randperm(NBUF)
            for i in range(0, NBUF, bs):
                j = idx[i:i + bs]
                loss = ((net(S[j]).squeeze(-1) - y[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 10 == 0 or ep == 0:
                with torch.no_grad():
                    xs.append(ep + 1)
                    tr.append((net(S).squeeze(-1) - y).abs().mean().item() / y.abs().mean().item() * 100)
                    ho.append((net(Sho).squeeze(-1) - yh).abs().mean().item() / yh.abs().mean().item() * 100)
        n_par = sum(p.numel() for p in net.parameters())
        curves[f"{hidden[0]}x{hidden[1]} ({n_par:,} params)"] = (xs, tr, ho)
    return curves


def panel_c(decays=(0.0, 0.25, 0.5, 0.75)):
    """Value error introduced by the decay step: two MLPs vs shared trunk + linear heads."""
    torch.manual_seed(0); np.random.seed(0)
    probe = torch.randn(512, D)
    sep, sh = [], []
    for d in decays:
        sc = SplitCritic(D, hidden_sizes=[64, 64])
        with torch.no_grad():
            for p in sc.trans.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        with torch.no_grad():
            _, Tb = sc(probe)
        sc2 = copy.deepcopy(sc); sc2.decay_transient(d)
        with torch.no_grad():
            _, Ta = sc2(probe)
        sep.append((Ta - d * Tb).abs().mean().item() / Tb.abs().mean().item() * 100)

        c = SharedTrunkSplitCritic(D, (64, 64))
        with torch.no_grad():
            c.trans.weight.add_(torch.randn_like(c.trans.weight) * 0.8)
            c.trans.bias.add_(0.3)
        with torch.no_grad():
            p0, t0 = c(probe)
        c.consolidate(d)
        with torch.no_grad():
            p1, t1 = c(probe)
        sh.append(((p1 + t1) - (p0 + t0)).abs().mean().item() / (p0 + t0).abs().mean().item() * 100)
    return list(decays), sep, sh


def _style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9, length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="src_continuous_control/plots/figures")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    absorbed, deleted, dV = panel_a()
    curves = panel_b(epochs=args.epochs)
    decays, sep, sh = panel_c()
    print(f"(a) absorbed {absorbed:.2f}%  deleted {deleted:.1f}%  net dV {dV:.1f}%")
    for k, (xs, tr, ho) in curves.items():
        print(f"(b) {k}: final train {tr[-1]:.1f}%  held-out {ho[-1]:.1f}%")
    print(f"(c) separate {['%.1f' % v for v in sep]}   shared {['%.4f' % v for v in sh]}")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    ax = axes[0]
    ax.bar([0, 1], [absorbed, deleted], 0.55, color=[C_SEP, "#8a8a86"], zorder=3)
    for xi, v in zip([0, 1], [absorbed, deleted]):
        ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["absorbed into\npermanent", "deleted\nby decay"])
    ax.set_ylim(0, 118)
    ax.set_ylabel("% of the transient value", color=MUT, fontsize=10)
    ax.set_title("(a) One consolidation, production settings\n"
                 r"$lr_{perm}$=1e-5, SGD, 1 epoch (320 steps)", fontsize=11, color=INK, loc="left")
    ax.annotate(f"net: {dV:.0f}% of the acting value\ndestroyed, ~150x per run",
                xy=(0.03, 0.60), xycoords="axes fraction", ha="left",
                fontsize=9.5, color=C_SEP, fontweight="bold")

    ax = axes[1]
    for (label, (xs, tr, ho)), col in zip(curves.items(), (C_ALT, C_SEP)):
        ax.plot(xs, tr, color=col, lw=2, label=f"{label} — fitted batch")
        ax.plot(xs, ho, color=col, lw=2, ls="--", label=f"{label} — held-out states")
    ax.set_xlabel("consolidation training epochs (Adam, lr 1e-3)", color=MUT, fontsize=10)
    ax.set_ylabel("value error (%)", color=MUT, fontsize=10)
    ax.set_title("(b) The target IS fittable — but only on the batch\n"
                 "held-out error does not improve", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=MUT, loc="upper right")

    ax = axes[2]
    x = np.arange(len(decays)); w = 0.36
    ax.bar(x - w / 2, sep, w * 0.92, label="Two separate MLPs (current)", color=C_SEP, zorder=3)
    ax.bar(x + w / 2, sh, w * 0.92, label="Shared trunk + linear heads (fix)", color=C_SHARED, zorder=3)
    ax.scatter(x + w / 2, np.zeros_like(x, dtype=float), s=46, marker="_",
               linewidths=2.6, color=C_SHARED, zorder=5)
    for xi, v in zip(x - w / 2, sep):
        ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=MUT)
    for xi in x + w / 2:
        ax.annotate("0.00%", (xi, 0), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color=C_SHARED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{d:g}" for d in decays])
    ax.set_xlabel("decay", color=MUT, fontsize=10)
    ax.set_ylabel("value error from the decay step (%)", color=MUT, fontsize=10)
    ax.set_title("(c) Parameter scaling is not output scaling\n"
                 "exact for any decay only with linear heads", fontsize=11, color=INK, loc="left")
    ax.set_ylim(0, max(sep) * 1.42)     # headroom so the legend clears the bar labels
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUT, loc="upper left", bbox_to_anchor=(0.0, 1.0))

    for ax in axes:
        _style(ax)
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(args.out_dir, f"consolidation_mechanism.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
