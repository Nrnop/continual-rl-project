"""Unified experiment logger: TensorBoard + Weights & Biases + per-seed pickle.

One interface so the agents don't care which backend is active. The pickle dump preserves the
baseline's offline-plotting workflow (results/<exp>_seed_<s>_returns.pkl), so plot_compare.py and
the paper's own matplotlib style both keep working.
"""
import os
import pickle

import numpy as np


class Logger:
    def __init__(
        self,
        exp_name,
        seed,
        backend="both",          # "tb" | "wandb" | "both" | "none"
        runs_dir="src_continuous_control/runs",
        results_dir="src_continuous_control/results",
        wandb_project="pt-continuous-control",
        config=None,
    ):
        self.exp_name = exp_name
        self.seed = seed
        self.backend = backend
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.tb = None
        self.wandb = None

        want_tb = backend in ("tb", "both")
        want_wandb = backend in ("wandb", "both")

        if want_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(runs_dir, f"{exp_name}_seed_{seed}")
                self.tb = SummaryWriter(log_dir=log_dir)
            except Exception as e:  # pragma: no cover - backend optional
                print(f"[logger] TensorBoard disabled: {e}")

        if want_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=wandb_project,
                    name=f"{exp_name}_seed_{seed}",
                    config=config or {},
                    reinit=True,
                )
            except Exception as e:  # pragma: no cover - backend optional
                print(f"[logger] W&B disabled ({e}); continuing without it.")
                self.wandb = None

    def log_scalar(self, name, value, step):
        value = float(value)
        if self.tb is not None:
            self.tb.add_scalar(name, value, step)
        if self.wandb is not None:
            self.wandb.log({name: value}, step=step)

    def log_scalars(self, scalar_dict, step):
        for name, value in scalar_dict.items():
            if value is None:
                continue
            self.log_scalar(name, value, step)

    def save_returns(self, returns_array, suffix="returns"):
        """Dump the per-step running-return curve as a pickle (baseline convention)."""
        fname = os.path.join(self.results_dir, f"{self.exp_name}_seed_{self.seed}_{suffix}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(np.asarray(returns_array, dtype=np.float32), f)
        return fname

    def save_checkpoint(self, state_dict, step):
        """Save network weights to checkpoints directory."""
        import torch
        ckpt_dir = os.path.join("src_continuous_control/checkpoints", f"{self.exp_name}_seed_{self.seed}")
        os.makedirs(ckpt_dir, exist_ok=True)
        fname = os.path.join(ckpt_dir, f"step_{step}.pt")
        torch.save(state_dict, fname)
        return fname

    def close(self):
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()
