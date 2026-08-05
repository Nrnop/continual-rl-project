"""Scalars must survive a run with every logging backend disabled.

Sweeps run with --no-tb --no-wandb. Before this was fixed, log_scalar was a no-op in that mode, so
the jumpstart and retention curves — the PRIMARY metrics for the PT comparison — were silently
discarded across the whole sweep. Nothing in the run's output revealed it.
"""
import pickle

import numpy as np

from src_continuous_control.utils.logger import Logger


def test_scalars_persist_with_no_backends(tmp_path):
    log = Logger("test_exp", 0, backend="none", results_dir=str(tmp_path))
    assert log.tb is None and log.wandb is None

    log.log_scalars({"boundary/jumpstart_mean": 12.5, "retention/mse_perm": 0.25}, step=100)
    log.log_scalars({"boundary/jumpstart_mean": 15.0, "retention/mse_perm": 0.50}, step=200)
    log.log_scalar("boundary/mean_drop", 3.0, step=200)

    fname = log.save_scalars()
    with open(fname, "rb") as f:
        got = pickle.load(f)

    assert set(got) == {"boundary/jumpstart_mean", "retention/mse_perm", "boundary/mean_drop"}
    js = got["boundary/jumpstart_mean"]
    assert js.shape == (2, 2)
    assert np.allclose(js[:, 0], [100, 200])      # steps
    assert np.allclose(js[:, 1], [12.5, 15.0])    # values
    assert got["boundary/mean_drop"].shape == (1, 2)


def test_none_values_are_skipped_not_recorded(tmp_path):
    """log_scalars drops None (a metric that has not fired yet) rather than writing NaN."""
    log = Logger("test_exp", 1, backend="none", results_dir=str(tmp_path))
    log.log_scalars({"a": 1.0, "b": None}, step=10)
    assert "a" in log.history and "b" not in log.history
