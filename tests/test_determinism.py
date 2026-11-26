"""The offline evaluation must not change the training RNG streams.

`_run_offline_eval` samples with `actor.act()`, which draws from the global torch generator.
Without isolation the same (agent, config, seed) would train differently depending on `--no-eval`.
"""
import random

import numpy as np
import torch

from src_continuous_control.train import _isolated_rng


def _draw():
    """One sample from each of the three global streams."""
    return (float(torch.randn(1)), float(np.random.rand()), random.random())


def test_isolated_rng_restores_all_three_streams():
    """_isolated_rng restores the torch, numpy and random streams after heavy use."""
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    expected = _draw()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    with _isolated_rng():
        # Burn a lot of randomness, as a 5000-step evaluation rollout would.
        for _ in range(5000):
            torch.randn(8)
            np.random.rand(8)
            random.random()
    got = _draw()

    assert got == expected, (
        "the RNG streams were not restored - an offline eval would shift every subsequent "
        f"training action ({got} != {expected})")


def test_without_isolation_the_streams_do_shift():
    """Without isolation the streams do shift, so the test above is not vacuous."""
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    expected = _draw()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    for _ in range(100):
        torch.randn(8); np.random.rand(8); random.random()
    got = _draw()

    assert got != expected, "expected unisolated draws to shift the streams"


def test_isolation_survives_an_exception():
    """A crash inside the eval must not leave the training streams corrupted."""
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    expected = _draw()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    try:
        with _isolated_rng():
            torch.randn(64); np.random.rand(64); random.random()
            raise RuntimeError("eval blew up")
    except RuntimeError:
        pass
    assert _draw() == expected
