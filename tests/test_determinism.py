"""The offline evaluation must be a READ-ONLY probe of training.

`_run_offline_eval` samples actions via `actor.act()`, which draws from the GLOBAL torch
generator — up to 5 episodes x 1000 steps every 50 updates. Without isolation those draws advance
the same stream that produces training actions, so the same (agent, config, seed) diverges
depending on whether `--no-eval` was passed. That silently broke reproducibility for paired
per-seed comparisons across this project.
"""
import random

import numpy as np
import torch

from src_continuous_control.train import _isolated_rng


def _draw():
    """One sample from each of the three global streams."""
    return (float(torch.randn(1)), float(np.random.rand()), random.random())


def test_isolated_rng_restores_all_three_streams():
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
        "the RNG streams were not restored — an offline eval would shift every subsequent "
        f"training action ({got} != {expected})")


def test_without_isolation_the_streams_do_shift():
    """Sanity: confirm the hazard is real, so the test above is actually testing something."""
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
