"""The Lipschitz1 -> Lipschitz2 contrast statistic.

This is the number that will carry the drift study's conclusion — "pt's lead grows when the world
gains a fast component" — so the statistic behind it is pinned here rather than trusted.

It is an INTERACTION test, not a comparison of two groups: pool each arm's runs across the two
settings, reassign the setting labels at random, and ask how often the difference-of-differences
comes out at least as large. Testing either gap on its own would answer a different question.
"""
import numpy as np
import pytest

from src_continuous_control.scripts.report_drift_multienv import _contrast_p


def _cell(pt_mu, base_mu, n=10, sd=20.0, seed=0):
    rng = np.random.RandomState(seed)
    return {"pt": np.stack([rng.normal(pt_mu, sd, n)] * 2, axis=1),
            "vanilla": np.stack([rng.normal(base_mu, sd, n)] * 2, axis=1)}


def _data(gap1, gap2, base=300.0, sd=20.0):
    return {("lipschitz1", "e"): _cell(base + gap1, base, sd=sd, seed=1),
            ("lipschitz2", "e"): _cell(base + gap2, base, sd=sd, seed=2)}


def test_no_interaction_gives_a_large_p():
    """Same gap in both settings -> nothing to detect."""
    p = _contrast_p(_data(gap1=20.0, gap2=20.0), "e", "vanilla", 0, n_perm=2000)
    assert p > 0.2


def test_a_large_interaction_is_detected():
    """pt's lead grows from +5 to +120: the effect the drift design predicts."""
    p = _contrast_p(_data(gap1=5.0, gap2=120.0), "e", "vanilla", 0, n_perm=2000)
    assert p < 0.05


def test_the_statistic_is_symmetric_in_direction():
    """A shrinking lead is as detectable as a growing one — the test is two-sided."""
    grow = _contrast_p(_data(gap1=5.0, gap2=120.0), "e", "vanilla", 0, n_perm=2000)
    shrink = _contrast_p(_data(gap1=120.0, gap2=5.0), "e", "vanilla", 0, n_perm=2000)
    assert grow < 0.05 and shrink < 0.05


def test_it_tests_the_interaction_not_the_gap():
    """Two big but EQUAL gaps must not be called significant.

    This is the trap the statistic exists to avoid: `pt` can be far ahead in both settings while
    the fast component changed nothing at all, and a naive test of "is pt ahead" would report a
    triumphant p-value for a prediction that failed.
    """
    p = _contrast_p(_data(gap1=150.0, gap2=150.0), "e", "vanilla", 0, n_perm=2000)
    assert p > 0.2


def test_p_is_never_zero():
    """A permutation p-value is (hits+1)/(perm+1), so it has a FLOOR and can never be reported as 0.

    The floor is 1/(n_perm+1). It is not asserted as an equality here: with a huge injected effect
    the permuted pools mix values from two very different distributions, so some reassignments do
    exceed the observed statistic and the count is not always zero. What matters for reporting is
    only that the p-value is strictly positive and bounded below.
    """
    p = _contrast_p(_data(gap1=0.0, gap2=5000.0), "e", "vanilla", 0, n_perm=500)
    assert p >= 1.0 / 501
    assert p > 0.0


def test_it_is_deterministic():
    """Same data, same answer — a statistic that moves between runs cannot be cited."""
    d = _data(gap1=5.0, gap2=60.0)
    assert (_contrast_p(d, "e", "vanilla", 0, n_perm=1000)
            == _contrast_p(d, "e", "vanilla", 0, n_perm=1000))
