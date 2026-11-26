"""The Lipschitz1 -> Lipschitz2 contrast statistic in scripts/report_drift_multienv.py.

It is an interaction test: pool each arm's runs across the two settings, permute the setting
labels, and count how often the difference of differences is at least as large as observed.
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
    """The same gap in both settings gives a large p-value."""
    p = _contrast_p(_data(gap1=20.0, gap2=20.0), "e", "vanilla", 0, n_perm=2000)
    assert p > 0.2


def test_a_large_interaction_is_detected():
    """A lead growing from +5 to +120 is detected."""
    p = _contrast_p(_data(gap1=5.0, gap2=120.0), "e", "vanilla", 0, n_perm=2000)
    assert p < 0.05


def test_the_statistic_is_symmetric_in_direction():
    """A shrinking lead is as detectable as a growing one (the test is two-sided)."""
    grow = _contrast_p(_data(gap1=5.0, gap2=120.0), "e", "vanilla", 0, n_perm=2000)
    shrink = _contrast_p(_data(gap1=120.0, gap2=5.0), "e", "vanilla", 0, n_perm=2000)
    assert grow < 0.05 and shrink < 0.05


def test_it_tests_the_interaction_not_the_gap():
    """Two large but equal gaps are not significant: the statistic tests the change in the gap."""
    p = _contrast_p(_data(gap1=150.0, gap2=150.0), "e", "vanilla", 0, n_perm=2000)
    assert p > 0.2


def test_p_is_never_zero():
    """A permutation p-value is (hits+1)/(n_perm+1), so it is at least 1/(n_perm+1).

    Equality is not asserted: with a huge effect some permutations still exceed the observed value.
    """
    p = _contrast_p(_data(gap1=0.0, gap2=5000.0), "e", "vanilla", 0, n_perm=500)
    assert p >= 1.0 / 501
    assert p > 0.0


def test_it_is_deterministic():
    """The same data gives the same p-value."""
    d = _data(gap1=5.0, gap2=60.0)
    assert (_contrast_p(d, "e", "vanilla", 0, n_perm=1000)
            == _contrast_p(d, "e", "vanilla", 0, n_perm=1000))
