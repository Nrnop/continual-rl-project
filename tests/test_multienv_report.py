"""Tests for carry_over, disruption and percent_of_ceiling in scripts/report_multienv.py.

Each case uses a transfer matrix whose answer is known by construction, so the definition of
carry-over stays fixed.
"""
import numpy as np
import pytest

from src_continuous_control.scripts.report_multienv import (
    carry_over,
    disruption,
    percent_of_ceiling,
)


def _matrix(row0, n=5):
    """A transfer matrix with the given row 0 and zeros elsewhere."""
    m = np.zeros((n, n), dtype=float)
    m[0] = row0
    return m


# Anchor points of the scale
def test_perfect_transfer_is_one():
    """Equal returns on every physics setting give 1.0, whatever the baselines."""
    m = _matrix([500.0] * 5)
    assert carry_over(m, [100.0] * 5) == pytest.approx(1.0)


def test_total_loss_is_zero():
    """Falling to the baseline on every other task gives 0.0 (not a raw return of zero)."""
    m = _matrix([500.0, 100.0, 100.0, 100.0, 100.0])
    assert carry_over(m, [100.0] * 5) == pytest.approx(0.0)


def test_worse_than_untrained_is_negative():
    """Scoring below the untrained baseline elsewhere reads below zero; it is not clipped."""
    m = _matrix([500.0, 50.0, 50.0, 50.0, 50.0])
    assert carry_over(m, [100.0] * 5) < 0.0


# Baseline correction
def test_baseline_correction_changes_the_answer():
    """Without the baseline correction, a high untrained score looks like transfer.

    walker-stand's random-init policy already scores about 139 by falling over slowly.
    """
    m = _matrix([500.0, 150.0, 150.0, 150.0, 150.0])
    corrected = carry_over(m, [140.0] * 5)
    uncorrected = carry_over(m, [0.0] * 5)
    assert corrected < 0.05, "with the floor removed, almost nothing survived"
    assert uncorrected > 0.25, "without the correction the same run looks like real transfer"


def test_per_task_baselines_are_used_not_a_single_scalar():
    """Baselines b_j are applied per task, since untrained scores differ between tasks."""
    m = _matrix([500.0, 300.0, 300.0, 300.0, 300.0])
    flat = carry_over(m, [100.0] * 5)
    varied = carry_over(m, [100.0, 250.0, 250.0, 250.0, 250.0])
    assert varied < flat


# Row 0 only
def test_only_row_zero_is_read():
    """Only row 0 is read; later rows mix transfer with extra training steps."""
    a = _matrix([500.0, 400.0, 400.0, 400.0, 400.0])
    b = a.copy()
    b[1:] = 12345.0                      # nonsense everywhere except row 0
    assert carry_over(a, [100.0] * 5) == carry_over(b, [100.0] * 5)


# Refusals
def test_untrained_task_zero_gives_nan_not_a_huge_ratio():
    """An untrained task 0 gives NaN instead of a huge ratio from a near-zero denominator."""
    m = _matrix([100.0, 400.0, 400.0, 400.0, 400.0])
    assert np.isnan(carry_over(m, [100.0] * 5))


def test_non_square_matrix_is_refused():
    """A non-square matrix raises ValueError."""
    with pytest.raises(ValueError, match="square"):
        carry_over(np.zeros((3, 5)), [0.0] * 5)


def test_wrong_number_of_baselines_is_refused():
    """A baselines list of the wrong length raises ValueError."""
    with pytest.raises(ValueError, match="baselines"):
        carry_over(np.zeros((5, 5)), [0.0] * 3)


# Disruption (the dynamic-range gate and a reported covariate)
def test_disruption_is_zero_when_the_physics_do_not_matter():
    """Disruption is zero when row 0 is flat across tasks."""
    assert disruption(_matrix([400.0] * 5)) == pytest.approx(0.0)


def test_disruption_grows_with_the_spread():
    """Disruption grows with the spread of row 0; the 0.20 gate separates the two cases."""
    small = disruption(_matrix([400.0, 380.0, 420.0, 380.0, 420.0]))
    large = disruption(_matrix([400.0, 100.0, 700.0, 100.0, 700.0]))
    assert large > small
    assert small < 0.20, "this one would fail the dynamic-range gate, as it should"
    assert large >= 0.20


def test_percent_of_ceiling_uses_the_dm_control_ceiling():
    """percent_of_ceiling divides by the dm_control ceiling of 1000."""
    assert percent_of_ceiling(750.0) == pytest.approx(75.0)
