"""The carry-over metric, pinned on matrices whose answer is known by construction.

WHY THIS MATTERS MORE THAN A USUAL UNIT TEST. Carry-over is the single measurement the project's
central explanation rests on — `pt` helps when consecutive tasks share structure, 0.56 on cartpole
against 0.23 on HalfCheetah — and the family study puts six of these on the x-axis of the plot that
justifies it. Until `scripts/report_multienv.py` was written the number was computed by no
committed script at all: it was worked out once by hand and survived as a parenthesis in
CARTPOLE_RESULTS.md. These tests are what stop the definition drifting the next time someone
recomputes it.
"""
import numpy as np
import pytest

from src_continuous_control.scripts.report_multienv import (
    carry_over,
    disruption,
    percent_of_ceiling,
)


def _matrix(row0, n=5):
    """A transfer matrix whose row 0 is what we care about; the rest is never read."""
    m = np.zeros((n, n), dtype=float)
    m[0] = row0
    return m


# ---------------------------------------------------------------------------
# The three anchor points of the scale
# ---------------------------------------------------------------------------
def test_perfect_transfer_is_one():
    """Equally competent on every physics setting -> 1.0, whatever the baselines are."""
    m = _matrix([500.0] * 5)
    assert carry_over(m, [100.0] * 5) == pytest.approx(1.0)


def test_total_loss_is_zero():
    """Back to the floor on every other task -> 0.0. NOT 'raw return is zero'."""
    m = _matrix([500.0, 100.0, 100.0, 100.0, 100.0])
    assert carry_over(m, [100.0] * 5) == pytest.approx(0.0)


def test_worse_than_untrained_is_negative():
    """A policy that actively hurts elsewhere must read below zero, not be clipped at it.

    ball_in_cup-catch measured ~0.00 on the gate runs and could go negative at full length; a
    metric that floored at zero would hide the most extreme point on the x-axis.
    """
    m = _matrix([500.0, 50.0, 50.0, 50.0, 50.0])
    assert carry_over(m, [100.0] * 5) < 0.0


# ---------------------------------------------------------------------------
# The baseline correction is doing real work
# ---------------------------------------------------------------------------
def test_baseline_correction_changes_the_answer():
    """Without it, a high floor masquerades as transfer.

    walker-stand's random-init policy already scores ~139 by falling over slowly. Uncorrected, an
    agent that lost everything would still look like it retained most of its competence.
    """
    m = _matrix([500.0, 150.0, 150.0, 150.0, 150.0])
    corrected = carry_over(m, [140.0] * 5)
    uncorrected = carry_over(m, [0.0] * 5)
    assert corrected < 0.05, "with the floor removed, almost nothing survived"
    assert uncorrected > 0.25, "without the correction the same run looks like real transfer"


def test_per_task_baselines_are_used_not_a_single_scalar():
    """b_j is per task. The tasks genuinely differ in how much an untrained policy scores."""
    m = _matrix([500.0, 300.0, 300.0, 300.0, 300.0])
    flat = carry_over(m, [100.0] * 5)
    varied = carry_over(m, [100.0, 250.0, 250.0, 250.0, 250.0])
    assert varied < flat


# ---------------------------------------------------------------------------
# It only reads row 0, and it says so
# ---------------------------------------------------------------------------
def test_only_row_zero_is_read():
    """Later rows have seen more tasks AND more steps; using them confounds transfer with budget."""
    a = _matrix([500.0, 400.0, 400.0, 400.0, 400.0])
    b = a.copy()
    b[1:] = 12345.0                      # nonsense everywhere except row 0
    assert carry_over(a, [100.0] * 5) == carry_over(b, [100.0] * 5)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_untrained_task_zero_gives_nan_not_a_huge_ratio():
    """"What fraction of nothing survived" has no answer, and must not be reported as one.

    A near-zero denominator would otherwise produce an enormous carry-over from an arm that simply
    never learned — the same trap as reporting backward transfer for an agent with peak return 0.
    """
    m = _matrix([100.0, 400.0, 400.0, 400.0, 400.0])
    assert np.isnan(carry_over(m, [100.0] * 5))


def test_non_square_matrix_is_refused():
    with pytest.raises(ValueError, match="square"):
        carry_over(np.zeros((3, 5)), [0.0] * 5)


def test_wrong_number_of_baselines_is_refused():
    with pytest.raises(ValueError, match="baselines"):
        carry_over(np.zeros((5, 5)), [0.0] * 3)


# ---------------------------------------------------------------------------
# Disruption — the dynamic-range gate and the reported covariate
# ---------------------------------------------------------------------------
def test_disruption_is_zero_when_the_physics_do_not_matter():
    assert disruption(_matrix([400.0] * 5)) == pytest.approx(0.0)


def test_disruption_grows_with_the_spread():
    small = disruption(_matrix([400.0, 380.0, 420.0, 380.0, 420.0]))
    large = disruption(_matrix([400.0, 100.0, 700.0, 100.0, 700.0]))
    assert large > small
    assert small < 0.20, "this one would fail the dynamic-range gate, as it should"
    assert large >= 0.20


def test_percent_of_ceiling_uses_the_dm_control_ceiling():
    """Every task's maximum is 1000 by construction; that is why returns average across the family."""
    assert percent_of_ceiling(750.0) == pytest.approx(75.0)
