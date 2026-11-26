"""Tests that the pre-flight physics gates take the right path for boundary and drift configs.

Under a continuous drift schedule `set_task` is a no-op, so the boundary gate would fail a correct
drift benchmark. The drift gate instead checks that the physics move with time and do not respond
to task boundaries.
"""
import argparse

import pytest

pytest.importorskip("dm_control", reason="the dm_control family needs dm_control")
pytest.importorskip("shimmy", reason="the dm_control family needs shimmy")

from src_continuous_control.scripts.preflight import (          # noqa: E402
    check_non_inert,
    check_physics_change,
)

BOUNDARY = {a: "multienv_walker_walk_%s" % a for a in ("vanilla", "ewc", "pt")}
DRIFT1 = {a: "multienv_lipschitz1_walker_walk_%s" % a for a in ("vanilla", "ewc", "pt")}
DRIFT2 = {a: "multienv_lipschitz2_walker_walk_%s" % a for a in ("vanilla", "ewc", "pt")}


def test_boundary_gate_passes_on_a_boundary_config():
    """The physics-change gate passes on a boundary (step-schedule) config."""
    assert check_physics_change("multienv:walker-walk", overlays=BOUNDARY)


@pytest.mark.parametrize("overlays", [DRIFT1, DRIFT2], ids=["lipschitz1", "lipschitz2"])
def test_drift_gate_passes_on_a_drift_config(overlays):
    """The physics-change gate passes on a continuous drift config."""
    assert check_physics_change("multienv:walker-walk", overlays=overlays)


@pytest.mark.parametrize("overlays", [DRIFT1, DRIFT2], ids=["lipschitz1", "lipschitz2"])
def test_drift_non_inert_gate_passes_on_a_drift_config(overlays):
    """The non-inert gate passes on a continuous drift config."""
    assert check_non_inert("multienv:walker-walk", overlays=overlays, n_steps=60)


def test_boundary_non_inert_gate_still_passes():
    """The non-inert gate passes on a boundary config."""
    assert check_non_inert("multienv:walker-walk", overlays=BOUNDARY, n_steps=60)


def test_the_two_gates_take_different_paths():
    """Routing depends on the merged config's `drift_schedule`, not on the overlay name."""
    from src_continuous_control.train import build_config
    boundary = build_config(argparse.Namespace(agent="vanilla", config=BOUNDARY["vanilla"]))
    drift = build_config(argparse.Namespace(agent="vanilla", config=DRIFT2["vanilla"]))
    assert boundary["drift_schedule"] == "step"
    assert drift["drift_schedule"] == "sin"
