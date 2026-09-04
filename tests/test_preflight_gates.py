"""The pre-flight gates must ask the RIGHT question of each setting.

WHY THIS FILE EXISTS. The drift sweep was shipped to a rented box and aborted on all ten cells
before spawning a single training process: `check_physics_change` called `set_task(0..4)` and
required the physics to differ, but under a continuous schedule `set_task` is a deliberate no-op —
the multiplier is a function of the clock — so a correct benchmark failed a gate that was asking a
boundary question of a boundary-free setting.

The gate was right to be strict and wrong about what to be strict about. These tests pin both
halves so the same class of mistake cannot come back:

  * the boundary gate still catches genuinely constant physics;
  * the drift gate catches physics that do not move with the clock, AND catches a drift config that
    still responds to task boundaries (which would be the boundary benchmark under a drift name).
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
    assert check_physics_change("multienv:walker-walk", overlays=BOUNDARY)


@pytest.mark.parametrize("overlays", [DRIFT1, DRIFT2], ids=["lipschitz1", "lipschitz2"])
def test_drift_gate_passes_on_a_drift_config(overlays):
    """The regression this file exists for: these aborted the whole sweep on the box."""
    assert check_physics_change("multienv:walker-walk", overlays=overlays)


@pytest.mark.parametrize("overlays", [DRIFT1, DRIFT2], ids=["lipschitz1", "lipschitz2"])
def test_drift_non_inert_gate_passes_on_a_drift_config(overlays):
    assert check_non_inert("multienv:walker-walk", overlays=overlays, n_steps=60)


def test_boundary_non_inert_gate_still_passes():
    assert check_non_inert("multienv:walker-walk", overlays=BOUNDARY, n_steps=60)


def test_the_two_gates_take_different_paths():
    """A drift config must not be silently routed through the boundary check, or vice versa.

    Routing is decided by the merged config's `drift_schedule`, so this asserts on the realised
    value rather than on the overlay's name.
    """
    from src_continuous_control.train import build_config
    boundary = build_config(argparse.Namespace(agent="vanilla", config=BOUNDARY["vanilla"]))
    drift = build_config(argparse.Namespace(agent="vanilla", config=DRIFT2["vanilla"]))
    assert boundary["drift_schedule"] == "step"
    assert drift["drift_schedule"] == "sin"
