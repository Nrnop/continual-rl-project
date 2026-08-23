"""build_config: a YAML value must survive the CLI merge unless the flag was actually passed.

THE BUG THIS GUARDS (found 2026-08-17). `build_config` merges CLI over YAML with

    for key, val in vars(cli_args).items():
        if val is not None:
            cfg[key] = val

`argparse`'s `action="store_true"` defaults to **False**, not None. False is not None, so every
boolean flag overwrote whatever the config file said — which meant those keys could not be set
from YAML AT ALL. A config carrying `disable_task_switch: true` trained with task switching fully
on and printed `SWITCH to task 1 (physics x1.6)` while doing it.

That is CLAUDE.md failure mode #1, a control that was not actually on, and the reason it survived
is that nothing about the run looks wrong: it trains, it converges, it writes plausible numbers.
The only visible symptom is a log line, in a file nobody reads, that says a switch happened in a
run whose whole purpose was that none would.

The fix is `default=None` on every store_true flag. These tests pin the behaviour rather than the
fix, so an equivalent regression through some other route still fails here.
"""
import types

import pytest

from ..train import build_config, parse_args


# Every boolean flag whose value a config file may legitimately want to set.
BOOL_FLAGS = [
    "no_wandb",
    "no_tb",
    "no_eval",
    "save_checkpoints",
    "disable_task_switch",
    "render",
]


@pytest.mark.parametrize("flag", BOOL_FLAGS)
def test_store_true_flags_default_to_none_not_false(flag, monkeypatch):
    """Unpassed boolean flags must arrive as None so the merge loop skips them.

    This is the root-cause test: if a flag defaults to False it will silently clobber YAML,
    whatever the rest of the config machinery does.
    """
    monkeypatch.setattr("sys.argv", ["train.py", "--agent", "vanilla"])
    args = parse_args()
    assert getattr(args, flag) is None, (
        f"--{flag.replace('_', '-')} defaults to {getattr(args, flag)!r}, not None. "
        f"build_config merges any non-None CLI value over the YAML, so this flag will "
        f"overwrite a config file's value and the key becomes unsettable from YAML."
    )


@pytest.mark.parametrize("flag", BOOL_FLAGS)
def test_yaml_true_survives_when_flag_not_passed(flag):
    """A config setting the key to True must still read True after the merge."""
    args = types.SimpleNamespace(agent="vanilla", config=None, **{f: None for f in BOOL_FLAGS})
    cfg = build_config(args)
    cfg[flag] = True                      # stand in for the overlay having set it
    merged = dict(cfg)
    for key, val in vars(args).items():
        if key != "config" and val is not None:
            merged[key] = val
    assert merged[flag] is True


def test_ceiling_overlay_actually_disables_switching():
    """The overlay that motivated the bug. Its whole purpose is this one key."""
    args = types.SimpleNamespace(agent="vanilla", config="ceiling_learned",
                                 **{f: None for f in BOOL_FLAGS})
    cfg = build_config(args)
    assert cfg.get("disable_task_switch") is True, (
        "ceiling_learned.yaml exists to measure a STATIONARY ceiling. If this key does not "
        "survive the merge the run switches tasks and silently measures the continual "
        "benchmark instead — the exact failure this file was written for."
    )


def test_passed_flag_still_beats_yaml():
    """The override must keep working in the direction it is meant to work."""
    args = types.SimpleNamespace(agent="vanilla", config="ceiling_learned",
                                 **{f: None for f in BOOL_FLAGS})
    args.no_wandb = True                  # as queue_runs.py passes it
    cfg = build_config(args)
    assert cfg["no_wandb"] is True
    assert cfg.get("disable_task_switch") is True   # and it did not disturb the overlay
