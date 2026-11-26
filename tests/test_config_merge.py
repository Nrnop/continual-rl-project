"""build_config: a YAML value must survive the CLI merge unless the flag was passed.

`build_config` copies every CLI value that is not None over the YAML. A plain store_true flag
defaults to False, which would overwrite the YAML value even when the flag is absent, so keys such
as `disable_task_switch` could not be set from a config file. Every store_true flag therefore uses
`default=None`. The tests check the merged result, so a regression by another route also fails.
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
    """Unpassed boolean flags arrive as None, so the merge loop skips them."""
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
    """ceiling_learned.yaml still disables task switching after the merge."""
    args = types.SimpleNamespace(agent="vanilla", config="ceiling_learned",
                                 **{f: None for f in BOOL_FLAGS})
    cfg = build_config(args)
    assert cfg.get("disable_task_switch") is True, (
        "ceiling_learned.yaml exists to measure a STATIONARY ceiling. If this key does not "
        "survive the merge the run switches tasks and silently measures the continual "
        "benchmark instead - the exact failure this file was written for."
    )


def test_passed_flag_still_beats_yaml():
    """The override must keep working in the direction it is meant to work."""
    args = types.SimpleNamespace(agent="vanilla", config="ceiling_learned",
                                 **{f: None for f in BOOL_FLAGS})
    args.no_wandb = True                  # as queue_runs.py passes it
    cfg = build_config(args)
    assert cfg["no_wandb"] is True
    assert cfg.get("disable_task_switch") is True   # and it did not disturb the overlay
