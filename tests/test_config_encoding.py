"""Every config file must be readable on a machine that is not this one.

WHY THIS EXISTS. Twelve configs were generated with `open(path, "w")` and no encoding. On Windows
that writes cp1252, so the em-dash in a comment header became byte 0x97. PyYAML decodes as UTF-8
and failed at byte 11 -- before parsing a single key -- so every one of 160 planned runs would have
crashed at config load on the Linux box they were sent to.

It could not be caught locally: `train.py`'s `_load_yaml` also opens without an encoding, so the
same wrong codec read them back and every local check passed. Six older `ln_*` configs turned out
to have the same latent defect and would have failed the same way.

The 141-test suite missed it because nothing loaded a config FILE -- `test_config_merge.py` and
`test_ewc_timer.py` both build configs from dicts. So this file reads the bytes off disk, which is
the only thing that reproduces the failure.

The rule is UTF-8, not ASCII: plenty of configs legitimately contain typographic characters and
decode fine everywhere. What is forbidden is a platform-default encoding leaking into a file that
crosses machines.
"""
import glob
import os

import pytest
import yaml

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
CONFIGS = sorted(glob.glob(os.path.join(CFG_DIR, "*.yaml")))


def test_there_are_configs_to_check():
    """A glob that silently matches nothing would make every test below vacuously pass."""
    assert len(CONFIGS) > 20, f"only found {len(CONFIGS)} configs in {CFG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p) for p in CONFIGS])
def test_config_is_valid_utf8(path):
    """Decode the raw bytes explicitly. Opening in text mode would use the platform default and
    hide exactly the bug this test exists for."""
    raw = open(path, "rb").read()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(
            f"{os.path.basename(path)} is not valid UTF-8: byte 0x{raw[exc.start]:02x} at "
            f"position {exc.start}. It was probably written with open(path, 'w') on Windows, "
            f"which uses cp1252. Write configs with an explicit encoding.")


@pytest.mark.parametrize("path", CONFIGS, ids=[os.path.basename(p) for p in CONFIGS])
def test_config_parses_as_yaml(path):
    """A file can be valid UTF-8 and still be unloadable; both failures block a run identically."""
    with open(path, "rb") as fh:
        loaded = yaml.safe_load(fh.read().decode("utf-8"))
    assert loaded is None or isinstance(loaded, dict), \
        f"{os.path.basename(path)} did not parse to a mapping"
