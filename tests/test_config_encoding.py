"""Every config file must be valid UTF-8 and parse as YAML on any platform.

A config written with `open(path, "w")` on Windows is cp1252, which PyYAML cannot decode as UTF-8.
Reading in text mode with the platform default would hide this, so the tests read raw bytes.
Non-ASCII characters are fine as long as the file is UTF-8.
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
    """The raw bytes decode as UTF-8 (text mode would use the platform default instead)."""
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
    """The file parses with yaml.safe_load to a mapping or None."""
    with open(path, "rb") as fh:
        loaded = yaml.safe_load(fh.read().decode("utf-8"))
    assert loaded is None or isinstance(loaded, dict), \
        f"{os.path.basename(path)} did not parse to a mapping"
