"""Compatibility shim to run the repository's top-level `train.py` as
`src_continuous_control.train` (so `python -m src_continuous_control.train` works).

This loader imports the top-level `train.py` file and executes it under the
package name `src_continuous_control.train` so that the relative imports in
`train.py` (e.g. `from .agents import ...`) resolve correctly.
"""
import importlib.util
import importlib
import os
import sys

_THIS_DIR = os.path.dirname(__file__)
_TRAIN_PY = os.path.abspath(os.path.join(_THIS_DIR, "..", "train.py"))

spec = importlib.util.spec_from_file_location("src_continuous_control.train", _TRAIN_PY)
module = importlib.util.module_from_spec(spec)
# Ensure relative imports inside train.py resolve to the shim package
module.__package__ = "src_continuous_control"
sys.modules["src_continuous_control.train"] = module

try:
    # Prefer importing the package properly if possible (ensures __path__ etc.)
    importlib.import_module("src_continuous_control")
except Exception:
    # Fall back to ensuring a minimal package module exists in sys.modules
    if "src_continuous_control" not in sys.modules:
        import types

        pkg = types.ModuleType("src_continuous_control")
        pkg.__path__ = [os.path.abspath(os.path.join(_THIS_DIR))]
        sys.modules["src_continuous_control"] = pkg

# Pre-register commonly-referenced top-level packages under the
# `src_continuous_control.*` names so relative imports like
# `from .envs.directional_half_cheetah import ...` work when the real
# packages live at the repository root (e.g. `envs`, `agents`, ...).
for _name in ("agents", "models", "envs", "utils"):
    try:
        _mod = importlib.import_module(_name)
        sys.modules[f"src_continuous_control.{_name}"] = _mod
    except Exception:
        # ignore absence; the called code will raise a clear error later
        pass

spec.loader.exec_module(module)


if __name__ == "__main__":
    if hasattr(module, "main"):
        module.main()
