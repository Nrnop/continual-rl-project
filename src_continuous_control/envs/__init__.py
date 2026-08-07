"""Re-export top-level ``envs`` modules under the compatibility package."""
import importlib
import pkgutil
import sys

_mod = importlib.import_module("envs")
for _name in dir(_mod):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_mod, _name)

for _info in pkgutil.iter_modules(_mod.__path__):
    _submodule = importlib.import_module(f"envs.{_info.name}")
    sys.modules[f"{__name__}.{_info.name}"] = _submodule

__all__ = [n for n in dir() if not n.startswith("_")]
