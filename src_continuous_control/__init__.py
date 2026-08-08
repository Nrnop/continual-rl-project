"""Compatibility shim package for tests expecting `src_continuous_control.*` imports.

This package re-exports the top-level modules (`agents`, `models`, `envs`, `utils`)
so older import paths like `src_continuous_control.agents` continue to work when the
codebase is laid out with those modules at the repo root.
"""

__all__ = ["agents", "models", "envs", "utils"]
