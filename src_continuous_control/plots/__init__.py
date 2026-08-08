"""Expose the repository-root plotting modules under the compatibility package."""
import importlib

_root_plots = importlib.import_module("plots")
__path__ = list(_root_plots.__path__)
