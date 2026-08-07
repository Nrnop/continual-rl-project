"""Expose repository-root tests for legacy ``src_continuous_control.tests`` imports."""
import os

__path__ = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))]
