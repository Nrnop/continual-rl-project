"""Boundary-focused metrics: value drift and return drop at task switches.

These quantify the supervisor's concern directly: how violently the value function moves at a task
boundary, and how much episodic return collapses right after a switch.
"""
import numpy as np


class ValueDriftProbe:
    """Tracks |V(s) - V_prev(s)| on a FIXED set of probe states.

    Call snapshot() once you have a probe set and a value_fn. Each subsequent call returns the mean
    absolute change in predicted value since the previous snapshot — a direct read on how much the
    critic moved. Spikes are expected at boundaries; the PT critic should spike less than vanilla.
    """

    def __init__(self):
        self.prev_values = None

    def snapshot(self, value_fn, probe_states):
        """value_fn: np.ndarray[states] -> np.ndarray[values]. Returns mean |delta| or None."""
        values = np.asarray(value_fn(probe_states), dtype=np.float32).reshape(-1)
        drift = None
        if self.prev_values is not None and self.prev_values.shape == values.shape:
            drift = float(np.mean(np.abs(values - self.prev_values)))
        self.prev_values = values
        return drift


class BoundaryReturnTracker:
    """Records the running return just before each switch and the trough just after it.

    drop = pre_switch_plateau - post_switch_trough  (larger = worse adaptation).
    """

    def __init__(self, post_window_steps):
        self.post_window = post_window_steps
        self.records = []          # list of dicts: {step, pre, trough, drop}
        self._pending = None       # dict being filled during the post-switch window

    def on_switch(self, step, current_return):
        # close any still-open record (back-to-back switches) before opening a new one
        if self._pending is not None:
            self._finalize()
        self._pending = {
            "step": step,
            "pre": float(current_return),
            "trough": float(current_return),
            "end_step": step + self.post_window,
        }

    def update(self, step, current_return):
        if self._pending is None:
            return None
        self._pending["trough"] = min(self._pending["trough"], float(current_return))
        if step >= self._pending["end_step"]:
            return self._finalize()
        return None

    def _finalize(self):
        rec = self._pending
        rec["drop"] = rec["pre"] - rec["trough"]
        self.records.append({k: rec[k] for k in ("step", "pre", "trough", "drop")})
        self._pending = None
        return self.records[-1]

    def mean_drop(self):
        if not self.records:
            return None
        return float(np.mean([r["drop"] for r in self.records]))
