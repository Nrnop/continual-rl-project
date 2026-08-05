"""Boundary-focused metrics: value drift, return drop, jumpstart, and retention.

These quantify the supervisor's concern directly — how violently the value function moves at a task
boundary, and how much return collapses right after a switch — and they are also the quantities the
PT theory actually makes predictions about:

  * `JumpstartTracker`  <- Theorem 6/8. Theorem 6: the permanent value function's fixed point
    E_tau[v_tau] optimises the JUMPSTART objective, i.e. performance on a new task *before*
    collecting data on it. Theorem 8: PT has a tighter error bound than TD only for `k <= k0`
    steps after a switch, and "as k -> infinity ... their upper bounds collapse to 0". So the
    predicted advantage lives in a WINDOW right after each boundary and decays to nothing.
    Averaging return over a whole 614k-step phase integrates that effect away, which is why the
    per-phase mean is the wrong primary metric for this method.

  * `RetentionProbe`  <- Theorem 7: there exists k0 such that for all k >= k0,
    E||V^(TD)_{t+k} - v_i||^2 > E||V^(P) - v_i||^2 — i.e. once enough samples from the new task
    have arrived, the single TD estimate has forgotten task i while the PERMANENT component still
    predicts it. This is the paper's second performance measure, the dotted "MSE on other tasks"
    line in Fig. 2, and we have never measured it.
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

    `post_window_steps` MUST span several PPO updates. `update()` is called once per update, so a
    window shorter than one batch (n_steps * num_envs) finalises on the very first post-switch
    sample and reports drop = 0.00 for every boundary — an artifact, not a measurement. Callers
    should compute it as `k_updates * n_steps * num_envs`; the assertion below makes the failure
    loud instead of silent.
    """

    def __init__(self, post_window_steps, min_useful_steps=None):
        if min_useful_steps is not None and post_window_steps < 2 * min_useful_steps:
            raise ValueError(
                f"post_window_steps={post_window_steps} spans < 2 updates of "
                f"{min_useful_steps} env steps; the tracker would finalise on its first sample "
                f"and report drop=0 by construction.")
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


class JumpstartTracker:
    """Return over a fixed window immediately after each task switch (Theorems 6 and 8).

    `window_steps` should be SHORT relative to the phase — this measures adaptation speed, not
    asymptote. Records, per boundary:

        first : return at the first update after the switch (the jumpstart proper — the value the
                permanent baseline hands the agent before it has adapted)
        mean  : mean return across the window (area under the recovery curve)
        end   : return at the end of the window
        gain  : end - first (how much of the recovery happened inside the window)
    """

    def __init__(self, window_steps):
        self.window = window_steps
        self.records = []
        self._pending = None

    def on_switch(self, step):
        if self._pending is not None:
            self._finalize()
        self._pending = {"step": step, "end_step": step + self.window, "vals": []}

    def update(self, step, current_return):
        if self._pending is None:
            return None
        self._pending["vals"].append(float(current_return))
        if step >= self._pending["end_step"]:
            return self._finalize()
        return None

    def _finalize(self):
        p = self._pending
        self._pending = None
        vals = p["vals"]
        if not vals:
            return None
        rec = {
            "step": p["step"],
            "first": vals[0],
            "mean": float(np.mean(vals)),
            "end": vals[-1],
            "gain": vals[-1] - vals[0],
        }
        self.records.append(rec)
        return rec

    def mean_jumpstart(self):
        if not self.records:
            return None
        return float(np.mean([r["mean"] for r in self.records]))


class RetentionProbe:
    """How well each component still predicts the value of a task it is no longer training on.

    Theorem 7 compares two quantities against the SAME reference v_i, the true value function of
    the finished task i:

        E||V^(TD)_{t+k} - v_i||^2   >   E||V^(P) - v_i||^2      for all k >= k0

    We approximate v_i by the agent's own converged ACTING value V^(PT) at the end of task i —
    the best estimate of it available — and store that single reference per task. Every component
    is then scored against it, which is the comparison the theorem makes.

    This is deliberately not "how much has V_perm moved since the switch": a permanent network with
    a tiny learning rate never moves and would score a perfect 0 while containing nothing about the
    task at all. Scored against v_i, a frozen permanent reports a large error in absolute terms.

    CONTROL BASELINES (added 2026-08-04, and they are NOT optional). Absolute error is not enough.
    On a symmetric reward-sign flip the two tasks' values are opposite in sign, so a permanent that
    never learns anything still beats an adapted estimate: `mse_perm < mse_full` is then a measure
    of INERTIA, not retention, and reads as a false confirmation of Theorem 7. (Measured: a
    permanent frozen at exactly zero scores 24.88 against an adapted 99.54.) Register static
    baselines with `set_baseline()` and every measurement is scored against them too:

        perm_init : V_perm evaluated at t=0, frozen. If `mse_perm` is not clearly BELOW
                    `mse_perm_init`, the permanent has learned nothing and any perm-vs-full
                    comparison is meaningless.
        zero      : the constant-zero function. If `mse_perm` is not below `mse_zero`, the
                    permanent is worse than storing nothing at all.

    Usage:
        probe.set_baseline("perm_init", v_perm_at_t0)            # once, at the start
        probe.snapshot(task_id, reference_fn, probe_states)      # at the END of task i
        probe.measure(active_task_id, {"perm": f1, "full": f2}, probe_states)

    For vanilla, `perm` and `full` are the same function — a single critic has no separately
    retained component, which is exactly the contrast Theorem 7 draws.
    """

    def __init__(self):
        self.snapshots = {}        # task_id -> np.ndarray, the converged V^(PT) on probe_states
        self.baselines = {}        # name -> np.ndarray, static reference predictions

    def snapshot(self, task_id, reference_fn, probe_states):
        """Store the converged acting value for `task_id` — the stand-in for v_i."""
        self.snapshots[task_id] = np.asarray(
            reference_fn(probe_states), dtype=np.float32).reshape(-1)

    def set_baseline(self, name, values):
        """Register a STATIC prediction (e.g. the permanent at initialisation) to score against."""
        self.baselines[name] = np.asarray(values, dtype=np.float32).reshape(-1)

    def measure(self, active_task_id, value_fns, probe_states):
        """MSE of each value fn AND each control baseline against every INACTIVE task's v_i."""
        others = [t for t in self.snapshots if t != active_task_id]
        if not others:
            return {}
        out = {}
        for name, fn in value_fns.items():
            cur = np.asarray(fn(probe_states), dtype=np.float32).reshape(-1)
            errs = [float(np.mean((cur - self.snapshots[t]) ** 2))
                    for t in others if self.snapshots[t].shape == cur.shape]
            if errs:
                out[name] = float(np.mean(errs))
        for name, vals in self.baselines.items():
            errs = [float(np.mean((vals - self.snapshots[t]) ** 2))
                    for t in others if self.snapshots[t].shape == vals.shape]
            if errs:
                out[name] = float(np.mean(errs))
        return out
