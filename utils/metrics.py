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


# ---------------------------------------------------------------------------
# Forward / backward transfer (Lopez-Paz & Ranzato 2017)
# ---------------------------------------------------------------------------
def evaluate_policy_on_tasks(policy_fn, env, set_task, n_tasks, n_episodes=10,
                             max_steps=None, restore_task=None, return_std=False):
    """Mean return of a FROZEN policy on each task, one row of the transfer matrix.

    Args:
        policy_fn: obs (np.ndarray) -> action (np.ndarray). Must be DETERMINISTIC — the transfer
            matrix measures competence, not exploration, so pass the policy mean, not a sample.
        env: a single (non-vector) evaluation env. Never the training env: `set_task` changes its
            physics, so evaluating on the training env would corrupt the run being measured.
        set_task: callable(i) that puts `env` on task i (e.g. the drift wrapper's `set_task`).
        n_tasks: how many tasks the sequence cycles through.
        n_episodes: episodes averaged per cell. 10 is the Phase 2 default.
        restore_task: task to leave `env` on afterwards. None leaves it on task n_tasks-1.

    Returns np.ndarray of shape (n_tasks,), or (means, standard_errors) with return_std=True.

    THE STANDARD ERROR IS NOT OPTIONAL BOOKKEEPING. Each cell is a mean over `n_episodes`
    episodes, so it carries sampling noise. If that noise is the same size as the differences
    between methods, the transfer figure is noise however carefully it is plotted — and the only
    way to know is to record it.

    The caller is responsible for RNG isolation — `env.reset()` consumes randomness even though a
    deterministic policy does not.
    """
    returns = np.zeros(n_tasks, dtype=np.float64)
    errors = np.zeros(n_tasks, dtype=np.float64)
    for j in range(n_tasks):
        set_task(j)
        per_episode = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total, done, steps = 0.0, False, 0
            while not done:
                obs, reward, terminated, truncated, info = env.step(policy_fn(obs))
                # The honest, un-normalized reward, exactly as the return curves use.
                total += float(info.get("directional_reward", reward))
                done = bool(terminated or truncated)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
            per_episode.append(total)
        returns[j] = float(np.mean(per_episode))
        errors[j] = (float(np.std(per_episode, ddof=1) / np.sqrt(len(per_episode)))
                     if len(per_episode) > 1 else 0.0)
    if restore_task is not None:
        set_task(restore_task)
    return (returns, errors) if return_std else returns


class TransferMatrix:
    """R[i, j] = mean return on task j after FINISHING task i (Lopez-Paz & Ranzato 2017).

    Rows are filled in as training passes each boundary, so no checkpoints have to be kept.

        BWT = mean over j < N of  R[N, j] - R[j, j]      retention on physics already seen
        FWT = mean over j > 0 of  R[j-1, j] - b[j]       zero-shot competence on physics not yet
                                                         trained on, against a random-init baseline

    with N the last task (0-indexed: N = n_tasks - 1).

    ONE CAVEAT SPECIFIC TO THIS BENCHMARK. The Phase 2 task sequence deliberately REVISITS —
    multipliers [1.0, 1.6, 0.6, 1.6, 0.6] make tasks 1/3 and 2/4 the same physics — because BWT is
    undefined on a sequence that never repeats. The cost is that FWT's "not yet seen" reading only
    holds for the first occurrence of each physics setting; for a revisit, R[j-1, j] - b[j] is
    measuring retention, not foresight. Report FWT over first occurrences when that distinction
    matters.
    """

    def __init__(self, n_tasks):
        if int(n_tasks) < 2:
            raise ValueError("a transfer matrix needs at least 2 tasks")
        self.n_tasks = int(n_tasks)
        self.matrix = np.full((self.n_tasks, self.n_tasks), np.nan, dtype=np.float64)
        # Per-cell standard error, same shape. Without it there is no way to tell a real method
        # difference from evaluation noise.
        self.errors = np.full((self.n_tasks, self.n_tasks), np.nan, dtype=np.float64)
        self.baselines = None       # b_j: a random-init policy's return on each task

    def set_baselines(self, returns):
        """b_j — a RANDOM-INIT policy's return per task, measured once before training."""
        b = np.asarray(returns, dtype=np.float64).reshape(-1)
        if b.shape != (self.n_tasks,):
            raise ValueError(f"expected {self.n_tasks} baseline returns, got {b.shape}")
        self.baselines = b

    def add_row(self, i, returns, errors=None):
        """Record the policy's returns on every task, measured after finishing task i."""
        row = np.asarray(returns, dtype=np.float64).reshape(-1)
        if row.shape != (self.n_tasks,):
            raise ValueError(f"expected {self.n_tasks} returns, got {row.shape}")
        if not 0 <= int(i) < self.n_tasks:
            raise ValueError(f"task index {i} outside 0..{self.n_tasks - 1}")
        self.matrix[int(i)] = row
        if errors is not None:
            err = np.asarray(errors, dtype=np.float64).reshape(-1)
            if err.shape != (self.n_tasks,):
                raise ValueError(f"expected {self.n_tasks} standard errors, got {err.shape}")
            self.errors[int(i)] = err

    def is_complete(self):
        return not np.isnan(self.matrix).any()

    def bwt(self):
        """Backward transfer. None until the last row and the whole diagonal are present."""
        last = self.n_tasks - 1
        diag = np.array([self.matrix[j, j] for j in range(last)])
        final = self.matrix[last, :last]
        if np.isnan(diag).any() or np.isnan(final).any():
            return None
        return float(np.mean(final - diag))

    def fwt(self):
        """Forward transfer against the random-init baselines. None until those are available."""
        if self.baselines is None:
            return None
        vals = [self.matrix[j - 1, j] - self.baselines[j] for j in range(1, self.n_tasks)]
        vals = np.asarray(vals, dtype=np.float64)
        if np.isnan(vals).any():
            return None
        return float(np.mean(vals))

    def repeat_noise(self, task_labels):
        """A second, assumption-free noise estimate: disagreement between REPEATED tasks.

        The Phase 2 sequence revisits — tasks 1/3 are the same physics, as are 2/4. Two columns
        with the same label should hold the same number in expectation, so their observed
        difference is pure evaluation noise. This needs no extra episodes at all.

        Returns the mean |difference| over same-label column pairs, or None.
        """
        labels = list(task_labels)
        diffs = []
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                if labels[a] != labels[b]:
                    continue
                pair = np.abs(self.matrix[:, a] - self.matrix[:, b])
                diffs.extend(pair[~np.isnan(pair)].tolist())
        return float(np.mean(diffs)) if diffs else None

    def summary(self):
        return {"transfer_matrix": self.matrix.copy(),
                "cell_standard_errors": self.errors.copy(),
                "baselines": None if self.baselines is None else self.baselines.copy(),
                "bwt": self.bwt(), "fwt": self.fwt()}


def evaluate_transfer_matrix(policy_fns, env, set_task, n_episodes=10, max_steps=None,
                             baseline_policy_fn=None):
    """Build the whole matrix offline from one frozen policy per finished task.

    `policy_fns[i]` is the policy as it stood at the END of task i, so len(policy_fns) is the
    number of tasks. Used for after-the-fact analysis from checkpoints; during training, fill the
    rows online with `evaluate_policy_on_tasks` + `TransferMatrix.add_row` instead, which needs no
    checkpoints at all.
    """
    n_tasks = len(policy_fns)
    tm = TransferMatrix(n_tasks)
    if baseline_policy_fn is not None:
        tm.set_baselines(evaluate_policy_on_tasks(
            baseline_policy_fn, env, set_task, n_tasks, n_episodes, max_steps))
    for i, fn in enumerate(policy_fns):
        tm.add_row(i, evaluate_policy_on_tasks(fn, env, set_task, n_tasks, n_episodes, max_steps))
    return tm
