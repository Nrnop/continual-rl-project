"""Task-boundary metrics (value drift, return drop, jumpstart, retention) and transfer.

JumpstartTracker follows Theorems 6 and 8 of Anand & Precup 2023: the permanent optimises the
jumpstart objective, and PT's error bound is tighter than TD's only for k <= k0 steps after a
switch, so the predicted advantage sits in a short window after each boundary.
RetentionProbe follows Theorem 7: after enough new-task samples, a single TD estimate has forgotten
task i while the permanent still predicts it (the "MSE on other tasks" curve in Fig. 2).
"""
import numpy as np


class ValueDriftProbe:
    """Mean |V(s) - V_prev(s)| on a fixed set of probe states between snapshots."""

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
    """Running return just before each switch and the trough just after it.

    drop = pre_switch_return - post_switch_trough (larger is worse).

    `update()` is called once per PPO update, so `post_window_steps` must span several updates
    (k_updates * n_steps * num_envs). A window shorter than one batch would finalise on the first
    sample and report drop = 0; the constructor raises in that case.
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
        # Close a still-open record (back-to-back switches) before opening a new one.
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
    """Return over a fixed window right after each task switch (Theorems 6 and 8).

    `window_steps` should be short relative to a task. Records per boundary:

        first : return at the first update after the switch (the jumpstart itself)
        mean  : mean return across the window (area under the recovery curve)
        end   : return at the end of the window
        gain  : end - first
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
    """MSE of each value component against finished tasks' values (Theorem 7).

    Theorem 7: E||V^(TD)_{t+k} - v_i||^2 > E||V^(P) - v_i||^2 for all k >= k0. v_i is
    approximated by the agent's acting value V^(PT) at the end of task i, stored once per task.
    Scoring against v_i, not against V_perm's own past values, keeps a frozen permanent from
    scoring a perfect 0.

    Static baselines registered with `set_baseline()` are scored too, because a permanent that
    learns nothing can still beat an adapted estimate (e.g. when the tasks' values have opposite
    signs):

        perm_init : V_perm at t=0. If mse_perm is not clearly below this, the permanent has
                    learned nothing.
        zero      : the constant-zero function.

    Usage:
        probe.set_baseline("perm_init", v_perm_at_t0)            # once, at the start
        probe.snapshot(task_id, reference_fn, probe_states)      # at the end of task i
        probe.measure(active_task_id, {"perm": f1, "full": f2}, probe_states)

    For vanilla, `perm` and `full` are the same function.
    """

    def __init__(self):
        self.snapshots = {}        # task_id -> np.ndarray, V^(PT) on probe_states at task end
        self.baselines = {}        # name -> np.ndarray, static reference predictions

    def snapshot(self, task_id, reference_fn, probe_states):
        """Store the acting value at the end of `task_id` as the stand-in for v_i."""
        self.snapshots[task_id] = np.asarray(
            reference_fn(probe_states), dtype=np.float32).reshape(-1)

    def set_baseline(self, name, values):
        """Register a static prediction (e.g. the permanent at initialisation) to score."""
        self.baselines[name] = np.asarray(values, dtype=np.float32).reshape(-1)

    def measure(self, active_task_id, value_fns, probe_states):
        """MSE of each value fn and each baseline against every inactive task's v_i."""
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


# Forward / backward transfer (Lopez-Paz & Ranzato 2017)
def evaluate_policy_on_tasks(policy_fn, env, set_task, n_tasks, n_episodes=10,
                             max_steps=None, restore_task=None, return_std=False):
    """Mean return of a frozen policy on each task: one row of the transfer matrix.

    Args:
        policy_fn: obs (np.ndarray) -> action (np.ndarray). Pass the deterministic policy mean.
        env: a single (non-vector) evaluation env, never the training env, since `set_task`
            changes its physics.
        set_task: callable(i) that puts `env` on task i.
        n_tasks: number of tasks in the sequence.
        n_episodes: episodes averaged per cell.
        restore_task: task to leave `env` on afterwards. None leaves it on task n_tasks-1.

    Returns np.ndarray of shape (n_tasks,), or (means, standard_errors) with return_std=True.
    The caller handles RNG isolation: `env.reset()` consumes randomness.
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
                # Un-normalized reward, as in the return curves.
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
    """R[i, j] = mean return on task j after finishing task i (Lopez-Paz & Ranzato 2017).

    Rows are filled in as training passes each boundary, so no checkpoints are needed.

        BWT = mean over j < N of  R[N, j] - R[j, j]      retention on physics already seen
        FWT = mean over j > 0 of  R[j-1, j] - b[j]       zero-shot return on the next task,
                                                         against a random-init baseline

    with N = n_tasks - 1. When the sequence revisits physics (e.g. multipliers
    [1.0, 1.6, 0.6, 1.6, 0.6]), FWT on a revisited task measures retention rather than zero-shot
    transfer; restrict FWT to first occurrences when that matters.
    """

    def __init__(self, n_tasks):
        if int(n_tasks) < 2:
            raise ValueError("a transfer matrix needs at least 2 tasks")
        self.n_tasks = int(n_tasks)
        self.matrix = np.full((self.n_tasks, self.n_tasks), np.nan, dtype=np.float64)
        # Per-cell standard error, same shape as the matrix.
        self.errors = np.full((self.n_tasks, self.n_tasks), np.nan, dtype=np.float64)
        self.baselines = None       # b_j: a random-init policy's return on each task

    def set_baselines(self, returns):
        """Set b_j, a random-init policy's return per task, measured once before training."""
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
        """Evaluation-noise estimate from columns of repeated tasks.

        Columns with the same label have the same physics, so their difference is evaluation
        noise. Returns the mean |difference| over same-label column pairs, or None.
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

    `policy_fns[i]` is the policy at the end of task i. For analysis from checkpoints; during
    training, rows are filled online with `evaluate_policy_on_tasks` and `TransferMatrix.add_row`.
    """
    n_tasks = len(policy_fns)
    tm = TransferMatrix(n_tasks)
    if baseline_policy_fn is not None:
        tm.set_baselines(evaluate_policy_on_tasks(
            baseline_policy_fn, env, set_task, n_tasks, n_episodes, max_steps))
    for i, fn in enumerate(policy_fns):
        tm.add_row(i, evaluate_policy_on_tasks(fn, env, set_task, n_tasks, n_episodes, max_steps))
    return tm
