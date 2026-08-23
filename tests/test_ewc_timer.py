"""EWC's timer-based consolidation — required for the boundary-free (Phase 2b) benchmark.

WHY THIS EXISTS. EWC accumulates its Fisher in `on_task_switch`. Under smooth Lipschitz drift there
are no task switches, so that hook never fires, the Fisher stays empty, the penalty is identically
zero, and the agent is BIT-IDENTICAL to vanilla PPO. Comparing `pt` against EWC in that state would
not be a weak comparison, it would be a meaningless one — EWC would simply be switched off.

`ewc_consolidate_every = n` accumulates and re-anchors every n PPO updates instead, so EWC and `pt`
can be run on the same cadence and the comparison is about the mechanism rather than the trigger.

Both directions are pinned here: OFF must leave every pre-existing (boundary-based) run unchanged,
ON must actually fire without any boundary.
"""
import numpy as np
import torch

from src_continuous_control.agents.ppo_ewc import PPOEWC


def _agent(cfg_extra, mock_cfg):
    cfg = dict(mock_cfg)
    cfg["agent"] = "ewc"
    cfg.update(cfg_extra)
    torch.manual_seed(0)
    return PPOEWC(17, 6, cfg, torch.device("cpu"))


def _fill_buffer(agent):
    """Put a rollout in the buffer so the Fisher estimate has something to consume."""
    n = agent.buffer.obs.shape[0] if agent.buffer.obs.ndim == 2 else agent.buffer.obs.shape[0]
    rng = np.random.RandomState(0)
    agent.buffer.obs[:] = torch.as_tensor(
        rng.randn(*agent.buffer.obs.shape).astype(np.float32))
    agent.buffer.actions[:] = torch.as_tensor(
        rng.randn(*agent.buffer.actions.shape).astype(np.float32))
    return n


def test_timer_off_by_default_leaves_fisher_empty(mock_cfg):
    """The default must reproduce every run made before the timer existed."""
    agent = _agent({}, mock_cfg)
    assert agent.ewc_consolidate_every == 0
    _fill_buffer(agent)
    for i in range(25):
        agent.post_update(i)
    assert agent.fisher == {}, "the timer fired with no boundary and no ewc_consolidate_every set"
    assert agent.anchor == {}


def test_timer_fires_without_any_boundary(mock_cfg):
    """With the timer on, EWC must consolidate on cadence and with no task switch at all."""
    every = 5
    agent = _agent({"ewc_consolidate_every": every}, mock_cfg)
    _fill_buffer(agent)

    for i in range(every - 1):
        agent.post_update(i)
    assert agent.fisher == {}, "consolidated before the timer elapsed"

    agent.post_update(every - 1)
    assert agent.fisher, "the timer elapsed but no Fisher was accumulated"
    assert agent.anchor, "the timer elapsed but the anchor was not set"


def test_timer_produces_a_nonzero_penalty(mock_cfg):
    """A Fisher that exists but yields a zero penalty would be an inert control."""
    agent = _agent({"ewc_consolidate_every": 2, "ewc_lambda": 100.0}, mock_cfg)
    _fill_buffer(agent)
    agent.post_update(0)
    agent.post_update(1)                      # fires here
    assert agent.fisher

    # Move the policy away from the anchor; the penalty must respond.
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(0.1)
    penalty = float(agent._ewc_penalty())
    assert np.isfinite(penalty)
    assert penalty > 0.0, "penalty is zero after consolidation — the control is inert"


def test_repeated_consolidation_is_gamma_decayed_not_summed(mock_cfg):
    """Online EWC keeps ONE running Fisher. A growing list makes the policy monotonically rigid,
    which is the bug already fixed once here — the timer must not reintroduce it."""
    agent = _agent({"ewc_consolidate_every": 1}, mock_cfg)
    _fill_buffer(agent)
    agent.post_update(0)
    first = {k: v.clone() for k, v in agent.fisher.items()}
    agent.post_update(1)
    # F_new = gamma * F_old + F_current, so with an identical buffer the trace grows by a bounded
    # factor rather than doubling every time.
    for k in first:
        assert torch.all(agent.fisher[k] <= (1.0 + agent.ewc_gamma) * first[k] + 1e-6)
    assert len(agent.past_tasks) >= 1
