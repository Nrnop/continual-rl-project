"""Control runs that separate the output-layer decay from the split.

Each test checks realised behaviour (weights, widths, decay counts), not only the config key.
"""
import os

import numpy as np
import torch
import yaml

from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_vanilla import PPOVanilla

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _vanilla(**extra):
    cfg = {"lr_actor": 1e-3, "n_steps": 4, "num_envs": 1, "gamma": 0.99, "gae_lambda": 0.95,
           "clip_coef": 0.2, "epochs": 1, "minibatch_size": 4, "ent_coef": 0.0,
           "max_grad_norm": 0.5, "normalize_advantage": False, "hidden_sizes": [8, 8]}
    cfg.update(extra)
    torch.manual_seed(0)
    return PPOVanilla(3, 2, cfg, "cpu")


def _out(agent):
    return (agent.actor.mean_net[-1].weight.clone(), agent.critic.net[-1].weight.clone())


def test_off_by_default_never_touches_the_outputs():
    """Without output_decay_every, vanilla never rescales its output layers."""
    agent = _vanilla()
    before = _out(agent)
    for i in range(1, 25):
        agent.post_update(i)
    agent.on_task_switch(0)
    after = _out(agent)
    assert agent.output_decays == 0
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_halves_both_output_layers_every_k_updates_and_at_a_boundary():
    """Vanilla halves both output layers every k updates and again at a task boundary."""
    agent = _vanilla(output_decay_every=10, output_decay_factor=0.5)
    w_actor, w_critic = _out(agent)
    for i in range(1, 10):
        agent.post_update(i)
    assert agent.output_decays == 0                        # not before the 10th update
    agent.post_update(10)
    assert agent.output_decays == 1
    a1, c1 = _out(agent)
    assert torch.allclose(a1, 0.5 * w_actor) and torch.allclose(c1, 0.5 * w_critic)
    agent.on_task_switch(0)                                 # PT also consolidates at a boundary
    assert agent.output_decays == 2 and agent._updates_since_decay == 0
    a2, _ = _out(agent)
    assert torch.allclose(a2, 0.25 * w_actor)


def test_hidden_layers_are_left_alone():
    """Output decay leaves the hidden layers unchanged."""
    agent = _vanilla(output_decay_every=1, output_decay_factor=0.5)
    hidden = agent.actor.mean_net[0].weight.clone()
    agent.post_update(1)
    assert torch.equal(agent.actor.mean_net[0].weight, hidden)


def _load(name, agent="pt"):
    """Merge default.yaml, ppo_<agent>.yaml and the named overlay."""
    cfg = {}
    for f in ("default.yaml", f"ppo_{agent}.yaml", name):
        cfg.update(yaml.safe_load(open(os.path.join(CONFIGS, f))) or {})
    return cfg


def _transient_outputs_after_one_consolidation(cfg):
    """Return the transient actor's output weights before and after one consolidation."""
    cfg = dict(cfg, n_steps=4, num_envs=1, consolidation_buffer_size=16, k=1, epochs=1,
               minibatch_size=4, consolidation_epochs=1)
    torch.manual_seed(0)
    agent = PPOPT(5, 1, cfg, "cpu")
    with torch.no_grad():                                   # give the transient a non-zero output
        agent.actor.trans_mean[-1].weight.fill_(0.3)
        agent.critic.trans[-1].weight.fill_(0.3)
    before = agent.actor.trans_mean[-1].weight.clone()
    rng = np.random.default_rng(0)
    for _ in range(4):
        obs = rng.normal(size=(1, 5)).astype(np.float32)
        vp, vt = agent.get_value(obs)
        agent.buffer.add(obs, np.zeros((1, 1), np.float32), np.zeros(1, np.float32),
                         np.zeros(1, np.float32), np.zeros(1, np.float32), vp, vt)
    agent.post_update(1)                                    # k = 1: consolidates now
    return before, agent.actor.trans_mean[-1].weight.clone()


def test_nodecay_config_really_keeps_the_transient_while_frozen_halves_it():
    """cartpole_pt_frozen halves the transient output; the nodecay control keeps it."""
    b0, a0 = _transient_outputs_after_one_consolidation(_load("cartpole_pt_frozen.yaml"))
    b1, a1 = _transient_outputs_after_one_consolidation(_load("cartpole_pt_frozen_nodecay.yaml"))
    assert torch.allclose(a0, 0.5 * b0)                     # the shipped frozen arm halves
    assert torch.equal(a1, b1)                              # the control does not


def test_small_vanilla_config_is_thirty_wide_in_the_built_networks():
    """cartpole_van_small builds 30-wide actor and critic networks."""
    agent = PPOVanilla(5, 1, _load("cartpole_van_small.yaml", agent="vanilla"), "cpu")
    widths = lambda net: [m.out_features for m in net if isinstance(m, torch.nn.Linear)]
    assert widths(agent.actor.mean_net) == [30, 30, 1]
    assert widths(agent.critic.net) == [30, 30, 1]


def test_outdecay_config_switches_the_control_on():
    """cartpole_van_outdecay turns output decay on (every 10 updates, factor 0.5)."""
    agent = PPOVanilla(5, 1, _load("cartpole_van_outdecay.yaml", agent="vanilla"), "cpu")
    assert agent.output_decay_every == 10 and agent.output_decay_factor == 0.5
