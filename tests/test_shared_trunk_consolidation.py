"""The shared-trunk PT critic must consolidate WITHOUT changing the acting value.

This is the property the two-MLP SplitCritic cannot satisfy: there, consolidation has to make
V_perm *learn* old_V_perm + V_trans by regression (representing the sum of two MLPs with one MLP),
which is lossy by construction. With a shared trunk and linear heads the sum is linear in the same
features, so consolidation is exact weight arithmetic and V = V_perm + V_trans is preserved exactly.
"""
import numpy as np
import torch

from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.models.critic import SharedTrunkSplitCritic


def _randomize_transient(critic, scale=0.8):
    """Give the transient head real signal, as it would have after training."""
    with torch.no_grad():
        critic.trans.weight.add_(torch.randn_like(critic.trans.weight) * scale)
        critic.trans.bias.add_(torch.randn_like(critic.trans.bias) * scale)


def test_shared_trunk_consolidation_preserves_value_for_any_decay():
    torch.manual_seed(0)
    probe = torch.randn(256, 17)
    for decay in (0.0, 0.25, 0.5, 0.75, 1.0):
        critic = SharedTrunkSplitCritic(17, hidden_sizes=(64, 64))
        _randomize_transient(critic)
        with torch.no_grad():
            p0, t0 = critic(probe)
        critic.consolidate(decay)
        with torch.no_grad():
            p1, t1 = critic(probe)
        before, after = p0 + t0, p1 + t1
        assert torch.allclose(before, after, atol=1e-5), (
            f"decay={decay}: acting value changed by "
            f"{(after - before).abs().max().item():.3e} (must be ~0)"
        )


def test_shared_trunk_consolidation_actually_transfers_to_permanent():
    """decay=0 must move the whole transient into the permanent head (not just zero it out)."""
    torch.manual_seed(0)
    probe = torch.randn(256, 17)
    critic = SharedTrunkSplitCritic(17, hidden_sizes=(64, 64))
    _randomize_transient(critic)
    with torch.no_grad():
        p0, t0 = critic(probe)
    critic.consolidate(0.0)
    with torch.no_grad():
        p1, t1 = critic(probe)
    # transient fully cleared, and the permanent absorbed exactly what it lost
    assert t1.abs().max().item() < 1e-6
    assert torch.allclose(p1 - p0, t0, atol=1e-5)


def test_pt_agent_shared_trunk_wiring(mock_cfg):
    """The agent builds the shared-trunk critic, skips the regression path, and preserves value."""
    cfg = dict(mock_cfg)
    cfg.update({"critic_arch": "shared_trunk", "hidden_sizes": [32, 32],
                "decay": 0.5, "k": 2, "n_steps": 8, "num_envs": 2})
    agent = PPOPT(17, 6, cfg, torch.device("cpu"))
    assert agent.shared_trunk is True
    assert agent.perm_optim is None, "shared-trunk variant must not create a permanent optimizer"
    _randomize_transient(agent.critic)

    probe = torch.randn(128, 17)
    with torch.no_grad():
        p0, t0 = agent.critic(probe)

    # post_update must NOT bank states (no consolidation buffer needed in this variant)
    agent.buffer.obs = np.random.randn(cfg["n_steps"], cfg["num_envs"], 17).astype(np.float32)
    agent.post_update(0)
    assert len(agent.consolidation_buffer) == 0
    agent.post_update(1)          # k=2 -> triggers _consolidate

    with torch.no_grad():
        p1, t1 = agent.critic(probe)
    assert torch.allclose(p0 + t0, p1 + t1, atol=1e-5), "consolidation changed the acting value"


def test_separate_arch_is_still_the_default(mock_cfg):
    """Back-compat: without critic_arch, PT keeps the original two-trunk behaviour."""
    agent = PPOPT(17, 6, dict(mock_cfg), torch.device("cpu"))
    assert agent.shared_trunk is False
    assert agent.perm_optim is not None
