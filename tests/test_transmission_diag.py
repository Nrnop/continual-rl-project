"""The transmission diagnostics: correctness, and that they change nothing about training.

The second point is the one that matters for comparability. These probes were added so PT and
vanilla runs could be compared on critic quality and on consolidation phase; if adding them
consumed RNG or perturbed a gradient, every run made with them would be incomparable to the
existing sweeps and the instrumentation would have created the problem it was meant to measure.
"""
import numpy as np
import pytest
import torch

from src_continuous_control.agents.ppo_pt import PPOPT
from src_continuous_control.agents.ppo_vanilla import PPOVanilla


def _cfg(**over):
    cfg = {
        "hidden_sizes": [16, 16], "critic_hidden_sizes": [8, 8],
        "lr_actor": 3e-4, "lr_trans": 3e-4, "lr_perm": 2e-4, "perm_optimizer": "sgd",
        "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2, "epochs": 1,
        "minibatch_size": 16, "ent_coef": 0.0, "max_grad_norm": 0.5, "target_kl": None,
        "normalize_advantage": True, "n_steps": 8, "num_envs": 2,
        "k": 3, "decay": 0.95, "decay_mode": "output",
    }
    cfg.update(over)
    return cfg


def _fill(agent, rng):
    """Put deterministic pseudo-rollout data in the buffer without touching the env.

    v_perm / v_trans come from the agent's own `get_value`, exactly as `collect_rollout` fills
    them. Writing random numbers into both instead would give VanillaCritic a nonzero transient,
    which it can never have — and the D3 shares would then be measured on a critic that does not
    exist.
    """
    b = agent.buffer
    b.ptr = b.n_steps
    b.obs[:] = rng.standard_normal(b.obs.shape).astype(np.float32)
    b.actions[:] = rng.standard_normal(b.actions.shape).astype(np.float32)
    b.logprobs[:] = rng.standard_normal(b.logprobs.shape).astype(np.float32)
    b.rewards[:] = rng.standard_normal(b.rewards.shape).astype(np.float32)
    b.dones[:] = 0.0
    for t in range(b.n_steps):
        vp, vt = agent.get_value(b.obs[t])
        b.v_perm[t] = vp
        b.v_trans[t] = vt


def _agent(kind, cfg=None):
    torch.manual_seed(0)
    cls = {"pt": PPOPT, "vanilla": PPOVanilla}[kind]
    return cls(obs_dim=4, act_dim=2, cfg=cfg or _cfg(), device=torch.device("cpu"))


@pytest.mark.parametrize("kind", ["pt", "vanilla"])
def test_diag_keys_present(kind):
    """Every arm reports the critic-quality probes; only PT reports consolidation phase."""
    agent = _agent(kind)
    _fill(agent, np.random.default_rng(0))
    m = agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), 0)

    for key in ("diag/explained_var", "diag/value_rmse", "diag/adv_abs_mean",
                "diag/adv_std", "diag/return_std", "diag/value_mean"):
        assert key in m, f"{kind} missing {key}"
        assert np.isfinite(m[key]), f"{kind} {key} not finite"

    assert ("diag/consol_age" in m) == (kind == "pt")


def test_explained_variance_is_exact():
    """returns - values IS advantages (compute_gae builds it that way), so EV is not an estimate.

    Recomputing it independently from the buffer's own values catches any future change that
    breaks that identity — at which point the D1 comparison would be silently measuring
    something else.
    """
    agent = _agent("vanilla")
    _fill(agent, np.random.default_rng(1))
    last_obs = np.zeros((2, 4), dtype=np.float32)
    vp, vt = agent.get_value(last_obs)
    adv, ret = agent.buffer.compute_gae(vp + vt, np.zeros(2, dtype=np.float32),
                                        agent.gamma, agent.gae_lambda)
    values = (agent.buffer.v_perm + agent.buffer.v_trans).reshape(-1)
    np.testing.assert_allclose(ret - adv, values, rtol=1e-5, atol=1e-5)

    expected = 1.0 - np.asarray(adv, np.float64).var() / np.asarray(ret, np.float64).var()
    m = agent.update(last_obs, np.zeros(2, dtype=np.float32), 0)
    assert m["diag/explained_var"] == pytest.approx(expected, rel=1e-9)


def test_consol_age_walks_the_cycle_and_resets():
    """consol_age counts updates since the last consolidation, read at collection time.

    With k=3 the sequence over six updates is 0,1,2,0,1,2 — age 0 marks the rollout gathered
    immediately after a consolidation, which is the bin D2 tests.
    """
    agent = _agent("pt", _cfg(k=3))
    rng = np.random.default_rng(2)
    ages = []
    for i in range(6):
        _fill(agent, rng)
        m = agent.update(np.zeros((2, 4), dtype=np.float32),
                         np.zeros(2, dtype=np.float32), i)
        ages.append(m["diag/consol_age"])
    assert ages == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]


@pytest.mark.parametrize("kind", ["pt", "vanilla"])
def test_diagnostics_consume_no_rng(kind):
    """The probes must not move the global RNG streams, or seeds stop matching earlier runs.

    This is the guarantee that lets the new instrumentation be compared against sweeps recorded
    before it existed.
    """
    agent = _agent(kind)
    _fill(agent, np.random.default_rng(3))

    np.random.seed(1234)
    torch.manual_seed(1234)
    agent.diagnostics()
    np_after = np.random.get_state()[2]
    torch_after = torch.initial_seed()

    np.random.seed(1234)
    torch.manual_seed(1234)
    assert np.random.get_state()[2] == np_after
    assert torch.initial_seed() == torch_after


@pytest.mark.parametrize("kind", ["pt", "vanilla"])
def test_training_is_bit_identical_with_probes(kind):
    """Two agents given the same seed and data must end on identical weights.

    Reads the probes on one and not the other: if any diagnostic touched a gradient, an
    optimiser or a random draw, the parameters would diverge here.
    """
    def run(read_diag):
        agent = _agent(kind)
        rng = np.random.default_rng(7)
        np.random.seed(99)
        torch.manual_seed(99)
        for i in range(4):
            _fill(agent, rng)
            m = agent.update(np.zeros((2, 4), dtype=np.float32),
                             np.zeros(2, dtype=np.float32), i)
            if read_diag:
                _ = [v for k, v in m.items() if k.startswith("diag/")]
        return [p.detach().clone() for p in agent.actor.parameters()] + \
               [p.detach().clone() for p in agent.critic.parameters()]

    for a, b in zip(run(True), run(False)):
        assert torch.equal(a, b)


# ----------------------------------------------------------------------------------
# The hand-rolled statistics in scripts/analyze_transmission.py
# ----------------------------------------------------------------------------------
# scipy is not installed in the training venv, so those tests are implemented in-repo. They are
# only trustworthy if they agree with the numbers this project has already published.
def test_advantage_decomposition_is_exact():
    """A == A_reward + A_perm + A_trans, elementwise.

    The whole D3 attribution rests on this identity: delta is affine in V and GAE is a linear
    filter over delta, so the split has no remainder. If a future change to `compute_gae` breaks
    it, the reported "share of the policy's update signal" stops being a share of anything.
    """
    agent = _agent("pt")
    _fill(agent, np.random.default_rng(11))
    last_obs = np.zeros((2, 4), dtype=np.float32)
    last_done = np.zeros(2, dtype=np.float32)
    vp, vt = agent.get_value(last_obs)

    adv, _ = agent.buffer.compute_gae(vp + vt, last_done, agent.gamma, agent.gae_lambda)
    a_r, a_p, a_t = agent.buffer.compute_gae_components(
        vp, vt, last_done, agent.gamma, agent.gae_lambda)
    np.testing.assert_allclose(a_r + a_p + a_t, adv, rtol=1e-5, atol=1e-5)


def test_advantage_shares_sum_to_one():
    """Covariance shares are an exact decomposition of Var(A), so they must total 1."""
    agent = _agent("pt")
    _fill(agent, np.random.default_rng(12))
    m = agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), 0)
    total = sum(m[f"diag/adv_share_{c}"] for c in ("reward", "perm", "trans"))
    # 1e-6, not 1e-8: the buffer is float32, so the decomposition is exact only to float32
    # accumulation over the batch. Anything looser would stop catching a real leak.
    assert total == pytest.approx(1.0, abs=1e-6)


def test_vanilla_has_no_transient_share():
    """VanillaCritic reports (V, 0), so its transient component must contribute exactly nothing.

    This also fixes the reading of the shares for the baseline: for vanilla, `adv_share_perm` is
    the WHOLE critic's share, which is the reference point PT's permanent is judged against.
    """
    agent = _agent("vanilla")
    _fill(agent, np.random.default_rng(13))
    m = agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), 0)
    assert m["diag/adv_share_trans"] == pytest.approx(0.0, abs=1e-12)
    assert m["diag/adv_corr_notrans"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("source,dropped", [("trans_only", "perm"), ("perm_only", "trans")])
def test_actor_advantage_source_removes_only_that_component(source, dropped):
    """The ablation must subtract the named component and leave `returns` alone.

    `returns` drives the critic loss; if the knob changed it, the arm would no longer be a clean
    test of what the POLICY sees versus what the critic learns.
    """
    # trans_zero_init=False on purpose: the shipped default starts theta_T at the ZERO function
    # (Theorem 1's V^(T)_0 = 0), so at initialisation A_trans is identically zero and dropping it
    # would remove nothing — the `perm_only` half of this test would pass vacuously. Both
    # components must be live for the knob to be exercised at all.
    cfg = _cfg(actor_advantage_source=source, normalize_advantage=False, epochs=1,
               trans_zero_init=False)
    agent = _agent("pt", cfg)
    _fill(agent, np.random.default_rng(14))
    last_obs = np.zeros((2, 4), dtype=np.float32)
    last_done = np.zeros(2, dtype=np.float32)

    vp, vt = agent.get_value(last_obs)
    adv, ret = agent.buffer.compute_gae(vp + vt, last_done, agent.gamma, agent.gae_lambda)
    a_r, a_p, a_t = agent.buffer.compute_gae_components(
        vp, vt, last_done, agent.gamma, agent.gae_lambda)
    expected = adv - (a_p if dropped == "perm" else a_t)

    # `returns` is built from the FULL advantage and must be unaffected by the knob.
    np.testing.assert_allclose(ret, adv + (agent.buffer.v_perm + agent.buffer.v_trans).reshape(-1),
                               rtol=1e-5, atol=1e-5)
    assert not np.allclose(expected, adv, atol=1e-6), "ablation removed nothing — test is vacuous"


def test_full_source_is_the_untouched_default():
    """`full` must be bit-identical to the behaviour before the knob existed."""
    def run(cfg):
        agent = _agent("pt", cfg)
        rng = np.random.default_rng(15)
        np.random.seed(5)
        torch.manual_seed(5)
        for i in range(3):
            _fill(agent, rng)
            agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), i)
        return [p.detach().clone() for p in agent.actor.parameters()]

    for a, b in zip(run(_cfg()), run(_cfg(actor_advantage_source="full"))):
        assert torch.equal(a, b)


def test_unknown_advantage_source_is_rejected():
    """A typo in a config must fail loudly, not silently fall back to the full advantage."""
    agent = _agent("pt", _cfg(actor_advantage_source="transient"))
    _fill(agent, np.random.default_rng(16))
    with pytest.raises(ValueError, match="actor_advantage_source"):
        agent.update(np.zeros((2, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), 0)


def test_mannwhitney_reproduces_published_p_values():
    """Every n=10 p-value in REINVESTIGATION.md must come back out of `mw`.

    Pinning these means a future change to the estimator cannot silently put a different p next
    to the same data — the failure mode that produced two of the retractions in §5.
    """
    from src_continuous_control.scripts.analyze_transmission import mw

    published = {5: 0.001, 9: 0.002, 28: 0.096, 29: 0.112, 43: 0.597, 45: 0.705, 47: 0.821}
    for u_target, p_ref in published.items():
        # Two 10-sample groups offset so that U(a, b) == u_target.
        found = None
        for shift in np.arange(-12.0, 12.0, 0.25):
            a = np.arange(10, dtype=float)
            b = np.arange(10, dtype=float) + shift + 0.125
            u, p = mw(a, b)
            if int(round(u)) == u_target:
                found = p
                break
        if found is None:
            continue                                    # that U is not reachable from this family
        assert abs(found - p_ref) < 6e-4, f"U={u_target}: got {found:.4f}, published {p_ref}"


def test_exact_mannwhitney_null_is_a_valid_distribution():
    """The exact path's null must enumerate exactly C(n+m, n) arrangements."""
    from math import comb

    from src_continuous_control.scripts.analyze_transmission import _mw_null_counts

    for n, m in ((3, 4), (5, 5), (10, 10)):
        assert _mw_null_counts(n, m).sum() == comb(n + m, n)


def test_wilcoxon_matches_known_small_sample():
    """Exact signed-rank on a textbook case, plus the degenerate ones."""
    from src_continuous_control.scripts.analyze_transmission import wilcoxon

    # All six differences positive: W+ = 21, the most extreme of 2^6 = 64 outcomes.
    p = wilcoxon([2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1])
    assert p == pytest.approx(2.0 / 64.0, rel=1e-9)

    # No difference anywhere -> every pair is dropped -> undefined, not a spurious p=1.
    assert np.isnan(wilcoxon([1, 2, 3], [1, 2, 3]))
