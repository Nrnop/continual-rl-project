"""LayerNorm (`layer_norm`, added in the shared `mlp()`) must keep two PT properties.

1. Decay: `decay_transient` (Alg. 2 line 9) scales the final Linear's weight and bias, which scales
   the output exactly only because that layer is affine and comes last.
2. Theorem 1 zero init: mu_T(s) == 0 and V_trans(s) == 0 at t=0 through a zeroed output layer.
Both are checked on network outputs. The flag is off by default.
"""
import torch

from ..models.actor import GaussianActor, SplitGaussianActor, mlp
from ..models.critic import SplitCritic, VanillaCritic

OBS, ACT, B = 17, 6, 32


def _obs():
    torch.manual_seed(0)
    return torch.randn(B, OBS)


def test_layer_norm_off_by_default_adds_no_modules():
    """With the flag off, mlp() contains no LayerNorm modules."""
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in mlp(OBS, [16, 16], ACT))


def test_layer_norm_on_adds_one_per_hidden_layer_and_none_after_the_output():
    """One LayerNorm per hidden layer, and the last module is still the output Linear."""
    net = mlp(OBS, [16, 16, 16], ACT, layer_norm=True)
    assert sum(isinstance(m, torch.nn.LayerNorm) for m in net) == 3
    # decay_transient indexes the output layer as [-1].
    assert isinstance(net[-1], torch.nn.Linear)
    assert net[-1].out_features == ACT


def test_decay_of_split_actor_is_still_exact():
    """decay_transient(f) gives mu_T -> f * mu_T for every state with LayerNorm on."""
    actor = SplitGaussianActor(OBS, ACT, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                               layer_norm=True)
    # Move the transient off zero first, or the test passes trivially.
    with torch.no_grad():
        actor.trans_mean[-1].weight.normal_(0, 0.5)
        actor.trans_mean[-1].bias.normal_(0, 0.5)
    obs = _obs()
    before = actor.trans_mean(obs).clone()
    actor.decay_transient(0.5)
    torch.testing.assert_close(actor.trans_mean(obs), 0.5 * before, rtol=1e-6, atol=1e-6)


def test_decay_of_split_critic_output_mode_is_still_exact():
    """Output-mode decay of SplitCritic scales V_trans exactly with LayerNorm on."""
    critic = SplitCritic(OBS, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                         trans_zero_init=False, layer_norm=True)
    obs = _obs()
    before = critic.trans(obs).clone()
    critic.decay_transient(0.25, mode="output")
    torch.testing.assert_close(critic.trans(obs), 0.25 * before, rtol=1e-6, atol=1e-6)


def test_theorem1_zero_init_survives_layer_norm():
    """V_trans(s) and mu_T(s) are exactly zero at init with LayerNorm on."""
    obs = _obs()
    critic = SplitCritic(OBS, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                         trans_zero_init=True, layer_norm=True)
    assert torch.count_nonzero(critic.trans(obs)) == 0
    actor = SplitGaussianActor(OBS, ACT, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                               layer_norm=True)
    assert torch.count_nonzero(actor.trans_mean(obs)) == 0
    # So the composed policy equals the permanent and KL(pi_PT || pi_P) == 0.
    torch.testing.assert_close(actor.act_deterministic(obs), actor.perm_forward(obs))
    assert float(actor.kl_to_prior(obs).abs().max()) == 0.0


def test_every_network_type_accepts_the_flag():
    """All four network types build LayerNorm modules when the flag is set."""
    obs = _obs()
    for net in (
        GaussianActor(OBS, ACT, hidden_sizes=[11, 11], layer_norm=True).mean_net,
        VanillaCritic(OBS, hidden_sizes=[11, 11], layer_norm=True).net,
        SplitGaussianActor(OBS, ACT, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                           layer_norm=True).perm_mean,
        SplitCritic(OBS, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                    layer_norm=True).perm,
    ):
        assert any(isinstance(m, torch.nn.LayerNorm) for m in net)
    # The nets also run and give finite outputs.
    assert torch.isfinite(GaussianActor(OBS, ACT, hidden_sizes=[11, 11],
                                        layer_norm=True).mean_net(obs)).all()


def test_layer_norm_actually_normalises_the_hidden_activations():
    """Hidden activations after LayerNorm have mean 0 and std ~1 whatever the input scale.

    Only upward scales are tested, the direction growing weight norms push. At small scales the
    eps=1e-5 pulls the std below 1 (0.906 at input scale 0.01), but observations are normalised
    and clipped to +-10, so that regime is not reached.
    """
    net = mlp(OBS, [64], 1, layer_norm=True)
    hidden = torch.nn.Sequential(*list(net)[:2])      # Linear -> LayerNorm
    stds = []
    for scale in (1.0, 10.0, 100.0, 1000.0):
        h = hidden(scale * _obs())
        assert abs(float(h.mean())) < 1e-5            # centring is exact regardless of scale
        stds.append(float(h.std(unbiased=False)))
    # A 1000x swing in input scale must leave the hidden scale essentially untouched at ~1.
    assert max(stds) - min(stds) < 1e-3, stds
    assert all(abs(s - 1.0) < 1e-3 for s in stds), stds

    # Without the flag the scale is not invariant, so the check above is meaningful.
    plain = torch.nn.Sequential(*list(mlp(OBS, [64], 1))[:1])
    raw = [float(plain(scale * _obs()).std(unbiased=False)) for scale in (1.0, 1000.0)]
    assert raw[1] / raw[0] > 900, raw
