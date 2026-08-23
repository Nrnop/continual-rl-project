"""LayerNorm must not break the two things the PT mechanism is built on.

`layer_norm` inserts nn.LayerNorm on hidden layers of every network in the project through the one
shared `mlp()` constructor. That is a small change with two places it could silently destroy the
method while every log line still looks healthy:

  1. THE DECAY. `decay_transient` implements Alg. 2 line 9 by scaling the final Linear's weight and
     bias, which scales the OUTPUT exactly only because that layer is affine. Normalise after it
     and the decay stops being a decay. `absorbed_frac` would still report a number.
  2. THEOREM 1's ZERO INIT. mu_T(s) == 0 and V_trans(s) == 0 at t=0 are enforced by zeroing the
     output layer. Anything upstream must leave that at exactly zero.

Both are asserted here at the value that matters (the network's OUTPUT), not at the config key.
The off-by-default path is pinned too, because every result on disk depends on it.
"""
import torch

from ..models.actor import GaussianActor, SplitGaussianActor, mlp
from ..models.critic import SplitCritic, VanillaCritic

OBS, ACT, B = 17, 6, 32


def _obs():
    torch.manual_seed(0)
    return torch.randn(B, OBS)


def test_layer_norm_off_by_default_adds_no_modules():
    """The default path must be structurally identical to before the flag existed."""
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in mlp(OBS, [16, 16], ACT))


def test_layer_norm_on_adds_one_per_hidden_layer_and_none_after_the_output():
    net = mlp(OBS, [16, 16, 16], ACT, layer_norm=True)
    assert sum(isinstance(m, torch.nn.LayerNorm) for m in net) == 3
    # The LAST module must still be the output Linear -- decay_transient indexes it as [-1].
    assert isinstance(net[-1], torch.nn.Linear)
    assert net[-1].out_features == ACT


def test_decay_of_split_actor_is_still_exact():
    """mu_T -> factor * mu_T for every state, which is what makes rho one knob and not two."""
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
    critic = SplitCritic(OBS, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                         trans_zero_init=False, layer_norm=True)
    obs = _obs()
    before = critic.trans(obs).clone()
    critic.decay_transient(0.25, mode="output")
    torch.testing.assert_close(critic.trans(obs), 0.25 * before, rtol=1e-6, atol=1e-6)


def test_theorem1_zero_init_survives_layer_norm():
    """V_trans(s) == 0 and mu_T(s) == 0 exactly at init, for every state."""
    obs = _obs()
    critic = SplitCritic(OBS, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                         trans_zero_init=True, layer_norm=True)
    assert torch.count_nonzero(critic.trans(obs)) == 0
    actor = SplitGaussianActor(OBS, ACT, hidden_sizes=[11, 11], trans_hidden_sizes=[8, 8],
                               layer_norm=True)
    assert torch.count_nonzero(actor.trans_mean(obs)) == 0
    # ...so the composed policy is exactly the permanent, and KL(pi_PT || pi_P) == 0.
    torch.testing.assert_close(actor.act_deterministic(obs), actor.perm_forward(obs))
    assert float(actor.kl_to_prior(obs).abs().max()) == 0.0


def test_every_network_type_accepts_the_flag():
    """One key must reach all four, or an arm silently keeps the old architecture."""
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
    # And the outputs are finite, i.e. the nets actually run.
    assert torch.isfinite(GaussianActor(OBS, ACT, hidden_sizes=[11, 11],
                                        layer_norm=True).mean_net(obs)).all()


def test_layer_norm_actually_normalises_the_hidden_activations():
    """The point of the change: hidden scale is invariant to input scale.

    Without this the flag could be wired everywhere and still do nothing useful -- a silent no-op
    is this project's second-most-common failure mode.

    The invariance is the property that matters for plasticity. Loss of plasticity works through
    growing weight norms inflating pre-activations until units saturate; a layer whose output
    scale does not depend on its input scale cannot be driven that way.

    The scales tested go UPWARD, which is the direction the failure actually runs: weight norms
    grow during training, so pre-activations grow. Downward is not symmetric, because
    nn.LayerNorm's eps=1e-5 is added to the variance and pulls the output std below 1 once the
    pre-norm activations are themselves tiny. Measured here: at input scale 0.01 the pre-norm std
    is ~0.007 and the output std is 0.906, a real 9% effect and not a defect. It is also not a
    regime this project reaches — observations are normalised and clipped to +-10, so hidden
    activations are O(1).
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

    # ...and without the flag it is NOT invariant, or the assertion above proves nothing.
    plain = torch.nn.Sequential(*list(mlp(OBS, [64], 1))[:1])
    raw = [float(plain(scale * _obs()).std(unbiased=False)) for scale in (1.0, 1000.0)]
    assert raw[1] / raw[0] > 900, raw
