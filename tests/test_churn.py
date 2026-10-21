"""The churned predictor, now that it lives in ``compass`` rather than in three
benchmark-local copies.

The equivalence test is the important one: every published churn measurement in
this repository (the partial-pooling K arms, the inverse-kinematics rules, the
bimodal arms, the exponential/Gaussian/Laplace twins) was produced by the
monkeypatched ``churned_dpm_sampler`` in
``Partial_Pooling/run_churn_comparison.py``. If the library version is not
bit-identical to it, those numbers stop being reproducible through the supported
path and every table in ``docs/`` silently refers to a sampler that no longer
exists.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Partial_Pooling"))

import compass
import compass.churn
from compass import ScoreBasedInferenceModel as SBIm
from compass.MultiObsSampler import MultiObsSampler


def _tiny_model():
    torch.manual_seed(0)
    return SBIm(nodes_size=4, hidden_size=16, depth=2, num_heads=2, sigma=5.0)


def _draw(model, **kwargs):
    torch.manual_seed(1234)
    return model.sample(
        x=torch.zeros(4, 2), condition_mask=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        multi_obs_inference=True, hierarchy=[0], timesteps=15, num_samples=32,
        correction="gauss_hierarchical", precision_est_samples=32,
        final_corrector_steps=0, verbose=False, **kwargs
    )


def test_churn_sigma_matches_karras_formula():
    sigma_hat, spread = compass.churn_sigma(1.0, 0.9, eta=2.0)
    gamma = min(2.0 * 0.1 / 1.0, compass.CHURN_GAMMA_CAP)
    assert sigma_hat == pytest.approx(1.0 * (1.0 + gamma))
    assert spread == pytest.approx(math.sqrt(sigma_hat**2 - 1.0))


def test_churn_sigma_saturates_at_the_cap():
    # The samplers use a geometric sigma grid, so the relative step is constant
    # and the cap is all-or-nothing: past eta_crit the result stops depending on
    # eta at all. This is why runs labelled eta=4 and eta=8 were bit-identical.
    a, _ = compass.churn_sigma(1.0, 0.9, eta=8.0)
    b, _ = compass.churn_sigma(1.0, 0.9, eta=64.0)
    assert a == b == pytest.approx(1.0 + compass.CHURN_GAMMA_CAP)
    # ... and raising the cap is what makes a genuine high-eta sweep possible.
    c, _ = compass.churn_sigma(1.0, 0.9, eta=8.0, gamma_cap=2.0)
    assert c > a


def test_eta_zero_is_the_stock_probability_flow_predictor():
    sigma_hat, spread = compass.churn_sigma(1.0, 0.9, eta=0.0)
    assert (sigma_hat, spread) == (1.0, 0.0)


def test_corrector_steps_sentinel_resolves_per_method():
    # "dpm" defaults to corrector-free, because churn replaces the correctors.
    assert compass.resolve_corrector_steps(None, "dpm", 2.0) == 0
    # "langevin" is corrector-only by definition and must not become a no-op.
    assert compass.resolve_corrector_steps(None, "langevin", 2.0) == 5
    # An explicit value is always honoured.
    assert compass.resolve_corrector_steps(3, "langevin", 2.0) == 3
    assert compass.resolve_corrector_steps(3, "dpm", 0.0) == 3


def test_churn_and_correctors_together_is_refused():
    with pytest.raises(ValueError, match="alternatives, not"):
        compass.resolve_corrector_steps(5, "dpm", 2.0)


def test_library_churn_is_bit_identical_to_the_published_monkeypatch():
    """Every published churn number in this repo came from that patch."""
    from run_churn_comparison import churned_dpm_sampler

    model = _tiny_model()
    library = _draw(model, churn_eta=2.0)

    original = MultiObsSampler._dpm_sampler
    MultiObsSampler._dpm_sampler = churned_dpm_sampler(2.0)
    try:
        reference = _draw(model, churn_eta=0.0)   # the patch does the churning
    finally:
        MultiObsSampler._dpm_sampler = original

    assert torch.equal(library, reference)


def test_churn_changes_the_draw():
    model = _tiny_model()
    assert not torch.equal(_draw(model, churn_eta=2.0), _draw(model, churn_eta=0.0))


def test_churn_leaves_observed_coordinates_untouched():
    """Observed values are conditioning data; churn must not perturb them.

    Asserted on ``apply_churn`` directly, because ``SBIm.sample`` returns only
    the latent coordinates and so cannot show the invariant.
    """
    from compass.SDE import VESDE

    sde = VESDE(sigma=5.0)
    data = torch.randn(3, 4, 4)
    condition_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
    latent = 1 - condition_mask
    t_now = torch.full((1, 1), 0.5)
    t_next = torch.full((1, 1), 0.4)

    moved, t_start = compass.apply_churn(
        data, latent, sde, t_now, t_next, eta=2.0,
        ceiling=compass.churn.schedule_ceiling(sde, data.device),
    )
    assert torch.equal(moved[..., 2:], data[..., 2:])      # observed: untouched
    assert not torch.equal(moved[..., :2], data[..., :2])  # latent: churned
    assert float(t_start) > float(t_now)                   # stepped up in noise


def test_single_observation_sampler_also_churns():
    """The n=1 route had no churn path at all before this."""
    model = _tiny_model()

    def draw(eta):
        torch.manual_seed(99)
        return model.sample(
            x=torch.zeros(1, 2), condition_mask=torch.tensor([0.0, 0.0, 1.0, 1.0]),
            timesteps=15, num_samples=16, final_corrector_steps=0,
            verbose=False, churn_eta=eta,
        )

    assert not torch.equal(draw(2.0), draw(0.0))
