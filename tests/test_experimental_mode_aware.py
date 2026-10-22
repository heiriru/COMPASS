"""Mode-aware composition (compass.experimental.mode_aware)."""
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.experimental.mode_aware import fit_gmm, mode_aware_sample
from compass.experimental.oracles import LinearGaussianHierarchy, oracle_model


def test_bic_picks_two_well_separated_modes_with_their_weights():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(700, 2, generator=g) * 0.3 + torch.tensor([2.0, 0.0])
    b = torch.randn(300, 2, generator=g) * 0.3 - torch.tensor([2.0, 0.0])
    labels, means, covs, weights = fit_gmm(torch.cat([a, b]), max_components=3)
    assert means.shape[0] == 2
    order = means[:, 0].argsort(descending=True)
    torch.testing.assert_close(weights[order], torch.tensor([0.7, 0.3], dtype=torch.float64), atol=0.02, rtol=0)


def test_bic_keeps_one_mode_for_a_gaussian():
    g = torch.Generator().manual_seed(1)
    _, means, _, _ = fit_gmm(torch.randn(1000, 2, generator=g), max_components=3)
    assert means.shape[0] == 1


def test_unimodal_posterior_falls_through_to_the_plain_rule():
    generator = torch.Generator().manual_seed(0)
    h = LinearGaussianHierarchy(torch.randn(1, 2, generator=generator, dtype=torch.float64) * 0.7,
                                torch.randn(2, 1, generator=generator, dtype=torch.float64),
                                torch.randn(2, 2, generator=generator, dtype=torch.float64) * 0.5,
                                tau=0.6, nu=0.5)
    model = oracle_model(h, sigma=10.0)
    data = torch.randn(4, 2)
    mask = torch.tensor([0.0, 0, 0, 1, 1])
    out, info = mode_aware_sample(MultiObsSampler(model), data, mask, [0, 1], 64,
                                  discovery_samples=400, prior=(torch.zeros(2), torch.ones(2)),
                                  correction="gauss_jacobian_FD", timesteps=15)
    assert info["modes"] == 1 and out.shape == (4, 64, 5)
