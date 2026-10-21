"""PF-ODE endpoint='fixed': stored with the model, and a normalised density (Gibbs)."""

import validation.common.runtime  # noqa: F401

import torch

from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm
from validation.b5_b6_pfode_accuracy.families import Family, LikelihoodOracle

MASK = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])


def test_base_moments_round_trip(tmp_path):
    m = SBIm(nodes_size=5, hidden_size=16, depth=1, num_heads=2, mlp_ratio=2)
    m.set_base_moments(torch.randn(100, 5) * 2 + 1)
    m.save(str(tmp_path), "m")
    back = SBIm.load(str(tmp_path / "m.pt"), device="cpu")
    assert all(torch.allclose(a, b) for a, b in zip(m.base_moments, back.base_moments))


def test_fixed_endpoint_cannot_beat_exact_nll():
    fam = Family("bimodal")
    m = SBIm(nodes_size=5, sde_type="vesde", sigma=25.0, hidden_size=8, depth=1, num_heads=1,
             mlp_ratio=1)
    m.model = LikelihoodOracle(fam, m.sde)
    th, x = fam.simulate(1024, torch.Generator().manual_seed(0))
    m.set_base_moments(torch.cat([th, x], 1))
    sig = float(m.sde.marginal_prob_std(torch.tensor(1e-3)))
    exact = -fam.log_lik(th, x, sig).mean()
    fixed = -m.log_prob(torch.cat([th, x], 1), MASK, timesteps=100, endpoint="fixed").double().mean()
    # A normalised density has expected NLL >= the truth (up to Monte Carlo noise,
    # ~0.02 here); the score-based endpoints carry no such guarantee.
    assert fixed > exact - 0.02
    assert fixed - exact < 0.2
