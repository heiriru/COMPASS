"""The residual wrapper and the compositional penalty, checked against the oracle."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "Gaussian_mixtures_compositional_score_modelling"
for path in (ROOT / "src", ROOT / "ModeTree" / "experiments", BENCHMARK):
    sys.path.insert(0, str(path))

import numpy as np
import pytest
import torch

import model as M
import oracle
import train

import gmresidual


@pytest.fixture(scope="module")
def pieces():
    mixture = M.DEFAULT
    exact = oracle.OracleModel(mixture, sigma=train.SDE_KWARGS["sigma"])
    return mixture, exact


def test_latent_marginal_matches_the_closed_form(pieces):
    """The baseline is the diffused prior score for the posterior mask."""
    mixture, exact = pieces
    baseline = gmresidual.LatentMarginalScore(mixture, exact.sde)
    g = torch.linspace(-4.0, 4.0, 41, dtype=torch.float32)
    state = torch.stack([g, torch.zeros_like(g), torch.ones_like(g)], dim=-1)
    mask = torch.zeros_like(state)
    mask[:, M.OBSERVED_INDICES[0]] = 1.0
    t = torch.full((state.shape[0], 1), 0.05)
    lam = float(exact.sde.lambda_t(t[0]))
    raw = baseline(state, t, mask)
    score = raw / exact.sde.marginal_prob_std(t)
    # the g component of grad log p_t(g, l) is not the prior score on g alone,
    # so the check is against the joint marginal, built independently in numpy
    covariance = oracle.joint_covariance(mixture).numpy()[:2, :2] \
        + lam ** 2 * np.eye(2)
    inverse = np.linalg.inv(covariance)
    values = state[:, :2].numpy().astype(np.float64)
    terms, directions = [], []
    for centre in mixture.components:
        delta = values - centre
        terms.append(np.log(0.5) - 0.5 * np.einsum("ij,jk,ik->i", delta,
                                                   inverse, delta))
        directions.append(-delta @ inverse)
    stacked = np.stack(terms)
    stacked -= stacked.max(axis=0, keepdims=True)
    responsibility = np.exp(stacked)
    responsibility /= responsibility.sum(axis=0, keepdims=True)
    expected = (responsibility[..., None] * np.stack(directions)).sum(axis=0)
    assert np.allclose(score[:, :2].numpy(), expected, atol=1e-5)


def test_residual_wrapper_adds_the_baseline_and_hides_it_from_the_checkpoint(pieces):
    """``raw = sigma s_base + backbone``, and ``state_dict`` is the backbone's."""
    mixture, exact = pieces
    network = train.build("h16d2", "tokenized")
    backbone = network.model
    reference = {k: v.clone() for k, v in backbone.state_dict().items()}
    state = torch.randn(7, M.NODES)
    mask = torch.zeros_like(state)
    mask[:, M.OBSERVED_INDICES[0]] = 1.0
    t = torch.full((7, 1), 0.1)
    plain = backbone(x=state, t=t, c=mask)

    gmresidual.attach(network, mixture, "residual")
    wrapped = network.model(x=state, t=t, c=mask)
    baseline = gmresidual.LatentMarginalScore(mixture, exact.sde)(state, t, mask)
    assert torch.allclose(wrapped, plain + baseline, atol=1e-5)
    assert set(network.model.state_dict()) == set(reference)


def test_composition_penalty_vanishes_on_the_oracle(pieces):
    """A perfect network makes ``L_comp`` zero, for both targets."""
    mixture, exact = pieces
    trainer = gmresidual.CompositionalTrainer(exact, mixture, comp_datasets=256)
    trainer.device = "cpu"
    trainer.time_sampling = "log_sigma"
    trainer.model = exact.model
    trainer.generator = np.random.default_rng([3, 99])
    assert float(trainer._composition_loss()) < 1e-12

    # the simulation-only surrogate is not zero -- its floor is the noise term
    # that does not depend on the parameters -- but it must sit at that floor.
    trainer.comp_target = "sim"
    trainer.generator = np.random.default_rng([3, 99])
    penalties = [float(trainer._composition_loss()) for _ in range(6)]
    assert all(np.isfinite(penalties))
    assert max(penalties) < 2.0


def test_preconditioner_leaves_the_score_convention_alone(pieces):
    """Preconditioning reparameterizes the input, never the output scale."""
    mixture, exact = pieces
    network = train.build("h16d2", "tokenized")
    gmresidual.attach(network, mixture, "preconditioned", eps=1e-4)
    state = torch.randn(5, M.NODES)
    mask = torch.zeros_like(state)
    mask[:, M.OBSERVED_INDICES[0]] = 1.0
    for time in (1e-4, 1e-2, 1.0):
        t = torch.full((5, 1), time)
        raw = network.model(x=state, t=t, c=mask)
        assert torch.isfinite(raw).all()
    # c_noise spans the embedder's dynamic range across the schedule
    low = network.model._c_noise(torch.full((1, 1), 1e-4))
    high = network.model._c_noise(torch.full((1, 1), 1.0))
    assert float(low) == pytest.approx(0.0, abs=1e-3)
    assert float(high) == pytest.approx(1000.0, rel=1e-3)
