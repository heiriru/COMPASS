"""The inverse-kinematics baseline score, against its closed form."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "ModeTree" / "experiments",
             ROOT / "inverse_kinematics"):
    sys.path.insert(0, str(path))

import numpy as np
import torch

import hierarchical as H
import train_model

import exp4_ik_transfer as T


def test_baseline_is_the_diffused_gaussian_prior_and_zero_on_observables():
    network = train_model.build("h16d2", "legacy")
    baseline = T.GaussianPriorScore(network.sde, H.PRIOR_STD, H.LATENT_INDICES,
                                    H.NODES)
    state = torch.randn(9, H.NODES)
    mask = torch.zeros_like(state)
    for index in H.OBSERVED_INDICES:
        mask[:, index] = 1.0
    t = torch.full((9, 1), 0.02)
    lam = float(network.sde.lambda_t(t[0]))
    score = baseline(state, t, mask) / network.sde.marginal_prob_std(t)

    variance = np.asarray(H.PRIOR_STD, dtype=np.float64) ** 2 + lam ** 2
    expected = -state[:, H.LATENT_INDICES].numpy() / variance
    assert np.allclose(score[:, H.LATENT_INDICES].numpy(), expected, atol=1e-5)
    # the observables have no closed-form marginal: the network keeps those
    assert np.allclose(score[:, H.OBSERVED_INDICES].numpy(), 0.0)


def test_conditioned_coordinates_get_no_baseline():
    """A coordinate that is observed carries data, not a diffused latent."""
    network = train_model.build("h16d2", "legacy")
    baseline = T.GaussianPriorScore(network.sde, H.PRIOR_STD, H.LATENT_INDICES,
                                    H.NODES)
    state = torch.randn(4, H.NODES)
    mask = torch.zeros_like(state)
    mask[:, H.HIERARCHY[0]] = 1.0          # p(l, x | g)
    for index in H.OBSERVED_INDICES:
        mask[:, index] = 0.0
    t = torch.full((4, 1), 0.1)
    score = baseline(state, t, mask)
    assert np.allclose(score[:, H.HIERARCHY].numpy(), 0.0)
    assert not np.allclose(score[:, H.LOCAL_INDICES].numpy(), 0.0)


def test_wrapper_hides_itself_from_the_checkpoint():
    network = train_model.build("h16d2", "legacy")
    reference = set(network.model.state_dict())
    T.attach(network, "preconditioned", 1e-4)
    assert set(network.model.state_dict()) == reference
