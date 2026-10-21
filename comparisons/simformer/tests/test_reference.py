"""The MCMC reference machinery, validated against the one analytic posterior.

An unvalidated reference set is a silent way to publish a wrong C2ST: badly
mixed chains move the score away from 0.5 and look exactly like a bad model.
Gaussian Linear is the only task here with a closed-form posterior, so it is the
only place the samplers can be checked against truth rather than against each
other -- which is precisely why the check belongs in the test suite and not in a
notebook.

Short recipes and 400 chains keep this under a few seconds. The published runs
use the paper's own step counts (``common/paper.py::REFERENCE_MCMC``) and 10,000
chains; passing here says the sampler is correct, not that a given production
run has converged. Check the acceptance rate reported by each ``reference``
stage as well.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common import tasks as task_module
from common.masks import posterior_mask
from common.reference import reference_conditional_samples

SHORT_SLICE_MH = {"slice_steps": 50, "mh_steps": 300, "mh_step_size": 0.1}
SHORT_HMC = {"hmc_steps": 200, "hmc_step_size": 0.05, "hmc_leapfrog": 20}


def _setup():
    task = task_module.get_task("gaussian_linear")
    generator = torch.Generator().manual_seed(0)
    theta, x = task.sample_joint(1, generator=generator)
    conditioned = torch.cat([theta, x], dim=-1)[0]
    mask = posterior_mask(task.nodes_size, task.theta_dim)
    mean, std = task.analytic_posterior(x)
    return task, mask, conditioned, mean[0], std[0]


def test_slice_then_mh_recovers_the_analytic_posterior():
    task, mask, conditioned, mean, std = _setup()
    samples, diagnostics = reference_conditional_samples(
        task, mask, conditioned, 400, recipe=SHORT_SLICE_MH, seed=1)
    assert samples.shape == (400, task.theta_dim)
    assert 0.15 < diagnostics["mh_accept_rate"] < 0.95
    # Tolerance is the Monte-Carlo error of 400 draws, not zero.
    assert (samples.mean(0) - mean).abs().max() < 0.4 * float(std[0])
    assert torch.allclose(samples.std(0) / std, torch.ones(task.theta_dim),
                          atol=0.15)


def test_hmc_recovers_the_analytic_posterior():
    task, mask, conditioned, mean, std = _setup()
    samples, diagnostics = reference_conditional_samples(
        task, mask, conditioned, 400, recipe=SHORT_HMC, seed=2)
    assert samples.shape == (400, task.theta_dim)
    assert diagnostics["hmc_accept_rate"] > 0.6
    assert (samples.mean(0) - mean).abs().max() < 0.4 * float(std[0])
    assert torch.allclose(samples.std(0) / std, torch.ones(task.theta_dim),
                          atol=0.15)


def test_a_mixed_condition_mask_targets_the_right_marginal():
    """Conditioning on half of x must widen the posterior, not shift its shape."""
    task, _, conditioned, _, std = _setup()
    partial = torch.zeros(task.nodes_size)
    partial[task.theta_dim: task.theta_dim + 5] = 1.0
    samples, _ = reference_conditional_samples(
        task, partial, conditioned, 400, recipe=SHORT_SLICE_MH, seed=3)
    # Latent coordinates are theta (10) plus the 5 unobserved x nodes.
    assert samples.shape == (400, 15)
    # The five theta coordinates whose x is unobserved fall back to the prior.
    unconstrained = samples[:, 5:10].std(0)
    assert (unconstrained > float(std[0])).all()
