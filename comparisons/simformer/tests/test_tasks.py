"""Cheap correctness checks for the transcribed task definitions.

A transcription error in ``common/tasks.py`` would not crash anything -- it would
silently produce a wrong reference posterior and therefore a wrong C2ST for
every arm. These tests are the guard, and they are the first thing to run in
this directory.

    /export/home/rheinric/COMPASS/.COMPASS/bin/python -m pytest \
        comparisons/simformer/tests -q
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common import tasks as task_module
from common.masks import (SimformerMaskSampler, likelihood_mask,
                          posterior_mask, random_conditional_masks)


def test_all_tasks_simulate_and_score():
    for name in task_module.TASKS:
        task = task_module.get_task(name)
        generator = torch.Generator().manual_seed(0)
        theta, x = task.sample_joint(256, generator=generator)
        assert theta.shape == (256, task.theta_dim), name
        assert x.shape == (256, task.x_dim), name
        z = torch.cat([theta, x], dim=-1)
        log_prob = task.joint_log_prob(z)
        assert log_prob.shape == (256,), name
        assert torch.isfinite(log_prob).all(), f"{name}: non-finite joint density"


def test_roles_partition_the_node_vector():
    for name in task_module.TASKS:
        task = task_module.get_task(name)
        covered = sorted(task.roles["theta"] + task.roles["x"])
        assert covered == list(range(task.nodes_size)), name


def test_gaussian_linear_analytic_posterior_is_conjugate():
    task = task_module.get_task("gaussian_linear")
    x = torch.linspace(-1.0, 1.0, task.x_dim).reshape(1, -1)
    mean, std = task.analytic_posterior(x)
    pv, lv = task.prior_std ** 2, task.likelihood_std ** 2
    expected_var = 1.0 / (1.0 / pv + 1.0 / lv)
    assert torch.allclose(std, torch.full_like(std, math.sqrt(expected_var)))
    assert torch.allclose(mean, x * (expected_var / lv))


def test_two_moons_likelihood_integrates_to_one():
    """The Jacobian term |det J| = r is the likely transcription error."""
    task = task_module.get_task("two_moons")
    theta = torch.tensor([[0.3, -0.2]])
    grid = torch.linspace(-1.5, 1.5, 601)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    log_like = task.log_likelihood(theta.expand(points.shape[0], -1), points)
    cell = (grid[1] - grid[0]) ** 2
    mass = torch.exp(log_like.clamp_min(-60.0)).sum() * cell
    assert abs(float(mass) - 1.0) < 0.02, f"two moons likelihood mass {mass}"


def test_slcp_likelihood_matches_multivariate_normal():
    task = task_module.get_task("slcp")
    generator = torch.Generator().manual_seed(3)
    theta, x = task.sample_joint(64, generator=generator)
    mean, cov = task._moments(theta)
    reference = torch.distributions.MultivariateNormal(mean, cov)
    draws = x.reshape(64, task.n_draws, 2)
    expected = reference.log_prob(draws.transpose(0, 1)).transpose(0, 1).sum(-1)
    assert torch.allclose(task.log_likelihood(theta, x), expected, atol=1e-4)


def test_hmm_prior_is_a_random_walk():
    task = task_module.get_task("hmm")
    generator = torch.Generator().manual_seed(5)
    theta = task.sample_prior(200_000, generator=generator)
    # Var(theta_i) = (i+1) * step_std^2 for a cumulative sum of i.i.d. steps.
    observed = theta.var(0)
    expected = torch.arange(1, task.theta_dim + 1) * task.step_std ** 2
    assert torch.allclose(observed, expected, rtol=0.05)


def test_tree_and_hmm_declare_their_edges():
    for name in ("tree", "hmm", "gaussian_linear"):
        task = task_module.get_task(name)
        assert task.edges, name
        for child, parents in task.edges.items():
            assert 0 <= child < task.nodes_size
            assert all(0 <= p < task.nodes_size for p in parents)


def test_posterior_and_likelihood_masks_are_complementary():
    task = task_module.get_task("slcp")
    post = posterior_mask(task.nodes_size, task.theta_dim)
    like = likelihood_mask(task.nodes_size, task.theta_dim)
    assert torch.equal(post + like, torch.ones(task.nodes_size))
    assert post[: task.theta_dim].sum() == 0
    assert post[task.theta_dim:].sum() == task.x_dim


def test_simformer_mask_sampler_matches_the_published_law():
    from compass import NodeSchema

    task = task_module.get_task("two_moons")
    schema = NodeSchema.legacy(task.nodes_size)
    generator = torch.Generator().manual_seed(11)
    sampler = SimformerMaskSampler(schema, task.theta_dim, generator=generator)
    described = sampler.describe()
    assert all(abs(p - 0.2) < 1e-6 for p in described.values())
    assert set(described) == set(sampler.PATTERNS)

    seen = set()
    for _ in range(400):
        mask = sampler.sample((8, task.nodes_size))
        assert mask.shape == (8, task.nodes_size)
        # per_batch=True: one pattern broadcast over the batch
        assert torch.equal(mask[0].expand_as(mask), mask)
        seen.add(tuple(mask[0].tolist()))
    # The three fixed patterns must all appear, including the all-latent joint.
    assert (0.0, 0.0, 0.0, 0.0) in seen, "joint mask never drawn"
    assert (0.0, 0.0, 1.0, 1.0) in seen, "posterior mask never drawn"
    assert (1.0, 1.0, 0.0, 0.0) in seen, "likelihood mask never drawn"


def test_random_conditional_masks_are_distinct_and_non_degenerate():
    masks = random_conditional_masks(10, 100,
                                     generator=torch.Generator().manual_seed(1))
    assert masks.shape == (100, 10)
    assert len({tuple(m.tolist()) for m in masks}) == 100
    assert (masks.sum(-1) > 0).all() and (masks.sum(-1) < 10).all()
