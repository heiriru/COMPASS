"""Analytic verification that grouping preserves the density and the composition.

A grouped arm changes what one composed factor contains, so two claims need
evidence rather than assertion: that the PF-ODE still integrates the right
density over the nested node vector, and that composing ``R`` factors of ``K``
subjects still targets the same posterior as composing ``R*K`` factors of one.

As in ``test_multiobs_analytic.py`` and ``test_pfode_vpsde_analytic.py``, a mock
"score network" returns the **exact** score of the diffused posterior for a
linear-Gaussian toy, so any discrepancy is attributable to the composition and
PF-ODE machinery rather than to training. The toy is partial pooling in
miniature:

    g       ~ N(mu0, sigma0^2)                  one group-level parameter
    l_j | g ~ N(g, tau^2)                       one parameter per subject
    x_j | l_j ~ N(l_j, sx^2)                    one observation per subject

laid out the way ``Group`` lays it out -- ``[g, (l_0, x_0), (l_1, x_1), ...]`` --
so the latent coordinates are deliberately *not* contiguous, which is exactly
the property that breaks the flat ``[theta, x]`` assumptions elsewhere.

Runs on CPU in about a minute.
"""

import math

import pytest
import torch

torch.set_num_threads(1)

from compass import Atomic, Group, IID, ScoreBasedInferenceModel
from compass.MultiObsSampler import MultiObsSampler
from compass.PFODE import PFODE
from compass.Sampler import Sampler
from compass.SDE import VESDE

MU0, SIGMA0 = -0.4, 0.9      # group-level prior
TAU = 0.7                    # subject-level spread around the group mean
SX = 0.5                     # observation noise


#############################################
# ----- The toy, in closed form -----
#############################################

def joint_precision(subjects):
    """Precision of ``p(g, l_1..l_K | x)`` -- an arrow matrix, as it should be."""
    size = subjects + 1
    precision = torch.zeros(size, size, dtype=torch.float64)
    precision[0, 0] = 1.0 / SIGMA0**2 + subjects / TAU**2
    for j in range(1, size):
        precision[0, j] = precision[j, 0] = -1.0 / TAU**2
        precision[j, j] = 1.0 / TAU**2 + 1.0 / SX**2
    return precision


def joint_covariance(subjects):
    """``p(g, l | x)`` has the same covariance whatever ``x`` is (it is Gaussian)."""
    return torch.linalg.inv(joint_precision(subjects))


def joint_means(observations, covariance):
    """Batched posterior means: ``(rows, subjects)`` observations in, ``(rows, 1+K)`` out."""
    observations = observations.to(torch.float64)
    linear = torch.cat((
        torch.full((observations.shape[0], 1), MU0 / SIGMA0**2, dtype=torch.float64),
        observations / SX**2,
    ), dim=1)
    return linear @ covariance.T


def joint_moments(observations):
    """``(mean, covariance)`` of ``p(g, l | x)`` for one group's observations."""
    covariance = joint_covariance(observations.numel())
    return joint_means(observations.reshape(1, -1), covariance)[0], covariance


def global_posterior(all_observations):
    """``p(g | every subject's x)``, marginalizing every ``l_j`` analytically.

    ``x_j | g ~ N(g, tau^2 + sx^2)``, so the whole hierarchy collapses to a
    one-dimensional conjugate update whose answer cannot depend on how the
    subjects were grouped. That independence is the point of the test.
    """
    flat = all_observations.reshape(-1).to(torch.float64)
    marginal = TAU**2 + SX**2
    precision = 1.0 / SIGMA0**2 + flat.numel() / marginal
    mean = (MU0 / SIGMA0**2 + flat.sum() / marginal) / precision
    return float(mean), float(1.0 / precision)


def node_indices(subjects):
    """Latent and observed positions in the ``[g, (l, x), (l, x), ...]`` layout."""
    latent = [0] + [1 + 2 * j for j in range(subjects)]
    observed = [2 + 2 * j for j in range(subjects)]
    return latent, observed


#############################################
# ----- The exact-score mock -----
#############################################

class GroupedExactScore(torch.nn.Module):
    """``sigma_t * `` the exact score of the diffused ``p(g, l | x)``.

    The latent coordinates are scattered through the node vector rather than
    sitting in a leading block, so this gathers and scatters by index -- which
    is what makes it a test of the grouped layout and not of a flat one.
    """

    def __init__(self, sde, subjects):
        super().__init__()
        self.sde = sde
        self.subjects = subjects
        self.latent, self.observed = node_indices(subjects)
        # The covariance depends only on K, and the diffusion time is shared by
        # every row of a call, so the whole batch needs one solve rather than
        # one per row. Without this the mock, not COMPASS, is the bottleneck.
        self.register_buffer("covariance", joint_covariance(subjects), persistent=False)
        self.register_buffer(
            "eye", torch.eye(subjects + 1, dtype=torch.float64), persistent=False)

    def forward(self, x, t, c, return_attn_weights=False):
        sigma = float(self.sde.marginal_prob_std(t).reshape(-1)[0])
        alpha = float(self.sde.alpha_t(t).reshape(-1)[0])
        theta_t = x[:, self.latent].to(torch.float64)
        mean = joint_means(x[:, self.observed], self.covariance)
        spread = alpha**2 * self.covariance + sigma**2 * self.eye
        residual = theta_t - alpha * mean
        score = -torch.linalg.solve(spread, residual.T).T
        out = torch.zeros_like(x)
        out[:, self.latent] = (sigma * score).to(out.dtype)
        if return_attn_weights:
            return out, torch.zeros(1)
        return out


class MockSBIm:
    """Minimal stand-in for ScoreBasedInferenceModel, as the sibling tests use."""

    def __init__(self, subjects):
        self.sde = VESDE(sigma=25.0)
        self.subjects = subjects
        self.nodes_size = 1 + 2 * subjects
        self.model = GroupedExactScore(self.sde, subjects)
        self.sampler = Sampler(self)
        self.multi_obs_sampler = MultiObsSampler(self)
        self.pfode = PFODE(self)

    def output_scale_function(self, t, x):
        return x / self.sde.marginal_prob_std(t).to(x.device)


def observations_for(rows, subjects, seed=0):
    generator = torch.Generator().manual_seed(seed)
    truth = MU0 + SIGMA0 * torch.randn(1, generator=generator)
    locals_ = truth + TAU * torch.randn(rows, subjects, generator=generator)
    return locals_ + SX * torch.randn(rows, subjects, generator=generator)


def node_rows(observations):
    """``(rows, subjects)`` observations -> full ``(rows, nodes)`` vectors."""
    rows, subjects = observations.shape
    latent, observed = node_indices(subjects)
    out = torch.zeros(rows, 1 + 2 * subjects)
    out[:, observed] = observations
    return out


#############################################
# ----- The PF-ODE density -----
#############################################

@pytest.mark.parametrize("subjects", [1, 3])
def test_pfode_log_prob_matches_the_closed_form(subjects):
    """The density is over the raw nested node vector, and it is the right one."""
    model = MockSBIm(subjects)
    latent, observed = node_indices(subjects)
    observations = observations_for(4, subjects, seed=1)
    condition_mask = torch.zeros(model.nodes_size)
    condition_mask[observed] = 1.0

    rows = node_rows(observations)
    generator = torch.Generator().manual_seed(2)
    rows[:, latent] = torch.randn(4, len(latent), generator=generator) * 0.3

    values = model.pfode.log_prob(
        data=rows, condition_mask=condition_mask, timesteps=256,
        divergence="exact", device="cpu", verbose=False,
    ).to(torch.float64)

    expected = []
    for row in range(rows.shape[0]):
        mean, covariance = joint_moments(observations[row])
        residual = rows[row, latent].to(torch.float64) - mean
        solved = torch.linalg.solve(covariance, residual)
        expected.append(
            -0.5 * float(residual @ solved)
            - 0.5 * float(torch.logdet(covariance))
            - 0.5 * len(latent) * math.log(2 * math.pi)
        )
    expected = torch.tensor(expected, dtype=torch.float64)
    assert torch.allclose(values, expected, atol=0.05), (
        f"PF-ODE log-density {values.tolist()} != analytic {expected.tolist()}"
    )


@pytest.mark.parametrize("subjects", [1, 3])
def test_exact_and_hutchinson_divergence_agree(subjects):
    """The two divergence estimators must see the same Jacobian."""
    model = MockSBIm(subjects)
    latent, observed = node_indices(subjects)
    observations = observations_for(3, subjects, seed=4)
    condition_mask = torch.zeros(model.nodes_size)
    condition_mask[observed] = 1.0
    rows = node_rows(observations)

    exact = model.pfode.log_prob(
        data=rows, condition_mask=condition_mask, timesteps=128,
        divergence="exact", device="cpu", verbose=False,
    )
    torch.manual_seed(0)
    stochastic = model.pfode.log_prob(
        data=rows, condition_mask=condition_mask, timesteps=128,
        divergence="hutchinson", device="cpu", verbose=False,
    )
    assert torch.allclose(exact, stochastic, atol=0.5), (
        f"exact {exact.tolist()} vs hutchinson {stochastic.tolist()}"
    )


#############################################
# ----- Composition over groups -----
#############################################

@pytest.mark.parametrize("correction, method", [
    ("gauss_hierarchical", "dpm"),
    ("gauss_jacobian", "dpm"),
])
@pytest.mark.parametrize("rows, subjects", [(8, 1), (4, 2), (2, 4)])
def test_composition_over_groups_recovers_the_analytic_global(
    correction, method, rows, subjects,
):
    """Eight subjects, grouped three ways, must give the same answer.

    ``(8, 1)`` is today's arrangement -- eight single-subject factors -- and
    ``(2, 4)`` is a grouped arm. The exact posterior over ``g`` is identical for
    all three, because it does not depend on how subjects were partitioned. Any
    disagreement is the composition approximation, which is precisely what the
    partition-sensitivity study measures at full scale.
    """
    model = MockSBIm(subjects)
    _, observed = node_indices(subjects)
    observations = observations_for(rows, subjects, seed=7)
    condition_mask = torch.zeros(model.nodes_size)
    condition_mask[observed] = 1.0

    mean, variance = global_posterior(observations)
    torch.manual_seed(11)
    draws = model.multi_obs_sampler.sample(
        world_size=1,
        data=observations, condition_mask=condition_mask, timesteps=128,
        num_samples=384, method=method, order=2, hierarchy=[0],
        correction=correction, prior=([MU0], [SIGMA0]),
        jacobian_refresh=1, precision_est_samples=256,
        precision_est_timesteps=64, corrector_steps=0,
        final_corrector_steps=0, device="cpu", verbose=False,
    )
    globals_ = draws[0, :, 0].to(torch.float64)
    assert torch.isfinite(globals_).all()
    # Every factor must agree on the shared coordinate.
    assert float((draws[:, :, 0] - draws[:1, :, 0]).abs().max()) < 1e-4

    estimated_mean = float(globals_.mean())
    estimated_std = float(globals_.std())
    # Measured errors across the six arrangements are <= 0.020 on the mean and
    # <= 0.015 on the sd, so these leave about 3x headroom -- enough for a
    # different BLAS, not enough to let a real regression through.
    assert abs(estimated_mean - mean) < 0.06, (
        f"rows={rows} K={subjects} {correction}: posterior mean "
        f"{estimated_mean:.4f} != analytic {mean:.4f}"
    )
    assert abs(estimated_std - math.sqrt(variance)) < 0.05, (
        f"rows={rows} K={subjects} {correction}: posterior sd "
        f"{estimated_std:.4f} != analytic {math.sqrt(variance):.4f}"
    )


#############################################
# ----- The real trunk's density -----
#############################################

def real_grouped_model(subjects=3, trials=4, seed=0):
    torch.manual_seed(seed)
    built = ScoreBasedInferenceModel(
        theta_size=2, hidden_size=8, num_heads=2, mlp_ratio=2, depth=2,
        x_spec=[Group("subjects", subjects, [
            Atomic("local", 2), IID("trials", 2, trials),
        ])],
        device="cpu",
    )
    built.model.eval()
    return built


def test_the_density_is_invariant_to_permuting_a_subjects_trials():
    """A real consequence of the one symmetry ``Group`` claims.

    The network is exactly equivariant under permuting trials inside a repeat,
    so the joint density it defines must be *invariant* under the same
    permutation. That ties the schema's promise to the PF-ODE's arithmetic: if
    the two disagreed about the layout, this would fail.
    """
    model = real_grouped_model()
    subjects, trials, item, local = 3, 4, 2, 2
    width = local + trials * item
    nodes = 2 + subjects * width
    condition_mask = torch.ones(nodes)
    condition_mask[:2] = 0.0
    for repeat in range(subjects):
        base = 2 + repeat * width
        condition_mask[base:base + local] = 0.0

    generator = torch.Generator().manual_seed(3)
    data = torch.randn(3, nodes, generator=generator)
    order = torch.tensor([2, 0, 3, 1])
    permuted = data.clone()
    base = 2 + width + local                       # repeat 1's trials
    view = data[:, base:base + trials * item].reshape(-1, trials, item)
    permuted[:, base:base + trials * item] = view[:, order].reshape(-1, trials * item)

    kwargs = dict(condition_mask=condition_mask, timesteps=64,
                  divergence="exact", device="cpu", verbose=False)
    original = model.log_prob(data=data, **kwargs)
    after = model.log_prob(data=permuted, **kwargs)
    assert torch.allclose(original, after, atol=1e-4), (
        f"density moved under a within-subject trial permutation: "
        f"{original.tolist()} vs {after.tolist()}"
    )


def test_the_density_moves_when_trials_change_subject():
    """The mirror: the nesting is load-bearing for the density too."""
    model = real_grouped_model()
    subjects, trials, item, local = 3, 4, 2, 2
    width = local + trials * item
    nodes = 2 + subjects * width
    condition_mask = torch.ones(nodes)
    condition_mask[:2] = 0.0
    for repeat in range(subjects):
        base = 2 + repeat * width
        condition_mask[base:base + local] = 0.0

    generator = torch.Generator().manual_seed(3)
    data = torch.randn(3, nodes, generator=generator)
    swapped = data.clone()
    first = slice(2 + local, 2 + local + trials * item)
    second = slice(2 + width + local, 2 + width + local + trials * item)
    swapped[:, first], swapped[:, second] = data[:, second], data[:, first]

    kwargs = dict(condition_mask=condition_mask, timesteps=64,
                  divergence="exact", device="cpu", verbose=False)
    assert not torch.allclose(
        model.log_prob(data=data, **kwargs),
        model.log_prob(data=swapped, **kwargs), atol=1e-4,
    )


def test_the_three_arrangements_agree_with_each_other():
    """The sharpest form of the claim: grouping does not change the target.

    The same eight subjects, partitioned as eight factors of one, four of two
    and two of four. The exact posterior over ``g`` is identical in all three --
    it is a one-dimensional conjugate update that cannot see the partition -- so
    any spread here is the composition approximation itself. This is the
    miniature of the partition-sensitivity study the plan runs at full scale.
    """
    estimates = {}
    for rows, subjects in ((8, 1), (4, 2), (2, 4)):
        model = MockSBIm(subjects)
        _, observed = node_indices(subjects)
        observations = observations_for(rows, subjects, seed=7)
        condition_mask = torch.zeros(model.nodes_size)
        condition_mask[observed] = 1.0
        torch.manual_seed(11)
        draws = model.multi_obs_sampler.sample(
            world_size=1,
            data=observations, condition_mask=condition_mask, timesteps=128,
            num_samples=384, method="dpm", order=2, hierarchy=[0],
            correction="gauss_hierarchical", prior=([MU0], [SIGMA0]),
            jacobian_refresh=1, precision_est_samples=256,
            precision_est_timesteps=64, corrector_steps=0,
            final_corrector_steps=0, device="cpu", verbose=False,
        )
        globals_ = draws[0, :, 0].to(torch.float64)
        estimates[(rows, subjects)] = (float(globals_.mean()), float(globals_.std()))

    means = [value[0] for value in estimates.values()]
    stds = [value[1] for value in estimates.values()]
    assert max(means) - min(means) < 0.05, (
        f"posterior mean depends on the partition: {estimates}"
    )
    assert max(stds) - min(stds) < 0.05, (
        f"posterior sd depends on the partition: {estimates}"
    )
