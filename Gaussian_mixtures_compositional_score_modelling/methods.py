#!/usr/bin/env python3
"""The composition rules under test, and the one exact reference.

Every rule here consumes the *same* single-observation network and the same tall
dataset; the only thing that varies is how the per-observation information is
combined into ``p(g, l_1..l_n | x_1..x_n)``.

Three arms are run by default -- ``langevin_fnpse``, ``gauss_hierarchical`` and
``gauss_jacobian``, the last two under the churned DPM2 predictor at
``churn_eta = 4``. Two further arms exist as diagnostics and have to be asked
for with ``--methods``; they answer *why* a rule misses rather than proposing a
rule anyone would deploy on this problem.

``gauss_hierarchical``
    GAUSS (Linhart et al.) generalised to a global/local hierarchy by arrow
    elimination, with each observation's backward covariance estimated from a
    pilot run of the network's own draws.

``gauss_jacobian``
    The same composition with the per-observation curvature read off the
    network's Jacobian by Tweedie's second-order identity -- no pilot run.

``langevin_fnpse``
    Annealed Langevin on Geffner et al.'s bridging densities; the cheapest rule,
    and the only one whose target is deliberately not the diffusion marginals.

``gauss_jacobian_mixture_prior``
    ``gauss_jacobian``, with one thing changed: the prior term. This is the arm
    the benchmark exists to isolate, and the reason is structural. Every
    score-composition rule subtracts ``(n - 1)`` copies of the *diffused prior
    score*, and COMPASS's sampler API accepts only a **Gaussian** prior. The
    honest translation of the bimodal prior here is its moment match,
    ``N(0, sqrt(mu_0^2 + tau_g^2)) = N(0, 2.53)``, which is not remotely the
    same function: its score is a single line through the origin where the true
    diffused prior score has three zeros and pushes *away* from the origin
    between the modes. At ``n = 32`` that error is multiplied by 31. The prior
    of a Gaussian mixture diffuses in closed form -- same weights, variance
    inflated by ``lambda_t^2`` -- so the exact term is available, and
    :func:`mixture_prior_score` patches it in. Everything else about the arm is
    identical to ``gauss_jacobian``, so the difference between the two is
    attributable to the prior term alone.

``marginal_product``
    No composed score at all. For one shared block with conditionally
    independent locals both of these are exact identities, not approximations::

        p(g | x_1..n)  ∝  p(g)^(1-n) prod_i p(g | x_i)
        p(g, l_1..n | x_1..n) = p(g | x_1..n) prod_i p(l_i | g, x_i)

    and both conditionals on the right are ones the network is trained to
    produce directly, so the whole tall posterior assembles from two ordinary
    amortized sampler calls. The shared block is one-dimensional here, so the
    product is done on a grid and the prior used is the **exact mixture**, not a
    moment match. Its cost is a density estimate raised to the ``n``-th power;
    see ``inverse_kinematics/experiments/marginal_product.py``, where this rule
    was first measured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import model as M  # noqa: E402

FLOOR = 1e-300


def _normalise_at(mixture, values):
    """The prior density evaluated at arbitrary points, not just on a grid."""
    return np.asarray(mixture.prior_density(np.asarray(values, dtype=np.float64)),
                      dtype=np.float64)

# Shared by every rule, so a comparison isolates the composition.
COMMON = {"timesteps": 100}

METHODS = {
    "langevin_fnpse": {
        "label": "Langevin + F-NPSE",
        "colour": "#2A9D8F",
        "churn": False,
        "kwargs": {"method": "langevin", "correction": "fnpe",
                   "corrector_steps": 10, "snr": 0.1},
    },
    "gauss_hierarchical": {
        "label": "DPM2 + churn + gauss_hierarchical",
        "colour": "#E76F51",
        "churn": True,
        "kwargs": {"method": "dpm", "order": 2,
                   "correction": "gauss_hierarchical", "corrector_steps": 0,
                   "final_corrector_steps": 0, "terminal_corrector_steps": 0},
    },
    "gauss_jacobian": {
        "label": "DPM2 + churn + gauss_jacobian",
        "colour": "#5E60CE",
        "churn": True,
        "kwargs": {"method": "dpm", "order": 2, "correction": "gauss_jacobian",
                   "corrector_steps": 0, "final_corrector_steps": 0,
                   "terminal_corrector_steps": 0},
    },
    "gauss_jacobian_mixture_prior": {
        "label": "gauss_jacobian + exact mixture prior",
        "colour": "#B5179E",
        "churn": True,
        "mixture_prior": True,
        "kwargs": {"method": "dpm", "order": 2, "correction": "gauss_jacobian",
                   "corrector_steps": 0, "final_corrector_steps": 0,
                   "terminal_corrector_steps": 0},
    },
    "marginal_product": {
        "label": "marginal product (exact factorisation)",
        "colour": "#F4A300",
        "churn": True,
        "kwargs": {},
    },
    "mixture_product": {
        "label": "mixture product (K inferred)",
        "colour": "#1B9E77",
        "churn": True,
        "kwargs": {},
    },
    "mixture_product_bagged": {
        "label": "mixture product + bagging",
        "colour": "#C71585",
        "churn": True,
        "kwargs": {},
    },
    "mixture_product_likelihood_bagged": {
        "label": "likelihoods + bagging",
        "colour": "#8E0152",
        "churn": True,
        "kwargs": {},
    },
    "tilt_product": {
        "label": "tilt product (d=2)",
        "style": {"color": "#0f766e", "linestyle": "-"},
    },
    "tilt_product_bagged": {
        "label": "tilt product + bagging",
        "style": {"color": "#0f766e", "linestyle": "--"},
    },
    "tilt_product_bic": {
        "label": "tilt product (d by BIC)",
        "style": {"color": "#7c2d12", "linestyle": "-"},
    },
    "mixture_product_capped": {
        "label": "mixture-hierarchical, prior-width cap",
        "colour": "#0B6E4F",
        "churn": True,
        "kwargs": {},
    },
    "mixture_product_likelihood": {
        "label": "likelihood product (weighted EM)",
        "colour": "#C71585",
        "churn": True,
        "kwargs": {},
    },
    "mixture_product_clipped": {
        "label": "mixture product, support-constrained",
        "colour": "#C71585",
        "churn": True,
        "kwargs": {},
    },
}

SCORE_METHODS = tuple(name for name in METHODS
                      if name not in ("marginal_product", "mixture_product",
                                      "mixture_product_clipped",
                                      "mixture_product_likelihood",
                                      "mixture_product_bagged",
                                      "mixture_product_capped", "mixture_product_likelihood_bagged",
         "tilt_product", "tilt_product_bagged", "tilt_product_bic"))

# The three arms the sweep runs by default: the two Gaussian composition rules
# under the churned DPM2 predictor, and the Langevin/F-NPSE baseline.
ORDER = ("langevin_fnpse", "gauss_hierarchical", "gauss_jacobian")

# Available with an explicit ``--methods``, not run by default. Both are
# diagnostics for *why* a rule misses rather than rules anyone would deploy
# here: the first isolates the prior term, the second replaces score
# composition with the exact factorisation.
EXTRA = ("gauss_jacobian_mixture_prior", "marginal_product",
         "mixture_product", "mixture_product_clipped",
         "mixture_product_likelihood", "mixture_product_bagged",
         "mixture_product_capped", "mixture_product_likelihood_bagged",
         "tilt_product", "tilt_product_bagged", "tilt_product_bic")
ALL = ORDER + EXTRA


# ---------------------------------------------------------------------------
# The exact diffused mixture prior score, as a patch on the sampler
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def mixture_prior_score(mixture=M.DEFAULT, enabled=True):
    """Replace the sampler's Gaussian prior score with the exact mixture one.

    ``MultiObsSampler._diffused_gaussian_prior_score`` is the single place every
    correction reads the prior term from, and both of its branches reduce, in
    the sampler's ``y = x / alpha`` coordinates, to the score of a Gaussian with
    variance ``tau^2 + lambda_t^2``. The mixture diffuses the same way -- same
    weights, every component variance inflated by ``lambda_t^2`` -- so the
    replacement is a closed form, evaluated at the same argument, with the same
    shape contract.

    Restricted to a one-dimensional shared block, which is what this benchmark
    has; the multivariate version is the same formula with responsibilities from
    a full quadratic form, and is not written here because nothing would test it.
    """
    from compass.MultiObsSampler import MultiObsSampler

    if not enabled:
        yield False
        return

    original = MultiObsSampler._diffused_gaussian_prior_score
    centres = torch.as_tensor(mixture.components, dtype=torch.float64)
    log_weights = torch.log(torch.as_tensor(mixture.weights, dtype=torch.float64))

    def patched(self, theta_y, t):
        if len(self.hierarchy) != 1:
            raise NotImplementedError(
                "the exact mixture prior score is written for a scalar shared "
                f"block; this run has {len(self.hierarchy)}."
            )
        variance = self.sde.lambda_t(t).to(theta_y.device).square().reshape(())
        total = float(mixture.tau_g) ** 2 + variance.to(torch.float64)
        location = centres.to(theta_y.device).reshape(-1, *([1] * theta_y.dim()))
        delta = theta_y.to(torch.float64).unsqueeze(0) - location
        terms = log_weights.to(theta_y.device).reshape(
            -1, *([1] * theta_y.dim())
        ) - 0.5 * delta.square() / total
        responsibility = torch.softmax(terms, dim=0)
        score = (responsibility * (-delta / total)).sum(dim=0)
        return score.to(theta_y.dtype)

    MultiObsSampler._diffused_gaussian_prior_score = patched
    try:
        yield True
    finally:
        MultiObsSampler._diffused_gaussian_prior_score = original


# ---------------------------------------------------------------------------
# Score-composition rules
# ---------------------------------------------------------------------------

def _priors(mixture):
    return (
        (torch.tensor([mixture.global_prior_mean], dtype=torch.float32),
         torch.tensor([mixture.global_prior_std], dtype=torch.float32)),
        (torch.tensor([mixture.local_prior_mean], dtype=torch.float32),
         torch.tensor([mixture.local_prior_std], dtype=torch.float32)),
    )


def draw_score_rule(network, x, method, arguments, device, mixture=M.DEFAULT):
    """Run one score-composition rule; returns ``(samples, seconds)``.

    ``samples`` is ``(n, draws, NODES)`` with the observables written back in,
    matching what :func:`draw_marginal_product` returns so one grader consumes
    both.
    """
    import time

    from compass.MultiObsSampler import MultiObsSampler

    entry = METHODS[method]
    kwargs = dict(entry["kwargs"])
    if kwargs.get("correction") == "gauss_hierarchical":
        kwargs.update(precision_est_samples=int(arguments.pilot_draws),
                      precision_est_timesteps=COMMON["timesteps"])
    if kwargs.get("correction") == "gauss_jacobian":
        kwargs.update(jacobian_refresh=int(arguments.jacobian_refresh))

    data = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(-1, 1),
                           dtype=torch.float32)
    prior, local_prior = _priors(mixture)
    sampler = MultiObsSampler(network)
    torch.manual_seed(int(arguments.seed))
    torch.cuda.manual_seed_all(int(arguments.seed))

    started = time.perf_counter()
    with mixture_prior_score(mixture, enabled=entry.get("mixture_prior", False)):
        samples = sampler.sample(
            world_size=1, data=data,
            condition_mask=torch.tensor(
                [0.0] * len(M.LATENT_INDICES) + [1.0] * len(M.OBSERVED_INDICES)
            ),
            num_samples=int(arguments.draws),
            hierarchy=list(M.HIERARCHY),
            prior=prior, local_prior=local_prior,
            churn_eta=float(arguments.churn_eta) if entry["churn"] else 0.0,
            denoise_clamp=float(arguments.denoise_clamp),
            device=device, verbose=False, **COMMON, **kwargs,
        )
    seconds = time.perf_counter() - started

    samples = samples.detach().cpu().numpy()
    # The shared coordinate must be identical across observation rows; if it is
    # not, the composition never actually shared it and every number computed
    # from these draws is about something else.
    spread = float(np.abs(samples[:, :, M.HIERARCHY]
                          - samples[:1, :, M.HIERARCHY]).max())
    if spread > 1e-4:
        raise AssertionError(
            f"{method}: shared draws are not synchronised across observations "
            f"({spread:.3e})."
        )
    return samples, seconds


# ---------------------------------------------------------------------------
# The exact factorisation
# ---------------------------------------------------------------------------

def _normalise(grid, density):
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("a density on the shared grid did not normalise")
    return density / area


def shared_marginal(draws, grid, prior, bandwidth=None):
    """``p(g | x_1..n)`` on ``grid`` from per-observation draws of ``g``.

    ``draws`` is ``(n, samples)``. The ``prior^(1-n)`` factor is the exact
    mixture prior, tabulated on the same grid as everything it is divided into.
    """
    from scipy.stats import gaussian_kde

    draws = np.asarray(draws, dtype=np.float64)
    densities = np.asarray([
        _normalise(grid, gaussian_kde(row, bw_method=bandwidth)(grid))
        for row in draws
    ])
    n = densities.shape[0]
    log_density = (np.log(np.maximum(densities, FLOOR)).sum(axis=0)
                   - (n - 1) * np.log(np.maximum(prior, FLOOR)))
    log_density -= log_density.max()
    return _normalise(grid, np.exp(log_density)), densities


def mixture_shared_marginal(draws, grid, prior, k_max=5, restarts=1, seed=0,
                            clip=None, support_quantile=None,
                            variance_ceiling=None):
    """``shared_marginal`` with each factor a fitted Gaussian mixture, not a KDE.

    The only change from :func:`shared_marginal` is how ``p(g | x_i)`` is turned
    from draws into a density: a Gaussian mixture whose component count is chosen
    per observation by BIC, instead of a kernel density estimate. Everything else
    -- the product, the exact mixture prior, the ``(n-1)`` power -- is identical,
    so a difference between the two arms is attributable to that estimator alone.

    Why this is worth doing: the product raises each factor to the ``n``-th power,
    so it raises each factor's *error* to the ``n``-th power too. A KDE's error is
    largest exactly where the density is small, which is where an ``n``-fold
    product is most sensitive.

    Why the composition is still done on a grid here, unlike in
    ``inverse_kinematics``: the prior of this benchmark is itself a two-component
    mixture, and a mixture raised to the power ``1-n`` is not a mixture. The
    analytic, grid-free form of this method needs a Gaussian prior (or the
    per-observation *likelihoods* rather than posteriors). With a mixture prior
    the division has to happen somewhere, and here it happens on the grid.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent
                            / "inverse_kinematics" / "experiments"))
    from mixture_inference import choose_and_fit

    draws = np.asarray(draws, dtype=np.float64)
    if variance_ceiling is None:
        fits, chosen = choose_and_fit(draws, k_max=k_max, restarts=restarts,
                                      seed=seed)
    else:
        from mixture_product import bic, fit_mixture
        fits, chosen = [], []
        for index, row in enumerate(np.asarray(draws, dtype=np.float64)):
            scores = {}
            for k in range(1, k_max + 1):
                fit = fit_mixture(row, k, restarts=restarts, seed=seed + index,
                                  variance_ceiling=variance_ceiling)
                scores[k] = (bic(row, *fit), fit)
            best = min(scores, key=lambda k: scores[k][0])
            fits.append(scores[best][1]); chosen.append(best)
    densities = np.asarray([
        _normalise(grid, (
            weights[None, :] / np.sqrt(2.0 * np.pi * variances[None, :])
            * np.exp(-0.5 * (grid[:, None] - means[None, :]) ** 2
                     / variances[None, :])
        ).sum(axis=1))
        for weights, means, variances in fits
    ])
    n = densities.shape[0]
    log_prior = np.log(np.maximum(prior, FLOOR))
    log_factors = np.log(np.maximum(densities, FLOOR))
    if clip is None:
        log_density = log_factors.sum(axis=0) - (n - 1) * log_prior
    else:
        # Compose *evidence* rather than posteriors:
        #     p(g | x_1..n)  =  p(g) prod_i [ p(g | x_i) / p(g) ]
        # Each bracket is the evidence one observation contributes, and where
        # that observation has no draws it is unmeasured -- the safe value is 1
        # (no evidence), not whatever the tail of a fitted density extrapolates
        # to. Clipping |log r_i| <= clip bounds the composed damage by n * clip,
        # because composed log-error is the *sum* of per-observation log-errors.
        #
        # This benchmark is where that matters most: its prior is a mixture of
        # N(+-2.5, 0.4^2), so g = 0 sits 6.25 sigma from either component and
        # -(n-1) log p(0) is about +140 nats at n = 8. Without the clip that
        # lift multiplies pure estimation error in a region no observation
        # constrains, and the product collapses onto a spike in the valley.
        log_density = log_prior + np.clip(
            log_factors - log_prior[None, :], -clip, clip
        ).sum(axis=0)
    if support_quantile is not None:
        # The composed support *is* the intersection of the per-observation
        # supports: p(g|x_1..n) is proportional to a product, so if one factor
        # vanishes at g the product does. That is an identity, and the draws
        # observe each support directly, so it costs nothing to enforce.
        #
        # It is the right constraint for this benchmark. The prior is a mixture
        # of N(+-2.5, 0.4^2), so the valley at g = 0 is 6.25 sigma from either
        # component and -(n-1) log p(0) reaches +140 nats at n = 8. No
        # observation has a draw anywhere near there, so every bit of density
        # the product builds in the valley is estimation error being amplified.
        lower = max(float(np.quantile(row, support_quantile)) for row in draws)
        upper = min(float(np.quantile(row, 1.0 - support_quantile))
                    for row in draws)
        if upper > lower:
            log_density = np.where((grid >= lower) & (grid <= upper),
                                   log_density, -np.inf)
    log_density -= log_density.max()
    return _normalise(grid, np.exp(log_density)), densities, chosen



def likelihood_shared_marginal(draws, grid, prior, prior_at, k_max=5,
                               restarts=1, seed=0, weight_clip=100.0):
    """Compose *likelihoods* rather than posteriors -- no ``prior^(1-n)`` term.

    The identity
        p(g | x_1..n)  proportional to  p(g) prod_i [ p(g | x_i) / p(g) ]
    has a division by the prior in every factor, and the usual form collects
    those into ``prior^(1-n)``. That single power is the whole problem: on this
    benchmark the prior is a mixture of ``N(+-2.5, 0.4^2)``, so ``-(n-1) log p(g)``
    reaches ``+140`` nats in the valley at ``n = 8``, applied exactly where each
    estimated factor is small and mostly error.

    The fix costs nothing. Draws from ``p(g | x_i)`` reweighted by ``1 / p(g)``
    are draws from a measure proportional to the **likelihood** ``L_i(g)``, so a
    *weighted* EM fit of the same draws estimates ``L_i`` directly. Then
        log p(g | x_1..n) = log p(g) + sum_i log L_hat_i(g) + const
    and the ``(n-1)`` cancellation never happens at all -- it is removed rather
    than bounded.

    Weights ``1/p(g)`` are unbounded in the prior's tails, which is where the
    accuracy is wanted but also where a handful of draws would otherwise carry
    the fit; ``weight_clip`` caps them at that multiple of the median weight.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent
                            / "inverse_kinematics" / "experiments"))
    from mixture_product import bic, fit_mixture

    draws = np.asarray(draws, dtype=np.float64)
    densities, chosen = [], []
    for row in draws:
        weight = 1.0 / np.maximum(prior_at(row), FLOOR)
        weight = np.minimum(weight, weight_clip * np.median(weight))
        weight = weight / weight.sum()
        # Weighted EM by resampling: drawing with probability proportional to
        # the weights turns the posterior sample into a likelihood sample, and
        # then the existing unweighted fitter applies unchanged.
        rng = np.random.default_rng(seed)
        resampled = rng.choice(row, size=len(row), replace=True, p=weight)
        candidates = {}
        for k in range(1, k_max + 1):
            fit = fit_mixture(resampled, k, restarts=restarts, seed=seed)
            candidates[k] = (bic(resampled, *fit), fit)
        best = min(candidates, key=lambda k: candidates[k][0])
        chosen.append(best)
        weights_k, means, variances = candidates[best][1]
        densities.append(_normalise(grid, (
            weights_k[None, :] / np.sqrt(2.0 * np.pi * variances[None, :])
            * np.exp(-0.5 * (grid[:, None] - means[None, :]) ** 2
                     / variances[None, :])
        ).sum(axis=1)))
    densities = np.asarray(densities)
    log_density = (np.log(np.maximum(prior, FLOOR))
                   + np.log(np.maximum(densities, FLOOR)).sum(axis=0))
    log_density -= log_density.max()
    return _normalise(grid, np.exp(log_density)), densities, chosen


def tilt_shared_marginal(draws, grid, prior, degrees=(2,), bag=0, seed=0):
    """Compose exponential tilts of the prior rather than estimated posteriors.

    See ``inverse_kinematics/experiments/tilt_fit.py`` for why. The short
    version: the composed posterior of this benchmark sits in the prior's
    valley, five sd into the tail of every single-observation posterior, where a
    20,000-draw sample has no draws at all -- so no estimate of those factors
    can be accurate there. The *likelihood* ratio is a single broad Gaussian in
    ``g``, log-quadratic and fixed by two numbers, and a product of exponential
    tilts of one base measure is the tilt by the sum of their parameters. So the
    whole composition is: fit two moments per observation, add the parameter
    vectors, tilt the exact prior once.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent
                            / "inverse_kinematics" / "experiments"))
    import tilt_fit

    draws = np.asarray(draws, dtype=np.float64)
    log_prior = np.log(np.maximum(prior, FLOOR))
    thetas, chosen, centre, scale, _ = tilt_fit.fit_all(
        draws, grid, log_prior, degrees=tuple(degrees))
    if bag:
        density = tilt_fit.compose_bagged(thetas, grid, log_prior, centre,
                                          scale, replicates=int(bag), seed=seed)
    else:
        density = tilt_fit.compose(thetas, grid, log_prior, centre, scale)
    return density, np.empty((0, len(grid))), chosen


def bagged_from_factors(densities, grid, prior, replicates=64, seed=0):
    """Average the composed density over bootstrap resamples of the observations.

    The per-observation factors are already estimated, so a replicate is a
    re-multiplication of the same ``n`` curves drawn with replacement -- no EM
    refit and no network call. Averaging *densities* is a convex combination, so
    a replicate whose product collapses onto one mode cannot cancel one that did
    not, and the average keeps both. That is aimed squarely at the measured
    high-n failure: the composed log-odds is a sum of one term per observation,
    so a per-observation error is multiplied by ``n``.
    """
    densities = np.asarray(densities, dtype=np.float64)
    n = densities.shape[0]
    log_factors = np.log(np.maximum(densities, FLOOR))
    log_prior = np.log(np.maximum(prior, FLOOR))
    rng = np.random.default_rng(seed)
    total = np.zeros(len(grid), dtype=np.float64)
    for _ in range(replicates):
        picked = rng.integers(0, n, size=n)
        log_density = (log_factors[picked].sum(axis=0) - (n - 1) * log_prior)
        log_density -= log_density.max()
        total += _normalise(grid, np.exp(log_density))
    return _normalise(grid, total / replicates)


def inverse_cdf_sample(grid, density, count, rng):
    """Draw from a density tabulated on a grid, by inverting its CDF."""
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    cumulative /= cumulative[-1]
    # Flat stretches of the CDF are not invertible and ``np.interp`` would map a
    # whole zero-density interval onto its left edge.
    keep = np.concatenate([[True], np.diff(cumulative) > 0])
    return np.interp(rng.random(int(count)), cumulative[keep], grid[keep])


def draw_marginal_product(network, x, arguments, device, mixture=M.DEFAULT,
                          grid=None, chunk=16384, estimator="kde"):
    """The two-call exact factorisation. Returns ``(samples, seconds, detail)``.

    ``estimator`` selects how each ``p(g | x_i)`` is estimated from its draws:
    ``"kde"`` is the original rule, ``"mixture"`` fits a Gaussian mixture with K
    chosen per observation by BIC.
    """
    import time

    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    count = int(arguments.draws)
    if grid is None:
        # The tilt arms integrate moments on this grid, and at n = 100 the
        # composed sd is a tenth of the prior's, so the default resolution is
        # the quadrature's accuracy rather than a cosmetic choice.
        grid = mixture.default_grid(
            x, points=int(getattr(arguments, "grid_points", 1601)))
    prior = _normalise(grid, mixture.prior_density(grid))
    observations = torch.as_tensor(x.reshape(-1, 1), dtype=torch.float32)
    common = dict(
        timesteps=COMMON["timesteps"], method="dpm", order=2,
        churn_eta=float(arguments.churn_eta), corrector_steps=0,
        final_corrector_steps=0, terminal_denoise=True, device=device,
        verbose=False,
    )

    started = time.perf_counter()
    # Stage 1: every single-observation posterior p(g, l | x_i).
    #
    # Chunked over *observations*, not done in one call. The single-observation
    # posteriors are independent given their own observation, so splitting the
    # batch changes nothing about what is computed -- but the sampler's peak
    # memory is (block * draws) rows of transformer activations rather than
    # (n * draws), and at n = 100 with 20,000 draws the unchunked call needs more
    # than the 10.5 GB on this card and dies. Each block is seeded from the run
    # seed and its own index so a re-run reproduces the same draws.
    single_count = int(getattr(arguments, "single_draws", 20_000))
    block = int(getattr(arguments, "stage1_chunk", 16) or n)
    # Stage 1 and stage 3 want different churn: churn perturbs the
    # single-observation marginals the tilt reads its moments from, while stage 3
    # draws locals conditioned on one shared value and benefits from it.
    # Measured on inverse kinematics; ``stage1_churn`` defaults to ``churn_eta``
    # so nothing changes unless it is set.
    stage1_common = dict(common)
    stage1_common["churn_eta"] = float(
        getattr(arguments, "stage1_churn", None)
        if getattr(arguments, "stage1_churn", None) is not None
        else arguments.churn_eta)
    pieces = []
    for start in range(0, n, block):
        torch.manual_seed(int(arguments.seed) + 7919 * start)
        torch.cuda.manual_seed_all(int(arguments.seed) + 7919 * start)
        pieces.append(network.sample(
            x=observations[start:start + block], num_samples=single_count,
            **stage1_common
        ).detach().cpu().numpy())
    single = np.concatenate(pieces, axis=0)
    shared_draws = single[:, :, 0].astype(np.float64)

    # Stage 2: the shared marginal by the exact product, then draws from it.
    chosen = None
    if estimator == "likelihood":
        density, per_observation, chosen = likelihood_shared_marginal(
            shared_draws, grid, prior,
            lambda values: _normalise_at(mixture, values),
            k_max=int(getattr(arguments, "k_max", 5)),
            restarts=int(getattr(arguments, "em_restarts", 1)),
            seed=int(arguments.seed),
        )
    elif estimator == "mixture_bagged":
        density, per_observation, chosen = mixture_shared_marginal(
            shared_draws, grid, prior,
            k_max=int(getattr(arguments, "k_max", 5)),
            restarts=int(getattr(arguments, "em_restarts", 1)),
            seed=int(arguments.seed),
        )
        density = bagged_from_factors(
            per_observation, grid, prior,
            replicates=int(getattr(arguments, "bag", 64)),
            seed=int(arguments.seed))
    elif estimator == "likelihood_bagged":
        density, per_observation, chosen = likelihood_shared_marginal(
            shared_draws, grid, prior,
            lambda values: _normalise_at(mixture, values),
            k_max=int(getattr(arguments, "k_max", 5)),
            restarts=int(getattr(arguments, "em_restarts", 1)),
            seed=int(arguments.seed),
        )
        # The likelihood form has no prior^(1-n) term, so a replicate is the
        # prior times a resampled product of the fitted likelihoods.
        log_prior = np.log(np.maximum(prior, FLOOR))
        log_factors = np.log(np.maximum(per_observation, FLOOR))
        rng_bag = np.random.default_rng(int(arguments.seed))
        total = np.zeros(len(grid))
        for _ in range(int(getattr(arguments, "bag", 64))):
            picked = rng_bag.integers(0, len(log_factors), size=len(log_factors))
            log_density = log_prior + log_factors[picked].sum(axis=0)
            log_density -= log_density.max()
            total += _normalise(grid, np.exp(log_density))
        density = _normalise(grid, total / int(getattr(arguments, "bag", 64)))
    elif estimator.startswith("tilt"):
        degrees = {"tilt": (2,), "tilt_bic": (2, 4, 6)}[
            estimator.removesuffix("_bagged")]
        density, per_observation, chosen = tilt_shared_marginal(
            shared_draws, grid, prior, degrees=degrees,
            bag=(int(getattr(arguments, "bag", 64))
                 if estimator.endswith("_bagged") else 0),
            seed=int(arguments.seed))
    elif estimator == "mixture_capped":
        density, per_observation, chosen = mixture_shared_marginal(
            shared_draws, grid, prior,
            k_max=int(getattr(arguments, "k_max", 5)),
            restarts=int(getattr(arguments, "em_restarts", 1)),
            seed=int(arguments.seed),
            variance_ceiling=float(mixture.tau_g) ** 2,
        )
    elif estimator in ("mixture", "mixture_clipped"):
        density, per_observation, chosen = mixture_shared_marginal(
            shared_draws, grid, prior,
            k_max=int(getattr(arguments, "k_max", 5)),
            restarts=int(getattr(arguments, "em_restarts", 1)),
            seed=int(arguments.seed),
            clip=None,
            support_quantile=(float(getattr(arguments, "support_quantile", 1e-3))
                              if estimator == "mixture_clipped" else None),
        )
    else:
        density, per_observation = shared_marginal(
            shared_draws, grid, prior, getattr(arguments, "kde_bandwidth", None)
        )
    rng = np.random.default_rng(int(arguments.seed))
    # Stage 3 costs ``n * count`` network rows and the *global* posterior never
    # uses it -- the composed density is the tilted prior on the grid and the
    # shared draws come from inverting its CDF. So the locals can be evaluated
    # at fewer draws than are returned for the shared coordinate.
    local_count = int(getattr(arguments, "local_draws", 0) or count)
    count = max(1, min(local_count, count))
    shared = inverse_cdf_sample(grid, density, count, rng)

    # Stage 3: p(l_i | g, x_i) with the shared coordinate *conditioned* at each
    # drawn value -- the third of the six conditionals, weight 7 in training.
    condition_mask = torch.zeros(M.NODES)
    condition_mask[M.HIERARCHY[0]] = 1.0
    condition_mask[M.OBSERVED_INDICES] = 1.0
    # ``Sampler._prepare_data`` scatters ``data`` into the masked positions, so
    # only the conditioned coordinates are passed -- here ``(g, x)``.
    conditioned = torch.stack([
        torch.as_tensor(np.tile(shared, n), dtype=torch.float32),
        torch.as_tensor(np.repeat(x, count), dtype=torch.float32),
    ], dim=1)
    filled = []
    for start in range(0, conditioned.shape[0], chunk):
        block = conditioned[start:start + chunk]
        filled.append(network.sample(
            x=block, condition_mask=condition_mask, num_samples=1, **common
        ).detach().cpu().numpy()[:, 0, :])
    locals_ = np.concatenate(filled, axis=0)
    seconds = time.perf_counter() - started

    if locals_.shape[1] != len(M.LOCAL_INDICES):
        raise AssertionError(
            f"expected {len(M.LOCAL_INDICES)} latent column(s) from "
            f"p(l | g, x), got {locals_.shape[1]}."
        )
    samples = np.zeros((n, count, M.NODES), dtype=np.float64)
    samples[:, :, M.HIERARCHY[0]] = shared[None, :]
    samples[:, :, M.LOCAL_INDICES] = locals_.reshape(n, count, -1)
    samples[:, :, M.OBSERVED_INDICES[0]] = x[:, None]
    return samples, seconds, {"grid": grid, "density": density,
                              "per_observation": per_observation,
                              "shared_draws": shared_draws,
                              "chosen_k": chosen}


def draw(network, x, method, arguments, device, mixture=M.DEFAULT):
    """Dispatch to the right rule. Returns ``(samples, seconds, detail)``."""
    if method == "marginal_product":
        return draw_marginal_product(network, x, arguments, device, mixture)
    if method == "mixture_product":
        return draw_marginal_product(network, x, arguments, device, mixture,
                                     estimator="mixture")
    if method == "mixture_product_bagged":
        return draw_marginal_product(network, x, arguments, device, mixture,
                                     estimator="mixture_bagged")
    if method == "mixture_product_likelihood_bagged":
        return draw_marginal_product(network, x, arguments, device, mixture,
                                     estimator="likelihood_bagged")
    if method in ("tilt_product", "tilt_product_bagged",
                  "tilt_product_bic", "tilt_product_bic_bagged"):
        return draw_marginal_product(
            network, x, arguments, device, mixture,
            estimator=method.replace("_product", "").replace("tilt", "tilt", 1))
    if method == "mixture_product_capped":
        return draw_marginal_product(network, x, arguments, device, mixture,
                                     estimator="mixture_capped")
    if method == "mixture_product_likelihood":
        return draw_marginal_product(network, x, arguments, device, mixture,
                                     estimator="likelihood")
    if method == "mixture_product_clipped":
        return draw_marginal_product(network, x, arguments, device, mixture,
                                     estimator="mixture_clipped")
    samples, seconds = draw_score_rule(network, x, method, arguments, device,
                                       mixture)
    return samples, seconds, {}
