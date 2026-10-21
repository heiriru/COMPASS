#!/usr/bin/env python3
"""Compose the shared posterior by an exact product, not by correcting scores.

For a hierarchical model with one shared block ``a`` and per-observation local
latents ``l_i``, the joint posterior factorises exactly::

    p(a, l_1..l_n | x_1..x_n) = p(a | x_1..x_n) * prod_i p(l_i | a, x_i)

and the shared marginal itself factorises exactly::

    p(a | x_1..x_n)  proportional to  p(a)^(1-n) * prod_i p(a | x_i)

Neither line is an approximation. Both conditionals on the right are ones the
network is trained to produce directly -- ``p(theta | x)`` is the ordinary
posterior, and ``p(l | a, x)`` is one of the six masks in the ``six_weighted``
menu -- so the whole composed posterior can be assembled from **two ordinary
amortized sampler calls**, with no composed score, no Gaussian summary of any
observation, and no correction rule.

What this buys, and what it costs
---------------------------------
The Gaussian composition rules summarise each observation's contribution by a
mean and a covariance. A single-observation posterior with two branches of
unequal weight is exactly what that summary cannot carry, so the rules return a
near-symmetric answer: measured on the arm at ``|x1| < 0.15``, the reference
puts 0.24 of its mass on the positive branch and ``gauss_jacobian`` returns
0.47. The product below carries the branch weights because it never summarises
anything -- it multiplies the densities themselves.

The cost is the density estimate. ``p(a | x_i)`` is tabulated on a grid, so this
is practical while the shared block is low-dimensional (one to three
coordinates); beyond that the grid is the binding constraint and a score-based
rule is the right tool again. On this benchmark the shared block is
one-dimensional, which is also true of the partial-pooling global block.

The KDE is the one approximation that can bite: a product of ``n`` densities
raises any smoothing bias to the n-th power. ``composition_attribution.py``
sweeps the bandwidth for exactly this reason; do not skip it when moving to a
new problem.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

import hierarchical as H  # noqa: E402

FLOOR = 1e-300


def _normalise(grid, density):
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("shared-marginal product did not normalise")
    return density / area


def shared_marginal(draws, grid, prior, bandwidth=None):
    """``p(a | x_1..n)`` on ``grid`` from per-observation draws of ``a``.

    ``draws`` is ``(n, samples)``: observation ``i``'s draws of the shared
    coordinate, from an ordinary single-observation posterior sample.
    """
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


def joint_shared_score(model, states, device, eps=1e-3, chunk=32768):
    """``d/da log p_t(a, l | x)`` at ``t = eps``, one value per row of ``states``.

    ``states`` are full node vectors with the observables already in place. The
    VESDE has ``alpha_t = 1``, so the sampler's y-space and data space coincide
    and no rescaling is needed here; this mirrors ``Sampler._get_score`` for a
    single-observation condition mask.
    """
    mask = H.CONDITION_MASK.to(device)
    values = []
    with torch.no_grad():
        for start in range(0, states.shape[0], chunk):
            block = states[start:start + chunk].to(device)
            t = torch.full((block.shape[0], 1), float(eps), device=device)
            raw = model.model(x=block, t=t,
                              c=mask.expand(block.shape[0], H.NODES))
            score = model.output_scale_function(t, raw)
            values.append(score[:, H.HIERARCHY[0]].detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def network_prior_score(model, grid, device, noise_scale=0.05, count=100_000,
                        bandwidth=None, support_floor=5.0, seed=0):
    """``d/da log p_sigma(a)`` from the network itself, by the same estimator.

    The composed score subtracts ``(n-1)`` copies of the prior score, which at
    n = 8 is seven times a term of the same size as the ones it is subtracted
    from. That cancellation is only exact if the per-observation scores carry
    the prior the way the analytic formula does. They carry it the way the
    *network* does, so the analytic prior leaves a residual proportional to the
    network's own score bias, multiplied by ``n - 1``. Estimating the prior
    score through the identical path -- same network, same noise scale, same
    regression -- makes any common multiplicative bias cancel instead.
    """
    draws = model.sample(
        num_samples=int(count), timesteps=100, method="dpm", order=2,
        churn_eta=2.0, corrector_steps=0, final_corrector_steps=0,
        terminal_denoise=True, device=device, verbose=False,
    ).detach().cpu().numpy().reshape(-1, H.NODES).astype(np.float64)
    rng = np.random.default_rng(seed)
    if noise_scale > 0.0:
        draws = draws + rng.normal(0.0, noise_scale, draws.shape)
        t = float(model.sde.time_of_sigma(torch.tensor(float(noise_scale))))
    else:
        t = 1e-3
    # Nothing is conditioned: this is the model's own joint p(a, l, x).
    states = torch.as_tensor(draws, dtype=torch.float32)
    values = []
    with torch.no_grad():
        for start in range(0, states.shape[0], 32768):
            block = states[start:start + 32768].to(device)
            time = torch.full((block.shape[0], 1), t, device=device)
            raw = model.model(x=block, t=time,
                              c=torch.zeros_like(block))
            values.append(model.output_scale_function(time, raw)[
                :, H.HIERARCHY[0]].detach().cpu().numpy())
    scores = np.concatenate(values).astype(np.float64)
    shared = draws[:, H.HIERARCHY[0]]
    if bandwidth is None:
        bandwidth = float(shared.std()) * len(shared) ** (-1.0 / 5.0)
    weights = np.exp(-0.5 * ((grid[:, None] - shared[None, :]) / bandwidth) ** 2)
    mass = weights.sum(axis=1)
    regression = np.divide(weights @ scores, mass,
                           out=np.zeros_like(grid), where=mass > 0)
    analytic = -grid / (float(H.GLOBAL_PRIOR_STD[0]) ** 2 + noise_scale ** 2)
    return np.where(mass > support_floor, regression, analytic)


def shared_score_field(model, single, observations, grid, device,
                       noise_scale=0.05, bandwidth=None, support_floor=5.0,
                       prior_score=None):
    """The composed shared score on ``grid``, with no density estimate anywhere.

    Fisher's identity gives the marginal score as a conditional expectation of
    the joint score::

        d/da log p(a | x_i) = E_{l ~ p(l | a, x_i)} [ d/da log p(a, l | x_i) ]

    and the draws from ``p(a, l | x_i)`` already in hand are, conditioned on
    their own ``a``, draws from ``p(l | a, x_i)``. So the marginal score is a
    Nadaraya-Watson regression of the network's own joint score on ``a`` --
    a smooth conditional mean, not a density.

    That matters because the composed object is a *sum* of ``n`` of these. A
    kernel density estimate enters the product as a log, where smoothing bias is
    raised to the n-th power and the tails are worst; a regression of the score
    has no such term, and its error averages down in the number of draws.

    Where an observation's draws do not reach a grid point there is nothing to
    regress, so that observation falls back to contributing only the prior --
    conservative, and it keeps the composed density decaying like the prior
    rather than following an extrapolated score.

    ``noise_scale`` is not a tuning knob but a necessity. The identity wants the
    score at zero noise, and the score network is worst exactly there: it
    predicts ``sigma_t * score``, so at ``t = eps`` the returned score is a tiny
    output divided by a tiny sigma. Measured against the numerical derivative of
    a KDE log-density on the same draws, the correlation is **0.579 at sigma=0**
    and **0.976-0.979 at sigma = 0.05-0.10**. So the field is evaluated at a
    small but finite noise scale, on draws diffused to match, which makes it the
    score of a mildly smoothed marginal -- the same kind of object a KDE with a
    comparable bandwidth returns, and the two are therefore composed on equal
    terms.
    """
    n, count = single.shape[0], single.shape[1]
    latents = single.astype(np.float64)
    if noise_scale > 0.0:
        latents = latents + np.random.default_rng(0).normal(
            0.0, noise_scale, latents.shape
        )
        t = float(model.sde.time_of_sigma(torch.tensor(float(noise_scale))))
    else:
        t = 1e-3
    states = torch.zeros(n * count, H.NODES, dtype=torch.float32)
    states[:, H.LATENT_INDICES] = torch.as_tensor(
        latents.reshape(n * count, -1), dtype=torch.float32
    )
    states[:, H.OBSERVED_INDICES] = torch.as_tensor(
        observations, dtype=torch.float32
    ).repeat_interleave(count, dim=0)
    scores = joint_shared_score(model, states, device, t).reshape(n, count)
    shared = latents[:, :, H.HIERARCHY[0]]

    if bandwidth is None:
        # Scott's rule on the pooled draws; the regressand is smooth in ``a``,
        # so this is a conservative (wide) choice rather than a tuned one.
        bandwidth = float(shared.std()) * count ** (-1.0 / 5.0)
    # The prior must be smoothed to the same scale as everything else, or the
    # (n-1) subtraction is taken against a sharper density than the terms it is
    # subtracted from.
    prior_variance = float(H.GLOBAL_PRIOR_STD[0]) ** 2 + float(noise_scale) ** 2
    if prior_score is None:
        prior_score = -grid / prior_variance

    total = np.zeros_like(grid)
    for index in range(n):
        weights = np.exp(-0.5 * ((grid[:, None] - shared[index][None, :])
                                 / bandwidth) ** 2)
        mass = weights.sum(axis=1)
        regression = np.divide(
            weights @ scores[index], mass,
            out=np.zeros_like(grid), where=mass > 0,
        )
        supported = mass > support_floor
        total += np.where(supported, regression, prior_score)
    return total - (n - 1) * prior_score


def density_from_score(grid, score_field):
    """Normalised density whose log-derivative is ``score_field``."""
    log_density = np.concatenate([[0.0], np.cumsum(
        0.5 * (score_field[1:] + score_field[:-1]) * np.diff(grid))])
    log_density -= log_density.max()
    return _normalise(grid, np.exp(log_density))


def inverse_cdf_sample(grid, density, count, rng):
    """Draw from a density tabulated on a grid, by inverting its CDF."""
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    cumulative /= cumulative[-1]
    # Deduplicate: flat regions of the CDF are not invertible, and np.interp
    # would map an entire zero-density interval onto its left edge.
    keep = np.concatenate([[True], np.diff(cumulative) > 0])
    return np.interp(rng.random(count), cumulative[keep], grid[keep])


def prior_density(grid):
    scale = float(H.GLOBAL_PRIOR_STD[0])
    return _normalise(grid, np.exp(-0.5 * (grid / scale) ** 2))


def draw(model, data, arguments, device, grid=None, chunk=8192,
         estimator=None):
    """Joint draws ``(n, samples, nodes)`` with the shared coordinate composed.

    Matches ``run_inference.draw``'s return shape so the existing graders can
    consume it unchanged.
    """
    observations = torch.as_tensor(data, dtype=torch.float32)
    n = int(observations.shape[0])
    count = int(arguments.draws)
    if grid is None:
        span = 4.0 * H.GLOBAL_PRIOR_STD[0]
        grid = np.linspace(-span, span, 801)
    prior = prior_density(grid)
    single_count = int(getattr(arguments, "single_draws", 20_000))
    bandwidth = getattr(arguments, "kde_bandwidth", None)
    common = dict(
        timesteps=int(arguments.timesteps), method="dpm", order=2,
        churn_eta=float(arguments.churn_eta), corrector_steps=0,
        final_corrector_steps=0, terminal_denoise=True,
        device=device, verbose=False,
    )
    # Stage 1 and stage 3 want *different* churn, which the single shared value
    # inherited from the composition rules cannot express. Measured at n = 32
    # with 6,000 stage-1 draws: churn 0 / 2 / 4 gives shared W1 0.156 / 0.203 /
    # 0.277 but endpoint residual 0.0246 / 0.0188 / 0.0183. Churn injects noise
    # and re-denoises, which perturbs the single-observation marginals the tilt
    # reads its two moments from -- so stage 1 is better without it -- while
    # stage 3 draws the locals conditioned on one shared value and is better
    # with it. ``stage1_churn`` defaults to ``churn_eta``, so nothing changes
    # unless it is set.
    stage1_common = dict(common)
    stage1_common["churn_eta"] = float(
        getattr(arguments, "stage1_churn", None)
        if getattr(arguments, "stage1_churn", None) is not None
        else arguments.churn_eta)

    # Stage 1: every single-observation posterior.
    #
    # Chunked over *observations*. They are independent given their own
    # observation, so splitting the batch changes nothing about what is
    # computed, but peak memory becomes (block * draws) rows of transformer
    # activations instead of (n * draws) -- which is what stops this stage from
    # being the thing that caps the draw budget at large n. Each block is seeded
    # from the run seed and its own index so a re-run reproduces the draws.
    block = int(getattr(arguments, "stage1_chunk", 16) or n)
    pieces = []
    for start in range(0, n, block):
        torch.manual_seed(int(arguments.seed) + 7919 * start)
        torch.cuda.manual_seed_all(int(arguments.seed) + 7919 * start)
        pieces.append(model.sample(
            x=observations[start:start + block], num_samples=single_count,
            **stage1_common
        ).detach().cpu().numpy())
    single = np.concatenate(pieces, axis=0)
    shared_draws = single[:, :, H.HIERARCHY[0]].astype(np.float64)

    # Stage 2: the shared marginal, by one of two estimators, then draws from it.
    estimator = estimator or getattr(arguments, "estimator", "density")
    per_observation = None
    if estimator == "score":
        noise_scale = float(getattr(arguments, "score_noise_scale", 0.05))
        prior_score = None
        if getattr(arguments, "score_prior", "analytic") == "network":
            prior_score = network_prior_score(
                model, grid, device, noise_scale=noise_scale,
            )
        field = shared_score_field(
            model, single, observations.detach().cpu().numpy(), grid, device,
            noise_scale=noise_scale,
            bandwidth=getattr(arguments, "score_bandwidth", None),
            prior_score=prior_score,
        )
        density = density_from_score(grid, field)
    elif estimator == "density":
        density, per_observation = shared_marginal(
            shared_draws, grid, prior, bandwidth
        )
    elif estimator == "mixture":
        # The Gaussian-mixture product with K inferred per observation and the
        # cancellation guard applied. Grid-free in principle: the grid below is
        # only used to hand the same array shape to the existing graders.
        from mixture_inference import infer as mixture_infer
        density, mixture_diagnostics = mixture_infer(
            shared_draws, grid, float(H.GLOBAL_PRIOR_STD[0]),
            k_max=int(getattr(arguments, "k_max", 5)),
            beam=int(getattr(arguments, "beam", 64)),
            restarts=int(getattr(arguments, "em_restarts", 1)),
            seed=int(arguments.seed),
            device=getattr(arguments, "em_device", None),
            bag=int(getattr(arguments, "bag", 0)),
        )
    elif estimator.startswith("tilt"):
        # Compose exponential tilts of the shared prior instead of estimated
        # posteriors: see ``tilt_fit`` for why that is the object to estimate.
        # Here the shared prior is a single Gaussian, so a degree-2 tilt is a
        # Gaussian and a degree-4 one is a double well -- which is the shape an
        # endpoint constraint actually induces in the shared angle, so the
        # degree is chosen per observation by BIC rather than fixed.
        import tilt_fit

        log_prior = np.log(np.maximum(prior, 1e-300))
        degrees = tuple(int(v) for v in
                        str(getattr(arguments, "tilt_degrees", "2,4,6")).split(","))
        thetas, tilt_degree, centre, scale, spread = tilt_fit.fit_all(
            shared_draws, grid, log_prior, degrees=degrees)
        # A separate knob from ``--bag``: that one switches bagging on inside
        # the mixture estimator, and an arm meant to isolate the tilt must not
        # silently change the arm it is being compared against.
        if "local" in estimator:
            # Two passes. The first tilt locates the composed posterior; the
            # second refits every factor with that density attached to both the
            # base measure and the empirical distribution, which leaves the fit
            # consistent but concentrates it where the composed answer actually
            # lives. At large n a factor only has to be right over a window of
            # width sigma/sqrt(n), and a fit that spends its degrees of freedom
            # on the rest of that factor is wasting them.
            first = tilt_fit.compose(thetas, grid, log_prior, centre, scale)
            weight = np.log(np.maximum(first, 1e-300))
            inflate = float(getattr(arguments, "tilt_localise", 1.0))
            thetas, tilt_degree, centre, scale, spread = tilt_fit.fit_all(
                shared_draws, grid, log_prior, degrees=degrees,
                log_weight=inflate * weight, centre=centre, scale=scale)
        bag = int(getattr(arguments, "tilt_bag", 64)) if estimator.endswith("_bagged") else 0
        # The plug-in composition treats n maximum-likelihood fits as exact and
        # is over-confident as a result; ``--tilt-uncertainty`` marginalises over
        # their sampling covariance instead. Off by default so the switch is
        # visible in a comparison.
        if getattr(arguments, "tilt_uncertainty", False):
            density = tilt_fit.compose_uncertain(
                thetas, grid, log_prior, centre, scale, spread,
                replicates=int(getattr(arguments, "tilt_uncertainty_draws", 64)),
                seed=int(arguments.seed))
        elif bag:
            density = tilt_fit.compose_bagged(thetas, grid, log_prior, centre,
                                              scale, replicates=bag,
                                              seed=int(arguments.seed))
        else:
            density = tilt_fit.compose(thetas, grid, log_prior, centre, scale)
        mixture_diagnostics = {"tilt_degree": [int(v) for v in tilt_degree],
                               "chosen_k": [int(v) for v in tilt_degree],
                               "cancellation_ratio": float("nan"),
                               "outside_support": float("nan")}
    else:
        raise ValueError(f"unknown estimator {estimator!r}")
    if per_observation is None:
        per_observation = np.empty((0, len(grid)))
    rng = np.random.default_rng(int(arguments.seed))
    # Stage 3 is the wall clock: it pushes ``n * count`` rows through the network
    # (64,000 at n = 32 with 2,000 draws), where stage 1 pushes ``n * S`` once
    # and the fit is free. And the *global* posterior does not use stage 3 at
    # all -- it is the tilted prior on the grid, and the shared draws come from
    # inverting its CDF, which costs nothing. So the number of draws the locals
    # are evaluated at can be decoupled from the number of shared draws
    # returned: ``local_draws`` caps the former. The locals' own marginals and
    # the endpoint residual need a few hundred, not a few thousand.
    local_count = int(getattr(arguments, "local_draws", 0) or count)
    local_count = max(1, min(local_count, count))
    count = local_count
    shared = inverse_cdf_sample(grid, density, count, rng)

    # Stage 3, fast path: reuse stage 1's draws instead of calling the network
    # again. Stage 1 already produced joint draws (a, l) ~ p(a, l | x_i); what
    # stage 3 needs is l_i ~ p(l_i | a*, x_i). Those are the same draws,
    # conditioned -- so a Nadaraya-Watson resampler over the stage-1 draws of
    # observation i, with a kernel on |a - a*|, answers it with no network call
    # at all. Stage 3 is 55% of this pipeline's wall clock, so removing it is
    # the difference between being slower and faster than Langevin F-NPSE.
    #
    # The approximation is a kernel width: with ``single_draws`` draws and a
    # one-dimensional conditioning variable there are many draws within any
    # sensible bandwidth, and the endpoint residual measures directly whether
    # the resulting locals still reproduce the data.
    if getattr(arguments, "local_estimator", "sampler") == "resample":
        rng_local = np.random.default_rng(int(arguments.seed) + 1)
        bandwidth = float(getattr(arguments, "resample_bandwidth", 0.0)) or (
            0.1 * float(np.std(shared_draws))
        )
        locals_ = np.empty((n, count, len(H.LOCAL_INDICES)), dtype=np.float64)
        for row in range(n):
            pool_shared = shared_draws[row]
            pool_local = single[row][:, H.LOCAL_INDICES].astype(np.float64)
            # One kernel-weighted pick per requested draw, in blocks so the
            # (count x single_draws) weight matrix never materialises whole.
            block = max(1, int(4_000_000 // max(len(pool_shared), 1)))
            for start in range(0, count, block):
                target = shared[start:start + block]
                logits = -0.5 * ((target[:, None] - pool_shared[None, :])
                                 / bandwidth) ** 2
                logits -= logits.max(axis=1, keepdims=True)
                weights = np.exp(logits)
                weights /= weights.sum(axis=1, keepdims=True)
                cumulative = np.cumsum(weights, axis=1)
                picks = (cumulative < rng_local.random((len(target), 1))).sum(axis=1)
                locals_[row, start:start + len(target)] = pool_local[
                    np.clip(picks, 0, len(pool_shared) - 1)]
        samples = np.zeros((n, count, H.NODES), dtype=np.float64)
        samples[:, :, H.HIERARCHY[0]] = shared[None, :]
        samples[:, :, H.LOCAL_INDICES] = locals_
        samples[:, :, H.OBSERVED_INDICES] = (
            observations.detach().cpu().numpy()[:, None, :])
        return samples, {"grid": grid, "density": density,
                         "per_observation": per_observation,
                         "shared_draws": shared_draws,
                         "mixture": locals().get("mixture_diagnostics")}

    # Stage 3, stratified: the same conditional, in a batch layout the sampler
    # is actually fast at.
    #
    # The plain path below builds ``n * count`` rows each carrying a distinct
    # ``a*`` and asks for ``num_samples=1`` from each. Measured at n = 8, that
    # layout costs 5.1 s for 32,000 trajectories while stage 1 does 160,000 in
    # 3.6 s -- about 7x worse per trajectory, because the work per call is a
    # long thin batch rather than a short wide one.
    #
    # Stratifying fixes the shape without changing the target: take ``strata``
    # quantiles of the composed shared posterior, draw ``count // strata``
    # locals at each, and the batch becomes ``(n * strata, count // strata)``.
    # Conditional on ``a``, the locals are independent of everything else, so
    # this is stratified sampling of exactly the same joint -- and stratified
    # sampling of a smooth conditional is if anything lower variance than i.i.d.
    strata = int(getattr(arguments, "local_strata", 0) or 0)
    if strata and strata < count:
        per_stratum = int(np.ceil(count / strata))
        probabilities = (np.arange(strata) + 0.5) / strata
        cumulative = np.concatenate([[0.0], np.cumsum(
            0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
        cumulative /= cumulative[-1]
        keep = np.concatenate([[True], np.diff(cumulative) > 0])
        knots = np.interp(probabilities, cumulative[keep], grid[keep])

        condition_mask = torch.zeros(H.NODES)
        condition_mask[H.HIERARCHY[0]] = 1.0
        condition_mask[H.OBSERVED_INDICES] = 1.0
        conditioned = torch.cat([
            torch.as_tensor(np.tile(knots, n), dtype=torch.float32)[:, None],
            observations.repeat_interleave(strata, dim=0),
        ], dim=1)
        drawn = model.sample(
            x=conditioned, condition_mask=condition_mask,
            num_samples=per_stratum, **common
        ).detach().cpu().numpy()
        drawn = drawn.reshape(n, strata * per_stratum, -1)[:, :count, :]
        shared = np.repeat(knots, per_stratum)[:count]
        samples = np.zeros((n, count, H.NODES), dtype=np.float64)
        samples[:, :, H.HIERARCHY[0]] = shared[None, :]
        samples[:, :, H.LOCAL_INDICES] = drawn
        samples[:, :, H.OBSERVED_INDICES] = (
            observations.detach().cpu().numpy()[:, None, :])
        return samples, {"grid": grid, "density": density,
                         "per_observation": per_observation,
                         "shared_draws": shared_draws,
                         "mixture": locals().get("mixture_diagnostics")}

    # Stage 3: p(l_i | a, x_i), the network's own conditional, with the shared
    # coordinate *conditioned* at each drawn value. One row per (observation,
    # draw); the shared column is identical across observations by construction.
    condition_mask = torch.zeros(H.NODES)
    condition_mask[H.HIERARCHY[0]] = 1.0
    condition_mask[H.OBSERVED_INDICES] = 1.0
    # ``Sampler._prepare_data`` scatters ``data`` into the masked positions, so
    # ``data`` carries only the conditioned coordinates -- here (a, x1, x2) --
    # not a full node vector. Passing all six silently mismatches the shapes.
    conditioned = torch.cat([
        torch.as_tensor(np.tile(shared, n), dtype=torch.float32)[:, None],
        observations.repeat_interleave(count, dim=0),
    ], dim=1)

    filled = []
    for start in range(0, conditioned.shape[0], chunk):
        block = conditioned[start:start + chunk]
        filled.append(model.sample(
            x=block, condition_mask=condition_mask, num_samples=1, **common
        ).detach().cpu().numpy()[:, 0, :])
    locals_ = np.concatenate(filled, axis=0)
    # ``SBIm.sample`` returns the *latent* columns only, so what comes back is
    # (height_arm, angle_2, angle_3) and the full node vector is rebuilt here.
    if locals_.shape[1] != len(H.LOCAL_INDICES):
        raise AssertionError(
            f"expected {len(H.LOCAL_INDICES)} latent columns from the "
            f"conditional, got {locals_.shape[1]}."
        )
    samples = np.zeros((n, count, H.NODES), dtype=np.float64)
    samples[:, :, H.HIERARCHY[0]] = shared[None, :]
    samples[:, :, H.LOCAL_INDICES] = locals_.reshape(n, count, -1)
    samples[:, :, H.OBSERVED_INDICES] = (
        observations.detach().cpu().numpy()[:, None, :]
    )
    return samples, {"grid": grid, "density": density,
                     "per_observation": per_observation,
                     "shared_draws": shared_draws,
                     "mixture": locals().get("mixture_diagnostics")}
