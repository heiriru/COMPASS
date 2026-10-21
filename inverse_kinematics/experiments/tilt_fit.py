"""Compose per-observation *likelihoods* as exponential tilts of the prior.

The problem this solves
-----------------------
Every composition rule measured so far estimates each ``p(g | x_j)`` from its
draws and multiplies. That fails in exactly one situation, and it is a situation
the hard benchmarks are built out of: when the composed posterior sits where no
single-observation posterior has any draws.

It is easy to see why that happens and why it is not a pathology. Take the
Gaussian-mixtures benchmark, prior ``0.5 N(-2.5, 0.4^2) + 0.5 N(2.5, 0.4^2)``.
A single observation barely moves that prior, so ``p(g | x_j)`` has its mass at
``g = +-2.19``. But ``n`` observations do overcome it, and when the data centre
near zero the composed posterior sits at ``g ~ 0`` -- about five sd into the
tail of *every* factor, where a 20,000-draw sample contains zero draws. No
density estimator can recover a log-density to 0.01 nats in a region it has
never seen a point in, and the composition needs exactly that, because the
composed log-density is a sum of ``n`` such numbers.

The observation that fixes it
-----------------------------
The factor whose tail we cannot estimate is the *posterior*. The object the
composition actually needs is the **likelihood ratio**

    L_j(g)  proportional to  p(g | x_j) / p(g),

and a likelihood in the shared parameter is a fundamentally smoother object than
a posterior: it carries none of the prior's structure. On the benchmark above,
``L_j(g)`` is ``N(x_j; g, v)`` -- one broad Gaussian, log-quadratic in ``g``,
completely determined by two numbers -- while ``p(g | x_j)`` is a sharply
bimodal mixture. Estimating the smooth object and keeping the prior exact is
strictly easier than estimating the structured one.

So model the likelihood as a log-polynomial and fit it as an exponential family
whose **base measure is the known prior**:

    q_theta(g)  =  p(g) exp(theta . phi(g)) / Z(theta),     phi(g) = (g, g^2, ..., g^d).

Three things follow, and they are why this is worth doing:

1. **It needs no density estimate.** ``theta`` is the maximum-likelihood fit to
   the draws, and for an exponential family that is pure moment matching:
   ``E_theta[phi] = mean of phi over the draws``. Only the first ``d`` sample
   moments of each observation's draws are ever used. No KDE, no EM, no
   component count, no bandwidth.

2. **The composition is exact and closed-form.** The target factorises as
   ``p(g) prod_j L_j(g)``, and a product of exponential tilts of the same base
   measure is the tilt by the *sum* of the parameters:

       log p(g | x_1..n)  =  log p(g) + sum_j theta_j . phi(g) + const.

   The ``prior^(1-n)`` power is gone -- not clipped, not masked, absent -- and
   with it the spike in the prior's valley that it used to build. Composition
   reduces to adding ``n`` short parameter vectors.

3. **On the Gaussian-mixtures benchmark it is consistent, not approximate.**
   There the true likelihood *is* log-quadratic, so ``d = 2`` contains the truth
   and the only error left is Monte-Carlo noise in two sample moments.

The cost is two or three reductions over draws that were already taken, plus a
Newton solve in ``d`` unknowns -- convex, with the Hessian equal to minus the
covariance of ``phi``, so it converges in a handful of steps. It is far cheaper
than the EM fit it replaces.

Where the degree matters
------------------------
``d = 2`` says the likelihood is Gaussian in the shared parameter. Higher even
degrees buy genuine multimodality: ``d = 4`` is a double well, which is what an
inverse-kinematics endpoint constraint actually looks like in the shared angle.
:func:`fit_all` selects ``d`` per observation by BIC over a supplied list, so a
benchmark whose likelihood is quadratic is not charged for flexibility it does
not need.
"""

import numpy as np

FLOOR = 1e-300


def design(grid, degree, centre, scale):
    """``phi(g)`` on a grid, in the standardised variable ``(g - centre)/scale``.

    Standardising is not cosmetic: raw powers of ``g`` up to degree 6 give a
    moment matrix with a condition number in the millions, and every observation
    must be fitted in the *same* variable or the parameter vectors could not be
    added.
    """
    u = (np.asarray(grid, dtype=np.float64) - centre) / scale
    return np.stack([u ** power for power in range(1, degree + 1)], axis=1)


def _moments(log_base, features, theta, weights):
    """``(log Z, E[phi], Cov[phi])`` under the tilt ``theta``, by grid quadrature."""
    log_density = log_base + features @ theta
    peak = log_density.max()
    density = np.exp(log_density - peak) * weights
    total = density.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("tilted density did not normalise")
    probability = density / total
    mean = probability @ features
    centred = features - mean
    covariance = (centred * probability[:, None]).T @ centred
    return peak + np.log(total), mean, covariance


def fit(draws, grid, log_prior, degree, centre, scale, *, ridge=1e-6,
        iterations=60, tolerance=1e-10, log_weight=None, sample_weight=None):
    """Maximum-likelihood tilt of the prior that reproduces ``draws``' moments.

    Returns ``(theta, log_likelihood, iterations_used, covariance)``. The
    objective is concave in ``theta``, so Newton with a covariance Hessian is the
    whole algorithm; the ridge only keeps the step finite when a high-degree
    feature is unidentified by the sample.

    ``covariance`` is ``Cov_theta[phi]`` at the solution -- the Fisher
    information per draw. It is returned because the plug-in composition treats
    ``theta_hat`` as if it were exact, and it is not; see :func:`compose`.
    """
    grid = np.asarray(grid, dtype=np.float64)
    weights = np.gradient(grid)          # trapezoid-equivalent quadrature
    features = design(grid, degree, centre, scale)
    sample_features = design(np.asarray(draws, dtype=np.float64), degree,
                             centre, scale)
    # Localisation: attaching the same weight function to the base measure and
    # to the empirical distribution leaves the estimator consistent -- if
    # p_j = p exp(theta . phi)/Z then p_j w is proportional to p w exp(theta . phi),
    # so the weighted fit recovers the same theta -- while concentrating the fit
    # where the weight is large. That is what makes it possible to be accurate
    # where the *composed* posterior lives rather than on average over a factor,
    # which is the whole difficulty at large n.
    base = log_prior if log_weight is None else log_prior + log_weight
    if sample_weight is None:
        target = sample_features.mean(axis=0)
    else:
        omega = np.asarray(sample_weight, dtype=np.float64)
        omega = np.maximum(omega, 0.0)
        total_weight = omega.sum()
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise FloatingPointError("localisation weights vanished")
        target = (omega[:, None] * sample_features).sum(axis=0) / total_weight

    theta = np.zeros(degree, dtype=np.float64)
    log_z, _, _ = _moments(base, features, theta, weights)
    # The concave objective, up to a constant: the average log-likelihood of the
    # draws under the tilt. Newton maximises this.
    objective = float(target @ theta - log_z)
    for step in range(int(iterations)):
        log_z, mean, covariance = _moments(base, features, theta, weights)
        gradient = target - mean
        if np.max(np.abs(gradient)) < tolerance:
            break
        scaled = covariance + ridge * np.eye(degree) * max(
            float(np.trace(covariance)) / degree, 1e-12)
        try:
            direction = np.linalg.solve(scaled, gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(scaled, gradient, rcond=None)[0]
        # Backtracking line search on the objective itself, not merely on
        # ``log_z`` being finite. A finite-but-worse step used to be accepted,
        # and for a factor whose draws sit far from the prior's centre the first
        # Newton step can overshoot enormously: on the Gaussian hierarchy twin
        # one observation out of thirty came back with a fitted precision of
        # 2.8e18 against the 0.497 of its neighbours, and since the composition
        # *sums* the parameters that single factor destroyed the result. A
        # concave objective guarantees a short enough step improves, so the
        # search either improves or stops.
        improved = False
        for shrink in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
            trial = theta + shrink * direction
            try:
                trial_z, *_ = _moments(base, features, trial, weights)
            except FloatingPointError:
                continue
            if not np.isfinite(trial_z):
                continue
            value = float(target @ trial - trial_z)
            if np.isfinite(value) and value > objective:
                theta, log_z, objective = trial, trial_z, value
                improved = True
                break
        if not improved:
            break

    # The attained log-likelihood of the draws, needed for the degree choice.
    base_at_draws = np.interp(np.asarray(draws, dtype=np.float64), grid, base)
    terms = base_at_draws + sample_features @ theta - log_z
    if sample_weight is None:
        log_likelihood = float(terms.sum())
    else:
        omega = np.asarray(sample_weight, dtype=np.float64)
        log_likelihood = float(len(terms) * (omega @ terms) / omega.sum())
    _, _, covariance = _moments(base, features, theta, weights)
    return theta, log_likelihood, step + 1, covariance


def fit_all(draws, grid, log_prior, degrees=(2,), *, ridge=1e-6,
            centre=None, scale=None, log_weight=None):
    """Fit every observation, choosing the degree per observation by BIC.

    ``draws`` is ``(n, samples)``. Returns
    ``(thetas, chosen, centre, scale, uncertainty)`` with ``thetas`` padded to
    ``max(degrees)`` so they can simply be summed -- a lower-degree fit is the
    same model with the high coefficients at zero.

    ``uncertainty`` is the covariance of the *summed* parameter,
    ``sum_j (S * Cov_j)^-1``, padded the same way. Each ``theta_hat_j`` is a
    maximum-likelihood estimate from ``S`` draws with asymptotic covariance
    ``(S * Cov_j)^-1``, and the sum of ``n`` independent estimates has the sum of
    their covariances. :func:`compose` can then stop pretending the fit is
    exact.
    """
    draws = np.asarray(draws, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    # The grid is the quadrature domain for E_theta[phi], so it has to cover the
    # *single-observation* posteriors -- the objects being fitted -- not the
    # composed one. Those are wider by roughly sqrt(n), and a grid sized for the
    # composed posterior truncates them silently: measured on the Gaussian
    # hierarchy twin, whose reference grid spans +-8 composed sd (= +-2.0) while
    # the factors have sd 0.82 and means out to -1.3, the fitted precisions came
    # back at -4e18 and the composed density collapsed onto the grid edge. That
    # is a caller error worth naming rather than a number worth returning.
    inside = ((draws >= grid[0]) & (draws <= grid[-1])).mean()
    if inside < 0.999:
        raise ValueError(
            f"the fit grid [{grid[0]:.3f}, {grid[-1]:.3f}] contains only "
            f"{100.0 * inside:.2f}% of the draws; it must cover the "
            "single-observation posteriors, not just the composed one"
        )
    if centre is None:
        centre = float(draws.mean())
    if scale is None:
        scale = float(draws.std()) or 1.0
    top = max(degrees)
    samples = draws.shape[1]

    thetas, chosen = [], []
    uncertainty = np.zeros((top, top), dtype=np.float64)
    for row in draws:
        best = None
        for degree in degrees:
            try:
                sample_weight = None
                if log_weight is not None:
                    at_draws = np.interp(row, grid, log_weight)
                    sample_weight = np.exp(at_draws - at_draws.max())
                theta, log_likelihood, _, covariance = fit(
                    row, grid, log_prior, degree, centre, scale, ridge=ridge,
                    log_weight=log_weight, sample_weight=sample_weight)
            except (FloatingPointError, np.linalg.LinAlgError):
                continue
            score = degree * np.log(samples) - 2.0 * log_likelihood
            if best is None or score < best[0]:
                best = (score, theta, degree, covariance)
        if best is None:
            thetas.append(np.zeros(top)); chosen.append(0); continue
        padded = np.zeros(top)
        padded[:len(best[1])] = best[1]
        thetas.append(padded)
        chosen.append(best[2])
        degree = best[2]
        # (S * Cov)^-1, pseudo-inverted: at a high degree the moment matrix is
        # near-singular because the sample does not identify the top features,
        # and a pseudo-inverse assigns those directions no uncertainty rather
        # than an enormous one.
        block = np.linalg.pinv(float(samples) * best[3], rcond=1e-10)
        uncertainty[:degree, :degree] += block
    return np.asarray(thetas), chosen, centre, scale, uncertainty


def compose(thetas, grid, log_prior, centre, scale, *, subset=None,
            uncertainty=None):
    """``p(g | x_1..n)`` on ``grid``: the prior tilted by the summed parameters.

    ``uncertainty`` switches off the plug-in assumption -- see
    :func:`compose_uncertain`, which is the *correct* way to do it. Passing a
    covariance here applies the naive analytic marginalisation

        E[exp(theta . phi)] = exp(theta_hat . phi + 0.5 phi^T V phi)

    which is **improper and must not be used**. It is exact pointwise, but
    ``phi`` contains ``u^d``, so the added term grows like ``u^(2d)`` and the
    result is not integrable in ``g``: measured at the degrees BIC selects
    (4 and 8), the correction is a harmless 0.09-0.14 nats across the posterior's
    support and reaches **1.1e7 nats** at the edge of the grid, which after
    exponentiation puts every bit of mass in the far tail. Kept only so the
    mistake is documented; it raises rather than silently misleading.
    """
    if uncertainty is not None:
        raise ValueError(
            "compose(uncertainty=...) is improper -- the phi^T V phi term grows "
            "like u^(2d) and the result does not integrate. Use "
            "compose_uncertain(), which averages normalised densities over "
            "sampled theta instead."
        )
    return _compose_plugin(thetas, grid, log_prior, centre, scale, subset=subset)


def _compose_plugin(thetas, grid, log_prior, centre, scale, *, subset=None):
    """``p(g)exp(theta_hat . phi)`` normalised: the plug-in composition."""
    thetas = np.asarray(thetas, dtype=np.float64)
    if subset is not None:
        thetas = thetas[subset]
    total = thetas.sum(axis=0)
    features = design(grid, len(total), centre, scale)
    log_density = log_prior + features @ total
    log_density -= log_density.max()
    density = np.exp(log_density)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("composed tilt did not normalise")
    return density / area


def compose_uncertain(thetas, grid, log_prior, centre, scale, uncertainty, *,
                      replicates=64, seed=0):
    """Average *normalised* tilted densities over the fit's own uncertainty.

    The plug-in composition evaluates the tilt at the estimate, treating ``n``
    maximum-likelihood fits as exact. They are not, and the result is
    over-confident. The fix has to average densities, not exponentiate an
    average: each ``theta`` drawn from ``N(sum theta_hat, V)`` gives a *proper*
    density, and a mixture of proper densities is proper -- whereas
    ``exp(theta_hat . phi + 0.5 phi^T V phi)`` is not integrable at all (see
    :func:`compose`).

    ``V`` is rank-deficient by construction -- the sample does not identify the
    top polynomial features -- so the square root is taken through an
    eigendecomposition with the non-positive directions dropped, which gives
    those directions no spread rather than an arbitrary one.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    total = thetas.sum(axis=0)
    matrix = np.asarray(uncertainty, dtype=np.float64)[:len(total), :len(total)]
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    keep = values > max(float(values.max()), 0.0) * 1e-8
    root = vectors[:, keep] * np.sqrt(values[keep])
    if root.shape[1] == 0:
        return _compose_plugin(thetas, grid, log_prior, centre, scale)
    rng = np.random.default_rng(int(seed))
    features = design(grid, len(total), centre, scale)
    accumulated = np.zeros(len(grid), dtype=np.float64)
    kept = 0
    for _ in range(int(replicates)):
        sample = total + root @ rng.standard_normal(root.shape[1])
        log_density = log_prior + features @ sample
        log_density -= log_density.max()
        density = np.exp(log_density)
        area = np.trapezoid(density, grid)
        if not np.isfinite(area) or area <= 0.0:
            continue
        accumulated += density / area
        kept += 1
    if kept == 0:
        return _compose_plugin(thetas, grid, log_prior, centre, scale)
    accumulated /= kept
    return accumulated / np.trapezoid(accumulated, grid)


def compose_bagged(thetas, grid, log_prior, centre, scale, *, replicates=64,
                   seed=0, uncertainty=None):
    """Average the composed density over bootstrap resamples of observations.

    In this parameterisation a replicate costs one vector sum and one exponential
    on the grid -- no refit at all -- because a bootstrap over observations is a
    bootstrap over the ``theta_j``.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    n = thetas.shape[0]
    rng = np.random.default_rng(int(seed))
    total = np.zeros(len(grid), dtype=np.float64)
    for _ in range(int(replicates)):
        picked = rng.integers(0, n, size=n)
        total += _compose_plugin(thetas, grid, log_prior, centre, scale,
                                 subset=picked)
    density = total / float(replicates)
    return density / np.trapezoid(density, grid)
