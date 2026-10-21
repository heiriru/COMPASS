"""Grading composed draws against a posterior that is known exactly.

There is no reference noise here -- the target is closed form -- so the only
Monte Carlo left is in the draws being graded. That makes the ``exact_mc`` arm
in :mod:`run_scaling` the honest floor for every number below: it draws the same
number of samples from the true posterior and is graded identically, so any
method within its band is indistinguishable from perfect *at this sample size*,
and a tolerance quoted below that band is meaningless.
"""

import numpy as np

import model as M


def wasserstein(draws, mixture, x, kind="global", index=0):
    """W1 between 1-D draws and the exact marginal, by quantile matching.

    ``kind="global"`` grades ``p(g | x_1..n)``; ``kind="local"`` grades
    ``p(l_index | x_1..n)``. Both marginals are two-component Gaussian mixtures
    with closed-form CDFs, so the quantiles are inverted on a fine grid rather
    than estimated.
    """
    draws = np.sort(np.asarray(draws, dtype=np.float64))
    probabilities = (np.arange(len(draws)) + 0.5) / len(draws)
    grid = mixture.default_grid(x, points=40_001)
    if kind == "global":
        cdf = mixture.global_cdf(grid, x)
    elif kind == "local":
        cdf = mixture.local_cdf(grid, x, index)
    else:
        raise ValueError(f"unknown marginal {kind!r}")
    # The CDF is monotone but numerically flat in the tails; keep the strictly
    # increasing part so np.interp inverts it rather than collapsing a plateau.
    keep = np.concatenate([[True], np.diff(cdf) > 0])
    quantiles = np.interp(probabilities, cdf[keep], grid[keep])
    return float(np.abs(draws - quantiles).mean())


def sample_exact(mixture, x, count, rng):
    """``count`` draws of ``(g, l_1..l_n)`` from the exact tall posterior."""
    x = np.asarray(x, dtype=np.float64).ravel()
    weights, _, _ = mixture.global_posterior(x)
    covariance = mixture.branch_covariance(len(x))
    branch = (rng.random(int(count)) < weights[1]).astype(int)
    draws = np.empty((int(count), len(x) + 1))
    for index in (0, 1):
        rows = branch == index
        if rows.any():
            draws[rows] = rng.multivariate_normal(
                mixture.branch_mean_vector(x, index), covariance, int(rows.sum())
            )
    return draws


def evaluate(samples, mixture, x, seconds=float("nan")):
    """Every score for one (method, dataset) pair.

    ``samples`` is ``(n, draws, NODES)`` as returned by :mod:`methods`.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    samples = np.asarray(samples, dtype=np.float64)
    shared = samples[0, :, M.HIERARCHY[0]]
    mean, sd = mixture.global_moments(x)
    weights, means, _ = mixture.global_posterior(x)

    # Branch mass: the draws' mass on the positive branch against the exact
    # weight. The midpoint between the two branch means is the decision
    # boundary; with a symmetric target it is zero, but it is not in general.
    boundary = 0.5 * (means[0] + means[1])
    branch_mass = float((shared > boundary).mean())

    local_w1 = [
        wasserstein(samples[index, :, M.LOCAL_INDICES[0]], mixture, x,
                    kind="local", index=index) / mixture.local_moments(x, index)[1]
        for index in range(len(x))
    ]
    return {
        "w1_over_sigma": wasserstein(shared, mixture, x) / sd,
        "mean_error_sigma": float((shared.mean() - mean) / sd),
        "width_ratio": float(shared.std() / sd),
        "branch_mass": branch_mass,
        "branch_mass_error": float(branch_mass - weights[1]),
        "reference_branch_mass": float(weights[1]),
        "local_w1_over_sigma": float(np.mean(local_w1)),
        "local_w1_worst": float(np.max(local_w1)),
        "posterior_mean": float(shared.mean()),
        "posterior_std": float(shared.std()),
        "reference_mean": mean,
        "reference_std": sd,
        "seconds": float(seconds),
    }


def summarise(values):
    """Median and interquartile range over replicates."""
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(values):
        return {"median": float("nan"), "q25": float("nan"),
                "q75": float("nan"), "n": 0}
    return {"median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)), "n": int(len(values))}
