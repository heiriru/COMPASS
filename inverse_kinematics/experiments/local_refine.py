"""Local refinement: compose the shared marginal where the posterior lives.

STATUS: measured 2026-09-15 and **kept as a negative result**. The premise below
-- that each factor is much wider than the composed posterior, so a narrow window
localises the fit -- does not hold on either benchmark: the composed sd is only
``2.84/sqrt(n)`` times the factor sd, so a window wide enough to contain draws is
never narrow relative to a factor and the moment deconvolution is
ill-conditioned. On inverse kinematics it gives 0.106/0.122/0.186 at n = 8/16/32
against the exponential tilt's 0.063/0.171/0.161. Use
``tilt_fit.py`` instead; see ``docs/exponential_tilt_composition.md``.

Why this exists
---------------
The exact factorisation of a one-block hierarchy is

    log p(g | x_1..n) = sum_j log p(g | x_j) - (n - 1) log p(g) + const,

so the composed log-density is a **sum of n estimated curves**. Whatever error
each estimated factor carries is therefore multiplied by n, while the composed
posterior's own width shrinks like 1/sqrt(n). In units of the composed sd a
systematic per-factor error of delta nats grows like n^1.5 and a random one like
n. That -- not the sampler, not the network -- is why every grid product
measured so far degrades monotonically in n.

The way out is not a better global density estimate. It is to notice that the
product only ever needs each factor **where the composed posterior has mass**,
and that region shrinks as the problem gets easier. A factor estimated to 0.01
nats over a window of width sigma_1/sqrt(n) is all the composition asks for,
even if the same estimator is hopeless in that factor's tails.

So: locate the modes on a cheap first pass, then re-estimate every factor
*locally* around each mode and compose in closed form.

The local estimator
-------------------
Near a mode, tilt each observation's draws by a Gaussian kernel of width ``h``
centred on the mode. If ``p(g | x_j)`` is locally ``N(mu_j, s_j^2)``, the tilted
sample is ``N(., .)`` with precision ``1/s_j^2 + 1/h^2``, so the two tilted
moments invert exactly:

    1/s_j^2 = 1/v_j - 1/h^2,      mu_j = s_j^2 (m_j/v_j - g0/h^2).

That is a *nonparametric* local fit -- no EM, no component count, no kernel
density estimate, no family assumption beyond local quadratic behaviour of the
log-density, which is exactly what the Bernstein-von Mises regime guarantees.
It costs two weighted moments per observation per mode: a couple of vectorised
passes over draws that were already taken.

The prior's ``(n - 1)`` power is then applied to its *local* quadratic, which is
the second reason this works. The measured high-n failure of the grid product on
the Gaussian-mixtures benchmark is a spike in the prior's valley, built by
``-(n-1) log p(g)`` lifting fitted-Gaussian tails in a region no observation
constrains. A local expansion never evaluates a factor there at all, so that
failure mode is removed by construction rather than clipped or masked.

What is refined and what is not
-------------------------------
Each mode's *location* and *width* come from the local fits. Its *mass* comes
from the first pass, because a mass is an integral over a region and no local
expansion can supply one. This is deliberately conservative: on a unimodal
target the refinement is the whole answer, and on a multimodal one it cannot be
worse than the pass it refines.
"""

import numpy as np

FLOOR = 1e-300
LOG_FLOOR = -700.0


def _weighted_quadratic(centres, values, weights, origin):
    """``a + b (g - origin) + c (g - origin)^2`` by weighted least squares."""
    shifted = centres - origin
    design = np.stack([np.ones_like(shifted), shifted, shifted ** 2], axis=1)
    root = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design * root, values * root[:, 0], rcond=None)
    return solution


def local_gaussian(draws, centre, bandwidth, min_ess=50.0):
    """Per-observation local ``(precision, mean)`` by kernel-tilted moments.

    ``draws`` is ``(n, samples)``. Returns ``(precision, mean, ess)`` each of
    length ``n``; a factor that is locally flat, or has too few draws in the
    window to say anything, comes back with precision ``0`` and is then simply
    absent from the composition -- the honest answer for an observation that
    does not resolve the window.
    """
    draws = np.asarray(draws, dtype=np.float64)
    tilt = 1.0 / bandwidth ** 2
    log_weight = -0.5 * ((draws - centre) / bandwidth) ** 2
    log_weight -= log_weight.max(axis=1, keepdims=True)
    weight = np.exp(log_weight)
    total = weight.sum(axis=1)
    ess = total ** 2 / np.maximum((weight ** 2).sum(axis=1), FLOOR)
    first = (weight * draws).sum(axis=1) / np.maximum(total, FLOOR)
    second = (weight * draws ** 2).sum(axis=1) / np.maximum(total, FLOOR)
    variance = np.maximum(second - first ** 2, FLOOR)

    precision = 1.0 / variance - tilt
    usable = (precision > 1e-8 * tilt) & (ess >= min_ess)
    mean = np.where(usable,
                    (first / variance - centre * tilt)
                    / np.where(usable, precision, 1.0),
                    centre)
    return np.where(usable, precision, 0.0), mean, ess


def _modes(grid, density, min_relative=1e-3):
    """Interior local maxima of ``density`` above ``min_relative`` of the peak."""
    density = np.asarray(density, dtype=np.float64)
    peak = density.max()
    if not np.isfinite(peak) or peak <= 0.0:
        return np.array([int(np.argmax(density))])
    interior = np.arange(1, len(grid) - 1)
    rising = density[interior] >= density[interior - 1]
    falling = density[interior] >= density[interior + 1]
    tall = density[interior] >= min_relative * peak
    found = interior[rising & falling & tall]
    if len(found) == 0:
        return np.array([int(np.argmax(density))])
    return found


def _basins(grid, density, mode_indices):
    """Split the grid at the minima between consecutive modes."""
    edges = [0]
    for left, right in zip(mode_indices[:-1], mode_indices[1:]):
        edges.append(int(left + np.argmin(density[left:right + 1])))
    edges.append(len(grid) - 1)
    return list(zip(edges[:-1], edges[1:]))


def refine(draws, grid, log_prior, pass1_density, *, inflate=4.0,
           iterations=3, min_relative=1e-3, min_ess=50.0, level="mass",
           bag=0, seed=0, floor_scale=None):
    """A composed density rebuilt from local expansions at ``pass1``'s modes.

    ``level`` is how each mode's mass is set: ``"mass"`` preserves the first
    pass's basin mass, ``"peak"`` preserves its peak height and lets the refined
    width change the mass (a Laplace correction, which is right if the peak
    heights are better estimated than the widths -- measured, not assumed).

    ``bag`` bootstraps the *observations* inside the precision sums, which is
    where the per-observation error enters: a replicate re-pools ``n`` local
    fits drawn with replacement, at no extra estimation cost at all.
    """
    draws = np.asarray(draws, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n = draws.shape[0]
    log_pass1 = np.log(np.maximum(pass1_density, FLOOR))
    mode_indices = _modes(grid, pass1_density, min_relative)
    basins = _basins(grid, pass1_density, mode_indices)
    rng = np.random.default_rng(int(seed))

    spread = float(np.median(draws.std(axis=1)))
    if floor_scale is None:
        floor_scale = 0.02 * spread

    records, pieces = [], []
    for index, basin in zip(mode_indices, basins):
        centre = float(grid[index])
        width = _pass1_width(grid, log_pass1, index, spread)
        for _ in range(int(iterations)):
            bandwidth = max(inflate * width, floor_scale)
            precision, mean, ess = local_gaussian(draws, centre, bandwidth,
                                                  min_ess=min_ess)
            prior_precision, prior_mean = _local_prior(grid, log_prior, centre,
                                                       bandwidth)
            total_precision = (precision.sum()
                               - (n - 1) * prior_precision)
            total_h = ((precision * mean).sum()
                       - (n - 1) * prior_precision * prior_mean)
            if total_precision <= 0.0:
                total_precision, total_h = None, None
                break
            new_centre = total_h / total_precision
            new_width = 1.0 / np.sqrt(total_precision)
            # Keep the window anchored to the data: a Newton step that leaves
            # every observation's draws behind is a step into extrapolation.
            if abs(new_centre - centre) > 3.0 * bandwidth:
                total_precision, total_h = None, None
                break
            centre, width = float(new_centre), float(new_width)
        if total_precision is None:
            # Refinement declined this mode; keep the first pass's shape there.
            pieces.append(np.where((grid >= grid[basin[0]])
                                   & (grid <= grid[basin[1]]),
                                   pass1_density, 0.0))
            records.append({"centre": float(grid[index]), "refined": False,
                            "precision": float("nan"),
                            "used": int(0)})
            continue

        if bag:
            centres, precisions = [], []
            for _ in range(int(bag)):
                picked = rng.integers(0, n, size=n)
                p_sum = precision[picked].sum() - (n - 1) * prior_precision
                h_sum = ((precision * mean)[picked].sum()
                         - (n - 1) * prior_precision * prior_mean)
                if p_sum > 0.0:
                    centres.append(h_sum / p_sum)
                    precisions.append(p_sum)
            if centres:
                component = np.zeros_like(grid)
                for c, p in zip(centres, precisions):
                    component += np.exp(-0.5 * p * (grid - c) ** 2) * np.sqrt(p)
                component /= len(centres)
            else:
                component = np.exp(-0.5 * total_precision
                                   * (grid - centre) ** 2)
        else:
            component = np.exp(-0.5 * total_precision * (grid - centre) ** 2)

        mass = _basin_mass(grid, pass1_density, basin)
        if level == "peak":
            mass = (np.exp(log_pass1[index])
                    * np.sqrt(2.0 * np.pi / total_precision))
        area = np.trapezoid(component, grid)
        if area <= 0.0 or not np.isfinite(area):
            pieces.append(np.where((grid >= grid[basin[0]])
                                   & (grid <= grid[basin[1]]),
                                   pass1_density, 0.0))
            records.append({"centre": centre, "refined": False,
                            "precision": float(total_precision),
                            "used": int((precision > 0).sum())})
            continue
        pieces.append(mass * component / area)
        records.append({"centre": centre, "refined": True,
                        "precision": float(total_precision),
                        "sd": float(1.0 / np.sqrt(total_precision)),
                        "mass": float(mass),
                        "used": int((precision > 0).sum())})

    density = np.sum(pieces, axis=0)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        return np.asarray(pass1_density, dtype=np.float64), {"modes": records,
                                                             "fell_back": True}
    return density / area, {"modes": records, "fell_back": False}


def _pass1_width(grid, log_density, index, fallback):
    """A starting scale for the window: the first pass's own local curvature."""
    left = max(index - 3, 0)
    right = min(index + 4, len(grid))
    if right - left < 3:
        return fallback
    solution = _weighted_quadratic(grid[left:right], log_density[left:right],
                                   np.ones(right - left), grid[index])
    curvature = -2.0 * solution[2]
    if curvature > 0.0 and np.isfinite(curvature):
        return float(min(1.0 / np.sqrt(curvature), fallback))
    return float(fallback)


def _local_prior(grid, log_prior, centre, bandwidth):
    """The prior's local ``(precision, mean)`` by a kernel-weighted quadratic.

    Fitted the same way as the factors so the ``(n - 1)`` subtraction is between
    two objects of the same kind. Where the prior is locally flat or convex the
    precision is clamped to zero, which makes the subtraction a no-op rather
    than a source of negative total precision.
    """
    weight = np.exp(-0.5 * ((grid - centre) / bandwidth) ** 2)
    keep = weight > 1e-6
    if keep.sum() < 5:
        return 0.0, centre
    solution = _weighted_quadratic(grid[keep], log_prior[keep], weight[keep],
                                   centre)
    curvature = -2.0 * solution[2]
    if not np.isfinite(curvature) or curvature <= 0.0:
        return 0.0, centre
    mean = centre + solution[1] / curvature
    return float(curvature), float(mean)


def _basin_mass(grid, density, basin):
    low, high = basin
    return float(np.trapezoid(density[low:high + 1], grid[low:high + 1]))
