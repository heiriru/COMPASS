"""Bridge *topology* diagnostics: modes, valleys, barriers and branch mass.

The previous oracle experiment graded bridges by endpoint ``W1``.  That is the
wrong instrument for the question left open at ``n = 128``, where every
construction failed: endpoint error says *that* a rule is wrong, not *where* it
went wrong, and the bridge pictures suggested the answer is the shape of the
intermediate distribution rather than its final one.  This module therefore
measures, at every noise level, the four quantities that decide whether a
reverse chain can end up with the right branch mass at all:

``tau_split``
    the first level, moving from the noise end towards the data end, at which a
    single central basin becomes two stable modes separated by a valley;
``m_k(tau)``
    the locations of those modes;
``Delta E_barrier(tau)``
    ``min_k [ log q(m_k) - log q(valley) ]`` -- how deep the wall between the
    branches is in log units, i.e. how hard it is for a sampler to re-balance
    the branches once it has split;
``w_+(tau)``
    the mass the bridge puts to the right of its own valley.

The one-dimensional reduction
-----------------------------
Every field here lives on the tall state ``theta = (g, l_1..l_n)``.  The
reduction to a curve in ``g`` is the *profiled slice* already used by the
previous experiment's bridge pictures: each local is pinned at its exact clean
conditional mode ``l_i(g) = a g + b x_i``, so

    slope(g) = S_g(theta(g)) + a sum_i S_{l_i}(theta(g))

is a genuine directional derivative of the composed field along a
one-dimensional cut, and its integral is that cut's log density up to a
constant.  The slice is used for *every* rule including ``true``, so the arms
are compared on identical footing; :func:`slice_validation` separately reports
how far the ``true`` slice sits from the closed-form diffused marginal, which
bounds the error the construction itself introduces.

Roots are located by sign change on a fine grid and refined with Brent, and
classified by the sign of the slope on either side: ``+ -> -`` is a mode,
``- -> +`` a valley.  Barrier depths and branch masses come from the trapezoid
integral of the slope on the same grid.

The bridge coordinate
---------------------
``tau`` here is **uniform in log lambda**, ``tau = 0`` at ``lambda_min`` and
``tau = 1`` at ``lambda_max``, which is the coordinate the samplers' own
schedules are uniform in (``_schedule`` is a ``logspace`` in lambda).  It is
*not* the coordinate ``ComposedField.tau`` uses internally for ``a(t)`` and
``d(t)``: those are COMPASS diffusion time, where ``a(t) = 1 - t`` is literally
Geffner's ``a = (T - t)/T``.  The two disagree strongly because ``lambda(t)``
is exponential in ``t``.  Nothing in this module changes any bridge's
definition; the log-lambda coordinate is used only to *index* levels and to
parameterise the ``alpha`` correction, which is a sampler-facing object.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import brentq


# A stationary point whose basin carries less than this share of the curve is
# a wrinkle, not a branch.  The choice matters: at 1e-6 a numerically tiny
# shoulder counts as a mode and every barrier statistic downstream becomes
# meaningless, while the spurious central modes the previous experiment found
# carried 8-99% of the mass and survive any floor in this range.
MASS_FLOOR = 1e-3


# ---------------------------------------------------------------------------
# Level indexing
# ---------------------------------------------------------------------------

def lam_of_tau(tau, lam_max, lam_min):
    """``lambda`` at the log-uniform bridge coordinate ``tau in [0, 1]``."""
    return float(math.exp(math.log(lam_min)
                          + float(tau) * (math.log(lam_max) - math.log(lam_min))))


def tau_of_lam(lam, lam_max, lam_min):
    lo, hi = math.log(lam_min), math.log(lam_max)
    return float(min(max((math.log(float(lam)) - lo) / (hi - lo), 0.0), 1.0))


def level_grid(count, lam_max, lam_min):
    """``count`` levels from the noise end to the data end, as ``(tau, lam)``."""
    taus = np.linspace(1.0, 0.0, int(count))
    return [(float(t), lam_of_tau(t, lam_max, lam_min)) for t in taus]


# ---------------------------------------------------------------------------
# The profiled slice
# ---------------------------------------------------------------------------

def slice_grid(oracle, lam, points=20001, pad=6.0):
    """A grid wide enough for both the clean branches and the diffused width."""
    tall = oracle.tall
    within = math.sqrt(tall.global_variance)
    centre = float(tall.global_means.mean())
    reach = (float(tall.global_means.abs().max() - centre)
             + pad * math.sqrt(within ** 2 + float(lam) ** 2))
    reach = max(reach, pad * float(lam))
    return torch.linspace(centre - reach, centre + reach, int(points),
                          dtype=torch.float64)


def slice_slope(oracle, field, grid, t, lam, chunk=4096):
    """``S_g + a sum_i S_{l_i}`` along ``l_i(g) = a g + b x_i``."""
    a, b = oracle.mixture.a, oracle.mixture.b
    x = torch.as_tensor(oracle.x, dtype=torch.float64)
    out = torch.empty_like(grid)
    for start in range(0, grid.numel(), chunk):
        piece = grid[start:start + chunk]
        locals_ = a * piece.unsqueeze(-1) + b * x
        state = torch.cat([piece.unsqueeze(-1), locals_], dim=-1)
        score = field(state, t, lam)
        out[start:start + chunk] = score[:, 0] + a * score[:, 1:].sum(-1)
    return out


def potential_from_slope(grid, slope):
    """Cumulative trapezoid of the slope: the slice log density, max-shifted."""
    step = grid[1] - grid[0]
    potential = torch.cat([
        torch.zeros(1, dtype=torch.float64),
        torch.cumsum(0.5 * (slope[1:] + slope[:-1]) * step, dim=0),
    ])
    return potential - potential.max()


# ---------------------------------------------------------------------------
# Topology of one curve
# ---------------------------------------------------------------------------

def _refine(function, left, right):
    try:
        return float(brentq(function, left, right, xtol=1e-10, rtol=1e-12,
                            maxiter=200))
    except (ValueError, RuntimeError):
        return 0.5 * (left + right)


def stationary_points(grid, slope, function=None, mass_floor=MASS_FLOOR,
                      potential=None):
    """Roots of the slope, refined with Brent and classified.

    ``function`` is the continuous slope, used for the Brent refinement; when
    it is ``None`` the grid bracket midpoint is used instead.  A root is a mode
    when the slope changes ``+ -> -`` and a valley when it changes ``- -> +``.

    Modes are filtered by the mass of their basin: a "mode" carrying less than
    ``mass_floor`` of the curve is a numerical wrinkle in the tail, not a
    branch, and counting it would make every barrier statistic meaningless.
    ``potential`` (if given) supplies the density used for that filter.
    """
    sign = torch.sign(slope)
    changes = torch.nonzero((sign[:-1] * sign[1:]) < 0).reshape(-1)
    roots, kinds = [], []
    for index in changes.tolist():
        left, right = float(grid[index]), float(grid[index + 1])
        root = _refine(function, left, right) if function is not None \
            else 0.5 * (left + right)
        roots.append(root)
        kinds.append("mode" if slope[index] > 0 else "valley")

    if potential is not None and roots:
        density = torch.exp(potential - potential.max())
        step = float(grid[1] - grid[0])
        total = float(density.sum()) * step
        if total > 0:
            keep_roots, keep_kinds = [], []
            positions = torch.as_tensor(roots, dtype=torch.float64)
            edges = [0]
            valleys = [r for r, k in zip(roots, kinds) if k == "valley"]
            for value in valleys:
                edges.append(int(torch.searchsorted(grid.contiguous(),
                                                    torch.tensor(value))))
            edges.append(len(grid))
            basin = 0
            for root, kind in zip(roots, kinds):
                if kind == "valley":
                    keep_roots.append(root)
                    keep_kinds.append(kind)
                    continue
                lo, hi = edges[basin], edges[basin + 1]
                basin += 1
                mass = float(density[lo:hi].sum()) * step / total
                if mass >= mass_floor:
                    keep_roots.append(root)
                    keep_kinds.append(kind)
            # Drop valleys that no longer separate two surviving modes.
            roots, kinds = [], []
            for position, kind in zip(keep_roots, keep_kinds):
                if kind == "valley":
                    before = any(k == "mode" for k in kinds)
                    after = any(k == "mode" and r > position
                                for r, k in zip(keep_roots, keep_kinds))
                    if not (before and after):
                        continue
                roots.append(position)
                kinds.append(kind)
            del positions
    return roots, kinds


def describe(grid, slope, function=None, mass_floor=MASS_FLOOR):
    """Full topology of one slice: modes, valleys, barriers and branch mass."""
    potential = potential_from_slope(grid, slope)
    roots, kinds = stationary_points(grid, slope, function,
                                     mass_floor=mass_floor,
                                     potential=potential)
    modes = [r for r, k in zip(roots, kinds) if k == "mode"]
    valleys = [r for r, k in zip(roots, kinds) if k == "valley"]

    density = torch.exp(potential)
    step = float(grid[1] - grid[0])
    total = float(density.sum()) * step
    normalisable = math.isfinite(total) and total > 0

    def value_at(position):
        index = int(torch.searchsorted(grid.contiguous(),
                                       torch.tensor(float(position))))
        index = min(max(index, 0), len(grid) - 1)
        return float(potential[index])

    result = {
        "num_modes": len(modes),
        "modes": modes,
        "valleys": valleys,
        "normalisable": bool(normalisable),
    }

    if len(modes) >= 2 and valleys:
        # The *principal* valley: the deepest one separating the outermost
        # pair of modes, which is the branch split the sampler has to cross.
        left, right = modes[0], modes[-1]
        inner = [v for v in valleys if left < v < right]
        split = min(inner, key=value_at) if inner else 0.5 * (left + right)
        depth_left = value_at(left) - value_at(split)
        depth_right = value_at(right) - value_at(split)
        result.update({
            "split_point": split,
            "mode_minus": left, "mode_plus": right,
            "barrier_minus": depth_left, "barrier_plus": depth_right,
            "barrier": min(depth_left, depth_right),
        })
        if normalisable:
            cut = int(torch.searchsorted(grid.contiguous(),
                                         torch.tensor(float(split))))
            below = float(density[:cut].sum()) * step / total
            result["w_minus"] = below
            result["w_plus"] = 1.0 - below
    else:
        result.update({"split_point": None, "barrier": 0.0,
                       "mode_minus": modes[0] if modes else None,
                       "mode_plus": modes[-1] if modes else None,
                       "barrier_minus": 0.0, "barrier_plus": 0.0})
        if normalisable and modes:
            cut = int(torch.searchsorted(grid.contiguous(),
                                         torch.tensor(float(modes[0]))))
            result["w_minus"] = float(density[:cut].sum()) * step / total
            result["w_plus"] = 1.0 - result["w_minus"]
    # Curvature-based width at the dominant mode: what a geometry-aware
    # Langevin step would read off the bridge without knowing the answer.
    if modes:
        dominant = max(modes, key=value_at)
        index = int(torch.searchsorted(grid.contiguous(),
                                       torch.tensor(float(dominant))))
        index = min(max(index, 1), len(grid) - 2)
        curvature = float((slope[index + 1] - slope[index - 1])
                          / (2.0 * step))
        result["mode_curvature"] = curvature
        result["mode_width"] = (math.sqrt(-1.0 / curvature)
                                if curvature < 0 else float("nan"))
    if normalisable:
        mean = float((grid * density).sum()) * step / total
        second = float((grid ** 2 * density).sum()) * step / total
        result["curve_mean"] = mean
        result["curve_sd"] = math.sqrt(max(second - mean ** 2, 0.0))
    return result


def profile(oracle, field, taus, lam_max, lam_min, points=20001,
            mass_floor=MASS_FLOOR, keep_curve=False, curve_stride=25):
    """Topology of one bridge at every level in ``taus``."""
    rows = {}
    for tau in taus:
        lam = lam_of_tau(tau, lam_max, lam_min)
        t = float(oracle.schedule.time_of_lam(lam))
        grid = slice_grid(oracle, lam, points)
        slope = slice_slope(oracle, field, grid, t, lam)

        def continuous(value, _t=t, _lam=lam):
            point = torch.tensor([float(value)], dtype=torch.float64)
            return float(slice_slope(oracle, field, point, _t, _lam)[0])

        entry = describe(grid, slope, continuous, mass_floor)
        entry.update({"tau": float(tau), "lambda": lam, "t": t})
        if keep_curve:
            potential = potential_from_slope(grid, slope)
            entry["grid"] = grid[::curve_stride].tolist()
            entry["curve"] = potential[::curve_stride].tolist()
        rows[f"tau={tau:.4f}"] = entry
    return rows


def split_level(rows):
    """``tau_split``: the highest-noise level at which two modes are stable.

    Rows are scanned from the noise end down.  The split level is reported as
    the first ``tau`` (largest, i.e. noisiest) whose curve has two or more
    modes *and* every level below it also does -- a single two-mode level
    surrounded by one-mode levels is a grid artefact, not a bifurcation.
    """
    ordered = sorted(rows.values(), key=lambda row: -row["tau"])
    best = None
    for index, row in enumerate(ordered):
        if row["num_modes"] < 2:
            continue
        if all(later["num_modes"] >= 2 for later in ordered[index:]):
            best = row["tau"]
            break
    return best


def slice_validation(oracle, taus, lam_max, lam_min, points=20001):
    """How far the ``true`` profiled slice sits from the exact marginal.

    The slice is a cut, not a marginal, so the two are different curves by
    construction.  What matters for the topology claims is that they agree on
    *topology*: same mode count, nearly the same mode locations, nearly the
    same branch mass.  This reports that gap so the reader can see how much of
    any measured discrepancy could be the reduction rather than the rule.
    """
    from gm_bridges import ComposedField

    field = ComposedField(oracle, "true", lam_max=lam_max, lam_min=lam_min)
    out = {}
    for tau in taus:
        lam = lam_of_tau(tau, lam_max, lam_min)
        t = float(oracle.schedule.time_of_lam(lam))
        grid = slice_grid(oracle, lam, points)
        slope = slice_slope(oracle, field, grid, t, lam)
        cut = describe(grid, slope)
        exact_log = oracle.tall.global_log_prob(grid, lam)
        exact_slope = oracle.tall.global_score(grid, lam)
        marginal = describe(grid, exact_slope)
        out[f"tau={tau:.4f}"] = {
            "tau": float(tau), "lambda": lam,
            "slice_modes": cut["num_modes"],
            "marginal_modes": marginal["num_modes"],
            "slice_w_plus": cut.get("w_plus"),
            "marginal_w_plus": marginal.get("w_plus"),
            "slice_barrier": cut.get("barrier"),
            "marginal_barrier": marginal.get("barrier"),
            "slice_mode_plus": cut.get("mode_plus"),
            "marginal_mode_plus": marginal.get("mode_plus"),
            "exact_log_range": float(exact_log.max() - exact_log.min()),
        }
    return out


# ---------------------------------------------------------------------------
# The topology objective (section 7)
# ---------------------------------------------------------------------------

def topology_loss(candidate, truth, within,
                  extra_mode_penalty=10.0, c_mode=1.0, c_barrier=0.25,
                  c_mass=4.0):
    """Hierarchical distance between a candidate topology and the true one.

    The hierarchy the experiment asks for is encoded as a lexicographic
    penalty rather than a weighted sum that could trade one level against
    another: an extra or missing mode costs ``extra_mode_penalty`` per mode,
    which dominates any achievable contribution from the remaining terms, and
    only within a matched mode count do positions, barrier depth and branch
    mass compete.  Mode positions are measured in within-branch sds so the
    number means the same thing at every ``n``; barrier depth is already in log
    units.

    **This objective never sees the endpoint.**  Selecting ``alpha`` on final
    ``W1`` would make the whole experiment circular -- the question is whether
    matching the *bridge* is enough to fix the endpoint, which can only be
    answered if the endpoint played no part in the choice.
    """
    loss = extra_mode_penalty * abs(candidate["num_modes"] - truth["num_modes"])
    if not candidate.get("normalisable", True):
        loss += 10.0 * extra_mode_penalty

    if truth["num_modes"] >= 2 and candidate["num_modes"] >= 2:
        for key in ("mode_minus", "mode_plus"):
            if candidate.get(key) is not None and truth.get(key) is not None:
                loss += c_mode * abs(candidate[key] - truth[key]) / within
        loss += c_barrier * abs(candidate.get("barrier", 0.0)
                                - truth.get("barrier", 0.0))
    elif truth["num_modes"] == 1 and candidate["num_modes"] == 1:
        if candidate.get("mode_plus") is not None and truth.get("mode_plus") is not None:
            loss += c_mode * abs(candidate["mode_plus"] - truth["mode_plus"]) / within

    if (candidate.get("w_plus") is not None
            and truth.get("w_plus") is not None):
        loss += c_mass * abs(candidate["w_plus"] - truth["w_plus"])
    elif truth.get("w_plus") is not None:
        loss += c_mass
    return float(loss)
