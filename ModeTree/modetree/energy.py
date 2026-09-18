"""Milestones D and E -- relative log density from the score field.

D: ``Delta_ab = log p(b) - log p(a) = int_0^1 S_0(a + u(b-a)) . (b-a) du``
evaluated with Gauss-Legendre quadrature, adaptively refined in ``Q``, with all
quadrature points evaluated in one batched call to the composed score.

E: because a learned score field is not exactly conservative, each mode pair is
integrated along several geometrically distinct paths, and the resulting
pairwise differences are reconciled into globally consistent mode energies by a
weighted graph-Laplacian least-squares problem.

A note on the "reversed path" check requested in the task: Gauss-Legendre nodes
are symmetric about ``u = 1/2``, so integrating ``b -> a`` visits *exactly* the
same points with the same weights and returns ``-Delta_ab`` to the last bit,
independently of how non-conservative the field is.  It is reported (and is
always ~1e-16 here), but it is a rounding check, not a physics check: the
informative diagnostics are the disagreement between geometrically distinct
paths and the closure of triangle cycles, both of which are computed below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math

import numpy as np
import torch


def gauss_legendre(q):
    nodes, weights = np.polynomial.legendre.leggauss(q)
    nodes = torch.as_tensor(0.5 * (nodes + 1.0), dtype=torch.float64)
    weights = torch.as_tensor(0.5 * weights, dtype=torch.float64)
    return nodes, weights


@dataclass
class PathIntegral:
    value: float
    q_used: int
    refinements: list = field(default_factory=list)
    evaluations: int = 0


def composite_nodes(q, panels):
    """Composite Gauss-Legendre nodes/weights on [0, 1] with ``panels`` panels."""
    nodes, weights = gauss_legendre(q)
    offsets = torch.arange(panels, dtype=torch.float64).unsqueeze(-1) / panels
    u = (offsets + nodes.unsqueeze(0) / panels).reshape(-1)
    w = (weights.unsqueeze(0).expand(panels, -1) / panels).reshape(-1)
    return u, w


def integrate_segment(score_fn, a, b, q=16, adaptive=True, tol=1e-10,
                      panel_ladder=(1, 2, 4, 8, 16, 32, 64), counter=None):
    """``int_a^b S . dtheta`` along the straight segment, in float64.

    Composite Gauss-Legendre, refined by *doubling the panel count* rather than
    by raising the order.  Raising the order alone is not enough here: between
    two well-separated modes the integrand reaches ``O(n * separation / s^2)``
    while the integral itself is ``O(1)``, so a single high-order panel leaves a
    truncation error that grows linearly in ``n`` -- measured at 0.73 nats for
    the ``empty_valley`` benchmark at ``n = 100`` with the order-only ladder.

    For the same reason the convergence test is relative to the integrand's own
    scale ``int |S . dtheta|`` and not to ``|value|``: the value of a symmetric
    mode pair is exactly zero, so a ``tol * (1 + |value|)`` test is a demand for
    absolute ``tol`` accuracy on a quantity assembled from terms thousands of
    times larger, and can never be met.
    """
    a = torch.as_tensor(a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
    delta = b - a
    ladder = list(panel_ladder) if adaptive else [panel_ladder[-1]]
    previous = None
    refinements = []
    evaluations = 0
    value, scale, panels = 0.0, 0.0, ladder[0]
    for panels in ladder:
        u, w = composite_nodes(q, panels)
        points = a.unsqueeze(0) + u.unsqueeze(-1) * delta.unsqueeze(0)
        integrand = (score_fn(points) * delta).sum(-1)
        evaluations += u.numel()
        value = float((w * integrand).sum())
        scale = float((w * integrand.abs()).sum())
        if previous is not None:
            refinements.append(abs(value - previous))
            if abs(value - previous) < tol * (1.0 + scale):
                break
        previous = value
    if counter is not None:
        counter["quadrature_points"] = counter.get("quadrature_points", 0) + evaluations
    return PathIntegral(value=value, q_used=q * panels, refinements=refinements,
                        evaluations=evaluations)


def integrate_polyline(score_fn, points, **kwargs):
    """Sum of segment integrals along a polyline of waypoints."""
    total, q_used, refinements, evaluations = 0.0, 0, [], 0
    for a, b in zip(points[:-1], points[1:]):
        part = integrate_segment(score_fn, a, b, **kwargs)
        total += part.value
        q_used = max(q_used, part.q_used)
        refinements.extend(part.refinements)
        evaluations += part.evaluations
    return PathIntegral(total, q_used, refinements, evaluations)


# ----------------------------------------------------------------------------
# Path construction
# ----------------------------------------------------------------------------

def saddle_waypoint(score_fn, a, b, samples=65):
    """The point of smallest ``||S||`` on the straight segment.

    For a two-mode problem this sits essentially on the separating saddle, so a
    path through it stays in the high-density ridge instead of cutting a
    low-density valley -- which is where a learned score is least reliable.
    """
    a = torch.as_tensor(a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
    u = torch.linspace(0.05, 0.95, samples, dtype=torch.float64).unsqueeze(-1)
    points = a + u * (b - a)
    norms = torch.linalg.vector_norm(score_fn(points), dim=-1)
    return points[int(torch.argmin(norms))]


def bowed_waypoint(score_fn, a, b, scale=0.5):
    """Midpoint pushed along the local score, giving a geometrically distinct path."""
    a = torch.as_tensor(a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
    mid = 0.5 * (a + b)
    s = score_fn(mid.unsqueeze(0)).reshape(-1)
    direction = b - a
    norm_sq = float((direction * direction).sum())
    if norm_sq > 0:
        s = s - direction * float((s * direction).sum()) / norm_sq   # orthogonal part
    magnitude = float(torch.linalg.vector_norm(s))
    if magnitude < 1e-12:
        # no orthogonal component (e.g. 1-D): bow along the segment instead by
        # inserting a displaced interior point, which still changes the
        # quadrature nodes and therefore still probes discretisation error.
        return mid + 0.13 * (b - a)
    return mid + scale * s / magnitude * float(torch.linalg.vector_norm(b - a))


def build_paths(score_fn, a, b, multiple=True):
    """Return a dict of named waypoint lists for the pair ``(a, b)``."""
    a = torch.as_tensor(a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
    paths = {"straight": [a, b]}
    if not multiple:
        return paths
    paths["saddle"] = [a, saddle_waypoint(score_fn, a, b), b]
    paths["bowed"] = [a, bowed_waypoint(score_fn, a, b), b]
    paths["thirds"] = [a, a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0, b]
    return paths


# ----------------------------------------------------------------------------
# Pairwise differences + cycle-consistent reconstruction
# ----------------------------------------------------------------------------

@dataclass
class EnergyGraph:
    energies: torch.Tensor                 # (B,) with energies[0] == 0
    pairwise: dict                         # (a, b) -> mean estimate
    path_estimates: dict                   # (a, b) -> {name: value}
    path_disagreement: dict                # (a, b) -> max-min across paths
    reverse_residual: dict                 # (a, b) -> |forward + reverse|
    sigma: dict                            # (a, b) -> uncertainty proxy
    cycle_residuals: dict                  # (a, b, c) -> D_ab + D_bc + D_ca
    residual: float                        # least-squares residual of the fit
    evaluations: int


def pairwise_energies(score_fn, modes, multiple_paths=True, tol=1e-10,
                      q_ladder=None, max_pairs=None):
    """All pairwise ``E_b - E_a`` estimates with their path spread."""
    modes = [torch.as_tensor(m, dtype=torch.float64).reshape(-1) for m in modes]
    B = len(modes)
    pairs = list(itertools.combinations(range(B), 2))
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    estimates, disagreement, reverse, sigma, means = {}, {}, {}, {}, {}
    evaluations = 0
    for a, b in pairs:
        paths = build_paths(score_fn, modes[a], modes[b], multiple=multiple_paths)
        values, refinement = {}, 0.0
        for name, waypoints in paths.items():
            part = integrate_polyline(
                score_fn, waypoints, adaptive=True, tol=tol
            )
            values[name] = part.value
            evaluations += part.evaluations
            if part.refinements:
                refinement = max(refinement, part.refinements[-1])
        back = integrate_polyline(
            score_fn, list(reversed(paths["straight"])), adaptive=True, tol=tol,
        )
        evaluations += back.evaluations
        reverse[(a, b)] = abs(values["straight"] + back.value)
        spread = max(values.values()) - min(values.values())
        estimates[(a, b)] = values
        disagreement[(a, b)] = spread
        means[(a, b)] = float(np.mean(list(values.values())))
        sigma[(a, b)] = max(spread, refinement, 1e-12)
    return estimates, means, disagreement, reverse, sigma, evaluations


def reconcile(means, sigma, B, weight_floor=1e-12):
    """Weighted graph-Laplacian least squares for ``E`` with ``E_0 = 0``."""
    if B <= 1:
        return torch.zeros(max(B, 0), dtype=torch.float64), 0.0
    rows, rhs, weights = [], [], []
    for (a, b), value in means.items():
        row = torch.zeros(B, dtype=torch.float64)
        row[b], row[a] = 1.0, -1.0
        rows.append(row)
        rhs.append(value)
        weights.append(1.0 / max(sigma[(a, b)], weight_floor) ** 2)
    A = torch.stack(rows)
    y = torch.tensor(rhs, dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64).sqrt()
    # gauge fix E_0 = 0 with a hard constraint row
    gauge = torch.zeros(1, B, dtype=torch.float64)
    gauge[0, 0] = 1.0
    big = 1e6 * float(w.max())
    A_full = torch.cat([A * w.unsqueeze(-1), big * gauge], dim=0)
    y_full = torch.cat([y * w, torch.zeros(1, dtype=torch.float64)])
    solution = torch.linalg.lstsq(A_full, y_full.unsqueeze(-1)).solution.reshape(-1)
    residual = float(torch.linalg.vector_norm(A @ solution - y))
    return solution, residual


def cycle_residuals(means, B):
    out = {}
    lookup = {}
    for (a, b), value in means.items():
        lookup[(a, b)] = value
        lookup[(b, a)] = -value
    for a, b, c in itertools.combinations(range(B), 3):
        if (a, b) in lookup and (b, c) in lookup and (c, a) in lookup:
            out[(a, b, c)] = lookup[(a, b)] + lookup[(b, c)] + lookup[(c, a)]
    return out


def mode_energies(score_fn, modes, multiple_paths=True, tol=1e-10,
                  q_ladder=None):
    """Full Milestone D+E pipeline: paths -> pairwise -> consistent energies."""
    estimates, means, disagreement, reverse, sigma, evaluations = pairwise_energies(
        score_fn, modes, multiple_paths=multiple_paths, tol=tol
    )
    energies, residual = reconcile(means, sigma, len(modes))
    return EnergyGraph(
        energies=energies, pairwise=means, path_estimates=estimates,
        path_disagreement=disagreement, reverse_residual=reverse, sigma=sigma,
        cycle_residuals=cycle_residuals(means, len(modes)),
        residual=residual, evaluations=evaluations,
    )
