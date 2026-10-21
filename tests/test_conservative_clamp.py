"""Tests for clamp_mode="metric" (conservative clamping of denoised predictions).

1. `_box_qp` solves the box-constrained quadratic program (against brute force).
2. It returns the prediction unchanged when it is already inside the box.
3. On a Gaussian posterior the coordinate clamp's score has an antisymmetric Jacobian part
   (D_ii - D_jj) H_ij, the metric clamp's has none, and the metric clamp equals the exact
   conjugate construction psi_B = (psi* + iota_B)*.
4. The sampler's per-row and composed clamps use the right metric and leave the default
   ("coordinate") path untouched.
"""
import itertools
import sys
from pathlib import Path

import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pd_study"))
import conservative_clamp as cc  # noqa: E402

box_qp = MultiObsSampler._box_qp


def brute_force(center, metric, lower, upper):
    """Enumerate free/lower/upper for every coordinate; keep the feasible KKT point."""
    d = center.shape[-1]
    best, best_value = None, float("inf")
    for pattern in itertools.product((0, 1, 2), repeat=d):
        z = center.clone()
        fixed = [i for i, p in enumerate(pattern) if p]
        free = [i for i, p in enumerate(pattern) if not p]
        for i in fixed:
            z[i] = lower[i] if pattern[i] == 1 else upper[i]
        if not torch.isfinite(z).all():
            continue
        if free:
            a_ff = metric[free][:, free]
            rhs = -metric[free][:, fixed] @ (z[fixed] - center[fixed]) if fixed else torch.zeros(len(free), dtype=z.dtype)
            z[free] = center[free] + torch.linalg.solve(a_ff, rhs)
        if (z < lower - 1e-12).any() or (z > upper + 1e-12).any():
            continue
        value = float((z - center) @ metric @ (z - center))
        if value < best_value:
            best, best_value = z, value
    return best


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_box_qp_matches_brute_force(dimension):
    generator = torch.Generator().manual_seed(dimension)
    for _ in range(40):
        root = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
        metric = root @ root.T + 0.05 * torch.eye(dimension, dtype=torch.float64)
        center = 3 * torch.randn(dimension, generator=generator, dtype=torch.float64)
        lower = -torch.rand(dimension, generator=generator, dtype=torch.float64) - 0.2
        upper = torch.rand(dimension, generator=generator, dtype=torch.float64) + 0.2
        lower[0] = float("-inf")    # an unbounded coordinate, as for a local without a prior
        got = box_qp(center, metric, lower, upper)
        torch.testing.assert_close(got, brute_force(center, metric, lower, upper), rtol=1e-7, atol=1e-7)


def test_box_qp_batched_and_identity_metric_is_coordinate_clip():
    generator = torch.Generator().manual_seed(0)
    center = 2 * torch.randn(5, 7, 3, generator=generator, dtype=torch.float64)
    lower, upper = -torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    got = box_qp(center, torch.eye(3, dtype=torch.float64), lower, upper)
    torch.testing.assert_close(got, center.clamp(-1, 1))


def test_inside_the_box_is_returned_exactly():
    center = torch.tensor([[0.3, -0.2], [0.1, 0.9]], dtype=torch.float64)
    metric = torch.tensor([[2.0, 1.5], [1.5, 2.0]], dtype=torch.float64)
    got = box_qp(center, metric, -torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    assert torch.equal(got, center)


# --- a correlated Gaussian posterior N(mu, S) smoothed by lambda^2 ---------------------------
MU = torch.tensor([0.4, -0.3], dtype=torch.float64)
S = torch.tensor([[1.0, 0.85], [0.85, 1.0]], dtype=torch.float64)
LAM2 = 0.5
LOWER = torch.tensor([-0.6, -0.6], dtype=torch.float64)
UPPER = torch.tensor([0.6, 0.6], dtype=torch.float64)
P = torch.linalg.inv(S + LAM2 * torch.eye(2, dtype=torch.float64))


def denoiser(y):
    return y - LAM2 * (y - MU) @ P.T


def denoiser_jacobian(y):
    j = torch.eye(2, dtype=torch.float64) - LAM2 * P
    return j.expand(y.shape[:-1] + (2, 2))


def clamped_score(kind):
    def field(y):
        m = denoiser(y)
        if kind == "coordinate":
            z = cc.coordinate_clamp(m, LOWER, UPPER)
        elif kind == "metric":
            z = cc.metric_clamp(m, denoiser_jacobian(y), LOWER, UPPER)
        else:
            z = m
        return (z - y) / LAM2
    return field


def points_with_one_coordinate_clamped():
    """Points whose denoised prediction leaves the box in exactly one coordinate."""
    grid = torch.stack(torch.meshgrid(torch.linspace(-3, 3, 41, dtype=torch.float64),
                                      torch.linspace(-3, 3, 41, dtype=torch.float64), indexing="ij"), -1).reshape(-1, 2)
    m = denoiser(grid)
    out = (m < LOWER) | (m > UPPER)
    keep = out.sum(-1) == 1
    margin = ((m - LOWER).abs().min(-1).values > 0.02) & ((m - UPPER).abs().min(-1).values > 0.02)
    return grid[keep & margin]


def test_coordinate_clamp_is_not_conservative_metric_clamp_is():
    y = points_with_one_coordinate_clamped()
    assert y.shape[0] > 50
    coordinate = cc.antisymmetry(cc.jacobian_of(clamped_score("coordinate"), y))
    metric = cc.antisymmetry(cc.jacobian_of(clamped_score("metric"), y))
    unclamped = cc.antisymmetry(cc.jacobian_of(clamped_score("none"), y))
    assert float(unclamped.max()) < 1e-12
    assert float(coordinate.median()) > 0.1            # (D_ii - D_jj) H_ij != 0
    assert float(metric.max()) < 1e-9


def test_coordinate_antisymmetry_matches_the_formula():
    """d s~_i/dy_j - d s~_j/dy_i = (D_ii - D_jj) (grad m)_ij / lambda^2 at a one-coordinate clamp."""
    y = points_with_one_coordinate_clamped()[:20]
    jac = cc.jacobian_of(clamped_score("coordinate"), y)
    m = denoiser(y)
    free = ((m >= LOWER) & (m <= UPPER)).to(torch.float64)          # D_ii
    grad_m = torch.eye(2, dtype=torch.float64) - LAM2 * P
    expected = (free[:, 0] - free[:, 1]) * grad_m[0, 1] / LAM2
    torch.testing.assert_close(jac[:, 0, 1] - jac[:, 1, 0], expected)


def test_metric_clamp_equals_the_conjugate_construction_for_a_gaussian():
    y = 2.5 * torch.randn(200, 2, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    m = denoiser(y)
    metric = cc.metric_clamp(m, denoiser_jacobian(y), LOWER, UPPER)
    exact = cc.conjugate_clamp(denoiser, denoiser_jacobian, y, LOWER, UPPER)
    torch.testing.assert_close(metric, exact, rtol=1e-9, atol=1e-9)
    inside = ((m >= LOWER) & (m <= UPPER)).all(-1)
    assert torch.equal(metric[inside], m[inside])
    assert ((metric >= LOWER - 1e-12) & (metric <= UPPER + 1e-12)).all()


def test_sampler_row_clamp_uses_the_joint_precision_metric():
    """_metric_row_clamp == box QP in Lambda_j; locals without a prior are unbounded."""
    sampler = MultiObsSampler.__new__(MultiObsSampler)
    sampler.hierarchy = [0]
    sampler.local_latent_indices = [1]
    sampler.full_gaussian_features = [0, 1]
    sampler.denoise_clamp = 1.0
    sampler.local_prior_mean = None
    generator = torch.Generator().manual_seed(4)
    x = 2 * torch.randn(3, 5, 3, generator=generator)
    scores = 2 * torch.randn(3, 5, 3, generator=generator)
    root = torch.randn(3, 5, 2, 2, generator=generator, dtype=torch.float64)
    metric = root @ root.mT + 0.1 * torch.eye(2, dtype=torch.float64)
    var_t = 0.7
    out = sampler._metric_row_clamp(scores.clone(), x, var_t, torch.tensor([-1.0]), torch.tensor([1.0]),
                                    [0, 1], metric)
    center = x[..., :2].double() + var_t * scores[..., :2].double()
    lower = torch.tensor([-1.0, float("-inf")], dtype=torch.float64)
    upper = torch.tensor([1.0, float("inf")], dtype=torch.float64)
    expected = (box_qp(center, metric, lower, upper) - x[..., :2].double()) / var_t
    torch.testing.assert_close(out[..., :2].double(), expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[..., 2], scores[..., 2])          # observed node untouched
    m = x[..., 0] + var_t * out[..., 0]
    assert float(m.abs().max()) <= 1.0 + 1e-5


def test_unknown_clamp_mode_is_rejected():
    sampler = MultiObsSampler.__new__(MultiObsSampler)
    with pytest.raises(ValueError):
        sampler.set_clamp_mode("box")


def test_on_a_mixture_only_the_conjugate_clamp_is_exactly_conservative():
    """psi is not quadratic: the metric clamp (its first Newton step) is approximately conservative,
    the globalised conjugate construction exactly, and both stay in the box."""
    import clamp_study as cs
    target = cs.TARGETS["mixture"]
    lam2 = 0.49
    lower, upper = -torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    axis = torch.linspace(-3.5, 3.5, 29, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    m = target.denoiser(lam2)(grid)
    gap = torch.minimum((m - lower).abs().min(-1).values, (m - upper).abs().min(-1).values)
    pts = grid[((m < lower) | (m > upper)).any(-1) & (gap > 0.02)]
    result = {}
    for kind in ("coordinate", "metric", "conjugate"):
        field = cs.clamped_score(kind, target, lam2, lower, upper)
        z = pts + lam2 * field(pts)
        assert ((z >= lower - 1e-9) & (z <= upper + 1e-9)).all()
        result[kind] = cc.antisymmetry(cs.fd_jacobian(field, pts))
    assert float(result["conjugate"].max()) < 1e-7
    assert float(result["metric"].median()) < 0.05 * float(result["coordinate"].median())
    assert float(result["metric"].max()) > 1e-3          # but not exact off a Gaussian
