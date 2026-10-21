"""Reference implementation of the exact conservative clamp (for oracle score fields).

For an exact score s on the lambda-smoothed density, the denoiser m(y) = y + lambda^2 s(y)
is the gradient of the convex potential psi(y) = |y|^2/2 + lambda^2 log p_t(y). The clamped
potential psi_B = (psi* + iota_B)* has gradient

    m_B(y) = argmax_{z in B} { y.z - psi*(z) } = m(y'),  where  z = m(y') in B  and
    y - y' in N_B(z)  (the normal cone of the box at z).

`conjugate_clamp` solves that system by Newton's method on y': each step takes the quadratic
model of psi at the current y' (Hessian J = grad m), whose clamped maximiser is the box QP
argmin_{z in B} (z - c)^T J^-1 (z - c), c = m(y') + J (y - y'), and moves y' so the model's
denoiser hits z. At the fixed point m(y') = z and y - y' lies in N_B(z).

The library's clamp_mode="metric" is the *first* step of this iteration from y' = y: it uses
the quadratic model at y only. It is exactly m_B when psi is quadratic (a Gaussian posterior)
and approximate otherwise; this file measures how far apart the two are.
"""
import torch

from compass.MultiObsSampler import MultiObsSampler

box_qp = MultiObsSampler._box_qp


def metric_clamp(m, jacobian, lower, upper):
    """clamp_mode="metric": the box QP in the metric J^-1 of the denoiser Jacobian at y."""
    return box_qp(m, torch.linalg.inv(jacobian), lower, upper)


def coordinate_clamp(m, lower, upper):
    return torch.maximum(torch.minimum(m, upper), lower)


def support(u, lower, upper):
    """sigma_B(u) = max_{z in B} u.z for the box B = [lower, upper]."""
    return torch.where(u > 0, u * upper, u * lower).nan_to_num(nan=0.0).sum(-1)


def conjugate_clamp(denoiser, jacobian_fn, y, lower, upper, psi=None, iterations=100, tol=1e-12):
    """Exact m_B(y) for a denoiser m = grad psi with symmetric PSD Jacobian. y: (..., D).

    psi_B(y) = min_{y'} psi(y') + sigma_B(y - y') (the inf-convolution form of
    (psi* + iota_B)*), and m_B(y) = m(y*) at the minimiser y*. Proximal Newton on y': the
    model step is the box QP above; with ``psi`` given, a backtracking (Armijo) line search
    on that convex objective makes it globally convergent, without it the step is undamped.
    """
    y = y.to(torch.float64)
    m = denoiser(y)
    z0 = coordinate_clamp(m, lower, upper)
    inside = (z0 == m).all(dim=-1, keepdim=True)
    objective = (lambda v: psi(v) + support(y - v, lower, upper)) if psi is not None else None
    yp = y.clone()
    for _ in range(iterations):
        m = denoiser(yp)
        j = jacobian_fn(yp)
        j_inv = torch.linalg.inv(j)
        center = m + (j @ (y - yp).unsqueeze(-1)).squeeze(-1)
        z = box_qp(center, j_inv, lower, upper)
        step = (j_inv @ (z - m).unsqueeze(-1)).squeeze(-1)
        if objective is not None:
            current = objective(yp)
            decrease = (m * step).sum(-1) + support(y - yp - step, lower, upper) - support(y - yp, lower, upper)
            alpha = torch.ones_like(current)
            for _ in range(50):
                trial = objective(yp + alpha.unsqueeze(-1) * step)
                ok = trial <= current + 1e-4 * alpha * decrease + 1e-13 * current.abs().clamp_min(1.0)
                if bool(ok.all()):
                    break
                alpha = torch.where(ok, alpha, 0.5 * alpha)
            step = alpha.unsqueeze(-1) * step
        yp = yp + step
        if float(step.abs().max()) < tol:
            break
    return torch.where(inside, denoiser(y), denoiser(yp))


def jacobian_of(field, y):
    """Batched Jacobian d field_i / d y_j at points y (N, D) by autograd."""
    y = y.detach().to(torch.float64).requires_grad_(True)
    out = field(y)
    rows = [torch.autograd.grad(out[:, i].sum(), y, retain_graph=True)[0] for i in range(out.shape[-1])]
    return torch.stack(rows, dim=-2)


def antisymmetry(jac):
    """||J - J^T||_F / ||J||_F per point: 0 for a conservative (gradient) field."""
    return torch.linalg.matrix_norm(jac - jac.mT) / torch.linalg.matrix_norm(jac).clamp_min(1e-300)
