"""
Maximum-likelihood point estimates through the PF-ODE likelihood.

Information criteria (AIC, AICc, BIC) are defined at the maximum of the
likelihood. Plugging in the posterior MAP instead under-estimates the fit term
whenever the prior is informative (validation/TODO.md, item A2). This module
maximizes the very quantity the criterion uses, the model's PF-ODE
log-likelihood ``log p(x_i | theta_i)``, by gradient-based optimization:

* the gradient d log p / d theta comes from autograd through the PF-ODE solver
  (``model.log_prob(..., differentiable=True)``), so it is the exact gradient of
  the discretized density, not of a surrogate;
* parameters are optimized in standardized coordinates
  ``u = (theta - theta_start) / scale`` (``scale`` ~ posterior sd), with
  scipy's L-BFGS-B so that box constraints are honoured;
* ``shared_dims`` (hierarchy/global parameters) are one value for all rows; all
  other parameter dims are free per row;
* every objective evaluation is recorded and the best one is kept -- per row when
  the rows are independent (no shared dims), for the total otherwise -- so the
  result is never worse than the starting point;
* several starts can be given (e.g. the MAP and posterior draws); the best is
  chosen by the same rule.

The function only calls ``model.log_prob(data, condition_mask, timesteps=...,
eps=..., device=..., differentiable=True, **log_prob_kwargs)``; anything with
that interface can be maximized.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy import optimize


@dataclass
class MaximumLikelihoodResult:
    theta: torch.Tensor           # (N, D) joint vectors, parameter dims at the optimum
    log_likelihood: torch.Tensor  # (N,) log p(x_i | theta_i) at the optimum (optimization grid)
    log_likelihood_start: torch.Tensor  # (N,) at the best *start*, before optimization
    at_bound: torch.Tensor        # (N, D) bool, parameter dims that sit on the box
    start_index: torch.Tensor     # (N,) index of the start that produced each row's optimum
    evaluations: int              # number of objective (PF-ODE) evaluations
    messages: list                # scipy termination messages, one per start
    history: list = None          # per start: summed log-likelihood at every evaluation


def maximize_likelihood(model, data, condition_mask, shared_dims=(), starts=None,
                        scale=None, lower=None, upper=None, max_iterations=50,
                        timesteps=32, eps=1e-3, device="cpu", log_prob_kwargs=None,
                        bound_tolerance=1e-4, keep_best_starts=None):
    """Maximize ``sum_i log p(x_i | theta_i)`` over the parameter dims.

    Args:
        model:          Object with a differentiable ``log_prob`` (an SBIm).
        data:           ``(N, D)`` joint vectors. Parameter dims (``condition_mask``
                        == 1) hold the first start, observed dims the data x_i.
        condition_mask: ``(D,)`` likelihood mask, 1 on the parameter dims (they
                        are conditioned on), 0 on the observed dims.
        shared_dims:    Parameter dims whose value is common to all rows.
        starts:         Optional ``(S, N, D)`` additional starts (parameter dims are
                        used). Shared dims are averaged over rows per start.
        keep_best_starts: If given and smaller than S, the S additional starts are
                        first screened by one gradient-free likelihood evaluation
                        and only the best ``keep_best_starts`` are optimized
                        (chosen per row without shared dims, by the total with).
                        Cheap insurance against multimodal likelihoods: screening
                        many posterior draws finds every well-populated basin.
        scale:          ``(N, D)`` or ``(D,)`` positive scale of each parameter dim
                        (e.g. the posterior sd). Defaults to 1.
        lower, upper:   ``(N, D)`` or ``(D,)`` box on the parameter dims; ``None``
                        for unbounded. Shared dims use the loosest box over rows.
        max_iterations: L-BFGS-B iterations per start.
        timesteps, eps: PF-ODE grid used during optimization.
        bound_tolerance: Distance (in scale units) below which a coordinate is
                        reported as sitting on the box.

    Returns:
        :class:`MaximumLikelihoodResult`.
    """
    log_prob_kwargs = dict(log_prob_kwargs or {})
    data = torch.as_tensor(data, dtype=torch.float32).detach().cpu()
    if data.dim() != 2:
        raise ValueError(f"data must be (N, D), got shape {tuple(data.shape)}")
    N, D = data.shape
    mask = torch.as_tensor(condition_mask, dtype=torch.float32).reshape(-1).cpu()
    if mask.numel() != D:
        raise ValueError("condition_mask must be a single (D,) mask shared by all rows.")
    params = torch.nonzero(mask > 0.5).flatten().tolist()
    shared = sorted(int(d) for d in shared_dims)
    if not set(shared) <= set(params):
        raise ValueError("shared_dims must be parameter (conditioned) dims.")
    local = [d for d in params if d not in shared]
    if not params:
        raise ValueError("condition_mask conditions on no parameter dims.")

    def per_row(value, fill):
        if value is None:
            return torch.full((N, D), float(fill))
        value = torch.as_tensor(value, dtype=torch.float32).cpu()
        return value.expand(N, D).clone() if value.dim() == 1 else value.clone()

    scale = per_row(scale, 1.0)
    if torch.any(scale[:, params] <= 0):
        raise ValueError("scale must be positive on the parameter dims.")
    lower = per_row(lower, -np.inf)
    upper = per_row(upper, np.inf)

    start_list = [data]
    if starts is not None:
        starts = torch.as_tensor(starts, dtype=torch.float32).cpu().clone()
        starts[:, :, [d for d in range(D) if d not in params]] = data[:, [d for d in range(D) if d not in params]]
        if keep_best_starts is not None and int(keep_best_starts) < starts.shape[0]:
            starts = _screen_starts(model, starts, mask, params, shared, lower, upper,
                                    int(keep_best_starts), timesteps, eps, device,
                                    log_prob_kwargs)
        start_list += [s for s in starts]

    s_shared = scale[:, shared].mean(0) if shared else torch.zeros(0)
    lo_shared = lower[:, shared].min(0).values if shared else torch.zeros(0)
    hi_shared = upper[:, shared].max(0).values if shared else torch.zeros(0)
    n_shared, n_local = len(shared), N * len(local)

    best_ll = torch.full((N,), -np.inf, dtype=torch.float64)
    best_theta = data.clone()
    best_start_ll = torch.full((N,), -np.inf, dtype=torch.float64)
    best_index = torch.zeros(N, dtype=torch.long)
    best_total = -np.inf
    evaluations = 0
    messages = []
    history = []

    for index, start in enumerate(start_list):
        center = data.clone()
        center[:, params] = start[:, params]
        if shared:
            center[:, shared] = start[:, shared].mean(0, keepdim=True)
        center[:, params] = torch.minimum(torch.maximum(center[:, params], lower[:, params]),
                                          upper[:, params])

        c_shared = center[0, shared] if shared else torch.zeros(0)
        c_local = center[:, local]
        s_local = scale[:, local]

        def unpack(v):
            theta = center.clone()
            if n_shared:
                theta[:, shared] = (c_shared + s_shared * v[:n_shared]).expand(N, n_shared)
            if n_local:
                theta[:, local] = c_local + s_local * v[n_shared:].reshape(N, len(local))
            return theta

        bounds = []
        for j in range(n_shared):
            bounds.append(_std_bound(lo_shared[j], hi_shared[j], c_shared[j], s_shared[j]))
        for i in range(N):
            for j, d in enumerate(local):
                bounds.append(_std_bound(lower[i, d], upper[i, d], c_local[i, j], s_local[i, j]))

        trace = []
        history.append(trace)
        record = {"best_ll": torch.full((N,), -np.inf, dtype=torch.float64),
                  "best_theta": center.clone(), "best_total": -np.inf, "first": None}

        def objective(v_np):
            nonlocal evaluations
            evaluations += 1
            v = torch.tensor(v_np, dtype=torch.float32, requires_grad=True)
            theta = unpack(v)
            ll = model.log_prob(theta.to(device), condition_mask=mask.to(device),
                                timesteps=timesteps, eps=eps, device=device,
                                differentiable=True, **log_prob_kwargs).reshape(-1)
            total = ll.sum()
            grad, = torch.autograd.grad(total, v)
            ll_d = ll.detach().cpu().double()
            theta_d = theta.detach()
            trace.append(ll_d.sum().item())
            if record["first"] is None:
                record["first"] = ll_d.clone()
            if not torch.all(torch.isfinite(ll_d)):
                return np.inf, np.zeros_like(v_np)
            if n_shared:
                if ll_d.sum().item() > record["best_total"]:
                    record["best_total"] = ll_d.sum().item()
                    record["best_ll"] = ll_d.clone()
                    record["best_theta"] = theta_d.clone()
            else:
                better = ll_d > record["best_ll"]
                record["best_ll"][better] = ll_d[better]
                record["best_theta"][better] = theta_d[better]
            return -total.item(), -grad.double().numpy()

        v0 = np.zeros(n_shared + n_local)
        result = optimize.minimize(objective, v0, jac=True, method="L-BFGS-B",
                                   bounds=bounds, options={"maxiter": int(max_iterations)})
        messages.append(str(result.message))

        if n_shared:
            if record["best_total"] > best_total:
                best_total = record["best_total"]
                best_ll, best_theta = record["best_ll"], record["best_theta"]
                best_start_ll = record["first"]
                best_index[:] = index
        else:
            better = record["best_ll"] > best_ll
            best_ll[better] = record["best_ll"][better]
            best_theta[better] = record["best_theta"][better]
            best_start_ll[better] = record["first"][better]
            best_index[better] = index

    at_bound = torch.zeros(N, D, dtype=torch.bool)
    tol = bound_tolerance * scale
    at_bound[:, params] = ((best_theta[:, params] - lower[:, params]) <= tol[:, params]) | \
                          ((upper[:, params] - best_theta[:, params]) <= tol[:, params])
    return MaximumLikelihoodResult(theta=best_theta, log_likelihood=best_ll.float(),
                                   log_likelihood_start=best_start_ll.float(),
                                   at_bound=at_bound, start_index=best_index,
                                   evaluations=evaluations, messages=messages,
                                   history=history)


def _screen_starts(model, starts, mask, params, shared, lower, upper, keep, timesteps,
                   eps, device, log_prob_kwargs):
    """Keep the ``keep`` best of ``starts`` by one gradient-free likelihood evaluation."""
    S, N, D = starts.shape
    candidates = starts.clone()
    if shared:
        candidates[:, :, shared] = candidates[:, :, shared].mean(1, keepdim=True)
    candidates[:, :, params] = torch.minimum(torch.maximum(candidates[:, :, params],
                                                           lower[:, params]), upper[:, params])
    with torch.no_grad():
        ll = model.log_prob(candidates.reshape(S * N, D).to(device), condition_mask=mask.to(device),
                            timesteps=timesteps, eps=eps, device=device,
                            **log_prob_kwargs).reshape(S, N).cpu()
    ll = torch.nan_to_num(ll, nan=-np.inf)
    if shared:
        order = ll.sum(1).argsort(descending=True)[:keep]
        return candidates[order]
    order = ll.argsort(dim=0, descending=True)[:keep]                  # (keep, N)
    return torch.gather(candidates, 0, order.unsqueeze(-1).expand(keep, N, D))


def _std_bound(low, high, center, scale):
    low, high, center, scale = (float(low), float(high), float(center), float(scale))
    return ((low - center) / scale if np.isfinite(low) else None,
            (high - center) / scale if np.isfinite(high) else None)
