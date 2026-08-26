#!/usr/bin/env python3
"""Exact single-observation scores for the symmetric hierarchy twins.

The counterpart of ``analytic_compare.ExactRowScoreNetwork``, for
``hierarchy_gauss`` and ``hierarchy_laplace``. Each class is a drop-in for
``SBIm.model``: ``forward(x, t, c)`` takes the state ``(g, l, x_obs)``, the
diffusion time and the condition mask, and returns the exact score of

    p_lam(g_t, l_t | x)

for **one** observation -- precisely what the network is trained to output. Every
error a sampler then makes is the composition rule, the initial law or the
integrator; none of it is network error.

``c[:, GLOBAL] == 1`` (stage 3's conditional local ascent) means ``g`` is given
rather than diffused, and each class collapses to its conditional branch.

Both must survive ``torch.func.jvp``: ``correction="gauss_jacobian"`` gets its
curvature by forward-differentiating the row score.

Gaussian: closed form, no quadrature
------------------------------------
``(g, l) | x`` is a bivariate Gaussian with precision

    [[1/sigma_g^2 + 1/s^2,  -1/s^2],
     [-1/s^2,                2/s^2]]

and mean ``Lambda^{-1} [mu_g/sigma_g^2, x/s^2]``. The VESDE kernel adds
``lam^2 I`` to its covariance, so the score is exactly linear:
``-(Sigma + lam^2 I)^{-1} (z - m)``.

Laplace: the same one-dimensional quadrature the exponential twin uses
---------------------------------------------------------------------
    log w(g) = -(g - mu_g)^2/(2 sigma_g^2) - (g_t - g)^2/(2 lam^2)
               + log K(g, l_t)

with ``log K`` and ``d/dl_t log K`` supplied in closed form by
``hierarchy_laplace.log_local_kernel``. Then
``d/dg_t log p = E_w[(g - g_t)]/lam^2`` and ``d/dl_t log p = E_w[d/dl_t log K]``.
Nodes are placed at ``m + s z`` where ``m, s`` are the mean and sd of the
Gaussian formed by the prior times the kernel -- i.e. where the mass is at every
noise level -- so a trapezoid over ``z`` converges quickly even though the
integrand has kinks at ``g = x``.

Usage:
    python symmetric_row_scores.py            # grade both against the reference
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hierarchy_gauss  # noqa: E402
import hierarchy_laplace  # noqa: E402


class _RowScoreBase(torch.nn.Module):
    """Shared plumbing: the SBIm.model interface and the alpha rescale."""

    def __init__(self, sde, module, chunk=8192):
        super().__init__()
        self.sde = sde
        self.problem = module
        self.chunk = int(chunk)

    def forward(self, x, t, c):
        module = self.problem
        time = torch.as_tensor(t).reshape(-1)[0]
        lam = float(self.sde.lambda_t(time))
        alpha = float(self.sde.alpha_t(time))
        state = x.to(torch.float64)
        mask = c.to(torch.float64)
        # The sampler hands the network alpha * y on latent coordinates and the
        # unscaled value on conditioned ones. The closed forms below live in the
        # y = x_t / alpha coordinate, where the VP kernel is the VE kernel at
        # lambda = sigma/alpha. For the VESDE alpha == 1 and this is a no-op.
        scale = torch.where(
            mask > 0.5, torch.ones_like(state), torch.full_like(state, alpha)
        )
        state = state / scale
        conditioned = mask[:, module.GLOBAL_INDEX] > 0.5
        if bool(conditioned.any()) and bool((~conditioned).any()):
            raise NotImplementedError(
                "mixed conditioning on the shared coordinate within one batch"
            )

        g_t = state[:, module.GLOBAL_INDEX]
        l_t = state[:, module.LOCAL_INDEX]
        obs = state[:, module.OBSERVED_INDEX]

        shared_parts, local_parts = [], []
        for start in range(0, state.shape[0], self.chunk):
            stop = min(start + self.chunk, state.shape[0])
            if bool(conditioned.all()):
                shared = torch.zeros_like(g_t[start:stop])
                local = self._conditioned(
                    g_t[start:stop], l_t[start:stop], obs[start:stop], lam
                )
            else:
                shared, local = self._diffused(
                    g_t[start:stop], l_t[start:stop], obs[start:stop], lam
                )
            shared_parts.append(shared)
            local_parts.append(local)

        score = torch.stack([
            torch.cat(shared_parts), torch.cat(local_parts),
            torch.zeros_like(g_t),
        ], dim=1) / scale
        return score.to(x.dtype)


class GaussianRowScoreNetwork(_RowScoreBase):
    """Exact -- and exactly linear -- row score for ``hierarchy_gauss``."""

    def _moments(self):
        module = self.problem
        s2 = module.NOISE_STD**2
        precision = torch.tensor(
            [[1.0 / module.SIGMA_G**2 + 1.0 / s2, -1.0 / s2],
             [-1.0 / s2, 2.0 / s2]], dtype=torch.float64,
        )
        return torch.linalg.inv(precision), s2

    def _conditioned(self, g, l_t, obs, lam):
        """``d/dl_t log p_lam(l_t | g, x)``: Normal((g + x)/2, s^2/2 + lam^2)."""
        s2 = self.problem.NOISE_STD**2
        return (0.5 * (g + obs) - l_t) / (s2 / 2.0 + lam**2)

    def _diffused(self, g_t, l_t, obs, lam):
        module = self.problem
        covariance, s2 = self._moments()
        covariance = covariance.to(g_t.device)
        # mean = Sigma @ [mu_g/sigma_g^2, x/s^2]; only the second entry varies.
        linear = torch.stack([
            torch.full_like(obs, module.MU_G / module.SIGMA_G**2), obs / s2
        ], dim=1)
        mean = linear @ covariance.T
        precision = torch.linalg.inv(
            covariance + lam**2 * torch.eye(2, dtype=torch.float64,
                                            device=g_t.device)
        )
        residual = torch.stack([g_t, l_t], dim=1) - mean
        score = -residual @ precision.T
        return score[:, 0], score[:, 1]


class LaplaceRowScoreNetwork(_RowScoreBase):
    """Quadrature row score for ``hierarchy_laplace``."""

    def __init__(self, sde, module, nodes=769, z_max=10.0, chunk=4096):
        super().__init__(sde, module, chunk=chunk)
        self.z_max = float(z_max)
        self.register_buffer(
            "z", torch.linspace(-z_max, z_max, int(nodes), dtype=torch.float64)
        )

    def _conditioned(self, g, l_t, obs, lam):
        """``d/dl_t log p_lam(l_t | g, x)``: the single-node kernel derivative."""
        _, derivative = self.problem.log_local_kernel(g, l_t, obs, lam)
        return derivative

    def _diffused(self, g_t, l_t, obs, lam):
        module = self.problem
        variance = lam**2
        prior_variance = module.SIGMA_G**2
        s_squared = 1.0 / (1.0 / prior_variance + 1.0 / variance)
        s = math.sqrt(s_squared)
        m = s_squared * (module.MU_G / prior_variance + g_t / variance)

        nodes = m[:, None] + s * self.z[None, :].to(g_t.device)
        log_kernel, derivative = module.log_local_kernel(
            nodes, l_t[:, None], obs[:, None], lam
        )
        # The Gaussian factor the nodes were placed under is already in the
        # spacing; what remains to weight by is its own exponent plus log K.
        log_weight = -0.5 * ((nodes - m[:, None]) / s) ** 2 + log_kernel
        weight = torch.softmax(log_weight, dim=1)

        score_g = (weight * (nodes - g_t[:, None])).sum(dim=1) / variance
        return score_g, (weight * derivative).sum(dim=1)


class ExactSingleObservationSampler:
    """Exact draws from ``p(g, l | x_j)``; the pilot ``gauss_hierarchical`` needs.

    Reuses the problem module's own validated ``sample_diffused`` at ``lam = 0``,
    so the pilot carries the same Monte-Carlo error the learned arm's pilot run
    would at the same ``precision_est_samples``, and no network error.
    """

    def __init__(self, module, seed=0):
        self.problem = module
        self.seed = int(seed)
        self.calls = 0

    def sample(self, data=None, num_samples=1, device="cpu", **kwargs):
        module = self.problem
        values = torch.as_tensor(data, dtype=torch.float64).reshape(
            torch.as_tensor(data).shape[0], -1
        )[:, -1].cpu().numpy()
        self.calls += 1
        draws = np.zeros((len(values), int(num_samples), module.NODES))
        for index, value in enumerate(values):
            _, _, g, local = module.sample_diffused(
                [value], 0.0, int(num_samples), self.seed + 977 * self.calls + index
            )
            draws[index, :, module.GLOBAL_INDEX] = g
            draws[index, :, module.LOCAL_INDEX] = local[:, 0]
            draws[index, :, module.OBSERVED_INDEX] = value
        return torch.as_tensor(draws, dtype=torch.float32, device=device)


class AnalyticModel:
    """The ``ScoreBasedInferenceModel`` surface ``MultiObsSampler`` uses."""

    def __init__(self, sde, row_score_class, module, device="cpu", nodes=769,
                 seed=0):
        self.sde = sde
        self.problem = module
        if row_score_class is LaplaceRowScoreNetwork:
            self.model = row_score_class(sde, module, nodes=nodes).to(device)
        else:
            self.model = row_score_class(sde, module).to(device)
        self.sampler = ExactSingleObservationSampler(module, seed=seed)

    @staticmethod
    def output_scale_function(t, scores):
        """Identity: the modules already return the score itself."""
        return scores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(module, row_score_class, name, lambdas=(0.05, 0.2, 1.0, 3.0),
             states=192, nodes=769, device="cpu"):
    """Row score against the module's own N = 1 reference quadrature.

    ``diffused_score`` reaches the same target by a different route -- a uniform
    grid over the numeric support rather than mass-placed nodes -- so agreement
    grades this module rather than restating it. For the Gaussian twin the
    reference is a closed-form linear-algebra route with no quadrature at all.
    """
    from compass.SDE import VESDE
    import compare

    sde = VESDE(sigma=compare.recipe.SDE_KWARGS["sigma"])
    model = AnalyticModel(sde, row_score_class, module, device=device,
                          nodes=nodes)
    _, _, x = module.observations(30, 0)
    mask = module.CONDITION_MASK.to(device)
    worst = 0.0
    print(f"{name}: row score vs the N = 1 reference")
    for lam in lambdas:
        errors, magnitudes = [], []
        for index, value in enumerate(np.asarray(x[:4], dtype=np.float64)):
            grid = module.quadrature_grid([value], device=device)
            shared, local, _, _ = module.sample_diffused(
                [value], lam, states, 99 + 1000 * index
            )
            exact_g, exact_l = module.diffused_score(
                shared, local, [value], lam, grid
            )
            state = torch.zeros(states, module.NODES, dtype=torch.float64,
                                device=device)
            state[:, module.GLOBAL_INDEX] = torch.as_tensor(shared, device=device)
            state[:, module.LOCAL_INDEX] = torch.as_tensor(local[:, 0],
                                                           device=device)
            state[:, module.OBSERVED_INDEX] = float(value)
            # float64 throughout: torch.tensor(float) defaults to float32, and
            # the lambda -> t -> lambda round trip would then cap the measurable
            # agreement at ~1e-7 -- a property of the probe, not of the module.
            t = sde.time_of_lambda(
                torch.tensor(float(lam), dtype=torch.float64)
            ).reshape(1, 1)
            with torch.no_grad():
                predicted = model.model(
                    x=state, t=t.to(device),
                    c=mask.unsqueeze(0).repeat(states, 1),
                ).to(torch.float64)
            exact = torch.stack([exact_g, exact_l[:, 0]], dim=1).to(device)
            errors.append((predicted[:, :2] - exact).pow(2).sum().item())
            magnitudes.append(exact.pow(2).sum().item())
        relative = math.sqrt(sum(errors) / max(sum(magnitudes), 1e-30))
        worst = max(worst, relative)
        print(f"  lam = {float(lam):<6g} relative RMS {relative:.3e}")
    return worst, model


def validate_jacobian(model, module, name, lam=0.2, count=8, device="cpu"):
    """``torch.func.jvp`` through the module against central differences."""
    from torch.func import jvp

    _, _, x = module.observations(30, 0)
    value = float(np.asarray(x).reshape(-1)[0])
    shared, local, _, _ = module.sample_diffused([value], lam, count, 3)
    state = torch.zeros(count, module.NODES, dtype=torch.float64, device=device)
    state[:, module.GLOBAL_INDEX] = torch.as_tensor(shared, device=device)
    state[:, module.LOCAL_INDEX] = torch.as_tensor(local[:, 0], device=device)
    state[:, module.OBSERVED_INDEX] = value
    mask = module.CONDITION_MASK.to(device).unsqueeze(0).repeat(count, 1)
    t = model.sde.time_of_lambda(
        torch.tensor(float(lam), dtype=torch.float64)
    ).reshape(1, 1).to(device)

    print(f"{name}: forward-mode differentiability (gauss_jacobian needs it)")
    worst = 0.0
    for feature in (module.GLOBAL_INDEX, module.LOCAL_INDEX):
        tangent = torch.zeros_like(state)
        tangent[:, feature] = 1.0
        _, forward = jvp(lambda s: model.model(x=s, t=t, c=mask), (state,),
                         (tangent,))
        step = 1e-4
        with torch.no_grad():
            plus = model.model(x=state + step * tangent, t=t, c=mask)
            minus = model.model(x=state - step * tangent, t=t, c=mask)
        difference = (plus - minus) / (2 * step)
        relative = float((forward[:, :2] - difference[:, :2]).pow(2).mean().sqrt()
                         / difference[:, :2].pow(2).mean().sqrt())
        worst = max(worst, relative)
        print(f"  jvp column {feature} vs central differences {relative:.3e}")
    return worst


def validate_conditional(model, module, name, lam=0.05, count=512, device="cpu"):
    """The conditioned branch against central differences of its own log density."""
    rng = np.random.default_rng(5)
    _, _, x = module.observations(30, 0)
    value = float(np.asarray(x).reshape(-1)[0])
    g = float(np.asarray(x).reshape(-1).mean())
    local = g + rng.standard_normal(count) * (1.0 + abs(value - g))
    state = torch.zeros(count, module.NODES, dtype=torch.float64, device=device)
    state[:, module.GLOBAL_INDEX] = g
    state[:, module.LOCAL_INDEX] = torch.as_tensor(local, device=device)
    state[:, module.OBSERVED_INDEX] = value
    mask = module.CONDITION_MASK.clone()
    mask[module.GLOBAL_INDEX] = 1.0
    t = model.sde.time_of_lambda(
        torch.tensor(float(lam), dtype=torch.float64)
    ).reshape(1, 1)
    with torch.no_grad():
        predicted = model.model(
            x=state, t=t.to(device),
            c=mask.to(device).unsqueeze(0).repeat(count, 1),
        ).to(torch.float64)[:, module.LOCAL_INDEX]

    step = 1e-5
    grid = np.linspace(-14.0, 14.0, 12001)

    def log_density(values):
        out = []
        for l_value in values:
            if module is hierarchy_gauss:
                s2 = module.NOISE_STD**2
                out.append(-0.5 * (l_value - 0.5 * (g + value)) ** 2
                           / (s2 / 2.0 + lam**2))
            else:
                inner = module._conditional_density(grid, g, value)
                kernel = np.exp(-0.5 * ((l_value - grid) / lam) ** 2)
                out.append(math.log(float((inner * kernel).sum()) + 1e-300))
        return np.array(out)

    reference = (log_density(local + step) - log_density(local - step)) / (2 * step)
    relative = float(np.sqrt(np.mean((predicted.cpu().numpy() - reference) ** 2)
                             / np.mean(reference**2)))
    print(f"{name}: conditioned branch (lam = {lam:g}) relative RMS "
          f"vs central differences {relative:.3e}")
    return relative


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    problems = [
        # The floor is ~1e-8 for both, and it is neither module's fault:
        # VESDE stores ``sigma`` as a float32 tensor, so ``log(sigma)`` carries
        # float32 precision into every lambda the probe reconstructs. Both twins
        # sit on that floor (9.60e-9 and 9.56e-9 at lam = 3), which is seven
        # orders of magnitude below the composition-rule errors being measured.
        ("gauss", hierarchy_gauss, GaussianRowScoreNetwork, 1e-7),
        ("laplace", hierarchy_laplace, LaplaceRowScoreNetwork, 1e-7),
    ]
    failures = []
    for name, module, row_score_class, tolerance in problems:
        print(f"\n=== {name} ===")
        worst, model = validate(module, row_score_class, name, device=device)
        if worst > tolerance:
            failures.append(f"{name}: row score {worst:.2e} > {tolerance:.0e}")
        jacobian = validate_jacobian(model, module, name, device=device)
        if jacobian > 1e-4:
            failures.append(f"{name}: jvp {jacobian:.2e}")
        conditional = validate_conditional(model, module, name, device=device)
        if conditional > 5e-3:
            failures.append(f"{name}: conditioned branch {conditional:.2e}")

    print()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print("all row-score checks passed")


if __name__ == "__main__":
    main()
