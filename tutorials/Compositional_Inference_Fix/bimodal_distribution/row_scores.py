#!/usr/bin/env python3
"""The *exact* single-observation score, as a drop-in for ``SBIm.model``.

Both benchmarks have a row posterior that is a Gaussian mixture with one shared
covariance,

    p(g, l | y) = sum_k w_k(y) N(m_k(y), C),

so under the VE kernel the diffused row posterior is the same mixture with
``C_lam = C + lam^2 I`` and

    grad log p_lam(x) = C_lam^{-1} [ sum_k r_k(x) m_k - x ],
    r_k(x) ~ w_k N(x; m_k, C_lam).

That is a closed form at every noise level: no quadrature, no Monte Carlo, no
learned network. Running the whole comparison twice -- once with the trained
network and once with this class -- separates "the composition rule is wrong at
finite noise" from "the network is wrong", which on a bimodal target is exactly
the ambiguity worth removing: a collapsed mode could be either.

``c[:, GLOBAL] == 1`` means ``g`` is given rather than diffused (the conditional
local ascent of the MAP stage), and the class switches to the conditioned branch
``p_lam(l_t | g, y)`` -- itself a mixture, with weights that are *not* equal,
because conditioning on ``g`` is what resolves the sign.

Everything below is plain differentiable torch, so ``torch.func.jvp`` goes
through it: ``correction="gauss_jacobian"`` gets its curvature by
forward-differentiating this score.

Usage:
    python row_scores.py               # grade both classes against the mixture
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

import mixture_reference as reference  # noqa: E402


class ExactRowScoreNetwork(torch.nn.Module):
    """Exact ``grad log p_lam(g_t, l_t | y)`` for one observation.

    The signature matches the trained network exactly -- ``forward(x, t, c)``
    with ``x`` the state ``(g, l, y)``, ``t`` the diffusion time and ``c`` the
    condition mask -- and the observed column is returned as zero, since the
    sampler only reads the latent columns.
    """

    def __init__(self, sde, module):
        super().__init__()
        self.sde = sde
        self.problem = module

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

        if bool(conditioned.all()):
            shared = torch.zeros_like(g_t)
            local = self._conditioned(g_t, l_t, obs, lam)
        else:
            shared, local = self._diffused(g_t, l_t, obs, lam)

        score = torch.stack([shared, local, torch.zeros_like(g_t)], dim=1)
        # d/dx_t = (1/alpha) d/dy on the coordinates that were rescaled above.
        return (score / scale).to(x.dtype)

    # -- the two branches ---------------------------------------------------

    def _diffused(self, g_t, l_t, obs, lam):
        """``grad log p_lam(g_t, l_t | y)``: a two-dimensional mixture score."""
        log_weights, means, covariance = self.problem.row_components(obs)
        smoothed = covariance + lam**2 * torch.eye(
            2, dtype=torch.float64, device=g_t.device
        )
        precision = torch.linalg.inv(smoothed)
        state = torch.stack([g_t, l_t], dim=1)                     # (B, 2)
        residual = state[:, None, :] - means                       # (B, K, 2)
        quadratic = torch.einsum("bki,ij,bkj->bk", residual, precision, residual)
        responsibility = torch.softmax(log_weights - 0.5 * quadratic, dim=1)
        centre = torch.einsum("bk,bki->bi", responsibility, means)
        score = (centre - state) @ precision.T
        return score[:, 0], score[:, 1]

    def _conditioned(self, g, l_t, obs, lam):
        """``d/dl_t log p_lam(l_t | g, y)``: the same rule in one dimension."""
        log_weights, means, variance = self.problem.conditional_local_components(
            g, obs
        )
        smoothed = variance + lam**2
        responsibility = torch.softmax(
            log_weights - 0.5 * (l_t[:, None] - means) ** 2 / smoothed, dim=1
        )
        return ((responsibility * means).sum(dim=1) - l_t) / smoothed


class ExactRowSampler:
    """Stand-in for ``SBIm.sampler``: exact draws from p(g, l | y_j).

    Only ``correction="gauss_hierarchical"`` uses it, and only to build the
    pilot covariance. Drawing from the exact row posterior is the noiseless
    limit of the pilot run the learned arm performs, so the pilot keeps the same
    Monte-Carlo error at the same ``precision_est_samples`` and carries no
    network error -- which is the whole point of the analytic arms.
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
        rng = np.random.default_rng(self.seed + 977 * self.calls)
        self.calls += 1
        draws = np.zeros((len(values), int(num_samples), module.NODES))
        for index, value in enumerate(values):
            pair = reference.row_sample(module, value, int(num_samples), rng)
            draws[index, :, module.GLOBAL_INDEX] = pair[:, 0]
            draws[index, :, module.LOCAL_INDEX] = pair[:, 1]
            draws[index, :, module.OBSERVED_INDEX] = value
        return torch.as_tensor(draws, dtype=torch.float32, device=device)


class AnalyticModel:
    """The ``ScoreBasedInferenceModel`` surface ``MultiObsSampler`` actually uses."""

    def __init__(self, sde, module, device="cpu", seed=0):
        self.sde = sde
        self.problem = module
        self.model = ExactRowScoreNetwork(sde, module).to(device)
        self.sampler = ExactRowSampler(module, seed=seed)

    @staticmethod
    def output_scale_function(t, scores):
        """Identity: the module already returns the score itself."""
        return scores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_diffused(network, module, values, lambdas, states=512, seed=99):
    """Relative RMS error against the ``n = 1`` case of the exact mixture score.

    ``mixture_reference.diffused_score`` reaches the same target through the
    joint arrow covariance rather than the 2x2 row form, so agreement grades
    this class rather than restating it.
    """
    worst = 0.0
    mask = module.CONDITION_MASK
    for lam in lambdas:
        errors, magnitudes = [], []
        for index, value in enumerate(values):
            shared, local, _, _ = reference.sample_diffused(
                module, [value], lam, states, seed + 1000 * index
            )
            exact_g, exact_l = reference.diffused_score(
                module, shared, local, [value], lam
            )
            state = torch.zeros(states, module.NODES, dtype=torch.float64)
            state[:, module.GLOBAL_INDEX] = torch.as_tensor(shared)
            state[:, module.LOCAL_INDEX] = torch.as_tensor(local[:, 0])
            state[:, module.OBSERVED_INDEX] = float(value)
            t = network.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1)
            with torch.no_grad():
                predicted = network.model(
                    x=state, t=t, c=mask.unsqueeze(0).repeat(states, 1)
                ).to(torch.float64)
            exact = torch.stack(
                [torch.as_tensor(exact_g), torch.as_tensor(exact_l[:, 0])], dim=1
            )
            errors.append((predicted[:, :2] - exact).pow(2).sum().item())
            magnitudes.append(exact.pow(2).sum().item())
        relative = math.sqrt(sum(errors) / max(sum(magnitudes), 1e-30))
        worst = max(worst, relative)
        print(f"    lam={float(lam):<6g} relative RMS {relative:.3e}")
    return worst


def validate_conditioned(network, module, value, count=512, lam=1e-2, seed=5,
                         step=1e-5):
    """The conditioned branch against a central difference of its own log density."""
    rng = np.random.default_rng(seed)
    g = 0.7 * float(np.abs(value))
    local = g + 0.8 * rng.standard_normal(count)
    state = torch.zeros(count, module.NODES, dtype=torch.float64)
    state[:, module.GLOBAL_INDEX] = g
    state[:, module.LOCAL_INDEX] = torch.as_tensor(local)
    state[:, module.OBSERVED_INDEX] = float(value)
    mask = module.CONDITION_MASK.clone()
    mask[module.GLOBAL_INDEX] = 1.0
    t = network.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1)
    with torch.no_grad():
        predicted = network.model(
            x=state, t=t, c=mask.unsqueeze(0).repeat(count, 1)
        ).to(torch.float64)[:, module.LOCAL_INDEX]

    weights, means, variance = module.conditional_local_mixture(
        np.full(count, g), np.full(count, float(value))
    )

    def log_density(position):
        exponent = -0.5 * (position[None, :] - means) ** 2 / (variance + lam**2)
        shifted = exponent + np.log(np.maximum(weights, 1e-300))
        top = shifted.max(axis=0)
        return top + np.log(np.exp(shifted - top[None, :]).sum(axis=0))

    finite = (log_density(local + step) - log_density(local - step)) / (2 * step)
    relative = float(
        np.sqrt(np.mean((predicted.numpy() - finite) ** 2))
        / np.sqrt(np.mean(finite**2))
    )
    print(f"    conditioned branch (lam={lam:g}) relative RMS "
          f"vs finite differences {relative:.3e}")
    return relative


def validate_jacobian(network, module, value, lam=0.2, count=8, step=1e-4):
    """``torch.func.jvp`` through the class, against central differences."""
    from torch.func import jvp

    rng = np.random.default_rng(3)
    state = torch.zeros(count, module.NODES, dtype=torch.float64)
    state[:, module.GLOBAL_INDEX] = torch.as_tensor(rng.standard_normal(count))
    state[:, module.LOCAL_INDEX] = torch.as_tensor(rng.standard_normal(count))
    state[:, module.OBSERVED_INDEX] = float(value)
    mask = module.CONDITION_MASK.unsqueeze(0).repeat(count, 1)
    t = network.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1)

    def scores(z):
        return network.model(x=z, t=t, c=mask)

    worst = 0.0
    for feature in (module.GLOBAL_INDEX, module.LOCAL_INDEX):
        tangent = torch.zeros_like(state)
        tangent[:, feature] = 1.0
        _, column = jvp(scores, (state,), (tangent,))
        with torch.no_grad():
            finite = (scores(state + step * tangent)
                      - scores(state - step * tangent)) / (2 * step)
        relative = float((column - finite).norm() / finite.norm())
        worst = max(worst, relative)
        print(f"    jvp column {feature} relative error {relative:.3e}")
    return worst


def main():
    import sign_model
    import control_model
    from compass.SDE import VESDE

    sde = VESDE(sigma=8.0)
    lambdas = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    failures = []
    for module in (sign_model, control_model):
        print(f"\n=== {module.TITLE} ===")
        network = AnalyticModel(sde, module)
        _, _, _, y = module.observations(4, seed=11)
        worst = validate_diffused(network, module, y, lambdas)
        conditioned = validate_conditioned(network, module, float(y[0]))
        jacobian = validate_jacobian(network, module, float(y[0]))
        if max(worst, conditioned, jacobian) > 1e-5:
            failures.append(module.NAME)
    if failures:
        raise SystemExit(f"row score validation failed for: {failures}")
    print("\nAll row scores agree with the exact mixture to quadrature precision.")


if __name__ == "__main__":
    main()
