"""Karras-style stochastic predictor ("churn") for the DPM samplers.

A Langevin corrector relaxes the ensemble onto the (composed) density where that
density is accurate. The reverse-time SDE has the same mechanism in its
*predictor*: take a short step **up** in noise, then the deterministic step back
down. Following Karras et al. (2022, Alg. 2),

    sigma_hat = sigma_i * (1 + gamma),
    gamma     = min(eta * (sigma_i - sigma_{i+1}) / sigma_i, gamma_cap),

with ``eta = 0`` the stock probability-flow ODE and ``eta = 1`` the exact reverse
SDE. It costs **no extra score evaluations**, unlike a corrector, which buys the
same relaxation with a full compositional score evaluation per iteration.

Why this is the default
-----------------------
Measured on the exactly-solvable ``g -> l_j -> x_j`` hierarchy
(``tutorials/Compositional_Inference_Fix/New_Attempt/exponential_score_comparison``),
shared-marginal W1 in units of the posterior sigma, corrector-free:

    eta = 0 (stock predictor-only)   2.598
    eta = 1 (exact reverse SDE)      0.314
    eta = 2                          0.049
    eta = 4                          0.034
    DPM2 + 2 Langevin correctors     0.057   (2.1x the score calls)

With a *learned* score the margin widens and the optimum moves to ``eta = 2``:
0.036 (``gauss_jacobian``) and 0.019 (``gauss_hierarchical``) against 0.265 and
0.270 for the same rules with two correctors. Hence ``DEFAULT_ETA = 2.0``.

The same ordering holds on matched-variance Gaussian and Laplace twins of that
hierarchy, so it is not a property of the exponential's hard wall; and on the
bimodal benchmarks corrector-free churned arms are the ones that *improve* as the
row score improves, while corrector-heavy arms degrade under a learned score and
are not rescued by more capacity.

The cap
-------
``gamma_cap = sqrt(2) - 1`` is the Karras stability cap and the value every
published COMPASS measurement used. Note that the samplers build a **geometric**
sigma grid, so the relative step ``(sigma_i - sigma_{i+1}) / sigma_i`` is the
same at every level and the clamp is therefore all-or-nothing: above
``eta_crit = gamma_cap / (1 - r)`` every step pins to the cap and the sampler
stops depending on ``eta`` at all. At 50 steps on the partial-pooling schedule
that threshold is ~3.2-4.0, which is why runs labelled ``eta = 4`` and
``eta = 8`` there were bit-identical. Raise ``churn_gamma_cap`` to sweep past it.
"""

import math

import torch

# Karras et al. (2022) cap the per-step noise-up ratio for stability.
GAMMA_CAP = math.sqrt(2.0) - 1.0

# Corrector-free churn at this strength beats a two-corrector arm at half the
# score-network evaluations on every benchmark it has been measured on.
DEFAULT_ETA = 2.0


def churn_sigma(sigma_now, sigma_next, eta, gamma_cap=GAMMA_CAP, ceiling=None,
                sigma_min=0.0, sigma_max=float("inf")):
    """Noise level to step up to, and the standard deviation of the noise to add.

    Returns ``(sigma_hat, spread)`` where ``spread`` is
    ``sqrt(sigma_hat**2 - sigma_now**2)``, the standard deviation of the noise
    that carries the state from ``sigma_now`` to ``sigma_hat``. ``spread == 0``
    means no churn applies at this level and the caller should take the ordinary
    deterministic step.

    ``sigma_min`` / ``sigma_max`` restrict churn to a band of the schedule --
    Karras' ``S_tmin`` / ``S_tmax``. The default band is the whole schedule.
    """
    if eta <= 0.0 or not (sigma_min <= sigma_now <= sigma_max):
        return float(sigma_now), 0.0
    gamma = min(eta * (sigma_now - sigma_next) / sigma_now, gamma_cap)
    sigma_hat = sigma_now * (1.0 + gamma)
    if ceiling is not None:
        # The schedule's own maximum is the largest noise level the score
        # network was trained at; stepping above it would query the model
        # outside its support.
        sigma_hat = min(sigma_hat, ceiling)
    return float(sigma_hat), math.sqrt(max(sigma_hat**2 - sigma_now**2, 0.0))


def resolve_corrector_steps(corrector_steps, method, churn_eta):
    """Resolve the ``corrector_steps=None`` sentinel, and refuse the mixed sampler.

    ``None`` means "whatever this method's default is": ``0`` under ``"dpm"``,
    where churn is on by default and replaces the correctors, and ``5`` under
    ``"langevin"``, which *is* a corrector-only sampler and would otherwise
    become a no-op.

    Churn and Langevin correctors together is a combination no COMPASS
    measurement covers, and the evidence says they are alternatives rather than
    complements, so asking for both explicitly is an error rather than a silent
    blend.
    """
    if method == "langevin":
        return 5 if corrector_steps is None else int(corrector_steps)
    if corrector_steps is None:
        return 0
    corrector_steps = int(corrector_steps)
    if corrector_steps > 0 and churn_eta > 0.0 and method == "dpm":
        raise ValueError(
            f"corrector_steps={corrector_steps} with churn_eta={churn_eta}: the "
            "churned predictor and Langevin correctors are alternatives, not "
            "complements, and no COMPASS measurement covers the combination. "
            "Pass churn_eta=0.0 for the legacy predictor-corrector sampler, or "
            "corrector_steps=0 for the corrector-free churned predictor "
            "(recommended -- see compass.churn)."
        )
    return corrector_steps


def apply_churn(data, latent, sde, t_now, t_next, eta, gamma_cap=GAMMA_CAP,
                ceiling=None, s_noise=1.0, noise_fn=None,
                sigma_min=0.0, sigma_max=float("inf")):
    """Noise ``data`` up one churn step; return ``(data, t_start)``.

    ``t_start`` is the diffusion time the deterministic step should start from,
    which is ``t_now`` when no churn applies. Only latent coordinates are
    touched: observed values are conditioning data and must not be perturbed.

    ``noise_fn`` defaults to ``torch.randn_like``; the multi-observation sampler
    passes its ``_shared_noise`` so the hierarchy coordinates stay synchronized
    across observation rows.
    """
    if eta <= 0.0:
        return data, t_now
    sigma_now = float(sde.lambda_t(t_now).reshape(-1)[0])
    sigma_next = float(sde.lambda_t(t_next).reshape(-1)[0])
    sigma_hat, spread = churn_sigma(
        sigma_now, sigma_next, eta, gamma_cap=gamma_cap, ceiling=ceiling,
        sigma_min=sigma_min, sigma_max=sigma_max,
    )
    if spread <= 0.0:
        return data, t_now
    noise = (noise_fn or torch.randn_like)(data)
    data = data + s_noise * spread * noise * latent
    t_start = sde.time_of_lambda(
        torch.as_tensor(sigma_hat, dtype=t_now.dtype, device=t_now.device)
    ).reshape(1, 1)
    return data, t_start


def schedule_ceiling(sde, device):
    """The schedule's maximum noise level: ``lambda_t(1)``."""
    return float(sde.lambda_t(torch.ones(1, device=device)))
