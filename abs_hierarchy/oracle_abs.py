"""The exact diffused row score of the folded hierarchy, in the network's place.

Mirrors ``Gaussian_mixtures_compositional_score_modelling/oracle.py`` (same
interface, same output convention ``marginal_prob_std(t) * score``) for

    g ~ N(0, s_g^2),  l | g ~ N(|g|, s_l^2),  x | l ~ N(l, s_x^2).

Closed form. Write u = s*g on the half-line s*g > 0 (s = +-1). There (u, l, x) is
jointly Gaussian with the GM covariance at mu_0 = 0, so the row posterior is

    p(g, l | x) = 1/2 sum_s  TN_s,      TN_s = N((g, l); m_s, C_s) restricted to s*g > 0,

with m_s = (s m_u, m_l) and C_s = C with Cov(g, l) multiplied by s; both halves
carry the same normaliser Phi(m_u / sqrt(C_uu)). Adding lambda^2 I to a truncated
Gaussian gives, exactly,

    p_lambda(y) ∝ N(y; m_s, C_s + lambda^2 I) * Phi(s mu_g(y) / sqrt(S_gg)),
    S = (C_s^-1 + I / lambda^2)^-1,   mu(y) = S (C_s^-1 m_s + y / lambda^2),

and the g-marginal is the same formula in one dimension, because the
truncation acts on g only. Scores are taken by autograd of these log densities.

``form``:
    "joint"         both outputs from log p_t(g, l | x): what gauss_hierarchical,
                    gauss_jacobian(_FD) and F-NPSE are derived for.
    "hierarchical"  g from log p_t(g | x) (marginal), l from log p_t(g, l | x):
                    what a hierarchical-attention network represents and what
                    arruda_direct needs.
``lean_model`` (l centred on |g| + a g) is the same with slope 1 + s a on branch s
and unequal branch weights N(x; 0, var_x_s); pass ``model=``.
Two masks are served: the posterior p(g, l | x) and p(l | g, x) (exactly
Gaussian; arruda_direct draws its locals from it). Any other pattern raises, so a rule silently asking for a different conditional cannot
get a wrong answer.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Gaussian_mixtures_compositional_score_modelling"))

import torch  # noqa: E402

from abs_model import DEFAULT  # noqa: E402

FORMS = ("joint", "hierarchical")


def row_constants(model=DEFAULT, slope=1.0):
    """``(gain, C, var_x)``: on a half-line l = slope * u + noise, (u, l, x) is Gaussian
    and the conditional of (u, l) on x has mean gain * x and covariance C.

    slope is 1 for ``abs_model``; ``lean_model`` (l centred on |g| + a g) has
    slope 1 + s a on branch s.
    """
    g = model.sigma_g ** 2
    k = float(slope)
    l = k * k * g + model.sigma_l ** 2
    xx = l + model.sigma_x ** 2
    cov = torch.tensor([[g, k * g, k * g], [k * g, l, l], [k * g, l, xx]], dtype=torch.float64)
    block_lo = cov[:2, 2:]
    gain = (block_lo / cov[2, 2]).reshape(2)              # mean = gain * x
    conditional = cov[:2, :2] - block_lo @ block_lo.T / cov[2, 2]
    return gain, conditional, xx


def _log_trunc_diffused(y, mean, cov, sign, variance):
    """log N(y; mean, cov + v I) + log Phi(sign * mu_g / sqrt(S_gg)), batched.

    y, mean: (rows, d); cov: (d, d) shared; variance: (rows,).
    """
    d = y.shape[1]
    eye = torch.eye(d, dtype=y.dtype, device=y.device)
    v = variance.reshape(-1, 1, 1)
    total = cov.unsqueeze(0) + v * eye
    delta = (y - mean).unsqueeze(-1)
    solved = torch.linalg.solve(total, delta)
    log_normal = -0.5 * ((delta * solved).sum((1, 2)) + torch.logdet(total))
    # S = (C^-1 + I/v)^-1 = C (C + v I)^-1 v ;  mu = S (C^-1 m + y / v) = m + C (C + vI)^-1 (y - m)
    gain = cov.unsqueeze(0) @ torch.linalg.inv(total)
    mu = mean + (gain @ delta).squeeze(-1)
    S = (gain @ (v * eye)).clone()
    z = sign * mu[:, 0] / S[:, 0, 0].clamp_min(1e-300).sqrt()
    return log_normal + torch.special.log_ndtr(z)


def log_density(y, x, variance, model=DEFAULT, marginal=False):
    """log p_t(y | x) up to a constant; ``marginal`` gives the g-only version."""
    lean = float(getattr(model, "a", 0.0))
    terms = []
    for sign in (1.0, -1.0):
        gain, conditional, var_x = row_constants(model, 1.0 + sign * lean)
        gain, conditional = gain.to(y), conditional.to(y)
        # Branch weight: p(x, branch s) ∝ N(x; 0, var_x_s) Phi_s, and Phi_s cancels the
        # truncated normaliser, leaving N(x; 0, var_x_s). Equal for both signs when a = 0.
        evidence = -0.5 * (x.reshape(-1) ** 2 / var_x + torch.log(torch.as_tensor(var_x)))
        flip = torch.tensor([sign, 1.0], dtype=y.dtype, device=y.device)
        mean = x.reshape(-1, 1) * gain * flip
        cov = conditional * torch.outer(flip, flip)
        if marginal:
            terms.append(evidence + _log_trunc_diffused(y[:, :1], mean[:, :1], cov[:1, :1], sign, variance))
        else:
            terms.append(evidence + _log_trunc_diffused(y, mean, cov, sign, variance))
    return torch.logsumexp(torch.stack(terms), 0)


def exact_score(y, x, variance, form="joint", model=DEFAULT):
    """``(rows, 2)`` score of p_t at latent state ``y = (g_t, l_t)``."""
    with torch.enable_grad():
        y = y.detach().to(torch.float64).requires_grad_(True)
        joint = torch.autograd.grad(log_density(y, x, variance, model).sum(), y,
                                    create_graph=torch.is_grad_enabled() and False)[0]
        if form == "joint":
            return joint.detach()
        yg = y.detach().clone().requires_grad_(True)
        marginal = torch.autograd.grad(
            log_density(yg, x, variance, model, marginal=True).sum(), yg)[0][:, 0]
        return torch.stack([marginal.detach(), joint[:, 1].detach()], 1)


class FoldedOracleScore(torch.nn.Module):
    def __init__(self, sde, form="joint", model=DEFAULT):
        super().__init__()
        if form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}")
        self.sde, self.form, self.folded = sde, form, model
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x, t, c, return_attn_weights=False, **unused):
        mask = c > 0.5
        if mask.dim() == 1:
            mask = mask.expand(x.shape[0], -1)
        variance = self.sde.lambda_t(t).to(x.device).reshape(-1).square().to(torch.float64)
        variance = variance.expand(x.shape[0])
        state = x.to(torch.float64)
        out = torch.zeros_like(state)
        posterior = torch.tensor([False, False, True], device=x.device)
        local = torch.tensor([True, False, True], device=x.device)
        rows = (mask == posterior).all(1)
        if rows.any():
            out[rows, :2] = exact_score(state[rows, :2], state[rows, 2], variance[rows],
                                        self.form, self.folded)
        rows_l = (mask == local).all(1)
        if rows_l.any():
            # p(l | g, x) = N(q (|g|/s_l^2 + x/s_x^2), q): Gaussian, so diffused in closed form.
            m = self.folded
            centre = state[rows_l, 0].abs() + float(getattr(m, "a", 0.0)) * state[rows_l, 0]
            mean = m.q * (centre / m.sigma_l ** 2 + state[rows_l, 2] / m.sigma_x ** 2)
            out[rows_l, 1] = -(state[rows_l, 1] - mean) / (m.q + variance[rows_l])
        if not bool((rows | rows_l).all()):
            raise NotImplementedError(
                "the folded oracle serves p(g, l | x) and p(l | g, x) only; got "
                f"{torch.unique(mask.int(), dim=0).tolist()}")
        result = (out * self.sde.marginal_prob_std(t).to(x.device).reshape(-1, 1)).to(x.dtype)
        return (result, None) if return_attn_weights else result


def oracle_model(form, device, sigma=15.0, model=DEFAULT):
    """An ``SBIm`` stand-in (GM's ``OracleModel``) with the folded score inside."""
    import oracle as gm_oracle

    network = gm_oracle.OracleModel(sigma=sigma, device=device)
    network.model = FoldedOracleScore(network.sde, form, model).to(device)
    network.mixture = None
    return network
