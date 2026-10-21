"""Simulators with exact likelihoods, and exact-score oracles for the likelihood direction.

Every family: theta ~ N(0, I_2), mean mu = B theta in R^3, and given theta the three
observed dims are independent with a 1-D noise law:

    gauss     x_j = mu_j + nu e,            e ~ N(0, 1)
    student   x_j = mu_j + nu t,            t ~ Student-t(df = 3)
    bimodal   x_j = mu_j + m s + nu e,      s = +-1 with prob 1/2   (two peaks, m = 1)

Because the dims are independent given theta, everything needed is 1-D:

* ``log_lik(theta, x, smooth)`` is the exact log density of the model *smoothed by
  N(0, smooth^2)* -- the quantity the PF-ODE targets at ``eps`` with
  ``smooth = sigma(eps)`` -- and the plain model density for ``smooth = 0``.
* :class:`LikelihoodOracle` returns the exact diffused score of p(x | theta) for the
  likelihood mask (theta conditioned, x latent) in COMPASS's network convention
  (``score * sigma_t``). Gaussian and mixture laws are closed form; the Student-t
  convolution uses a fixed quadrature grid.
* :class:`HybridNetwork` uses a trained network above a noise level ``lambda_c`` and
  the oracle below it, which attributes the trained network's density error to
  noise bands.
"""

import math

import torch
import torch.nn as nn

B = torch.tensor([[1.0, 0.0], [0.5, 1.0], [-0.3, 0.8]])
THETA_DIM, X_DIM = 2, 3
FAMILIES = ("gauss", "student", "bimodal")
STUDENT_DF = 3.0
BIMODAL_SHIFT = 1.0


class Family:
    def __init__(self, name, nu=0.5):
        if name not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.name, self.nu = name, float(nu)

    @property
    def tag(self):
        return f"{self.name}_nu{self.nu:g}"

    # -- simulation ----------------------------------------------------------
    def simulate(self, n, generator):
        theta = torch.randn(n, THETA_DIM, generator=generator)
        mu = theta @ B.T
        if self.name == "gauss":
            noise = self.nu * torch.randn(n, X_DIM, generator=generator)
        elif self.name == "student":
            z = torch.randn(n, X_DIM, generator=generator)
            # chi^2_3 as a sum of three squared normals, drawn from ``generator`` so the
            # test sets are reproducible (torch.distributions.Chi2 uses the global RNG).
            chi2 = (torch.randn(n, X_DIM, int(STUDENT_DF), generator=generator) ** 2).sum(-1)
            noise = self.nu * z / torch.sqrt(chi2 / STUDENT_DF)
        else:
            s = torch.where(torch.rand(n, X_DIM, generator=generator) < 0.5, -1.0, 1.0)
            noise = BIMODAL_SHIFT * s + self.nu * torch.randn(n, X_DIM, generator=generator)
        return theta, mu + noise

    # -- exact (smoothed) log density -----------------------------------------
    def log_lik(self, theta, x, smooth=0.0):
        """Exact ``log (p(. | theta) * N(0, smooth^2))(x)`` summed over the 3 dims (float64)."""
        theta = torch.as_tensor(theta, dtype=torch.float64)
        x = torch.as_tensor(x, dtype=torch.float64)
        r = x - theta @ B.double().T
        return self._log_density_1d(r, float(smooth)).sum(-1)

    def _log_density_1d(self, r, smooth):
        nu = self.nu
        if self.name == "gauss":
            return _log_normal(r, math.sqrt(nu ** 2 + smooth ** 2))
        if self.name == "bimodal":
            sd = math.sqrt(nu ** 2 + smooth ** 2)
            return torch.logaddexp(_log_normal(r - BIMODAL_SHIFT, sd),
                                   _log_normal(r + BIMODAL_SHIFT, sd)) - math.log(2.0)
        if smooth == 0.0:
            return _log_student(r, nu)
        return torch.log(_student_convolved(r, nu, smooth)[0])

    # -- exact diffused score of the likelihood ---------------------------------
    def diffused_score_1d(self, r, lam):
        """d/dr log (p * N(0, lam^2))(r); ``r``, ``lam`` broadcastable float64 tensors."""
        nu = self.nu
        if self.name == "gauss":
            return -r / (nu ** 2 + lam ** 2)
        if self.name == "bimodal":
            var = nu ** 2 + lam ** 2
            a = _log_normal(r - BIMODAL_SHIFT, var.sqrt())
            b = _log_normal(r + BIMODAL_SHIFT, var.sqrt())
            w = torch.sigmoid(a - b)
            return -(r - BIMODAL_SHIFT * (2 * w - 1)) / var
        dens, dderiv = _student_convolved(r, nu, lam)
        return dderiv / dens


def _log_normal(r, sd):
    sd = torch.as_tensor(sd, dtype=torch.float64)
    return -0.5 * torch.log(2 * math.pi * sd ** 2) - 0.5 * (r / sd) ** 2


def _log_student(r, nu, df=STUDENT_DF):
    return (math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
            - math.log(nu) - (df + 1) / 2 * torch.log1p((r / nu) ** 2 / df))


_GRID = None


_GH = None


def _student_gh(r, lam, nu, df=STUDENT_DF, nodes=48):
    """``(q, q')`` by Gauss-Hermite in the kernel variable: accurate when the kernel is
    narrow relative to the local scale of the t density (|r| + nu)."""
    global _GH
    if _GH is None:
        import numpy as np
        xi, w = np.polynomial.hermite.hermgauss(nodes)
        _GH = (torch.tensor(xi), torch.tensor(w) / math.sqrt(math.pi))
    xi, w = (t.to(r.device) for t in _GH)
    y = r.unsqueeze(-1) - math.sqrt(2.0) * lam.unsqueeze(-1) * xi
    dens = torch.exp(_log_student(y, nu, df))
    g = -(df + 1) * y / (df * nu ** 2 + y ** 2)
    return (dens * w).sum(-1), (dens * g * w).sum(-1)


def _student_convolved(r, nu, lam, df=STUDENT_DF):
    """Dispatch: Gauss-Hermite around r for narrow kernels, the y-grid for wide ones."""
    r = torch.as_tensor(r, dtype=torch.float64)
    lam_t = torch.as_tensor(lam, dtype=torch.float64)
    lam_t = lam_t.expand_as(r) if lam_t.dim() else torch.full_like(r, float(lam_t))
    if float(lam_t.max()) == 0.0:
        return torch.exp(_log_student(r, nu, df)), None
    narrow = lam_t < 0.2 * (r.abs() + nu)
    q_gh, d_gh = _student_gh(r, lam_t, nu, df)
    if bool(narrow.all()):
        return q_gh, d_gh
    q_grid, d_grid = _student_grid(r, nu, lam_t, df)
    return torch.where(narrow, q_gh, q_grid), torch.where(narrow, d_gh, d_grid)


def _student_grid(r, nu, lam, df=STUDENT_DF):
    """``(q(r), q'(r))`` with ``q = t_nu * N(0, lam^2)``, by quadrature over the t variable.

    q(r) = int t(y) N(r - y; 0, lam^2) dy. The grid covers +-400 nu with a fine centre
    (spacing nu/200 inside +-20 nu); the t3 mass beyond 400 nu is < 1e-8.
    """
    global _GRID
    if _GRID is None or _GRID[0] != nu:
        inner = torch.linspace(-20 * nu, 20 * nu, 8001, dtype=torch.float64)
        outer = torch.logspace(math.log10(20 * nu), math.log10(400 * nu), 1500,
                               dtype=torch.float64)[1:]
        y = torch.cat([-outer.flip(0), inner, outer])
        w = torch.zeros_like(y)
        dy = y[1:] - y[:-1]
        w[1:] += dy / 2
        w[:-1] += dy / 2                                 # trapezoid weights
        wt = w * torch.exp(_log_student(y, nu, df))
        _GRID = (nu, y, wt)
    _, y, wt = _GRID
    r = torch.as_tensor(r, dtype=torch.float64)
    y, wt = y.to(r.device), wt.to(r.device)
    lam = torch.as_tensor(lam, dtype=torch.float64).expand_as(r) if torch.is_tensor(lam) \
        else torch.full_like(r, float(lam))
    if float(lam.max()) == 0.0:
        return torch.exp(_log_student(r, nu, df)), None
    out_q = torch.empty_like(r)
    out_d = torch.empty_like(r)
    flat_r, flat_l = r.reshape(-1), lam.reshape(-1)
    q_flat, d_flat = out_q.reshape(-1), out_d.reshape(-1)
    for start in range(0, flat_r.numel(), 256):
        rr = flat_r[start:start + 256, None]
        ll = flat_l[start:start + 256, None]
        u = rr - y[None, :]
        kern = torch.exp(-0.5 * (u / ll) ** 2) / (math.sqrt(2 * math.pi) * ll)
        q_flat[start:start + 256] = (kern * wt).sum(1)
        d_flat[start:start + 256] = (kern * (-u / ll ** 2) * wt).sum(1)
    return out_q, out_d


class LikelihoodOracle(nn.Module):
    """Exact diffused score of p(x | theta) for the mask theta-conditioned, x-latent (VESDE)."""

    def __init__(self, family, sde):
        super().__init__()
        self.family, self.sde = family, sde

    def forward(self, x, t, c, return_attn_weights=False):
        c0 = c[0] if c.dim() == 2 else c
        expected = torch.tensor([1.0] * THETA_DIM + [0.0] * X_DIM, device=c.device)
        if c.dim() == 2 and not torch.all(c == c0):
            raise ValueError("LikelihoodOracle needs one mask for the whole batch.")
        if not torch.equal(c0.float(), expected):
            raise ValueError("LikelihoodOracle only supports the likelihood mask.")
        z = x.to(torch.float64)
        sigma = self.sde.marginal_prob_std(t).to(z.device, torch.float64).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(z.shape[0])
        theta, xs = z[:, :THETA_DIM], z[:, THETA_DIM:]
        r = xs - theta @ B.to(z.device, torch.float64).T
        score = self.family.diffused_score_1d(r, sigma.unsqueeze(1).expand_as(r))
        out = torch.zeros_like(z)
        out[:, THETA_DIM:] = score * sigma.unsqueeze(1)
        out = out.to(x.dtype)
        return (out, torch.zeros(1, 1, 1)) if return_attn_weights else out


class HybridNetwork(nn.Module):
    """Trained network for lambda_c < lambda(t) < lambda_hi, the exact oracle elsewhere.

    ``lambda_hi`` just below lambda_max swaps in the oracle at the t = 1 endpoint only.
    """

    def __init__(self, trained, oracle, sde, lambda_c, lambda_hi=float("inf")):
        super().__init__()
        self.trained, self.oracle, self.sde = trained, oracle, sde
        self.lambda_c, self.lambda_hi = float(lambda_c), float(lambda_hi)

    def forward(self, x, t, c, return_attn_weights=False):
        lam = float(self.sde.lambda_t(t.reshape(-1)[:1]).item())
        net = self.trained if self.lambda_c < lam < self.lambda_hi else self.oracle
        return net(x, t, c, return_attn_weights) if return_attn_weights else net(x, t, c)
