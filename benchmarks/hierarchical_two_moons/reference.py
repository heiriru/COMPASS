"""Exact reference posterior for hierarchical Two Moons: collapsed SMC-MCMC.

Why not hsbibm's own NUTS reference
-----------------------------------
``HierarchicalTwoMoons._likelihood`` is not the simulator's likelihood:

* it assigns ``log L = invalid_log_prob = 1e-10`` (i.e. L ~ 1, not 1e-10) to
  parameters that cannot produce x (u < 0), while a valid point peaks at
  log L ~ 3.7, so impossible regions keep e^-3.7 of the peak density;
* it drops the polar Jacobian 1/r and the uniform density 1/pi of the angle.

The posterior of one local is also two arcs only 0.01 wide (the r noise),
which NUTS on the full (4 + 2N)-dimensional space mixes across poorly. This
reference uses the exact likelihood and integrates every local out exactly.

The exact likelihood, as a change of variables
----------------------------------------------
Given x_i, every theta_i that can produce it is reached by exactly one
(a, r, s), s = sign(z_0):

    z_0 = s (r cos a + 0.25 - x_0),  z_1 = x_1 - r sin a,  theta = R^T z,

valid iff r cos a + 0.25 - x_0 >= 0. The Jacobian |d theta / d(a, r)| = r
cancels the 1/r in the density of p, so for any f

    int f(theta) p(x_i | theta) d theta = sum_{s=+-1} E_{a~U, r~N}[ f(theta(a, r, s)) ].

Hence
    p(x_i | g)        = sum_s E_{a,r}[ TN(theta(a,r,s); mu, sigma) ]    (quadrature)
    p(theta_i | g, x) is drawn exactly by importance resampling of fresh
                      (a, r, s) draws with weights TN(theta; mu, sigma).

The sampler
-----------
Globals live in probit coordinates z (prior exactly N(0, I)). Their collapsed
posterior p(z | x_1:N) ∝ N(z; 0, I) prod_i p(x_i | g(z)) is sampled by a
likelihood-tempered SMC with adaptive random-walk Metropolis moves (an MCMC
kernel at every temperature, many more at the end), robust to the multimodality
the two moons can induce. Locals are then drawn exactly given each global draw.
Exact up to quadrature (256 angles x 3 radii x 2 branches) and Monte-Carlo error;
``self_check`` measures the latter on two independent runs.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import task as T

ANGLE_NODES = 256
RADIUS_NODES = 3
LOG_2PI = math.log(2 * math.pi)


def _log_diff_ndtr(lo, hi):
    """log(Phi(hi) - Phi(lo)) for lo < hi, stable in both tails."""
    flip = lo > 0
    lo_, hi_ = torch.where(flip, -hi, lo), torch.where(flip, -lo, hi)
    log_hi = torch.special.log_ndtr(hi_)
    log_lo = torch.special.log_ndtr(lo_)
    return log_hi + torch.log1p(-torch.exp((log_lo - log_hi).clamp(max=-1e-12)))


def physical(z):
    """Probit z (..., 4) -> (mu (..., 2), sigma (..., 2)) as torch tensors."""
    u = torch.special.ndtr(z)
    mu = T.MU_LOW + (T.MU_HIGH - T.MU_LOW) * u[..., :2]
    sigma = T.SIGMA_LOW + (T.SIGMA_HIGH - T.SIGMA_LOW) * u[..., 2:]
    return mu, sigma


def theta_of(a, r, s, x):
    """theta(a, r, s) for observations x (..., 2); returns (theta, valid)."""
    c, sn = math.cos(T.ROTATION), math.sin(T.ROTATION)
    abs_z0 = r * torch.cos(a) + T.BASE_OFFSET - x[..., 0]
    z0 = s * abs_z0
    z1 = x[..., 1] - r * torch.sin(a)
    # theta = R^T z with R = [[c, -s], [s, c]]
    theta = torch.stack([c * z0 + sn * z1, -sn * z0 + c * z1], dim=-1)
    return theta, abs_z0 >= 0


class Reference:
    def __init__(self, x, device="cpu", dtype=torch.float64):
        self.device, self.dtype = device, dtype
        self.x = torch.as_tensor(np.asarray(x), device=device, dtype=dtype)   # (N, 2)
        # Midpoint rule in a (the integrand has kinks where the arc leaves the
        # box or the valid region), Gauss-Hermite in r.
        a = A = (torch.arange(ANGLE_NODES, device=device, dtype=dtype) + 0.5) / ANGLE_NODES
        a = T.A_LOW + (T.A_HIGH - T.A_LOW) * A
        gh_x, gh_w = np.polynomial.hermite_e.hermegauss(RADIUS_NODES)
        r = torch.as_tensor(T.R_LOC + T.R_SCALE * gh_x, device=device, dtype=dtype)
        w_r = torch.as_tensor(gh_w / gh_w.sum(), device=device, dtype=dtype)
        aa, rr = torch.meshgrid(a, r, indexing="ij")
        ww = (w_r[None, :] / ANGLE_NODES).expand_as(aa)
        aa, rr, ww = aa.reshape(-1), rr.reshape(-1), ww.reshape(-1)
        nodes, logw = [], []
        for s in (1.0, -1.0):
            theta, valid = theta_of(aa[None, :], rr[None, :], s, self.x[:, None, :])
            inside = valid & (theta.abs() <= 1).all(-1)
            nodes.append(theta)
            logw.append(torch.where(inside, torch.log(ww)[None, :].expand_as(valid),
                                    torch.full_like(valid, -math.inf, dtype=dtype)))
        self.nodes = torch.cat(nodes, dim=1)            # (N, K, 2)
        self.logw = torch.cat(logw, dim=1)              # (N, K)
        keep = torch.isfinite(self.logw).any(0)
        self.nodes, self.logw = self.nodes[:, keep], self.logw[:, keep]
        # float32 is ample for the node sums (per-local logsumexp, ~1e-5 nats);
        # the sum over locals and the constants stay in float64.
        th = self.nodes.reshape(-1, 2)
        self.features = torch.stack([th[:, 0] ** 2, th[:, 1] ** 2, th[:, 0], th[:, 1]],
                                    dim=-1).to(torch.float32)
        self.logw_flat = self.logw.reshape(-1).to(torch.float32)

    # ------------------------------------------------------------------
    def log_likelihood(self, z, chunk=None):
        """sum_i log p(x_i | g(z)) for z (P, 4).

        -0.5 sum_d ((theta_d - mu_d)/sigma_d)^2 is linear in the node features
        (theta_0^2, theta_1^2, theta_0, theta_1), so all nodes x particles are one
        matrix product; only the logsumexp over nodes touches every pair.
        """
        n, k = self.logw.shape
        chunk = chunk or max(1, int(6e7 // (n * k)))
        out = []
        for start in range(0, z.shape[0], chunk):
            mu, sigma = physical(z[start:start + chunk])           # (B, 2)
            log_z = _log_diff_ndtr((-1 - mu) / sigma, (1 - mu) / sigma)
            const = -(torch.log(sigma) + 0.5 * LOG_2PI + log_z).sum(-1) \
                - 0.5 * ((mu / sigma) ** 2).sum(-1)                  # (B,)
            coeff = torch.cat([-0.5 / sigma ** 2, mu / sigma ** 2], dim=-1)   # (B, 4)
            log_tn = (self.features @ coeff.T.to(self.features.dtype)
                      + self.logw_flat[:, None])                    # (N*K, B)
            per_local = torch.logsumexp(log_tn.reshape(n, k, -1), dim=1)     # (N, B)
            out.append(per_local.to(self.dtype).sum(0) + n * const)
        return torch.cat(out)

    @staticmethod
    def log_prior(z):
        return -0.5 * (z ** 2).sum(-1)

    # ------------------------------------------------------------------
    def _mh(self, z, ll, beta, steps, scale, generator):
        """Random-walk Metropolis on prior * L^beta, proposal from the particle covariance."""
        cov = torch.cov(z.T) + 1e-9 * torch.eye(4, device=self.device, dtype=self.dtype)
        chol = torch.linalg.cholesky(cov)
        accepted = 0.0
        for _ in range(steps):
            noise = torch.randn(z.shape, generator=generator, device=self.device,
                                dtype=self.dtype)
            prop = z + scale * noise @ chol.T
            ll_prop = self.log_likelihood(prop)
            log_ratio = (self.log_prior(prop) + beta * ll_prop
                         - self.log_prior(z) - beta * ll)
            u = torch.rand(z.shape[0], generator=generator, device=self.device,
                           dtype=self.dtype)
            accept = torch.log(u) < log_ratio
            z = torch.where(accept[:, None], prop, z)
            ll = torch.where(accept, ll_prop, ll)
            rate = float(accept.to(self.dtype).mean())
            accepted += rate
            # Robbins-Monro style adaptation towards ~0.3 acceptance.
            scale = scale * math.exp(rate - 0.3)
        return z, ll, scale, accepted / max(steps, 1)

    def sample_globals(self, draws, seed, particles=1024, moves=8, final_moves=30, thin=10):
        """Tempered SMC to beta=1, then the particles run on as RWM chains.

        Returns (z draws (draws, 4) numpy, diagnostics). Draws are collected from
        all chains every ``thin`` steps after ``final_moves`` burn-in steps.
        """
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        z = torch.randn(particles, 4, generator=generator, device=self.device,
                        dtype=self.dtype)
        ll = self.log_likelihood(z)
        beta, scale, stages, rates = 0.0, 2.38 / 2.0, 0, []
        while beta < 1.0:
            # next temperature: ESS of the incremental weights = particles / 2
            def ess(b):
                w = torch.softmax((b - beta) * ll, 0)
                return float(1.0 / (w ** 2).sum())
            lo, hi = beta, 1.0
            if ess(1.0) >= particles / 2:
                new = 1.0
            else:
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if ess(mid) >= particles / 2 else (lo, mid)
                new = lo if lo > beta else hi
            w = torch.softmax((new - beta) * ll, 0)
            beta = new
            # systematic resampling
            u = (torch.rand(1, generator=generator, device=self.device, dtype=self.dtype)
                 + torch.arange(particles, device=self.device, dtype=self.dtype)) / particles
            index = torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=particles - 1)
            z, ll = z[index], ll[index]
            z, ll, scale, rate = self._mh(z, ll, beta, moves if beta < 1 else final_moves,
                                          scale, generator)
            rates.append(rate)
            stages += 1
            if stages > 500:
                raise RuntimeError("SMC did not reach beta=1")
        collected = [z]
        while sum(c.shape[0] for c in collected) < draws:
            z, ll, scale, rate = self._mh(z, ll, 1.0, thin, scale, generator)
            rates.append(rate)
            collected.append(z)
        chains = torch.stack(collected)                         # (C, P, 4)
        # Agreement of the first and last collection (Monte-Carlo noise check).
        first, last = chains[0], chains[-1]
        drift = ((first.mean(0) - last.mean(0)).abs()
                 / torch.sqrt(0.5 * (first.var(0) + last.var(0)) * 2 / particles)).max()
        flat = chains.reshape(-1, 4)
        pick = torch.randperm(flat.shape[0], generator=generator, device=self.device)[:draws]
        return flat[pick].cpu().numpy(), {
            "stages": stages, "final_acceptance": rates[-1], "particles": particles,
            "collections": len(collected), "max_mean_drift_z": float(drift)}

    def sample_locals(self, z, seed, proposals=2048, chunk=64):
        """theta_i | g(z), x_i exactly by importance resampling: (S, N, 2) numpy."""
        generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        z = torch.as_tensor(z, device=self.device, dtype=self.dtype)
        n = self.x.shape[0]
        out = []
        for start in range(0, z.shape[0], chunk):
            mu, sigma = physical(z[start:start + chunk])                   # (B, 2)
            b = mu.shape[0]
            shape = (b, n, proposals)
            a = T.A_LOW + (T.A_HIGH - T.A_LOW) * torch.rand(
                shape, generator=generator, device=self.device, dtype=self.dtype)
            r = T.R_LOC + T.R_SCALE * torch.randn(
                shape, generator=generator, device=self.device, dtype=self.dtype)
            s = torch.where(torch.rand(shape, generator=generator, device=self.device,
                                       dtype=self.dtype) < 0.5, 1.0, -1.0)
            theta, valid = theta_of(a, r, s, self.x[None, :, None, :])      # (B, N, M, 2)
            inside = valid & (theta.abs() <= 1).all(-1)
            q = ((theta - mu[:, None, None, :]) / sigma[:, None, None, :]) ** 2
            logits = torch.where(inside, -0.5 * q.sum(-1),
                                 torch.full_like(q[..., 0], -math.inf))
            probs = torch.softmax(logits.reshape(-1, proposals), dim=-1)
            pick = torch.multinomial(probs, 1, generator=generator)[:, 0]
            chosen = theta.reshape(-1, proposals, 2)[torch.arange(b * n, device=self.device), pick]
            out.append(chosen.reshape(b, n, 2))
        return torch.cat(out).cpu().numpy()

    def sample(self, draws, seed, **kwargs):
        """Joint reference draws: (globals physical (S, 4), locals (S, N, 2), diag)."""
        z, diagnostics = self.sample_globals(draws, seed, **kwargs)
        locals_ = self.sample_locals(z, seed)
        return T.z_to_globals(z), locals_, diagnostics
