"""Exact-score stand-ins for a trained network, for isolating composition rules.

``LinearGaussianHierarchy`` is a linear-Gaussian two-level model with node layout
``[g (H), l (L), x (X)]``:

    g ~ N(0, s_g^2 I),   l = A g + tau eps_l,   x = B l + C g + nu eps_x.

It is multivariate on purpose. With ``H >= 2`` a single subject's posterior
over ``g`` is correlated, so a mean-field factor is visibly wrong. Every
conditional that a mask can ask for is Gaussian, so :class:`OracleNetwork`
returns the *exact* diffused score of that conditional in the output
convention of a COMPASS network (``score * sigma_t``, since
``output_scale_function`` divides by ``sigma_t``). Setting ``mean_field=True``
returns only the per-coordinate marginal scores. That reproduces the
production shared trunk, whose latents never attend each other.
"""

import torch
import torch.nn as nn


class LinearGaussianHierarchy:
    def __init__(self, A, B, C, s_g=1.0, tau=0.5, nu=0.5):
        self.A = torch.as_tensor(A, dtype=torch.float64)     # (L, H)
        self.B = torch.as_tensor(B, dtype=torch.float64)     # (X, L)
        self.C = torch.as_tensor(C, dtype=torch.float64)     # (X, H)
        self.L, self.H = self.A.shape
        self.X = self.B.shape[0]
        self.s_g, self.tau, self.nu = float(s_g), float(tau), float(nu)
        H, L, X = self.H, self.L, self.X
        # z = M eps with eps ~ N(0, I_(H+L+X)).
        Mg = torch.cat([self.s_g * torch.eye(H), torch.zeros(H, L + X)], 1).double()
        Ml = self.A @ Mg + torch.cat([torch.zeros(L, H), self.tau * torch.eye(L),
                                      torch.zeros(L, X)], 1).double()
        Mx = self.B @ Ml + self.C @ Mg + torch.cat([torch.zeros(X, H + L),
                                                    self.nu * torch.eye(X)], 1).double()
        M = torch.cat([Mg, Ml, Mx], 0)
        self.covariance = M @ M.T
        self.nodes = H + L + X
        self.global_indices = list(range(H))
        self.local_indices = list(range(H, H + L))
        self.observed_indices = list(range(H + L, H + L + X))

    @property
    def roles(self):
        return {"g": self.global_indices, "l": self.local_indices, "x": self.observed_indices}

    def conditional(self, mask):
        """Latent indices and ``(gain, covariance)`` of ``latent | observed`` for a 0/1 mask."""
        mask = torch.as_tensor(mask).bool()
        U = torch.where(~mask)[0]
        O = torch.where(mask)[0]
        S = self.covariance
        if len(O) == 0:
            return U, torch.zeros(len(U), 0, dtype=torch.float64), S[U][:, U]
        gain = S[U][:, O] @ torch.linalg.inv(S[O][:, O])
        cov = S[U][:, U] - gain @ S[O][:, U]
        return U, gain, 0.5 * (cov + cov.T)

    def simulate(self, count, generator=None):
        eps = torch.randn(count, self.nodes, generator=generator, dtype=torch.float64)
        L = torch.linalg.cholesky(self.covariance + 1e-12 * torch.eye(self.nodes, dtype=torch.float64))
        return eps @ L.T

    def global_posterior(self, observations):
        """Exact ``p(g | x_1..x_J)`` mean and covariance for ``observations (J, X)``."""
        S = self.covariance
        g, x = self.global_indices, self.observed_indices
        Sgg, Sxg, Sxx = S[g][:, g], S[x][:, g], S[x][:, x]
        D = Sxg @ torch.linalg.inv(Sgg)
        R = Sxx - D @ Sgg @ D.T
        Rinv = torch.linalg.inv(R)
        obs = torch.as_tensor(observations, dtype=torch.float64)
        precision = torch.linalg.inv(Sgg) + obs.shape[0] * D.T @ Rinv @ D
        cov = torch.linalg.inv(precision)
        mean = cov @ (D.T @ Rinv @ obs.sum(0))
        return mean, cov

    def local_posterior(self, observations):
        """Exact per-subject ``p(l_j | x_1..x_J)`` means ``(J, L)`` and covariances ``(J, L, L)``."""
        obs = torch.as_tensor(observations, dtype=torch.float64)
        mean_g, cov_g = self.global_posterior(obs)
        mask = torch.zeros(self.nodes)
        mask[self.global_indices + self.observed_indices] = 1
        U, gain, cov = self.conditional(mask)
        H = self.H
        F, G = gain[:, :H], gain[:, H:]
        means = (F @ mean_g).unsqueeze(0) + obs @ G.T
        covs = (F @ cov_g @ F.T + cov).expand(obs.shape[0], -1, -1)
        return means, covs


class OracleNetwork(nn.Module):
    """Exact diffused conditional scores of a :class:`LinearGaussianHierarchy` (VESDE only)."""

    def __init__(self, hierarchy, sde, mean_field=False):
        super().__init__()
        self.hierarchy = hierarchy
        self.sde = sde
        self.mean_field = bool(mean_field)
        self._cache = {}
        self.calls = 0

    def forward(self, x, t, c, return_attn_weights=False):
        self.calls += 1
        dtype = x.dtype
        z = x.to(torch.float64)
        c = c.to(z.device)
        sigma = self.sde.marginal_prob_std(t).to(z.device, torch.float64).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(z.shape[0])
        out = torch.zeros_like(z)
        patterns, inverse = torch.unique(c.round().to(torch.int64), dim=0, return_inverse=True)
        for p_index, pattern in enumerate(patterns):
            rows = torch.where(inverse == p_index)[0]
            key = tuple(pattern.tolist())
            if key not in self._cache:
                self._cache[key] = self.hierarchy.conditional(pattern.cpu())
            U, gain, cov = self._cache[key]
            U, gain, cov = U.to(z.device), gain.to(z.device), cov.to(z.device)
            O = torch.where(pattern.bool())[0]
            mean = z[rows][:, O] @ gain.T if len(O) else torch.zeros(len(rows), len(U), dtype=torch.float64, device=z.device)
            delta = z[rows][:, U] - mean
            var = sigma[rows] ** 2
            if self.mean_field:
                score = -delta / (torch.diagonal(cov).unsqueeze(0) + var.unsqueeze(1))
            else:
                eye = torch.eye(len(U), dtype=torch.float64, device=z.device)
                cov_t = cov.unsqueeze(0) + var.reshape(-1, 1, 1) * eye
                score = -torch.linalg.solve(cov_t, delta.unsqueeze(-1)).squeeze(-1)
            block = out[rows]
            block[:, U] = score * sigma[rows].unsqueeze(1)
            out[rows] = block
        out = out.to(dtype)
        if return_attn_weights:
            return out, torch.zeros(1, 1, 1)
        return out


def oracle_model(hierarchy, mean_field=False, sigma=25.0):
    """A ``ScoreBasedInferenceModel`` whose network is the exact oracle."""
    from ..ScoreBasedInferenceModel import ScoreBasedInferenceModel
    model = ScoreBasedInferenceModel(nodes_size=hierarchy.nodes, sde_type="vesde",
                                     sigma=sigma, hidden_size=8, depth=1, num_heads=1,
                                     mlp_ratio=1)
    model.model = OracleNetwork(hierarchy, model.sde, mean_field=mean_field)
    return model
