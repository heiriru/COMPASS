"""Covariance head for the Jacobian compositions: predict, don't differentiate.

``gauss_jacobian`` gets every row's backward precision from Tweedie's second-order
identity, ``Lambda_j = [lam^2 (I + lam^2 grad s_j)]^-1``, i.e. from the derivative of
the score network. Nothing in denoising score matching constrains ``grad s``, so the
Jacobian is noisier than the score (measured 0.82 vs 0.56 relative error on the GM
benchmark), is 30% antisymmetric, and costs D extra network passes (FD or JVP).

This module learns the same quantity directly. A small post-hoc network ``K(y, x, lam)``
is fitted to a *frozen* score network by the Gaussian negative log-likelihood of the
denoising residual (second-order DSM)::

    D(y)   = y + lam^2 s(y, x, lam)                (frozen network's Tweedie mean)
    Lambda = K + lam^-2 I,   K = L L^T             (L lower-triangular, softplus diagonal)
    loss   = 1/2 (theta_0 - D)^T Lambda (theta_0 - D) - 1/2 log det Lambda,

with ``y = theta_0 + lam eps``. At the optimum ``Lambda^-1 = Cov(theta_0 | y, x)`` (around
the network's own mean), which is exactly what the Jacobian rule estimates. ``K`` is the
lambda-free information the JAC refresh cache already stores; for a Gaussian posterior
``N(mu, C)`` it is ``C^-1`` at every ``(y, lam)``. ``K = L L^T`` is symmetric and PSD by
construction, which is the production curvature clamp ``H >= 0`` built in.

Usage::

    head = train_covariance_head(sbim, simulate, latent_nodes, observed_nodes, ...)
    sampler.set_covariance_head(head)        # MultiObsSampler, Jacobian corrections
    # or: torch.save(head, path); env COMPASS_COVARIANCE_HEAD=path

Remove the feature by deleting this module, its test and the three-line hook in
``MultiObsSampler._jacobian_joint_precision``.
"""
import math

import torch
import torch.nn as nn


class CovarianceHead(nn.Module):
    """``(y_latent (B, D), x_observed (B, X), lam (B,)) -> K (B, D, D)``, PSD."""

    def __init__(self, latent_nodes, observed_nodes, hidden=128, depth=3,
                 latent_scale=None, observed_scale=None):
        super().__init__()
        self.latent_nodes = [int(i) for i in latent_nodes]
        self.observed_nodes = [int(i) for i in observed_nodes]
        D, X = len(self.latent_nodes), len(self.observed_nodes)
        self.D = D
        self.register_buffer("latent_scale", torch.ones(D) if latent_scale is None
                             else torch.as_tensor(latent_scale, dtype=torch.float32))
        self.register_buffer("observed_scale", torch.ones(X) if observed_scale is None
                             else torch.as_tensor(observed_scale, dtype=torch.float32))
        layers, width = [], D + X + 4
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, D * (D + 1) // 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        rows, cols = torch.tril_indices(D, D)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)

    def forward(self, y, x, lam):
        lam = lam.reshape(-1, 1).to(y.dtype)
        s = self.latent_scale.to(y.dtype)
        # EDM-style input scaling: unit variance at every noise level
        y_in = y / torch.sqrt(s ** 2 + lam ** 2)
        log_lam = torch.log(lam)
        features = torch.cat([y_in, x / self.observed_scale.to(y.dtype), log_lam / 4.0,
                              torch.sin(log_lam), torch.cos(log_lam), 1.0 / (1.0 + lam ** 2)], -1)
        raw = self.out(self.trunk(features))
        L = torch.zeros(y.shape[0], self.D, self.D, dtype=raw.dtype, device=raw.device)
        L[:, self.rows, self.cols] = raw
        diag = torch.arange(self.D, device=raw.device)
        # softplus diagonal, in units of the prior scale (K ~ 1/s^2 when uninformative)
        L[:, diag, diag] = nn.functional.softplus(raw[:, self.cols == self.rows] + 0.5) + 1e-4
        L = L / s.reshape(1, -1, 1)
        return L @ L.mT

    def precision(self, y, x, lam):
        """``Lambda = K + lam^-2 I`` in float64."""
        K = self.forward(y.float(), x.float(), lam.float()).double()
        eye = torch.eye(self.D, dtype=torch.float64, device=K.device)
        return K + eye / lam.reshape(-1, 1, 1).double() ** 2


def gaussian_nll(theta, denoised, information, lam):
    """Per-row NLL of ``theta - denoised`` under precision ``K + lam^-2 I`` (constants dropped)."""
    D = theta.shape[-1]
    eye = torch.eye(D, dtype=information.dtype, device=information.device)
    precision = information + eye / lam.reshape(-1, 1, 1) ** 2
    r = (theta - denoised).unsqueeze(-1)
    quad = (r.mT @ precision @ r).reshape(-1)
    logdet = torch.linalg.cholesky(precision).diagonal(dim1=-2, dim2=-1).log().sum(-1) * 2
    return 0.5 * quad - 0.5 * logdet


def network_denoiser(sbim, latent_nodes, observed_nodes):
    """``denoise(y, x, lam) -> y + lam^2 s`` of a frozen COMPASS model, y coordinates."""
    nodes = sbim.nodes_size if hasattr(sbim, "nodes_size") else len(latent_nodes) + len(observed_nodes)

    @torch.no_grad()
    def denoise(y, x, lam):
        sde = sbim.sde
        t = sde.time_of_lambda(lam.float()).reshape(-1, 1)
        alpha = sde.alpha_t(t).to(y.device)
        state = torch.zeros(y.shape[0], nodes, device=y.device)
        state[:, latent_nodes] = alpha * y.float()
        state[:, observed_nodes] = x.float()
        mask = torch.zeros(nodes, device=y.device)
        mask[observed_nodes] = 1.0
        s = sbim.model(x=state, t=t, c=mask.expand(y.shape[0], -1))
        s = sbim.output_scale_function(t, s)
        return y + lam.reshape(-1, 1) ** 2 * (alpha * s[:, latent_nodes])
    return denoise


def train_covariance_head(sbim, simulate, latent_nodes, observed_nodes, lam_min, lam_max,
                          steps=20_000, batch=2048, lr=1e-3, hidden=128, depth=3,
                          device="cpu", seed=0, log_every=1000, denoiser=None, verbose=True):
    """Fit a :class:`CovarianceHead` to a frozen score network.

    ``simulate(count, generator) -> (theta (count, D), x (count, X))`` draws fresh joint
    samples (latents in ``latent_nodes`` order). ``lam`` is drawn log-uniformly in
    ``[lam_min, lam_max]``. ``denoiser`` overrides the network denoiser (e.g. an oracle).
    """
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    theta, x = simulate(20_000, generator)
    head = CovarianceHead(latent_nodes, observed_nodes, hidden=hidden, depth=depth,
                          latent_scale=theta.float().std(0), observed_scale=x.float().std(0)).to(device)
    denoise = denoiser or network_denoiser(sbim, latent_nodes, observed_nodes)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    log_min, log_max = math.log(lam_min), math.log(lam_max)
    history = []
    running = 0.0
    for step in range(1, steps + 1):
        theta, x = simulate(batch, generator)
        theta, x = theta.to(device, torch.float32), x.to(device, torch.float32)
        lam = torch.exp(log_min + (log_max - log_min) * torch.rand(batch, generator=generator)).to(device)
        y = theta + lam.reshape(-1, 1) * torch.randn(theta.shape, generator=generator).to(device)
        denoised = denoise(y, x, lam)
        K = head(y, x, lam)
        loss = gaussian_nll(theta.double(), denoised.double(), K.double(), lam.double()).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
        optimizer.step()
        schedule.step()
        running += float(loss)
        if step % log_every == 0:
            history.append((step, running / log_every))
            if verbose:
                print(f"[covariance head] step {step} nll {running / log_every:.4f}", flush=True)
            running = 0.0
    head.history = history
    return head.eval()
