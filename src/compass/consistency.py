"""Fisher-identity consistency penalty for the globals (roadmap 2.8; experimental).

For noisy globals ``g_t`` (noise level ``lambda``, ``y = x / alpha`` coordinates) the
posterior score averaged over the predictive is the diffused prior score::

    E_{x ~ p(x | g_t)} [ grad_g log p_t(g_t | x) ] = grad_g log p_t(g_t)     (*)

because ``grad log p_t(g_t | x) = grad log p_t(g_t) + grad log p_t(x | g_t)`` and a
score has mean zero. The prior is known, so (*) is a constraint on the network
that needs no oracle. It targets the bias ``b(g_t) = E[s_hat(g_t | x) | g_t] -
s_prior,t(g_t)`` that the composition multiplies by ``n``
(``sum_j s_hat(g | x_j) - (n - 1) s_prior``), which plain DSM leaves uncontrolled:
DSM fits the score on average over the joint, not its predictive mean at each
``g_t``.

Estimator, with only the training set (no simulator)
----------------------------------------------------
Training rows ``(g0_j, l0_j, x_j)`` are draws of the training joint, so under
``g_t = g0 + lambda eps`` the conditional ``p(g0, l0, x | g_t)`` is the training
joint reweighted by ``N(g_t; g0_j, lambda^2 I)``. For each of ``anchors`` noisy
anchors, ``neighbours`` rows are resampled (with replacement) from those
self-normalised weights over the *whole* training set -- keeping only the nearest
rows instead truncates ``p(g0 | g_t)`` at high noise, where it is as wide as the
prior, and biases the average. The network scores ``g_t`` against each drawn
row's ``x_j`` (the
row's other latents at their own noisy draw ``l0_j + lambda eps'``, so the identity
holds for a joint-score network too: the conditional-on-``l_t`` term has mean zero
under ``l_t | g_t, x``), and

    penalty = weight * mean_a lambda_a^2 || mean_k s_hat(g_t^a | x_k) - s_prior(g_t^a) ||^2

in epsilon units like the DSM loss. Anchors whose effective sample size is below
``min_ess`` are dropped (at small lambda the weights degenerate and the estimate
is biased). The prior score must be the *training* prior's (a wide training prior
is what the rows were drawn from).
"""

import torch


class FisherConsistency:
    """Callable penalty; pass as ``SBIm.train(..., fisher_consistency=...)``.

    Args:
        prior_score: ``f(g_y (A, k), lambda (A, 1)) -> (A, k)``, the diffused
            training-prior score of the globals in ``y`` coordinates.
        indices: node indices of the globals.
        observed: node indices conditioned in the posterior mask (the data);
            every other node is latent.
        weight: multiplier of the penalty (per anchor, in epsilon units).
        anchors: noisy anchors per training step.
        neighbours: training rows resampled from the importance weights per anchor.
        min_ess: anchors with fewer effective rows are not penalised.
        every: apply on every ``every``-th training step.
    """

    def __init__(self, prior_score, indices, observed, weight=1.0, anchors=8,
                 neighbours=64, min_ess=4.0, every=1):
        self.prior_score = prior_score
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.observed = torch.as_tensor(observed, dtype=torch.long)
        self.weight = float(weight)
        self.anchors = int(anchors)
        self.neighbours = int(neighbours)
        self.min_ess = float(min_ess)
        self.every = max(1, int(every))
        self.step = 0
        self.history = []          # (mean ess, kept anchors, mean lambda^2 |b|^2) per call

    def bind(self, trainer, train_data):
        self.pool = train_data
        self._device_pool = None

    def _pool_on(self, device):
        if self._device_pool is None or self._device_pool.device != torch.device(device):
            self._device_pool = self.pool.to(device)
            self._pool_globals = self._device_pool[:, self.indices.to(device)]
        return self._device_pool, self._pool_globals

    def __call__(self, trainer, device):
        self.step += 1
        if self.step % self.every:
            return torch.zeros((), device=device)
        pool, pool_g = self._pool_on(device)
        idx = self.indices.to(device)
        nodes = pool.shape[1]
        count = min(self.neighbours, pool.shape[0])
        anchors = torch.randint(0, pool.shape[0], (self.anchors,), device=device)
        t = trainer._sample_timesteps(self.anchors).to(device).reshape(-1, 1)
        sde = trainer.sde
        lam = sde.lambda_t(t).reshape(-1, 1).to(pool.dtype)
        alpha = sde.alpha_t(t).reshape(-1, 1).to(pool.dtype)
        with torch.no_grad():
            g_t = pool_g[anchors] + lam * torch.randn_like(pool_g[anchors])        # (A, k)
            log_w = -torch.cdist(g_t, pool_g).square() / (2.0 * lam.square())       # (A, N)
            full = torch.softmax(log_w, dim=1)
            ess = 1.0 / full.square().sum(1)                                        # (A,)
            rows = torch.multinomial(full, count, replacement=True)                 # (A, K)
            keep = ess >= self.min_ess
        self.history.append((float(ess.mean()), int(keep.sum()), float("nan")))
        if not bool(keep.any()):
            return torch.zeros((), device=device)

        anchors_kept = int(keep.sum())
        g_t, lam, alpha, t = g_t[keep], lam[keep], alpha[keep], t[keep]
        rows = rows[keep]
        data = pool[rows]                                                           # (A, K, nodes)
        mask = torch.zeros(nodes, device=device, dtype=pool.dtype)
        mask[self.observed.to(device)] = 1.0
        latent = (1.0 - mask)
        lam3, alpha3 = lam.unsqueeze(1), alpha.unsqueeze(1)
        noisy = alpha3 * (data + lam3 * torch.randn_like(data))                    # x coordinates
        noisy[:, :, idx] = (alpha3 * g_t.unsqueeze(1)).expand(-1, count, -1)
        inputs = latent * noisy + mask * data
        flat = inputs.reshape(-1, nodes)
        times = t.repeat_interleave(count, dim=0)
        masks = mask.expand(flat.shape[0], -1)
        score = trainer._get_score(flat, times, masks)                              # x coords
        score_y = alpha.repeat_interleave(count, dim=0) * score[:, idx]
        mean_score = score_y.reshape(anchors_kept, count, -1).mean(1)
        with torch.no_grad():
            prior = self.prior_score(g_t, lam).to(mean_score.dtype)
        bias = mean_score - prior
        per_anchor = lam.square().reshape(-1) * bias.square().sum(-1)
        value = per_anchor.mean()
        self.history[-1] = (float(ess.mean()), anchors_kept, float(value.detach()))
        return self.weight * value


def gaussian_prior_score(mean, std):
    """Diffused score of an independent ``N(mean, std^2)`` prior, for ``prior_score``."""
    mean = torch.as_tensor(mean, dtype=torch.float32)
    var = torch.as_tensor(std, dtype=torch.float32).square()

    def score(g_y, lam):
        return -(g_y - mean.to(g_y)) / (var.to(g_y) + lam.square())

    return score
