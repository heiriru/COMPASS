"""Residual parameterization, compositional loss and preconditioning, for GM.

Three interventions, each isolated so the ablation can attribute an effect to
one of them, and each implemented as a *wrapper* around the stock COMPASS
network rather than an edit to it -- the backbone, the optimizer, the mask menu
and the DSM loss stay bit-for-bit what the baseline uses.

1. :class:`ResidualScore` -- Part IX's ``s_phi = s_prior + r_phi``
--------------------------------------------------------------------
COMPASS trains ``s_post`` and composes with ``Delta s = s_post - s_prior``.
Experiment 1 measured ``|s_post| / |Delta s| ~ 7`` at the states the sampler
visits, so the quantity composition consumes is a small difference of two large
learned/known numbers. This supplies the large one exactly and lets the network
carry only the difference.

The algebra is a one-liner in the network's *raw* convention. COMPASS reads a
score as ``raw / sigma(t)``, so writing

    raw_phi = sigma(t) * s_base + backbone(x, t, c)

gives ``s_phi = s_base + backbone / sigma``. The denoising objective is
``|| eps + sigma * s_phi ||^2 = || eps + sigma * s_base + backbone ||^2``, whose
minimizer is ``backbone = sigma * (s_post - s_base)``. So the *unchanged* DSM
loss trains the residual directly -- no new objective, no new parameters, and at
initialization the model is exactly the prior, which is the right answer for an
uninformative observation.

``s_base`` is the diffused score of the latent block's *marginal*, ``p_t(theta_L)``
with the conditioning dropped, which is the mixture prior for the mask COMPASS
composes with and is well defined for every other mask in the menu too. It is
the benchmark's own closed form (:class:`oracle.OracleScore`, conditioning set
to empty), so no approximation enters.

2. :class:`CompositionalTrainer` -- Part III's ``L_comp``
---------------------------------------------------------
Experiment 1 measured the composed error growing as ``n^1.00`` with ``r^2 =
1.000``: the per-row errors add *coherently*, so they are a systematic bias and
not noise. Single-row DSM cannot see that, because it penalizes the mean of the
squared per-row error, in which a shared bias and an independent wobble of the
same size are worth exactly the same.

``L_comp`` penalizes the square of the *summed* error instead. Note that the
prior term cancels in the difference of two compositions over the same ``m``,

    S_m,phi - S_m^*  =  sum_i [ s_phi,i - s^*_i ] ,

so this is literally the squared coherent component of the row error at a shared
global state, divided by ``m`` so that no ``m`` in the menu dominates the batch.

Two targets are available. ``"oracle"`` uses the exact composition ``S_m^*``:
the best a perfect network could do under this rule, so the network is never
asked to absorb the rule's own error. ``"sim"`` replaces ``lambda s^*_i`` with
its unbiased denoising surrogate ``-eps_i``, needs no closed form at all and is
therefore the form that transfers to inverse kinematics; its minimizer is the
exact tall diffused score rather than the exact composition, which is the same
thing wherever the rule error is small.

3. :class:`PreconditionedScore` -- Part IV's low-noise problem
--------------------------------------------------------------
Two derived, parameter-free changes, both forced by this model's own numbers.

*Time conditioning.* ``TimestepEmbedder`` builds ``cos/sin(t * freqs)`` with
``freqs`` in ``[1e-4, 1]``, an embedding designed for ``t`` in the low thousands.
Here ``t`` lives in ``(1e-3, 1]`` and the band the composition needs, ``lambda``
in ``[0.03, 0.2]``, is ``t`` in ``[1e-3, 4e-2]`` -- every argument is then below
``0.04``, so every ``cos`` is 1 and every ``sin`` is its argument: the embedding
is affine, tiny and nearly constant exactly where the score has to be sharp.
Replacing ``t`` by a log-``lambda`` coordinate rescaled onto the embedder's
dynamic range spends the whole embedding on the whole schedule.

*Input scaling.* EDM's ``c_in = 1/sqrt(sigma^2 + sigma_data^2)`` keeps the
network's input at unit scale across the schedule. Its ``c_skip`` is not used:
``c_skip`` is exactly a *Gaussian* residual parameterization, and
:class:`ResidualScore` already supplies the true mixture one, which is strictly
better here.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import model as M
import oracle
from compass.Trainer import Trainer


class LatentMarginalScore(oracle.OracleScore):
    """``sigma(t) * grad log p_t(theta_L)``: the oracle with no conditioning.

    Same closed form, same output convention, ``observed`` forced to empty, so
    the two can never drift apart. ``scale`` is inherited but unused.
    """

    def forward(self, x, t, c, return_attn_weights=False, **unused):
        variance = self.sde.lambda_t(t).to(x.device).reshape(-1, 1).square()
        variance = variance.expand(x.shape[0], 1).to(torch.float64)
        state = x.to(torch.float64)
        mask = (c > 0.5)
        out = torch.zeros_like(state)
        empty = torch.zeros(0, dtype=torch.long, device=x.device)
        for pattern in torch.unique(mask, dim=0):
            rows = (mask == pattern).all(dim=1)
            latent = torch.nonzero(~pattern, as_tuple=False).reshape(-1)
            if len(latent) == 0:
                continue
            out[rows] = self._score(state[rows], variance[rows], empty, latent)
        result = (out * self.sde.marginal_prob_std(t).to(x.device)).to(x.dtype)
        return (result, None) if return_attn_weights else result


from residual_score import PreconditionedScore, ResidualScore  # noqa: E402,F401


def attach(sbim, mixture, kind="residual", eps=1e-3, embedding_span=1000.0):
    """Wrap ``sbim.model`` in place and return ``sbim``."""
    if kind == "plain":
        return sbim
    baseline = LatentMarginalScore(mixture, sbim.sde).to(
        next(sbim.model.parameters()).device)
    if kind == "residual":
        sbim.model = ResidualScore(sbim.model, baseline)
        return sbim
    if kind != "preconditioned":
        raise ValueError(f"unknown wrapper {kind!r}")
    joint = oracle.joint_covariance(mixture).numpy()
    sigma_data = np.sqrt(np.diag(joint) + mixture.mu_0 ** 2)
    lam_min = float(sbim.sde.lambda_t(torch.tensor([eps])))
    lam_max = float(sbim.sde.lambda_t(torch.tensor([1.0])))
    sbim.model = PreconditionedScore(sbim.model, baseline, sbim.sde, sigma_data,
                                     (lam_min, lam_max), embedding_span)
    return sbim


# ---------------------------------------------------------------------------


class CompositionalTrainer(Trainer):
    """``Trainer`` with the tall-composition penalty and a configurable ``eps``.

    Only :meth:`_run_batch` and :meth:`_sample_timesteps` change; the data
    loader, optimizer, early stopping, mask sampler and DSM loss are the base
    class's, so an arm with ``comp_weight = 0`` and the stock ``eps`` is the
    baseline recipe exactly.
    """

    def __init__(self, sbim, mixture, comp_weight=1.0, comp_target="oracle",
                 comp_datasets=64, comp_menu=(1, 2, 4, 8, 16), eps=1e-3,
                 seed=11):
        super().__init__(sbim)
        self.mixture = mixture
        self.comp_weight = float(comp_weight)
        self.comp_target = comp_target
        self.comp_datasets = int(comp_datasets)
        self.comp_menu = tuple(int(m) for m in comp_menu)
        self.eps_override = float(eps)
        self.generator = np.random.default_rng([seed, 99])
        self.oracle_rows = oracle.OracleScore(mixture, sbim.sde)
        self.last_terms = (0.0, 0.0)

    def _sample_timesteps(self, batch_size):
        self.eps = self.eps_override
        return super()._sample_timesteps(batch_size)

    def _composition_batch(self, m):
        """One batch of tall datasets: ``(state, x, noise)`` for ``m`` rows."""
        count = self.comp_datasets
        rng = self.generator
        g = self.mixture.sample_prior(count, rng)
        locals_ = g[:, None] + self.mixture.tau_l * rng.standard_normal((count, m))
        x = locals_ + self.mixture.sigma_x * rng.standard_normal((count, m))
        return (torch.as_tensor(g, dtype=torch.float32),
                torch.as_tensor(locals_, dtype=torch.float32),
                torch.as_tensor(x, dtype=torch.float32))

    def _composition_loss(self):
        m = int(self.comp_menu[self.generator.integers(len(self.comp_menu))])
        g, locals_, x = self._composition_batch(m)
        count = g.shape[0]
        device = self.device
        g, locals_, x = g.to(device), locals_.to(device), x.to(device)

        t = self._sample_timesteps(count)                    # (count, 1)
        lam = self.sde.lambda_t(t).to(device).reshape(count, 1)
        noise_g = torch.randn(count, 1, device=device)
        noise_l = torch.randn(count, m, device=device)
        y_g = g.reshape(count, 1) + lam * noise_g
        y_l = locals_ + lam * noise_l

        # (count * m, 3) in node order, one row per (dataset, observation)
        state = torch.stack([
            y_g.expand(count, m).reshape(-1), y_l.reshape(-1), x.reshape(-1),
        ], dim=-1)
        mask = torch.zeros_like(state)
        mask[:, M.OBSERVED_INDICES[0]] = 1.0
        times = t.reshape(count, 1).expand(count, m).reshape(-1, 1)

        raw = self.model(x=state, t=times, c=mask)
        score = self.SBIm.output_scale_function(times, raw)
        composed = score[:, M.HIERARCHY[0]].reshape(count, m).sum(dim=1)

        if self.comp_target == "sim":
            # -eps_g is an unbiased draw of lambda * s^*_g for every row, and
            # eps_g is shared across the rows of a dataset, so the surrogate is
            # the summed denoising residual. Its expectation differs from the
            # oracle form by a term that does not depend on the parameters.
            residual = (lam.reshape(count) * composed
                        + m * noise_g.reshape(count))
        else:
            with torch.no_grad():
                exact = self.oracle_rows(state, times, mask)
                exact = self.SBIm.output_scale_function(times, exact)
                target = exact[:, M.HIERARCHY[0]].reshape(count, m).sum(dim=1)
            residual = lam.reshape(count) * (composed - target)
        return torch.mean((residual / m) ** 2)

    def _run_batch(self, batch):
        loss = super()._run_batch(batch)
        if self.comp_weight <= 0.0:
            return loss
        composition = self._composition_loss()
        # `loss_fn` returns a batch *sum*, which `_run_epoch` divides by the
        # batch size; matching that scaling keeps `comp_weight` a ratio of
        # per-example losses rather than a batch-size-dependent knob.
        scaled = self.comp_weight * composition * batch[0].shape[0]
        self.last_terms = (float(loss.detach()), float(scaled.detach()))
        return loss + scaled
