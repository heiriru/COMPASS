"""Drive ModeTree from a real COMPASS score source (section 27's third rung).

Everything in :mod:`modetree.diffusion` runs on an *oracle* row score. This
module supplies the same interface from ``MultiObsSampler``'s own two entry
points, so the mode tree consumes exactly what the sampler consumes:

* :meth:`MultiObsSampler._raw_row_scores` -- the per-observation network score
  in the sampler's ``y = x / alpha`` coordinates;
* :meth:`MultiObsSampler._row_score_jacobian` -- the exact per-row Jacobian on
  the latent block by forward-mode JVPs, ``D`` passes independent of ``n``.

Because both are taken from the sampler rather than reimplemented, the analytic
and the learned arm differ only in which object is behind ``SBIm.model``: the
benchmark's exact diffused score, or the trained checkpoint. Nothing else in the
pipeline knows which.

The COMPASS row state is ``(rows, samples, features)`` with the observed columns
carrying the data and the latent columns the current diffused value. ModeTree's
state is the flat ``(d_g + n * d_l,)`` vector, so this module owns the
translation between the two and nothing above it has to.
"""

from __future__ import annotations

import torch

from .diffusion import VESchedule


class CompassRowSource:
    """``DiffusedOracle``'s interface, backed by a COMPASS score source.

    Parameters
    ----------
    sbim : anything ``MultiObsSampler`` accepts -- a ``ScoreBasedInferenceModel``
        or the benchmark's ``OracleModel``. Only ``.model``, ``.sde`` and
        ``.output_scale_function`` are used, through the sampler.
    data : ``(n, n_observed)`` observed values, one row per observation.
    hierarchy, local_indices, observed_indices : the feature-index layout.
    prior : an object with ``.score(g, t, lambda_t)`` and ``.covariance`` --
        :class:`compass.priors.GaussianMixturePrior` for a multimodal prior, or
        :class:`compass.priors.GaussianPrior` for the historical tuple.
    """

    def __init__(self, sbim, data, hierarchy, local_indices, observed_indices,
                 prior, schedule=None, device="cpu"):
        from compass.MultiObsSampler import MultiObsSampler

        self.sbim = sbim
        self.sampler = MultiObsSampler(sbim)
        self.device = device
        self.hierarchy = list(hierarchy)
        self.local_indices = list(local_indices)
        self.observed_indices = list(observed_indices)
        self.features = len(self.hierarchy) + len(self.local_indices) \
            + len(self.observed_indices)
        self.d_g = len(self.hierarchy)
        self.d_l = len(self.local_indices)
        self.dim_row = self.d_g + self.d_l
        self.prior = prior

        data = torch.as_tensor(data, dtype=torch.float64).reshape(-1, len(observed_indices))
        self.n = data.shape[0]
        self.data = data.to(device)

        self.schedule = schedule or VESchedule(
            sigma=float(getattr(sbim.sde, "sigma", 25.0))
        )
        # the sampler's helpers read these off the instance
        self.sampler.hierarchy = self.hierarchy
        self.sampler.local_latent_indices = self.local_indices
        self.sampler.full_gaussian_features = self.hierarchy + self.local_indices
        self.feature_order = self.hierarchy + self.local_indices

        try:
            self.dtype = next(sbim.model.parameters()).dtype
        except (StopIteration, AttributeError):          # pragma: no cover
            self.dtype = torch.float32

        self.score_calls = 0
        self.jacobian_calls = 0
        self.network_forwards = 0

    # -- state translation ----------------------------------------------------
    def _expand(self, rows_state):
        """``(..., n, d_g + d_l)`` -> the COMPASS ``(n, samples, features)`` state.

        ModeTree evaluates batches of states (quadrature nodes, probe sets), and
        the sampler's row axis is the observation axis, so the batch has to ride
        on the *sample* axis. A leading batch of any shape is flattened into it
        and restored on the way out.
        """
        rows_state = torch.as_tensor(rows_state, dtype=torch.float64)
        batch_shape = rows_state.shape[:-2]
        flat = rows_state.reshape(-1, self.n, self.dim_row)      # (S, n, D)
        samples = flat.shape[0]
        z = torch.zeros(self.n, samples, self.features, dtype=torch.float64,
                        device=self.device)
        for position, feature in enumerate(self.hierarchy + self.local_indices):
            z[:, :, feature] = flat[:, :, position].mT
        for position, feature in enumerate(self.observed_indices):
            z[:, :, feature] = self.data[:, position].reshape(-1, 1)
        return z, batch_shape, samples

    def _condition_mask(self, samples):
        mask = torch.zeros(self.features, dtype=torch.float64, device=self.device)
        for feature in self.observed_indices:
            mask[feature] = 1.0
        return mask.reshape(1, 1, -1).expand(self.n, samples, self.features)

    def _restore(self, values, batch_shape):
        """``(n, samples, k)`` -> ``(..., n, k)``."""
        k = values.shape[-1]
        return values.permute(1, 0, 2).reshape(batch_shape + (self.n, k))

    # -- the DiffusedOracle interface ----------------------------------------
    def row_scores(self, rows_state, lam):
        t = self.schedule.time_of_lambda(lam)
        z, batch_shape, samples = self._expand(rows_state)
        mask = self._condition_mask(samples)
        time = torch.tensor(t, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            scores = self.sampler._raw_row_scores(
                z.to(self.dtype), time, mask.to(self.dtype)
            )
        self.score_calls += 1
        self.network_forwards += self.n * samples
        latent = scores.to(torch.float64)[:, :, self.feature_order]
        return self._restore(latent, batch_shape)

    def row_jacobians(self, rows_state, lam):
        t = self.schedule.time_of_lambda(lam)
        z, batch_shape, samples = self._expand(rows_state)
        mask = self._condition_mask(samples)
        time = torch.tensor(t, dtype=torch.float64, device=self.device)
        with torch.enable_grad():
            jacobian = self.sampler._row_score_jacobian(
                z.to(self.dtype), time, mask.to(self.dtype), self.feature_order
            )
        self.jacobian_calls += 1
        self.network_forwards += self.n * samples * len(self.feature_order)
        jacobian = jacobian.to(torch.float64)                # (n, samples, D, D)
        k = jacobian.shape[-1]
        flat = jacobian.reshape(self.n, samples, k * k)
        return self._restore(flat, batch_shape).reshape(
            batch_shape + (self.n, k, k)
        )

    # -- prior ----------------------------------------------------------------
    def prior_covariance_t(self, lam):
        return self.prior.covariance.to(torch.float64) + (lam ** 2) * torch.eye(
            self.d_g, dtype=torch.float64
        )

    def prior_score(self, g, lam):
        g = torch.as_tensor(g, dtype=torch.float64)
        t = self.schedule.time_of_lambda(lam)
        return self.prior.score(g, t, lam).to(torch.float64)

    def prior_hessian(self, g, lam):
        """Exact ``grad^2 log p_t(g)`` of the prior -- mixture or Gaussian.

        The composed curvature subtracts ``(n-1)`` copies of this, so it has to
        be the *real* prior curvature and not the moment-matched Gaussian's.
        See :meth:`compass.priors.GaussianMixturePrior.hessian`.
        """
        g = torch.as_tensor(g, dtype=torch.float64)
        t = self.schedule.time_of_lambda(lam)
        return self.prior.hessian(g, t, lam).to(torch.float64)

    def prior_backward_precision(self, lam):
        """``Lambda_prior(t) = Sigma_prior^-1 + lambda^-2 I``.

        For a mixture prior ``Sigma_prior`` is the moment-matched covariance,
        which is what the precision *weighting* of the GAUSS/JAC family needs;
        the prior **score** above stays exact. The approximation is confined to
        the weighting, exactly as documented in ``compass.priors``.
        """
        eye = torch.eye(self.d_g, dtype=torch.float64)
        precision = torch.cholesky_inverse(
            torch.linalg.cholesky(self.prior.covariance.to(torch.float64))
        )
        return precision + eye / lam ** 2
