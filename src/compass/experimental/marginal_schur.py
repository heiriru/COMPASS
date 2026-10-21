"""Experimental: Arruda-style marginal-global composition on the all-in-one network.

Derivation: ``docs/arruda_schur_derivation.md``. Per subject ``j`` the network
gives the joint diffused score ``S = (S_g, S_l)`` of ``p_t(g, l | x_j)``. Arruda
et al. compose the *marginal* global score ``nabla_g log p_t(g | x_j)`` over
subjects on a state that holds ``g`` only, and sample the locals afterwards
from ``p(l | g, x_j)``. With the score Jacobian blocks ``A = dS_g/dl`` and
``H_ll = -dS_l/dl``, three per-subject estimators of that marginal score are
offered:

``"direct"``
    ``s_hat = S_g`` at the carried state. It is the marginal score when the
    network's ``g`` output is marginal by construction: the production mean-field
    trunk (per coordinate), or the ``global`` / ``hierarchical`` latent-attention
    modes (joint over ``g``). For a ``full`` network it is a one-sample Fisher
    estimate ``E_{l~p_t(l|g)}[S_g]`` at the carried ``l``, which is exactly what the
    joint-state composition does.
``"schur"``
    ``s_hat = S_g + A H_ll^-1 S_l`` at the carried state. It is the gradient of the
    Schur-complement (Gaussian) marginalisation of a local quadratic model of
    ``log p_t(g, l)``, and it is exact for any ``l`` when ``p_t(g, l | x_j)`` is
    jointly Gaussian. It is biased when the conditional precision of ``l``
    depends on ``g`` (every scale parameter), because it drops the ``l``-volume
    term.
``"laplace"``
    Newton to the conditional mode ``l*(g)`` (warm-started across solver
    steps), then ``s_hat = S_g(g, l*) - 1/2 grad_g log det H_ll(g, l*)``. It is
    exact when ``l | g`` is Gaussian with a ``g``-dependent precision. That is
    the partial-pooling structure ``l ~ N(mu, sigma^2)`` at low noise, and at
    high noise every conditional is close to Gaussian anyway.

Composition over the ``n`` subjects:

``correction="arruda_schur"``
    ``s = d(t) [ (1-n) w(t) s_prior,t(g) + sum_j s_hat_j ]``. The prior is exact
    and diffused. ``w = 1`` (default) or ``1-t`` (Geffner's bridge). ``d(t) = 1``
    unless ``damping=True``, which uses the Arruda schedule set by
    ``damping_at_data`` / ``damping_at_noise``.
``correction="arruda_schur_gauss"``
    GAUSS / Tweedie precision weighting of the marginal factors, with
    ``Lambda_j`` from the Schur complement ``H_gg - H_gl H_ll^-1 H_lg`` of the
    curvature.

The local coordinates of the state keep their network score ``S_l``. They are
auxiliary: they enter the composed drift only through ``s_hat``. With
``ancestral_locals=True`` (the default) the returned locals are redrawn from
``p(l | g, x_j)`` with the single-observation sampler. That is Arruda et al.'s
Eq. 12.
"""

import math

import torch

from ..MultiObsSampler import MultiObsSampler

ESTIMATORS = ("direct", "schur", "laplace")
PRIOR_WEIGHTINGS = ("diffused", "bridge")


class MarginalSchurSampler(MultiObsSampler):
    """``MultiObsSampler`` plus the ``arruda_schur`` corrections.

    Install on a model with :func:`install` so ``SBIm.sample`` and
    ``ModelTransfuser.compare`` route through it. Every production correction
    is delegated to the parent untouched.
    """

    MARGINAL_CORRECTIONS = frozenset({"arruda_schur", "arruda_schur_gauss"})
    VALID_CORRECTIONS = MultiObsSampler.VALID_CORRECTIONS | MARGINAL_CORRECTIONS

    def __init__(self, SBIm, estimator="laplace", prior_weighting="diffused",
                 damping=False, ancestral_locals=True, local_scale=1.0,
                 conditional_width=10.0, logdet_fd_step=None, newton_steps=1,
                 newton_trust=3.0,
                 cold_newton_steps=4, ancestral_timesteps=None,
                 ancestral_batch_rows=65536):
        super().__init__(SBIm)
        self.configure(estimator=estimator, prior_weighting=prior_weighting,
                       damping=damping, ancestral_locals=ancestral_locals,
                       local_scale=local_scale, conditional_width=conditional_width,
                       logdet_fd_step=logdet_fd_step, newton_steps=newton_steps,
                       newton_trust=newton_trust,
                       cold_newton_steps=cold_newton_steps,
                       ancestral_timesteps=ancestral_timesteps,
                       ancestral_batch_rows=ancestral_batch_rows)
        self._reset_marginal_state()

    def configure(self, **settings):
        """Update the marginal-composition settings (see module docstring)."""
        for key, value in settings.items():
            setattr(self, "marginal_" + key, value)
        if self.marginal_estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}.")
        if self.marginal_prior_weighting not in PRIOR_WEIGHTINGS:
            raise ValueError(f"prior_weighting must be one of {PRIOR_WEIGHTINGS}.")
        if not self.marginal_conditional_width > 0:
            raise ValueError("conditional_width must be positive.")
        if int(self.marginal_newton_steps) < 1 or int(self.marginal_cold_newton_steps) < 1:
            raise ValueError("Newton step counts must be at least 1.")
        return self

    def _reset_marginal_state(self):
        self._marginal_mode = None
        self._marginal_logdet_cache = None
        self._marginal_call_index = 0
        self._marginal_jacobian_evaluations = 0

    #############################################
    # ----- Sampling entry point -----
    #############################################

    def sample(self, world_size, data, condition_mask=None, **kwargs):
        correction = kwargs.get("correction", "gauss")
        if correction not in self.MARGINAL_CORRECTIONS:
            return super().sample(world_size, data, condition_mask=condition_mask, **kwargs)
        if world_size > 1:
            raise NotImplementedError("arruda_schur runs with world_size=1.")
        if kwargs.get("composition_batch_size") is not None:
            raise NotImplementedError(
                "Observation mini-batching is not implemented for arruda_schur yet: "
                "the carried local state of unselected subjects would go stale."
            )
        self._reset_marginal_state()
        samples = super().sample(world_size, data, condition_mask=condition_mask, **kwargs)
        if self.marginal_ancestral_locals and self.local_latent_indices:
            samples = self._ancestral_locals(
                samples, condition_mask,
                timesteps=self.marginal_ancestral_timesteps or kwargs.get("timesteps", 50),
                eps=kwargs.get("eps", 1e-3),
                device=kwargs.get("device", "cpu"),
                churn_eta=kwargs.get("churn_eta"),
            )
        return samples

    #############################################
    # ----- Composition -----
    #############################################

    def _compositional_score(self, scores, x, t, num_observations=None,
                             minibatch_selected=False, condition_mask=None):
        if self.correction not in self.MARGINAL_CORRECTIONS:
            return super()._compositional_score(
                scores, x, t, num_observations=num_observations,
                minibatch_selected=minibatch_selected, condition_mask=condition_mask,
            )
        h = list(self.hierarchy)
        n = scores.shape[0]
        var_t = self.sde.lambda_t(t).to(x.device) ** 2
        variance = float(var_t.reshape(-1)[0])
        theta_h = x[:1, :, h]

        # The same tail protection as every production rule.
        if self.denoise_clamp is not None:
            lo = self.prior_mean - self.denoise_clamp * self.prior_std
            hi = self.prior_mean + self.denoise_clamp * self.prior_std
            x0 = torch.clamp(theta_h + var_t * scores[:, :, h], min=lo, max=hi)
            scores[:, :, h] = (x0 - theta_h) / var_t
        scores = self._clamp_local_scores(scores, x, var_t)

        marginal, precision = self._marginal_scores(scores, x, t, condition_mask, variance)
        prior_score = self._diffused_gaussian_prior_score(theta_h, t).to(torch.float64)
        prior_weight = 1.0 if self.marginal_prior_weighting == "diffused" else float(1.0 - t)

        if self.correction == "arruda_schur":
            composed = (1 - n) * prior_weight * prior_score + marginal.sum(dim=0, keepdim=True)
            if self.marginal_damping:
                composed = self._damping_factor(t).to(composed) * composed
        else:
            inverse_variance = 1.0 / variance
            effective = precision
            delta = self._project_information(effective, inverse_variance, len(h))
            if delta is not None:
                effective = effective + delta
            precision_prior_t = self._prior_precision_at(inverse_variance, len(h), effective)
            weighted = torch.einsum("nsij,nsj->nsi", effective, marginal).sum(0, keepdim=True)
            composed_precision = effective.sum(0) + (1 - n) * prior_weight * precision_prior_t
            numerator = weighted + (1 - n) * prior_weight * torch.einsum(
                "ij,bsj->bsi", precision_prior_t, prior_score
            )
            composed, _, _, _ = self._solve_composed_global(
                composed_precision, numerator, marginal
            )

        composed = composed.to(scores.dtype)
        if self.denoise_clamp is not None and not self.marginal_damping:
            x0 = torch.clamp(theta_h + var_t * composed, min=lo, max=hi)
            composed = (x0 - theta_h) / var_t
        if not torch.all(torch.isfinite(composed)):
            raise RuntimeError(f"The {self.correction} composed score became non-finite.")

        result = scores if scores.shape[0] == x.shape[0] else torch.zeros_like(x)
        result[:, :, h] = composed
        return result

    #############################################
    # ----- Per-subject marginal scores -----
    #############################################

    def _fd_scores_and_jacobian(self, z, t, condition_mask, order):
        """Base scores ``(n, S, D)`` and Jacobian ``(n, S, |order|, |order|)`` in one batch."""
        rows, samples, features = z.shape
        lam = float(self.sde.lambda_t(t).reshape(-1)[0])
        step = float(self.jacobian_fd_step) * lam
        copies = [z]
        for feature in order:
            shifted = z.clone()
            shifted[:, :, feature] += step
            copies.append(shifted)
        per_call = max(1, self.FD_JACOBIAN_MAX_ROWS // max(1, rows * samples))
        mask_per_sample = condition_mask.dim() == 3 and condition_mask.shape[1] == samples
        outputs = []
        with torch.no_grad():
            for start in range(0, len(copies), per_call):
                chunk = copies[start:start + per_call]
                mask = (condition_mask.repeat(1, len(chunk), 1) if mask_per_sample
                        else condition_mask)
                out = self._raw_row_scores(torch.cat(chunk, dim=1), t, mask)
                outputs.append(out.reshape(rows, len(chunk), samples, features))
        stacked = torch.cat(outputs, dim=1).to(torch.float64)
        base = stacked[:, 0]
        columns = (stacked[:, 1:][..., order] - stacked[:, :1][..., order]) / step
        self._marginal_jacobian_evaluations += 1
        return base, columns.permute(0, 2, 3, 1).contiguous()

    def _local_width(self, variance, like):
        """Upper bound on the conditional sd of each diffused local: sqrt(s_l^2 + lambda^2)."""
        scale = getattr(self, "local_prior_std", None)
        if scale is None:
            scale = self.marginal_local_scale
        scale = torch.as_tensor(scale, dtype=like.dtype, device=like.device)
        return torch.sqrt(scale ** 2 + variance)

    def _local_curvature_bound(self, variance):
        # The marginal local prior scale when the caller supplied one (it is in
        # the sampler's own units), else the configured fallback.
        scale = getattr(self, "local_prior_std", None)
        if scale is None:
            scale = self.marginal_local_scale
        scale = torch.as_tensor(scale, dtype=torch.float64).cpu()
        return float((1.0 / ((self.marginal_conditional_width * scale) ** 2 + variance)).min())

    def _clamped_eigh(self, curvature, variance, lower):
        """Symmetrise and clamp a curvature block into ``[lower, (1-floor)/lambda^2]``."""
        curvature = 0.5 * (curvature + curvature.mT)
        eigenvalues, eigenvectors = self._batched_eigh(curvature)
        floor = float(self.jacobian_curvature_floor)
        return eigenvalues.clamp(min=lower, max=(1.0 - floor) / variance), eigenvectors

    def _marginal_scores(self, scores, x, t, condition_mask, variance):
        """Per-subject marginal global scores ``(n, S, H)`` and, for
        ``arruda_schur_gauss``, Tweedie precisions ``(n, S, H, H)``."""
        h = list(self.hierarchy)
        loc = list(self.local_latent_indices)
        H = len(h)
        estimator = self.marginal_estimator
        gauss = self.correction == "arruda_schur_gauss"
        index = self._marginal_call_index
        self._marginal_call_index = index + 1

        if not loc or (estimator == "direct" and not gauss):
            if gauss:
                _, jacobian = self._fd_scores_and_jacobian(x, t, condition_mask, h)
                return scores[:, :, h].to(torch.float64), self._tweedie_precision(
                    -jacobian, variance)
            return scores[:, :, h].to(torch.float64), None

        lower = self._local_curvature_bound(variance)
        order = h + loc
        if estimator == "laplace":
            z = x.clone()
            mode = self._marginal_mode
            cold = mode is None or mode.shape != x[:, :, loc].shape
            if not cold:
                z[:, :, loc] = mode.to(z)
            steps = int(self.marginal_cold_newton_steps if cold else self.marginal_newton_steps)
            for _ in range(steps):
                base, jacobian = self._fd_scores_and_jacobian(z, t, condition_mask, order)
                eig_l, vec_l = self._clamped_eigh(-jacobian[..., H:, H:], variance, lower)
                inverse_ll = (vec_l / eig_l.unsqueeze(-2)) @ vec_l.mT
                newton = torch.einsum("nslm,nsm->nsl", inverse_ll, base[:, :, loc])
                # Trust region. The mode of p_t(l | g_t) lies within a few
                # conditional sds of any conditional sample, and that sd is at
                # most sqrt(s_l^2 + lambda^2). A learned network can report a
                # near-flat H_ll, making H_ll^-1 S_l arbitrarily large, and the
                # first-order S_g(l*) extrapolation is then meaningless. The cap
                # is a no-op wherever the step is sane.
                radius = self.marginal_newton_trust * self._local_width(variance, newton)
                newton = torch.maximum(torch.minimum(newton, radius), -radius)
                z[:, :, loc] = z[:, :, loc] + newton.to(z)
            # One Newton step leaves S_g(l*) = S_g(l) + A dl to first order in the
            # final (small, warm-started) step, which is exact where the step is.
            marginal = base[:, :, h] + torch.einsum(
                "nsgl,nsl->nsg", jacobian[..., :H, H:], newton
            )
            self._marginal_mode = z[:, :, loc].detach()
            refresh = self._jacobian_refresh_at(1.0 / variance)
            cache = self._marginal_logdet_cache
            if cache is None or cache.shape[:2] != marginal.shape[:2] or index % refresh == 0:
                cache = self._logdet_gradient(z, t, condition_mask, variance, lower)
                self._marginal_logdet_cache = cache
            marginal = marginal - 0.5 * cache
        else:
            # "schur" (and "direct" under the precision-weighted rule).
            _, jacobian = self._fd_scores_and_jacobian(x, t, condition_mask, order)
            s_g = scores[:, :, h].to(torch.float64)
            if estimator == "schur":
                eig_l, vec_l = self._clamped_eigh(-jacobian[..., H:, H:], variance, lower)
                inverse_ll = (vec_l / eig_l.unsqueeze(-2)) @ vec_l.mT
                gain = jacobian[..., :H, H:] @ inverse_ll
                marginal = s_g + torch.einsum("nsgl,nsl->nsg", gain,
                                              scores[:, :, loc].to(torch.float64))
            else:
                marginal = s_g

        precision = None
        if gauss:
            curvature = -0.5 * (jacobian + jacobian.mT)
            eig_l, vec_l = self._clamped_eigh(curvature[..., H:, H:], variance, lower)
            inverse_ll = (vec_l / eig_l.unsqueeze(-2)) @ vec_l.mT
            marginal_curvature = curvature[..., :H, :H] - curvature[..., :H, H:] @ (
                inverse_ll @ curvature[..., H:, :H])
            precision = self._tweedie_precision(marginal_curvature, variance)
        return marginal, precision

    def _tweedie_precision(self, curvature, variance):
        """Backward-kernel precision ``[lambda^2 (I - lambda^2 H)]^-1`` of a curvature."""
        eig, vec = self._clamped_eigh(curvature, variance, 0.0)
        precision = 1.0 / (variance * (1.0 - variance * eig))
        return (vec * precision.unsqueeze(-2)) @ vec.mT

    def _logdet_gradient(self, z, t, condition_mask, variance, lower):
        """``grad_g log det H_ll`` by forward differences in each shared coordinate.

        The baseline and every shifted curvature are taken at the *same* local
        state ``z``. Reusing a curvature from before the last Newton step as the
        baseline turns each column into (log det at the new l - log det at the
        old l) / step: identical in every coordinate and divergent as the step
        shrinks with lambda. It is invisible whenever H_ll does not depend on l.
        """
        h = list(self.hierarchy)
        loc = list(self.local_latent_indices)
        lam = math.sqrt(variance)
        step = float(self.marginal_logdet_fd_step or self.jacobian_fd_step) * lam
        _, block = self._fd_scores_and_jacobian(z, t, condition_mask, loc)
        eig_base, _ = self._clamped_eigh(-block, variance, lower)
        base = torch.log(eig_base).sum(-1)
        columns = []
        for coordinate in h:
            shifted = z.clone()
            shifted[:, :, coordinate] += step
            _, block = self._fd_scores_and_jacobian(shifted, t, condition_mask, loc)
            eig, _ = self._clamped_eigh(-block, variance, lower)
            columns.append((torch.log(eig).sum(-1) - base) / step)
        return torch.stack(columns, dim=-1)

    #############################################
    # ----- Ancestral locals (Arruda Eq. 12) -----
    #############################################

    def _ancestral_locals(self, samples, condition_mask, timesteps, eps, device,
                          churn_eta=None):
        """Redraw each subject's locals from ``p(l | g, x_j)`` at every global draw.

        ``samples`` is the full node vector the composed sampler returns, so
        every index here is a node index.
        """
        cm = torch.as_tensor(condition_mask)
        cm = cm if cm.dim() == 1 else cm[0]
        h = list(self.hierarchy)
        loc = list(self.local_latent_indices)
        n, S, nodes = samples.shape
        if nodes != cm.numel():
            raise ValueError("Expected full node-width samples from the composed sampler.")
        mask_gx = cm.clone().float()
        mask_gx[h] = 1.0
        observed = torch.where(mask_gx == 1)[0].tolist()
        rows = samples[:, :, observed].detach().cpu().float().reshape(n * S, len(observed))
        kwargs = dict(condition_mask=mask_gx, timesteps=timesteps, eps=eps,
                      num_samples=1, device=device, verbose=False, capture_attention=False)
        if churn_eta is not None:
            kwargs["churn_eta"] = churn_eta
        batch = int(self.marginal_ancestral_batch_rows)
        chunks = []
        for start in range(0, rows.shape[0], batch):
            draw = torch.as_tensor(self.SBIm.sample(x=rows[start:start + batch], **kwargs))
            # SBIm.sample returns the latent columns only (here: exactly the
            # locals, in node order); a raw Sampler returns the full node vector.
            if draw.shape[-1] == len(loc):
                chunks.append(draw[:, 0, :].cpu())
            elif draw.shape[-1] == nodes:
                chunks.append(draw[:, 0, loc].cpu())
            else:
                raise ValueError(f"Unexpected ancestral draw width {draw.shape[-1]}.")
        locals_ = torch.cat(chunks).reshape(n, S, len(loc))
        out = samples.clone()
        self.carried_local_samples = samples[:, :, loc].detach().clone()
        out[:, :, loc] = locals_.to(out)
        return out


def install(model, **settings):
    """Route ``model.sample(multi_obs_inference=True, ...)`` through the new sampler."""
    model.multi_obs_sampler = MarginalSchurSampler(model, **settings)
    return model.multi_obs_sampler
