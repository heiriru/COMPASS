"""Non-Gaussian priors for compositional inference.

``MultiObsSampler.sample`` has always accepted ``prior=(mean, scale)`` and built
the diffused prior score internally from it.  That is enough for a unimodal
problem but not for multimodal work: the composition subtracts ``(n-1)`` copies
of the prior score, so a prior that is wrong about where the mass sits is
amplified ``n``-fold.

This module adds a prior *object* interface alongside the tuple.  A prior object
supplies

    score(theta_y, t)            -- the diffused prior score in y = x/alpha
                                    coordinates, shape in == shape out
    covariance                   -- a second-moment matrix used only to build
                                    ``Lambda_prior(t)`` for the precision-weighted
                                    composition rules
    mean                         -- first moment, used for the denoise clamp
    log_prob(theta)              -- optional, for diagnostics
    marginal_std                 -- optional, per-coordinate scale

The tuple form keeps working unchanged and is converted to a
:class:`GaussianPrior`, so existing callers are bit-for-bit unaffected.

A bare prior-score *callable* is accepted through :class:`ScorePrior`::

    prior = ScorePrior(lambda theta_y, lam: my_diffused_score(theta_y, lam),
                       mean=m, covariance=C)            # or ScorePrior.from_samples(fn, draws)
    model.sample(x=x, multi_obs_inference=True, prior=prior, ...)

so any prior whose diffused score you can write down (mixtures, truncations,
priors in transformed coordinates, a learned prior score) enters the composition
exactly. The moments are still needed, but only for the denoise clamp and the
precision *weighting* of the covariance-aware rules -- never for the score.

Exact diffused mixture score
----------------------------
For ``p_0(g) = sum_k pi_k N(g; mu_k, Sigma_k)`` under the additive-noise
(``y_t = y_0 + lambda_t eps``) formulation every sampler here uses,

    p_t(g) = sum_k pi_k N(g; mu_k, Sigma_k + lambda_t^2 I)

so the score is available in closed form,

    s_t(g) = sum_k r_k(g, t) [ -(Sigma_k + lambda_t^2 I)^-1 (g - mu_k) ]
    r_k    = softmax_k [ log pi_k + log N(g; mu_k, Sigma_k + lambda_t^2 I) ]

computed with ``logsumexp`` and Cholesky solves.  No Gaussian approximation is
made anywhere in the score itself.

A note on the weighting for the covariance-aware rules.  Those rules need a
single ``Lambda_prior(t) = Sigma_prior^-1 + lambda^-2 I``, which for a genuine
mixture is state-dependent.  :class:`GaussianMixturePrior` supplies the
mixture's *moment-matched* covariance there while keeping the score exact.  That
is an approximation of the precision **weighting**, not of the target: it is the
same second-moment quantity GAUSS already estimates by pilot sampling, and the
composed score it weights is the exact mixture score.
"""

import math

import torch


class GaussianPrior:
    """The historical ``prior=(mean, scale)`` behaviour, as an object."""

    def __init__(self, mean, covariance):
        self.mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
        covariance = torch.as_tensor(covariance, dtype=torch.float64)
        if covariance.dim() == 1:
            covariance = torch.diag(covariance)
        self.covariance = covariance
        self.marginal_std = torch.diagonal(covariance).sqrt()

    def score(self, theta_y, t, lambda_t):
        identity = torch.eye(
            self.covariance.shape[-1], dtype=theta_y.dtype, device=theta_y.device
        )
        covariance_t = self.covariance.to(theta_y) + (lambda_t ** 2) * identity
        delta = theta_y - self.mean.to(theta_y)
        return -torch.linalg.solve(
            covariance_t, delta.unsqueeze(-1)
        ).squeeze(-1)

    def log_prob(self, theta):
        return GaussianMixturePrior(
            [1.0], self.mean.unsqueeze(0), self.covariance.unsqueeze(0)
        ).log_prob(theta)

    def hessian(self, theta_y, t, lambda_t):
        """``grad^2 log p_t``: constant ``-(Sigma + lambda^2 I)^-1`` here."""
        identity = torch.eye(
            self.covariance.shape[-1], dtype=torch.float64, device=theta_y.device
        )
        covariance_t = self.covariance.to(torch.float64) + (lambda_t ** 2) * identity
        precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance_t))
        return -precision.expand(theta_y.shape[:-1] + precision.shape)


class GaussianMixturePrior:
    """``sum_k pi_k N(mu_k, Sigma_k)`` with the exact diffused score."""

    def __init__(self, weights, means, covariances):
        weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        self.log_weights = torch.log(weights / weights.sum())
        self.means = torch.as_tensor(means, dtype=torch.float64)
        if self.means.dim() == 1:
            self.means = self.means.unsqueeze(-1)
        covariances = torch.as_tensor(covariances, dtype=torch.float64)
        if covariances.dim() == 2 and covariances.shape[0] == self.means.shape[0]:
            covariances = torch.diag_embed(covariances)
        self.covariances = covariances
        self.dim = self.means.shape[-1]
        weights_ = self.log_weights.exp()
        self.mean = (weights_.unsqueeze(-1) * self.means).sum(0)
        delta = self.means - self.mean
        # moment-matched second moment: within-component plus between-component
        self.covariance = (
            (weights_[:, None, None] * self.covariances).sum(0)
            + (weights_[:, None, None] * delta.unsqueeze(-1) * delta.unsqueeze(-2)).sum(0)
        )
        self.marginal_std = torch.diagonal(self.covariance).sqrt()

    def _component_terms(self, theta, lambda_t):
        covariances = self.covariances.to(theta) + (lambda_t ** 2) * torch.eye(
            self.dim, dtype=theta.dtype, device=theta.device
        )
        factor = torch.linalg.cholesky(covariances)
        delta = theta.unsqueeze(-2) - self.means.to(theta)
        solved = torch.cholesky_solve(delta.unsqueeze(-1), factor).squeeze(-1)
        quadratic = (delta * solved).sum(-1)
        logdet = 2.0 * torch.log(
            torch.diagonal(factor, dim1=-2, dim2=-1)
        ).sum(-1)
        log_component = (
            self.log_weights.to(theta)
            - 0.5 * self.dim * math.log(2.0 * math.pi)
            - 0.5 * logdet
            - 0.5 * quadratic
        )
        return log_component, -solved

    def score(self, theta_y, t, lambda_t):
        log_component, direction = self._component_terms(theta_y, lambda_t)
        responsibility = torch.softmax(log_component, dim=-1)
        return (responsibility.unsqueeze(-1) * direction).sum(-2)

    def log_prob(self, theta, lambda_t=0.0):
        log_component, _ = self._component_terms(theta, lambda_t)
        return torch.logsumexp(log_component, dim=-1)

    def responsibilities(self, theta, lambda_t=0.0):
        log_component, _ = self._component_terms(theta, lambda_t)
        return torch.softmax(log_component, dim=-1)

    def hessian(self, theta_y, t, lambda_t):
        """Exact ``grad^2 log p_t`` of the diffused mixture.

        This is **not** interchangeable with the moment-matched Gaussian's
        curvature, and the difference is not second order.  Near a component the
        exact curvature is that component's, ``-1/tau_k^2``; the moment-matched
        one is the inverse of the *whole* mixture's variance, which for
        well-separated components is dominated by the between-component term and
        is therefore far flatter.  A compositional rule subtracts ``(n-1)``
        copies of the prior curvature, so using the moment-matched value there
        biases the composed precision by ``(n-1)`` times that difference --
        measured on the Gaussian-mixture benchmark at ``n = 8``, a posterior
        standard deviation of 0.134 against an exact 0.274.
        """
        theta_y = torch.as_tensor(theta_y, dtype=torch.float64)
        log_component, direction = self._component_terms(theta_y, lambda_t)
        responsibility = torch.softmax(log_component, dim=-1)
        covariances = self.covariances.to(torch.float64) + (lambda_t ** 2) * \
            torch.eye(self.dim, dtype=torch.float64, device=theta_y.device)
        precisions = torch.cholesky_inverse(torch.linalg.cholesky(covariances))
        outer = direction.unsqueeze(-1) * direction.unsqueeze(-2)
        first = (responsibility[..., None, None] * (-precisions + outer)).sum(-3)
        mean_direction = (responsibility.unsqueeze(-1) * direction).sum(-2)
        return first - mean_direction.unsqueeze(-1) * mean_direction.unsqueeze(-2)


class ScorePrior:
    """A prior given by its diffused score function.

    ``score(theta_y, lambda_t)`` returns ``grad log p_t(theta_y)`` for the prior
    diffused to noise level ``lambda_t`` (``p_t = p_0 * N(0, lambda_t^2 I)``, in the
    ``y = x / alpha`` coordinates every sampler uses), same shape as ``theta_y``.
    A three-argument ``score(theta_y, t, lambda_t)`` is accepted too.

    ``mean`` / ``covariance`` are the prior's first two moments. They are used for
    the denoise clamp box and for ``Lambda_prior(t)`` in the precision-weighted
    rules, which need one matrix; the composed score itself only ever reads
    ``score``. ``log_prob(theta)`` is optional and needed only by
    ``certify_composition``.
    """

    def __init__(self, score, mean, covariance, log_prob=None):
        if not callable(score):
            raise TypeError("ScorePrior needs a callable score(theta_y, lambda_t).")
        self._score = score
        self._takes_t = _positional_arity(score) >= 3
        self.mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
        covariance = torch.as_tensor(covariance, dtype=torch.float64)
        if covariance.dim() == 1:
            covariance = torch.diag(covariance)
        if covariance.shape != (self.mean.numel(), self.mean.numel()):
            raise ValueError(
                f"covariance must be {self.mean.numel()}x{self.mean.numel()} (or a "
                f"variance vector), got {tuple(covariance.shape)}."
            )
        self.covariance = covariance
        self.marginal_std = torch.diagonal(covariance).sqrt()
        self._log_prob = log_prob

    @classmethod
    def from_samples(cls, score, samples, log_prob=None):
        """Moments from prior draws ``(num, dim)``; the score is still ``score``."""
        samples = torch.as_tensor(samples, dtype=torch.float64)
        if samples.dim() != 2 or samples.shape[0] < 2:
            raise ValueError("samples must be (num >= 2, dim) prior draws.")
        return cls(score, samples.mean(0), torch.cov(samples.T).reshape(
            samples.shape[1], samples.shape[1]), log_prob=log_prob)

    def score(self, theta_y, t, lambda_t):
        out = (self._score(theta_y, t, lambda_t) if self._takes_t
               else self._score(theta_y, lambda_t))
        out = torch.as_tensor(out)
        if out.shape != theta_y.shape:
            raise ValueError(
                f"prior score returned shape {tuple(out.shape)} for input "
                f"{tuple(theta_y.shape)}; it must preserve the shape."
            )
        return out

    def log_prob(self, theta):
        if self._log_prob is None:
            raise NotImplementedError(
                "This ScorePrior has no log_prob; pass log_prob= to use it in "
                "certify_composition."
            )
        return self._log_prob(theta)


def _positional_arity(function):
    import inspect
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return 2
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return 2
    return sum(p.kind in kinds for p in parameters)


def resolve_prior_object(prior, width):
    """Return a prior object for ``prior``, or ``None`` for the tuple form.

    ``None`` means "the caller should keep the existing tuple path", which is
    what preserves the historical behaviour exactly.
    """
    if prior is None or isinstance(prior, (tuple, list)):
        return None
    if hasattr(prior, "score"):
        if not hasattr(prior, "covariance") or not hasattr(prior, "mean"):
            raise ValueError(
                "A prior object must expose .score(theta_y, t, lambda_t), "
                ".mean and .covariance."
            )
        if prior.covariance.shape[-1] != width:
            raise ValueError(
                f"Prior covariance has width {prior.covariance.shape[-1]}, "
                f"expected {width} (= len(hierarchy))."
            )
        return prior
    if callable(prior):
        raise ValueError(
            "A bare prior-score callable carries no moments, which the denoise clamp "
            "and the precision-weighted rules need. Wrap it: "
            "compass.priors.ScorePrior(fn, mean=..., covariance=...) or "
            "ScorePrior.from_samples(fn, prior_draws)."
        )
    raise ValueError(
        "prior must be (mean, scale), a ScorePrior, or an object with a .score "
        f"method; got {type(prior).__name__}."
    )
