#!/usr/bin/env python3
"""A *perfect* score network: the exact diffused score, in the network's place.

Why this exists
---------------
Every number the sweep reports is the composition of a rule with a trained
network, and a failure there has two possible causes the sweep cannot separate:
the network's score error amplified by the composition, or the composition
itself. ``prior_ablation.py`` removes the network analytically but only for the
plain sum-of-scores rule at zero noise. This module removes it for **every rule,
inside the real sampler**: the trained network is replaced by an object with the
same interface whose score is exact at every diffusion time, so whatever the
sampler then returns is the rule's own error and nothing else.

The closed form
---------------
The joint is a two-component Gaussian mixture in three dimensions::

    (g, l, x) | C = c   ~   N( c mu_0 * 1,  Sigma )

    Sigma = [[ t_g^2,  t_g^2,          t_g^2                 ],
             [ t_g^2,  t_g^2 + t_l^2,  t_g^2 + t_l^2         ],
             [ t_g^2,  t_g^2 + t_l^2,  t_g^2 + t_l^2 + s_x^2 ]]

with ``t_g = tau_g``, ``t_l = tau_l``, ``s_x = sigma_x``. Everything COMPASS asks
a score network for follows from that by Gaussian conditioning, for **any**
condition mask:

* split the coordinates into observed ``O`` and latent ``L``;
* per component, ``mu_{L|O} = mu_L + S_LO S_OO^-1 (x_O - mu_O)`` and
  ``S_{L|O} = S_LL - S_LO S_OO^-1 S_OL``;
* the component weight is reweighted by ``N(x_O; mu_O, S_OO)``;
* the diffusion adds ``lambda_t^2`` to the latent block only, since the observed
  coordinates are conditioned and never noised;
* the score of the resulting mixture is the responsibility-weighted average of
  the component scores.

So ``p_t(y_L | x_O)`` is again a two-component Gaussian mixture, exactly, at
every ``t`` -- which is the property that makes this benchmark able to grade a
sampler rather than only a rule.

The returned tensor follows the network's own convention: COMPASS reads a score
as ``output_scale_function(t, raw) = raw / marginal_prob_std(t)``, so this module
returns ``marginal_prob_std(t) * score``. Observed coordinates get zero, which
the samplers mask out anyway; ``tests/test_oracle.py`` pins that the composed
single-observation sample is the exact posterior, which would fail if any of
this were off.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import model as M  # noqa: E402


def joint_covariance(mixture):
    """``Sigma`` above: the covariance of ``(g, l, x)`` within one component."""
    g = mixture.tau_g ** 2
    l = g + mixture.tau_l ** 2
    x = l + mixture.sigma_x ** 2
    return torch.tensor([[g, g, g], [g, l, l], [g, l, x]], dtype=torch.float64)


class OracleScore(torch.nn.Module):
    """``model(x, t, c)`` with the exact answer instead of a learned one.

    ``x`` is ``(rows, 3)`` with the conditioned coordinates in place and the
    latent ones at their current diffused value, ``t`` is ``(rows, 1)`` or a
    scalar time, and ``c`` is the condition mask (1 = observed). Rows are
    grouped by mask pattern, so a batch may mix patterns.

    Written in torch and differentiable, so ``correction="gauss_jacobian"`` --
    which takes second derivatives of this through ``torch.autograd`` -- gets the
    exact curvature too rather than a finite-difference stand-in.
    """

    def __init__(self, mixture=M.DEFAULT, sde=None):
        super().__init__()
        self.mixture = mixture
        self.sde = sde
        self.register_buffer("covariance", joint_covariance(mixture))
        self.register_buffer(
            "means",
            torch.tensor([[c] * M.NODES for c in mixture.components],
                         dtype=torch.float64),
        )
        self.register_buffer(
            "log_weights",
            torch.log(torch.tensor(mixture.weights, dtype=torch.float64)),
        )
        # One parameter so ``.parameters()`` is non-empty and anything counting
        # parameters or moving the module to a device behaves normally.
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x, t, c, return_attn_weights=False, **unused):
        # ``Sampler`` asks the transformer for attention maps; there are none
        # here, and the flag is accepted so the call site needs no special case.
        variance = self.sde.lambda_t(t).to(x.device).reshape(-1, 1).square()
        variance = variance.expand(x.shape[0], 1).to(torch.float64)
        state = x.to(torch.float64)
        mask = (c > 0.5)
        out = torch.zeros_like(state)
        # Rows sharing a mask pattern share every matrix inverse below.
        patterns = torch.unique(mask, dim=0)
        for pattern in patterns:
            rows = (mask == pattern).all(dim=1)
            observed = torch.nonzero(pattern, as_tuple=False).reshape(-1)
            latent = torch.nonzero(~pattern, as_tuple=False).reshape(-1)
            if len(latent) == 0:
                continue
            out[rows] = self._score(state[rows], variance[rows], observed,
                                    latent)
        scale = self.sde.marginal_prob_std(t).to(x.device)
        result = (out * scale).to(x.dtype)
        return (result, None) if return_attn_weights else result

    def _score(self, state, variance, observed, latent):
        covariance = self.covariance.to(state)
        means = self.means.to(state)
        log_weights = self.log_weights.to(state)

        block_ll = covariance[latent][:, latent]
        values = state[:, latent]
        result = torch.zeros_like(state)
        terms, deltas, precisions = [], [], []
        for index in range(means.shape[0]):
            mean = means[index]
            if len(observed):
                block_oo = covariance[observed][:, observed]
                block_lo = covariance[latent][:, observed]
                residual = state[:, observed] - mean[observed]
                solved = torch.linalg.solve(
                    block_oo, residual.mT
                ).mT                                   # (rows, |O|)
                conditional_mean = mean[latent] + solved @ block_lo.mT
                conditional = block_ll - block_lo @ torch.linalg.solve(
                    block_oo, block_lo.mT
                )
                # log N(x_O; mu_O, S_OO), the component's evidence.
                sign, log_determinant = torch.linalg.slogdet(block_oo)
                evidence = -0.5 * (
                    len(observed) * float(np.log(2.0 * np.pi))
                    + log_determinant + (residual * solved).sum(dim=1)
                )
            else:
                conditional_mean = mean[latent].expand(state.shape[0],
                                                       len(latent))
                conditional = block_ll
                evidence = torch.zeros(state.shape[0], dtype=state.dtype,
                                       device=state.device)

            identity = torch.eye(len(latent), dtype=state.dtype,
                                 device=state.device)
            # The diffusion noises the latent block only.
            diffused = conditional.unsqueeze(0) + variance.reshape(-1, 1, 1) * identity
            delta = (values - conditional_mean).unsqueeze(-1)
            solved = torch.linalg.solve(diffused, delta)
            quadratic = (delta * solved).sum(dim=(1, 2))
            sign, log_determinant = torch.linalg.slogdet(diffused)
            terms.append(log_weights[index] + evidence
                         - 0.5 * (log_determinant + quadratic))
            deltas.append(delta)
            precisions.append(solved.squeeze(-1))

        stacked = torch.stack(terms)
        responsibility = torch.softmax(stacked, dim=0)
        score = sum(responsibility[index].unsqueeze(-1) * (-precisions[index])
                    for index in range(len(precisions)))
        result[:, latent] = score
        return result


class OracleModel:
    """The small slice of ``ScoreBasedInferenceModel`` the samplers actually use.

    ``MultiObsSampler`` and ``Sampler`` reach into ``SBIm`` for exactly four
    things: ``.model(x, t, c)``, ``.output_scale_function``, ``.sde`` and the
    usual ``.model.eval()/.to()``. Supplying those is enough to run every rule
    in this directory against an exact score.
    """

    def __init__(self, mixture=M.DEFAULT, sigma=15.0, device="cpu"):
        from compass.SDE import VESDE

        self.sde = VESDE(sigma=sigma)
        self.device = device
        self.nodes_size = M.NODES
        self.model = OracleScore(mixture, self.sde).to(device)
        self.mixture = mixture

    @property
    def sampler(self):
        """``correction="gauss_hierarchical"`` runs a pilot through this."""
        from compass.Sampler import Sampler

        return Sampler(self)

    def output_scale_function(self, t, x):
        return x / self.sde.marginal_prob_std(t).to(x.device)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def sample(self, theta=None, x=None, condition_mask=None, device="cpu",
               **kwargs):
        """``SBIm.sample``'s single-observation path, on the oracle.

        The mask-inference convenience of ``SBIm.sample`` is reproduced for the
        two calls this directory makes -- ``x=`` alone (the posterior) and an
        explicit ``condition_mask`` -- rather than the whole method, so nothing
        here can silently diverge from what the real class does on a path this
        module never exercises.
        """
        from compass.Sampler import Sampler

        if theta is not None:
            raise NotImplementedError("the oracle is used for posteriors only")
        data = x
        if condition_mask is None:
            condition_mask = torch.cat([
                torch.zeros(self.nodes_size - data.shape[-1]),
                torch.ones(data.shape[-1]),
            ])
        world_size = torch.cuda.device_count() if device == "cuda" else 1
        # There is no attention to capture in a closed form, and ``Sampler``
        # stacks whatever the model returns for it.
        kwargs.setdefault("capture_attention", False)
        samples = Sampler(self).sample(
            world_size=world_size, data=data, condition_mask=condition_mask,
            device=device, **kwargs,
        )
        # ``SBIm.sample`` returns the latent columns only; callers here (notably
        # ``methods.draw_marginal_product``) rely on that, so match it.
        mask = torch.as_tensor(condition_mask, dtype=torch.float32)
        return samples[:, :, (1 - mask).bool()]


def exact_conditional(mixture, x, observed, latent, variance=0.0):
    """Reference for the tests: ``(weights, means, covariance)`` of ``p_t``.

    The same Gaussian conditioning as :class:`OracleScore`, in NumPy, written
    independently so a shared bug cannot make the test pass.
    """
    covariance = joint_covariance(mixture).numpy()
    centres = np.asarray(mixture.components, dtype=np.float64)
    observed = list(observed)
    latent = list(latent)
    weights, means, covariances = [], [], []
    for weight, centre in zip(mixture.weights, centres):
        mean = np.full(M.NODES, centre)
        if observed:
            block_oo = covariance[np.ix_(observed, observed)]
            block_lo = covariance[np.ix_(latent, observed)]
            residual = np.asarray(x, dtype=np.float64)[observed] - mean[observed]
            solved = np.linalg.solve(block_oo, residual)
            conditional_mean = mean[latent] + block_lo @ solved
            conditional = block_lo @ np.linalg.solve(block_oo, block_lo.T)
            conditional = covariance[np.ix_(latent, latent)] - conditional
            sign, log_determinant = np.linalg.slogdet(block_oo)
            evidence = -0.5 * (len(observed) * np.log(2.0 * np.pi)
                               + log_determinant + residual @ solved)
        else:
            conditional_mean = mean[latent]
            conditional = covariance[np.ix_(latent, latent)]
            evidence = 0.0
        weights.append(np.log(weight) + evidence)
        means.append(conditional_mean)
        covariances.append(conditional + variance * np.eye(len(latent)))
    weights = np.array(weights)
    weights -= weights.max()
    weights = np.exp(weights)
    return weights / weights.sum(), np.array(means), np.array(covariances)
