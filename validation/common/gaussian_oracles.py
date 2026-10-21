"""Zero-mean linear-Gaussian models with exact-score COMPASS networks.

A model is a linear structural equation ``z = M eps`` with ``eps ~ N(0, I)`` over
the node vector ``z = [parameters..., observations...]``. Every conditional a
condition mask can ask for is Gaussian, so
:class:`compass.experimental.oracles.OracleNetwork` returns its *exact* diffused
score. Everything a COMPASS pipeline computes on such a model -- posterior
draws, MAP, PF-ODE likelihood, MLE, criteria -- can therefore be checked
against closed forms, and any disagreement is a pipeline error, not a network
error.
"""

import math

import torch

from compass.experimental.oracles import OracleNetwork
from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel


class GaussianJoint:
    """``z = M eps``; ``n_params`` leading nodes are parameters, the rest observations."""

    def __init__(self, M, n_params, names=None):
        self.M = torch.as_tensor(M, dtype=torch.float64)
        self.covariance = self.M @ self.M.T
        self.nodes = self.M.shape[0]
        self.n_params = int(n_params)
        self.param_indices = list(range(self.n_params))
        self.observed_indices = list(range(self.n_params, self.nodes))
        self.names = names or [f"z{i}" for i in range(self.nodes)]

    def conditional(self, mask):
        """Latent indices and ``(gain, covariance)`` of ``latent | observed``."""
        mask = torch.as_tensor(mask).bool()
        U = torch.where(~mask)[0]
        O = torch.where(mask)[0]
        S = self.covariance
        if len(O) == 0:
            return U, torch.zeros(len(U), 0, dtype=torch.float64), S[U][:, U]
        gain = S[U][:, O] @ torch.linalg.inv(S[O][:, O])
        cov = S[U][:, U] - gain @ S[O][:, U]
        return U, gain, 0.5 * (cov + cov.T)

    def conditional_log_density(self, values, mask):
        """Exact ``log p(z_latent | z_observed)`` for rows of ``values`` (float64)."""
        values = torch.as_tensor(values, dtype=torch.float64)
        U, gain, cov = self.conditional(mask)
        O = torch.where(torch.as_tensor(mask).bool())[0]
        mean = values[:, O] @ gain.T if len(O) else torch.zeros(len(values), len(U), dtype=torch.float64)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        return dist.log_prob(values[:, U])


def oracle_sbim(joint, sigma=25.0):
    """A ``ScoreBasedInferenceModel`` whose network is the exact VESDE oracle of ``joint``."""
    model = ScoreBasedInferenceModel(nodes_size=joint.nodes, sde_type="vesde", sigma=sigma,
                                     hidden_size=8, depth=1, num_heads=1, mlp_ratio=1)
    model.model = OracleNetwork(joint, model.sde)
    return model


# ---------------------------------------------------------------------------
# The nested pair used by A1/A2 (validation/TODO.md, E-A1.2 / E-A2.3)
# ---------------------------------------------------------------------------

def nested_offset_pair(s_g=1.0, s_l=1.0, nu=1.0):
    """M1: rows ``x = (l + nu e1, g + nu e2)`` with a global offset ``g``.
    M0: the same with ``g = 0`` removed.

    Node layouts: M1 ``[g, l, x1, x2]`` (global dim 0, local dim 1),
    M0 ``[l, x1, x2]`` (local dim 0). Priors ``g ~ N(0, s_g^2)``,
    ``l ~ N(0, s_l^2)``, independent.
    """
    M1 = torch.tensor([
        [s_g, 0.0, 0.0, 0.0],   # g
        [0.0, s_l, 0.0, 0.0],   # l
        [0.0, s_l, nu, 0.0],    # x1 = l + nu e1
        [s_g, 0.0, 0.0, nu],    # x2 = g + nu e2
    ])
    M0 = torch.tensor([
        [s_l, 0.0, 0.0],        # l
        [s_l, nu, 0.0],         # x1
        [0.0, 0.0, nu],         # x2 = nu e2
    ])
    return GaussianJoint(M1, 2, ["g", "l", "x1", "x2"]), GaussianJoint(M0, 1, ["l", "x1", "x2"])


def chi2_sf(value, dof=1):
    """Survival function of chi^2 (scipy-free for dof 1, scipy otherwise)."""
    if dof == 1:
        return math.erfc(math.sqrt(max(value, 0.0) / 2.0))
    from scipy.stats import chi2
    return float(chi2.sf(value, dof))


def wilson_interval(successes, trials, z=2.576):
    """Wilson score interval (default 99%)."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denom
    return (centre - half, centre + half)
