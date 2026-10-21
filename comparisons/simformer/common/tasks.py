"""The Simformer paper's tabular tasks, defined exactly as in Appendix A2.2.

Six tasks: the four sbibm benchmark tasks used in Fig. 4a (``gaussian_linear``,
``gaussian_mixture``, ``two_moons``, ``slcp``) and the two tasks they introduce
for the all-conditional evaluation in Fig. 4b (``tree``, ``hmm``).

Every task exposes an **exact joint log-density** over the concatenated node
vector ``z = [theta, x]``. That is what makes the all-conditional protocol
possible: a reference sample for *any* condition mask is an MCMC run against
``joint_log_prob`` with the conditioned coordinates pinned (see
``common/reference.py``). It is also a check on the simulators -- ``tests/``
verifies that simulated draws agree with the density.

Conventions
-----------
* ``theta`` first, then ``x``, in the flat node vector -- the layout COMPASS's
  ``ScoreBasedInferenceModel`` and every mask helper assume.
* Priors are stated in the paper's own parameterisation. Where the paper writes
  ``N(0, 0.1 * I)`` the 0.1 is read as a **variance** (sbibm's convention);
  this is flagged in the docstring of the task it affects.
* No parameter transform is applied. Note this differs from
  ``Compositional_score_testing/Gauss_test/tasks.py``, which maps bounded priors
  to a standard normal latent ``z`` because the Gaussian composition rules need
  a Gaussian prior. Here the target is the paper's protocol, which trains on the
  untransformed parameters, so C2ST is computed in the same space they report.
"""

import math

import torch

TWO_PI = 2.0 * math.pi


def _log_normal(x, mean, std):
    """Elementwise Gaussian log-density; ``mean``/``std`` may be floats."""
    std = torch.as_tensor(std, dtype=x.dtype)
    var = std ** 2
    return -0.5 * (((x - mean) ** 2) / var + torch.log(TWO_PI * var))


class Task:
    """Common behaviour: flat node vector, joint density, simulation."""

    name = None
    theta_dim = None
    x_dim = None
    #: Directed dependency structure of the generative process, as
    #: ``{child_node_index: [parent_node_indices]}`` over the flat vector. This
    #: is Fig. A4's "directed base mask". COMPASS cannot consume it today (no
    #: edge mask); e02 records why.
    edges = None

    @property
    def nodes_size(self):
        return self.theta_dim + self.x_dim

    @property
    def roles(self):
        """``{role: node indices}`` partition used by the mask samplers."""
        return {
            "theta": list(range(self.theta_dim)),
            "x": list(range(self.theta_dim, self.nodes_size)),
        }

    # -- to implement ------------------------------------------------------
    def sample_prior(self, n, generator=None):
        raise NotImplementedError

    def log_prior(self, theta):
        raise NotImplementedError

    def simulate(self, theta, generator=None):
        raise NotImplementedError

    def log_likelihood(self, theta, x):
        raise NotImplementedError

    # -- derived -----------------------------------------------------------
    def sample_joint(self, n, generator=None):
        theta = self.sample_prior(n, generator=generator)
        x = self.simulate(theta, generator=generator)
        return theta, x

    def joint_log_prob(self, z):
        """Log p(theta, x) for a batch of flat node vectors ``z``."""
        theta, x = z[..., : self.theta_dim], z[..., self.theta_dim:]
        return self.log_prior(theta) + self.log_likelihood(theta, x)


class GaussianLinear(Task):
    """A2.2: prior N(0, 0.1 I), likelihood N(x; theta, 0.1 I), theta, x in R^10.

    The 0.1 is a variance (sbibm's ``gaussian_linear``), so the standard
    deviation is sqrt(0.1). Fully factorised -- this is the task where the
    paper's structured attention mask helps most, and the one task where NPE
    beats Simformer at 10k simulations (Sec. 4.1).
    """

    name = "gaussian_linear"
    theta_dim = 10
    x_dim = 10
    prior_std = math.sqrt(0.1)
    likelihood_std = math.sqrt(0.1)

    def __init__(self):
        self.edges = {self.theta_dim + i: [i] for i in range(self.theta_dim)}

    def sample_prior(self, n, generator=None):
        return self.prior_std * torch.randn(n, self.theta_dim, generator=generator)

    def log_prior(self, theta):
        return _log_normal(theta, 0.0, self.prior_std).sum(-1)

    def simulate(self, theta, generator=None):
        noise = torch.randn(theta.shape, generator=generator)
        return theta + self.likelihood_std * noise

    def log_likelihood(self, theta, x):
        return _log_normal(x, theta, self.likelihood_std).sum(-1)

    def analytic_posterior(self, x):
        """Exact Gaussian posterior (mean, std) -- the reference for this task."""
        pv, lv = self.prior_std ** 2, self.likelihood_std ** 2
        var = 1.0 / (1.0 / pv + 1.0 / lv)
        return x * (var / lv), torch.full_like(x, math.sqrt(var))


class GaussianMixture(Task):
    """A2.2: prior U(-10, 10)^2; x|theta ~ 0.5 N(theta, I) + 0.5 N(theta, 0.01 I)."""

    name = "gaussian_mixture"
    theta_dim = 2
    x_dim = 2
    low, high = -10.0, 10.0
    stds = (1.0, 0.1)          # sqrt of I and of 0.01 I

    def __init__(self):
        self.edges = {self.theta_dim + i: list(range(self.theta_dim))
                      for i in range(self.x_dim)}

    def sample_prior(self, n, generator=None):
        u = torch.rand(n, self.theta_dim, generator=generator)
        return self.low + (self.high - self.low) * u

    def log_prior(self, theta):
        inside = ((theta >= self.low) & (theta <= self.high)).all(-1)
        density = -self.theta_dim * math.log(self.high - self.low)
        return torch.where(inside, torch.full_like(theta[..., 0], density),
                           torch.full_like(theta[..., 0], -float("inf")))

    def simulate(self, theta, generator=None):
        n = theta.shape[0]
        pick = (torch.rand(n, 1, generator=generator) < 0.5).to(theta.dtype)
        std = pick * self.stds[0] + (1 - pick) * self.stds[1]
        return theta + std * torch.randn(theta.shape, generator=generator)

    def log_likelihood(self, theta, x):
        parts = torch.stack([
            _log_normal(x, theta, std).sum(-1) + math.log(0.5) for std in self.stds
        ], dim=0)
        return torch.logsumexp(parts, dim=0)


class TwoMoons(Task):
    """A2.2: prior U(-1, 1)^2; x = [r cos a + 0.25, r sin a] + shift(theta),
    with a ~ U(-pi/2, pi/2) and r ~ N(0.1, 0.012^2).

    The paper writes ``r ~ N(0.1, 0.01^2)`` in one place and the sbibm task uses
    0.01 as the *standard deviation*; both readings coincide here because sbibm's
    ``two_moons`` draws ``r ~ N(0.1, 0.01)`` with 0.01 as the std. We use
    ``R_STD = 0.01``.

    The likelihood is exact: for fixed theta the map (a, r) -> x is a
    diffeomorphism with |det J| = r, so
    ``p(x|theta) = p(a) p(r) / r`` on the half-plane reachable by a in
    [-pi/2, pi/2], and zero elsewhere.
    """

    name = "two_moons"
    theta_dim = 2
    x_dim = 2
    low, high = -1.0, 1.0
    R_MEAN, R_STD = 0.1, 0.01

    def __init__(self):
        self.edges = {self.theta_dim + i: list(range(self.theta_dim))
                      for i in range(self.x_dim)}

    def sample_prior(self, n, generator=None):
        u = torch.rand(n, self.theta_dim, generator=generator)
        return self.low + (self.high - self.low) * u

    def log_prior(self, theta):
        inside = ((theta >= self.low) & (theta <= self.high)).all(-1)
        density = -self.theta_dim * math.log(self.high - self.low)
        return torch.where(inside, torch.full_like(theta[..., 0], density),
                           torch.full_like(theta[..., 0], -float("inf")))

    @staticmethod
    def _shift(theta):
        t1, t2 = theta[..., 0], theta[..., 1]
        return torch.stack([-torch.abs(t1 + t2) / math.sqrt(2.0),
                            (-t1 + t2) / math.sqrt(2.0)], dim=-1)

    def simulate(self, theta, generator=None):
        n = theta.shape[0]
        a = -math.pi / 2 + math.pi * torch.rand(n, generator=generator)
        r = self.R_MEAN + self.R_STD * torch.randn(n, generator=generator)
        base = torch.stack([r * torch.cos(a) + 0.25, r * torch.sin(a)], dim=-1)
        return base + self._shift(theta)

    def log_likelihood(self, theta, x):
        p = x - self._shift(theta)
        u, v = p[..., 0] - 0.25, p[..., 1]
        r = torch.sqrt(u ** 2 + v ** 2)
        reachable = u >= 0.0                       # a in [-pi/2, pi/2] <=> cos a >= 0
        log_r = _log_normal(r, self.R_MEAN, self.R_STD)
        out = log_r - math.log(math.pi) - torch.log(r.clamp_min(1e-12))
        return torch.where(reachable, out, torch.full_like(out, -float("inf")))


class SLCP(Task):
    """A2.2: prior U(-3, 3)^5; four i.i.d. draws x_i ~ N(mu_theta, Sigma_theta).

    mu = (theta1, theta2);
    Sigma = [[s1^2, rho s1 s2], [rho s1 s2, s2^2]] with s1 = theta3^2,
    s2 = theta4^2, rho = tanh(theta5).

    The paper writes the off-diagonal as ``tanh(theta5) * theta3^2 * theta4^2``,
    i.e. exactly rho * s1 * s2 in these symbols. theta in R^5, x in R^8
    (four 2-vectors, laid out draw-major).
    """

    name = "slcp"
    theta_dim = 5
    x_dim = 8
    n_draws = 4
    low, high = -3.0, 3.0

    def __init__(self):
        self.edges = {self.theta_dim + i: list(range(self.theta_dim))
                      for i in range(self.x_dim)}

    def sample_prior(self, n, generator=None):
        u = torch.rand(n, self.theta_dim, generator=generator)
        return self.low + (self.high - self.low) * u

    def log_prior(self, theta):
        inside = ((theta >= self.low) & (theta <= self.high)).all(-1)
        density = -self.theta_dim * math.log(self.high - self.low)
        return torch.where(inside, torch.full_like(theta[..., 0], density),
                           torch.full_like(theta[..., 0], -float("inf")))

    @staticmethod
    def _moments(theta):
        mean = theta[..., :2]
        s1 = theta[..., 2] ** 2
        s2 = theta[..., 3] ** 2
        rho = torch.tanh(theta[..., 4])
        cov = torch.stack([
            torch.stack([s1 ** 2, rho * s1 * s2], dim=-1),
            torch.stack([rho * s1 * s2, s2 ** 2], dim=-1),
        ], dim=-2)
        return mean, cov

    def simulate(self, theta, generator=None):
        mean, cov = self._moments(theta)
        chol = torch.linalg.cholesky(cov)
        eps = torch.randn(theta.shape[0], self.n_draws, 2, generator=generator)
        draws = mean.unsqueeze(1) + torch.einsum("bij,bkj->bki", chol, eps)
        return draws.reshape(theta.shape[0], self.x_dim)

    def log_likelihood(self, theta, x):
        mean, cov = self._moments(theta)
        draws = x.reshape(*x.shape[:-1], self.n_draws, 2)
        delta = draws - mean.unsqueeze(-2)
        chol = torch.linalg.cholesky(cov)
        solved = torch.linalg.solve_triangular(
            chol.unsqueeze(-3), delta.unsqueeze(-1), upper=False).squeeze(-1)
        quad = (solved ** 2).sum(-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return (-0.5 * (quad + logdet.unsqueeze(-1) + 2 * math.log(TWO_PI))).sum(-1)


class Tree(Task):
    """A2.2: nonlinear tree-shaped task with highly multimodal conditionals.

    theta0 ~ N(0, 1); theta1 ~ N(theta0, 1); theta2 ~ N(theta0, 1)
    x0 ~ N(sin(theta1)^2, 0.2^2)
    x1 ~ N(0.1 theta1^2, 0.2^2)
    x2 ~ N(0.1 theta2^2, 0.6^2)
    x3 ~ N(cos(theta2)^2, 0.1^2)

    The paper's typography for theta1 reads ``theta1 ~ N(theta0; 1.)``, which is
    the same N(theta0, 1) as theta2 up to the typo; both children hang off
    theta0, which is what makes it a tree.
    """

    name = "tree"
    theta_dim = 3
    x_dim = 4
    x_stds = (0.2, 0.2, 0.6, 0.1)

    def __init__(self):
        self.edges = {3: [1], 4: [1], 5: [2], 6: [2], 1: [0], 2: [0]}

    def sample_prior(self, n, generator=None):
        t0 = torch.randn(n, 1, generator=generator)
        t1 = t0 + torch.randn(n, 1, generator=generator)
        t2 = t0 + torch.randn(n, 1, generator=generator)
        return torch.cat([t0, t1, t2], dim=-1)

    def log_prior(self, theta):
        t0, t1, t2 = theta[..., 0], theta[..., 1], theta[..., 2]
        return (_log_normal(t0, 0.0, 1.0) + _log_normal(t1, t0, 1.0)
                + _log_normal(t2, t0, 1.0))

    def _means(self, theta):
        t1, t2 = theta[..., 1], theta[..., 2]
        return torch.stack([torch.sin(t1) ** 2, 0.1 * t1 ** 2,
                            0.1 * t2 ** 2, torch.cos(t2) ** 2], dim=-1)

    def simulate(self, theta, generator=None):
        std = torch.tensor(self.x_stds)
        return self._means(theta) + std * torch.randn(
            theta.shape[0], self.x_dim, generator=generator)

    def log_likelihood(self, theta, x):
        std = torch.tensor(self.x_stds, dtype=x.dtype)
        return _log_normal(x, self._means(theta), std).sum(-1)


class HMM(Task):
    """A2.2: Markovian parameters with a bimodal, correlated posterior.

    theta0 ~ N(0, 0.5^2); theta_{i+1} ~ N(theta_i, 0.5^2) for i = 0..8
    x_i ~ N(theta_i^2, 0.5^2) for i = 0..9.  theta, x in R^10.

    This is the sharpest test of COMPASS's key-padding convention: the posterior
    couples the ten latent theta nodes into a chain, and under
    ``key_padding_mask = 1 - c`` no latent theta token may attend to any other.
    """

    name = "hmm"
    theta_dim = 10
    x_dim = 10
    step_std = 0.5
    obs_std = 0.5

    def __init__(self):
        self.edges = {i: [i - 1] for i in range(1, self.theta_dim)}
        self.edges.update({self.theta_dim + i: [i] for i in range(self.theta_dim)})

    def sample_prior(self, n, generator=None):
        steps = self.step_std * torch.randn(n, self.theta_dim, generator=generator)
        return torch.cumsum(steps, dim=-1)

    def log_prior(self, theta):
        first = _log_normal(theta[..., 0], 0.0, self.step_std)
        rest = _log_normal(theta[..., 1:], theta[..., :-1], self.step_std).sum(-1)
        return first + rest

    def simulate(self, theta, generator=None):
        return theta ** 2 + self.obs_std * torch.randn(
            theta.shape, generator=generator)

    def log_likelihood(self, theta, x):
        return _log_normal(x, theta ** 2, self.obs_std).sum(-1)


TASKS = {
    "gaussian_linear": GaussianLinear,
    "gaussian_mixture": GaussianMixture,
    "two_moons": TwoMoons,
    "slcp": SLCP,
    "tree": Tree,
    "hmm": HMM,
}

#: The four sbibm tasks scored in Fig. 4a / A5a (posterior C2ST).
POSTERIOR_TASKS = ("gaussian_linear", "gaussian_mixture", "two_moons", "slcp")
#: The four tasks scored in Fig. 4b / A5b (C2ST over arbitrary conditionals).
CONDITIONAL_TASKS = ("tree", "hmm", "two_moons", "slcp")


def get_task(name):
    """Instantiate a task by its paper name."""
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}; choose from {sorted(TASKS)}.")
    return TASKS[name]()
