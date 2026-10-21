"""A hierarchical Gaussian mixture whose tall posterior is known in closed form.

The generative model
--------------------
One scalar global parameter ``g`` with a two-component Gaussian-mixture prior,
one local parameter per observation, one observation per local::

    C ~ {-1, +1}      with weights (pi_-, pi_+)
    g | C = c   ~  N(c * mu_0, tau_g^2)
    l_i | g     ~  N(g, tau_l^2)
    x_i | l_i   ~  N(l_i, sigma_x^2)

so the graph is ``C -> g -> l_i -> x_i``, with the concrete setting
``mu_0 = 2.5``, ``tau_g = 0.4``, ``tau_l = 0.7``, ``sigma_x = 0.8``.

Why this is the right test bed for compositional scaling
--------------------------------------------------------
Marginalising the local parameter gives ``x_i | g ~ N(g, v)`` with
``v = tau_l^2 + sigma_x^2``, so conditioned on the mixture component the whole
model is jointly Gaussian. The tall posterior

    p(g, l_1..l_n | x_1..x_n)

is therefore an **exact two-component Gaussian mixture in ``n + 1`` dimensions,
for every ``n``**: the number of modes does not grow with the number of
observations, so anything a composition rule gets wrong at large ``n`` is the
rule's doing and not an increase in the difficulty of the target. Every quantity
a grader could want -- the shared marginal, each local marginal, the full mean
vector and covariance of either branch, the branch weights -- is available here
in closed form, with no Monte Carlo reference and hence no reference noise
floor. That is the one thing ``inverse_kinematics`` cannot offer.

The branch weights, and how the asymmetry is controlled
-------------------------------------------------------
Writing ``T = sum_i x_i``, the posterior log-odds between the two branches are
exactly (see :func:`branch_log_odds`)

    log(w_+ / w_-) = log(pi_+ / pi_-) + 2 mu_0 T / (v + n tau_g^2).

Two consequences drive the whole experiment design in :mod:`datasets`:

* ``(n, T)`` is a **sufficient statistic for the shared posterior**: ``V``
  depends only on ``n``, and ``m_c`` and the weights only on ``T``. Two datasets
  with the same ``n`` and the same ``T`` have literally the same
  ``p(g | x_1..n)``, however differently their observations are arranged -- which
  is what :mod:`datasets` turns into a path-dependence test;
* with a symmetric prior the shared posterior is **exactly symmetric iff
  ``T = 0``** -- so a symmetric target is a measure-zero coincidence, not the
  generic case, and any rule graded only on ``T = 0`` data is graded on the one
  configuration where a symmetrising bias is invisible
  (``docs/nonsymmetric_composition.md`` in ``inverse_kinematics`` is the
  measurement that made this point);
* the map from ``T`` to the branch mass is invertible, so a dataset with a
  *prescribed* branch split can be constructed at any ``n``. This is what lets
  the ``n``-sweep hold the shape of the target fixed while only the number of
  observations changes; under plain prior-predictive data the posterior
  collapses onto one branch by ``n = 4`` and the sweep silently stops testing
  the thing it was built to test.
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Node layout
# ---------------------------------------------------------------------------
# COMPASS wants one flat vector per observation, shared coordinates first, then
# the row-local latents, then the conditioned observables:
#
#     index   0   1   2
#             g   l   x
#             <s> <l> <obs>

GLOBAL_NAMES = ["g"]
LOCAL_NAMES = ["l"]
OBSERVED_NAMES = ["x"]
NODE_NAMES = GLOBAL_NAMES + LOCAL_NAMES + OBSERVED_NAMES

NODES = len(NODE_NAMES)          # 3
HIERARCHY = [0]                  # g
LOCAL_INDICES = [1]              # l
OBSERVED_INDICES = [2]           # x
LATENT_INDICES = HIERARCHY + LOCAL_INDICES

LOG_TWO_PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class Mixture:
    """The generative model's parameters and every derived constant.

    ``weights`` are the *prior* mixture weights ``(pi_-, pi_+)``. They default
    to the balanced pair in the specification; an unbalanced pair is the second,
    model-level way of breaking the symmetry (the first being the data, see
    :mod:`datasets`) and shifts :func:`branch_log_odds` by a constant that does
    not decay with ``n``.
    """

    mu_0: float = 2.5
    tau_g: float = 0.4
    tau_l: float = 0.7
    sigma_x: float = 0.8
    weights: tuple = (0.5, 0.5)

    # -- derived scalars ---------------------------------------------------
    @property
    def v(self):
        """Var(x_i | g) -- the local parameter marginalised out."""
        return self.tau_l ** 2 + self.sigma_x ** 2

    @property
    def q(self):
        """Var(l_i | g, x_i) = (1/tau_l^2 + 1/sigma_x^2)^-1."""
        return 1.0 / (1.0 / self.tau_l ** 2 + 1.0 / self.sigma_x ** 2)

    @property
    def a(self):
        """Weight of ``g`` in E[l_i | g, x_i] = a g + b x_i."""
        return self.q / self.tau_l ** 2

    @property
    def b(self):
        """Weight of ``x_i`` in E[l_i | g, x_i]."""
        return self.q / self.sigma_x ** 2

    @property
    def components(self):
        """Component locations ``(-mu_0, +mu_0)``, in the order of ``weights``."""
        return np.array([-self.mu_0, self.mu_0], dtype=np.float64)

    @property
    def global_prior_std(self):
        """Marginal sd of ``g`` under the mixture prior -- NOT ``tau_g``.

        ``sqrt(mu_0^2 + tau_g^2)``, i.e. 2.53 at the default setting against a
        within-component 0.4. Every Gaussian-prior API in COMPASS is handed this
        moment-matched number, which is the honest translation of a bimodal
        prior into an interface that only accepts a Gaussian one -- and one of
        the things this benchmark is here to measure the cost of.
        """
        mean = float(self.weights[0] * -self.mu_0 + self.weights[1] * self.mu_0)
        second = float(self.mu_0 ** 2 + self.tau_g ** 2)
        return float(np.sqrt(second - mean ** 2))

    @property
    def global_prior_mean(self):
        return float(self.weights[0] * -self.mu_0 + self.weights[1] * self.mu_0)

    @property
    def local_prior_std(self):
        """Marginal sd of ``l_i``: the prior on ``g`` widened by ``tau_l``."""
        return float(np.sqrt(self.global_prior_std ** 2 + self.tau_l ** 2))

    @property
    def local_prior_mean(self):
        return self.global_prior_mean

    # -- simulation --------------------------------------------------------
    def sample_prior(self, count, rng):
        """``count`` draws of ``g`` from the mixture prior."""
        component = rng.random(int(count)) < self.weights[1]
        centre = np.where(component, self.mu_0, -self.mu_0)
        return centre + self.tau_g * rng.standard_normal(int(count))

    def simulate(self, count, rng):
        """``count`` independent ``(g, l, x)`` rows in node order.

        One fresh ``g`` per row, deliberately: sharing ``g`` is a property of
        the tall dataset the composition rules are evaluated on, not of the
        training set. The network only ever sees the single-observation joint;
        every cross-observation statement is the composition rule's doing.
        """
        count = int(count)
        g = self.sample_prior(count, rng)
        l = g + self.tau_l * rng.standard_normal(count)
        x = l + self.sigma_x * rng.standard_normal(count)
        return np.stack([g, l, x], axis=1)

    def simulate_dataset(self, observations, rng, g=None):
        """One tall dataset: a single shared ``g`` and ``n`` observations.

        Returns ``(g, locals, x)`` with ``locals`` and ``x`` shaped ``(n,)``.
        """
        observations = int(observations)
        if g is None:
            g = float(self.sample_prior(1, rng)[0])
        locals_ = g + self.tau_l * rng.standard_normal(observations)
        x = locals_ + self.sigma_x * rng.standard_normal(observations)
        return float(g), locals_, x

    # -- exact posterior ---------------------------------------------------
    def posterior_variance(self, n):
        """``V = (1/tau_g^2 + n/v)^-1``: the within-branch variance of ``g``."""
        return 1.0 / (1.0 / self.tau_g ** 2 + int(n) / self.v)

    def branch_means(self, x):
        """``(m_-, m_+)``: the within-branch posterior means of ``g``."""
        x = np.asarray(x, dtype=np.float64).ravel()
        variance = self.posterior_variance(len(x))
        total = x.sum()
        return np.array([
            variance * (component / self.tau_g ** 2 + total / self.v)
            for component in self.components
        ])

    def branch_log_evidence(self, x):
        """``log p(x_1..n | C = c)`` for both components.

        Conditioned on ``C = c`` the data are jointly Gaussian with mean
        ``c mu_0 1`` and covariance ``tau_g^2 J + v I``. The determinant and
        inverse of that rank-one update are written out rather than formed, so
        the cost is O(n) and the expression stays exact at ``n = 32``.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        ratio = n * self.tau_g ** 2 / self.v          # the rank-one correction
        log_determinant = n * np.log(self.v) + np.log1p(ratio)
        values = []
        for component in self.components:
            delta = x - component
            quadratic = (
                (delta ** 2).sum() / self.v
                - (self.tau_g ** 2 / self.v ** 2) * delta.sum() ** 2 / (1.0 + ratio)
            )
            values.append(-0.5 * (n * LOG_TWO_PI + log_determinant + quadratic))
        return np.array(values)

    def branch_weights(self, x):
        """Posterior branch weights ``(w_-, w_+)``, normalised."""
        log_weights = np.log(np.asarray(self.weights, dtype=np.float64))
        log_weights = log_weights + self.branch_log_evidence(x)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        return weights / weights.sum()

    def branch_log_odds(self, x):
        """``log(w_+ / w_-)``, in the closed form used to design the datasets.

        Every quadratic term in :func:`branch_log_evidence` that does not
        involve ``T = sum_i x_i`` is common to the two branches and cancels,
        leaving ``2 mu_0 T / (v + n tau_g^2)`` plus the prior log-odds. The
        agreement between this and ``log`` of the ratio from
        :func:`branch_weights` is pinned in the tests -- it is the identity
        :mod:`datasets` inverts, so a slip here would silently mis-target every
        prescribed branch split.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        prior = float(np.log(self.weights[1] / self.weights[0]))
        return prior + 2.0 * self.mu_0 * x.sum() / (
            self.v + len(x) * self.tau_g ** 2
        )

    def total_for_log_odds(self, log_odds, n):
        """The ``T = sum_i x_i`` that produces a prescribed branch log-odds."""
        prior = float(np.log(self.weights[1] / self.weights[0]))
        return (float(log_odds) - prior) * (
            self.v + int(n) * self.tau_g ** 2
        ) / (2.0 * self.mu_0)

    def global_posterior(self, x):
        """``(weights, means, variance)`` of the exact ``p(g | x_1..n)``."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return (self.branch_weights(x), self.branch_means(x),
                self.posterior_variance(len(x)))

    def global_density(self, grid, x):
        """The exact shared posterior tabulated on ``grid``."""
        weights, means, variance = self.global_posterior(x)
        grid = np.asarray(grid, dtype=np.float64)
        density = np.zeros_like(grid)
        for weight, mean in zip(weights, means):
            density += weight * np.exp(
                -0.5 * (grid - mean) ** 2 / variance
            ) / np.sqrt(2.0 * np.pi * variance)
        return density

    def global_cdf(self, points, x):
        """The exact shared posterior CDF -- used for W1 and for rank statistics."""
        from scipy.special import erf

        weights, means, variance = self.global_posterior(x)
        points = np.asarray(points, dtype=np.float64)
        total = np.zeros_like(points)
        scale = np.sqrt(2.0 * variance)
        for weight, mean in zip(weights, means):
            total += weight * 0.5 * (1.0 + erf((points - mean) / scale))
        return total

    def global_moments(self, x):
        """``(mean, sd)`` of the exact shared posterior, from the mixture law."""
        weights, means, variance = self.global_posterior(x)
        mean = float((weights * means).sum())
        second = float((weights * (means ** 2 + variance)).sum())
        return mean, float(np.sqrt(max(second - mean ** 2, 0.0)))

    def global_quantiles(self, probabilities, x, grid=None):
        """Quantiles of the exact shared posterior, by inverting a fine CDF."""
        if grid is None:
            grid = self.default_grid(x, points=20_001)
        cdf = self.global_cdf(grid, x)
        return np.interp(np.asarray(probabilities, dtype=np.float64), cdf, grid)

    def default_grid(self, x, points=1601, pad=5.0):
        """A grid wide enough for both branches plus ``pad`` prior sds."""
        span = self.mu_0 + pad * max(self.tau_g, np.sqrt(self.posterior_variance(1)))
        centre = np.asarray(x, dtype=np.float64).mean() if np.size(x) else 0.0
        low = min(-span, centre - span)
        high = max(span, centre + span)
        return np.linspace(low, high, int(points))

    def prior_density(self, grid):
        """The mixture prior on ``g``, tabulated on ``grid``."""
        grid = np.asarray(grid, dtype=np.float64)
        density = np.zeros_like(grid)
        for weight, centre in zip(self.weights, self.components):
            density += weight * np.exp(
                -0.5 * (grid - centre) ** 2 / self.tau_g ** 2
            ) / np.sqrt(2.0 * np.pi * self.tau_g ** 2)
        return density

    def diffused_prior_score(self, g, variance):
        """``d/dg log p_sigma(g)`` for the prior convolved with ``N(0, variance)``.

        A Gaussian mixture convolved with a Gaussian is a Gaussian mixture with
        the same weights and inflated variance, so the diffused prior score is
        the responsibility-weighted average of the component scores. This is the
        term every score-composition rule subtracts ``n - 1`` copies of; the
        Gaussian-prior version of it that COMPASS's samplers use by default is
        the thing ``methods.mixture_prior_score`` replaces.
        """
        g = np.asarray(g, dtype=np.float64)
        total = self.tau_g ** 2 + float(variance)
        log_terms = np.stack([
            np.log(weight) - 0.5 * (g - centre) ** 2 / total
            for weight, centre in zip(self.weights, self.components)
        ])
        log_terms -= log_terms.max(axis=0, keepdims=True)
        responsibility = np.exp(log_terms)
        responsibility /= responsibility.sum(axis=0, keepdims=True)
        scores = np.stack([
            -(g - centre) / total for centre in self.components
        ])
        return (responsibility * scores).sum(axis=0)

    # -- the exact tall posterior, in full ---------------------------------
    def branch_mean_vector(self, x, component_index):
        """``E[(g, l_1..l_n) | x, C = c]`` for one branch, length ``n + 1``."""
        x = np.asarray(x, dtype=np.float64).ravel()
        mean_g = self.branch_means(x)[int(component_index)]
        return np.concatenate([[mean_g], self.a * mean_g + self.b * x])

    def branch_covariance(self, n):
        """``Cov[(g, l_1..l_n) | x, C = c]`` -- independent of ``x`` and of ``c``.

        ``Var(g) = V``, ``Cov(g, l_i) = a V``, ``Var(l_i) = q + a^2 V`` and
        ``Cov(l_i, l_j) = a^2 V``: the locals are conditionally independent
        given ``g`` and inherit all of their mutual dependence through it.
        """
        n = int(n)
        variance = self.posterior_variance(n)
        covariance = np.full((n + 1, n + 1), self.a ** 2 * variance)
        covariance[0, 0] = variance
        covariance[0, 1:] = self.a * variance
        covariance[1:, 0] = self.a * variance
        covariance[1:, 1:] += np.eye(n) * self.q
        return covariance

    def local_marginal(self, x, index):
        """``(weights, means, variance)`` of the exact ``p(l_index | x_1..n)``."""
        x = np.asarray(x, dtype=np.float64).ravel()
        weights, means, variance = self.global_posterior(x)
        return (weights, self.a * means + self.b * x[int(index)],
                self.q + self.a ** 2 * variance)

    def local_moments(self, x, index):
        weights, means, variance = self.local_marginal(x, index)
        mean = float((weights * means).sum())
        second = float((weights * (means ** 2 + variance)).sum())
        return mean, float(np.sqrt(max(second - mean ** 2, 0.0)))

    def local_cdf(self, points, x, index):
        from scipy.special import erf

        weights, means, variance = self.local_marginal(x, index)
        points = np.asarray(points, dtype=np.float64)
        total = np.zeros_like(points)
        scale = np.sqrt(2.0 * variance)
        for weight, mean in zip(weights, means):
            total += weight * 0.5 * (1.0 + erf((points - mean) / scale))
        return total

    def joint_log_density(self, values, x):
        """``log p(g, l_1..l_n | x_1..n)`` at ``values`` -- the full mixture."""
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        covariance = self.branch_covariance(n)
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise RuntimeError("the branch covariance is not positive definite")
        weights = self.branch_weights(x)
        terms = []
        for index, weight in enumerate(weights):
            delta = values - self.branch_mean_vector(x, index)
            quadratic = np.einsum(
                "ij,jk,ik->i", delta, np.linalg.inv(covariance), delta
            )
            terms.append(np.log(weight) - 0.5 * (
                (n + 1) * LOG_TWO_PI + log_determinant + quadratic
            ))
        stacked = np.stack(terms)
        peak = stacked.max(axis=0)
        return peak + np.log(np.exp(stacked - peak).sum(axis=0))


DEFAULT = Mixture()
