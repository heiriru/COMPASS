"""The four tall datasets the composition rules are graded on.

The design problem
------------------
``model.branch_log_odds`` says the posterior branch split depends on the data
through the single number ``T = sum_i x_i``:

    log(w_+ / w_-) = log(pi_+ / pi_-) + 2 mu_0 T / (v + n tau_g^2).

Under plain prior-predictive data ``T`` grows like ``n g_true``, so the log-odds
grow like ``2 mu_0 g_true n / (v + n tau_g^2) -> 2 mu_0 g_true / tau_g^2``, which
at the default setting is ``31 g_true``: by ``n = 4`` the posterior has put
everything on one branch and the target is a plain unimodal Gaussian. A sweep
over ``n`` on such data measures how well a rule handles a target that is
getting *easier*, and the near-symmetric answer a Gaussian rule returns off
symmetry -- the failure mode measured in ``inverse_kinematics`` -- never gets a
chance to show.

So three of the four regimes fix the *shape* of the target and vary only ``n``,
by choosing the data offset that solves the identity above for a prescribed
branch split. Shifting every ``x_i`` by a constant is a change of which dataset
we condition on, not a change of the model: the posterior remains exact for the
data actually handed to the sampler.

The regimes
-----------
``symmetric``
    ``T = 0`` exactly, observations scattered around zero. Both branches carry
    mass 0.5 and no single observation prefers either. **This is the case that
    flatters a symmetrising rule** and it is included to show that.

``antisymmetric``
    ``T = 0`` exactly again -- so the same perfectly symmetric target -- but the
    observations come in ``+-`` pairs near ``+-mu_0``. Every single-observation
    posterior is then sharply one-sided, and half of them point the wrong way;
    the symmetry of the answer is a property of the *composition*, not of any
    term in it. A rule that averages evidence rather than multiplying it will
    look identical to ``symmetric`` here and it is not.

``asymmetric``
    ``T`` chosen so the branch split is exactly ``--branch-mass`` (default 0.75)
    at every ``n``. The two branches stay at a fixed, unequal weight as ``n``
    grows, which is precisely what a mean-plus-covariance summary of each
    observation cannot represent.

``natural``
    An honest prior-predictive draw: one ``g`` from the mixture prior, ``n``
    observations, no offset. The realistic case, kept because it is the one a
    practitioner meets -- and because it shows the collapse onto one branch that
    motivates the other three.
"""

import numpy as np

from model import DEFAULT

REGIMES = ("symmetric", "antisymmetric", "asymmetric", "natural")


def _shift_to_total(x, total):
    """Shift every observation by a constant so ``sum_i x_i == total``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return x + (float(total) - x.sum()) / len(x)


def make_dataset(regime, observations, rng, mixture=DEFAULT, branch_mass=0.75,
                 spread=1.0):
    """One tall dataset.

    Returns a dict with the observations ``x``, the truth ``g`` where the regime
    has one, and the exact branch weights the construction targeted.

    ``spread`` scales the scatter of the observations about their construction
    centre. It does not affect ``T`` -- the offset is applied afterwards -- so it
    changes how informative the individual observations are without changing the
    branch split.
    """
    observations = int(observations)
    if observations < 1:
        raise ValueError("a dataset needs at least one observation")
    scale = spread * np.sqrt(mixture.v)

    if regime == "natural":
        g, _, x = mixture.simulate_dataset(observations, rng)
        return {"regime": regime, "x": np.asarray(x, dtype=np.float64),
                "g_true": float(g), "target_mass": None,
                "observations": observations}

    if regime in ("symmetric", "asymmetric"):
        # Centred at zero: no single observation is evidence for either branch
        # beyond the noise, so the branch split is set entirely by the offset.
        base = scale * rng.standard_normal(observations)
    elif regime == "antisymmetric":
        # ``+-`` pairs near the two prior modes. An odd count leaves one
        # observation at the centre so the pairing stays exact for the rest.
        half = observations // 2
        magnitude = mixture.mu_0 + 0.25 * scale * rng.standard_normal(half)
        pairs = np.concatenate([magnitude, -magnitude])
        if observations % 2:
            pairs = np.concatenate([pairs, 0.25 * scale * rng.standard_normal(1)])
        base = rng.permutation(pairs)
    else:
        raise ValueError(f"unknown regime {regime!r}; choose from {REGIMES}")

    if regime == "asymmetric":
        mass = float(branch_mass)
        if not 0.0 < mass < 1.0:
            raise ValueError("branch_mass must lie strictly inside (0, 1)")
        log_odds = np.log(mass / (1.0 - mass))
    else:
        log_odds = 0.0
    total = mixture.total_for_log_odds(log_odds, observations)
    x = _shift_to_total(base, total)

    achieved = mixture.branch_weights(x)
    target = np.array([1.0 - float(branch_mass), float(branch_mass)]) \
        if regime == "asymmetric" else np.array([0.5, 0.5])
    if not np.allclose(achieved, target, atol=1e-9):
        raise AssertionError(
            f"{regime} at n={observations} targeted branch weights {target} "
            f"but the constructed data gives {achieved}; the closed-form "
            "log-odds and the evidence integral disagree."
        )
    return {"regime": regime, "x": x, "g_true": None,
            "target_mass": float(target[1]), "observations": observations}


def describe(dataset, mixture=DEFAULT):
    """A one-line summary of what the exact posterior for a dataset looks like."""
    x = dataset["x"]
    weights, means, variance = mixture.global_posterior(x)
    mean, sd = mixture.global_moments(x)
    return (
        f"{dataset['regime']:<14} n={len(x):<3} T={x.sum():+.4f}  "
        f"w=({weights[0]:.3f}, {weights[1]:.3f})  "
        f"modes=({means[0]:+.3f}, {means[1]:+.3f})  "
        f"branch sd={np.sqrt(variance):.3f}  posterior sd={sd:.3f}"
    )
