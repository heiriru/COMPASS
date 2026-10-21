"""Inverse-kinematics simulator: a 4-parameter robot arm on a vertical rail.

Adapted from the BayesFlow "Diffusion Models" example
(https://bayesflow.org/v2.0.13/_examples/Diffusion_Models.html), which in turn
takes the benchmark from Ardizzone et al., "Training Normalizing Flows with the
Information Bottleneck for Competitive Generative Classification"
(https://arxiv.org/pdf/2101.10763.pdf).

The equations, the prior scales and the segment lengths are reproduced exactly.
Two deliberate differences from the source notebook:

* **No BayesFlow, Keras or JAX dependency.** The prior and the forward model are
  a handful of trigonometric operations, so they are written directly against
  NumPy. Nothing here needs an inference framework, and taking one on would make
  this package unrunnable in the repository's environment.
* **Explicit named generators.** The notebook calls the global
  ``np.random.normal``, which cannot be reproduced across processes. Every draw
  here comes from a ``numpy.random.Generator`` derived from one root seed by an
  explicit name, so the training draws, the illustration draws and the
  rejection-sampling draws occupy independent, individually reproducible
  streams -- the same convention ``Partial_Pooling`` uses.

The parameter vector is ``(height_arm, angle_1, angle_2, angle_3)``: the first
coordinate slides the shoulder along a vertical rail, the remaining three are
joint angles measured cumulatively from the rail.
"""

import hashlib
import os

import numpy as np

# Segment lengths, from the source notebook.
L1 = 0.5
L2 = 0.5
L3 = 1.0

# Spherical Gaussian prior, zero mean. The height is a quarter as wide as the
# angles, which is what keeps the shoulder near the rail's origin.
# ``IK_SD_H`` overrides the height scale for the wide-height study
# (recovery/wide_height.py); every module reads the scale from here.
PRIOR_SCALES = np.array([float(os.environ.get("IK_SD_H") or 0.25), 0.5, 0.5, 0.5])

PARAMETER_NAMES = ["height_arm", "angle_1", "angle_2", "angle_3"]
PARAMETER_LABELS = [" ".join(name.title().split("_")) for name in PARAMETER_NAMES]

ROOT_SEED = 2026


def generator(name, seed=ROOT_SEED):
    """A named, independent ``Generator`` derived from ``seed``.

    Deriving the stream from a hash of the name rather than from a counter means
    adding a new draw site later cannot shift an existing one, so figures stay
    reproducible as this package grows.
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def sample_prior(count, rng):
    """``count`` draws from the 4-dimensional spherical Gaussian prior."""
    return rng.normal(loc=0.0, scale=PRIOR_SCALES, size=(int(count), 4))


def forward(parameters):
    """End-effector coordinates for one or many parameter vectors.

    Vectorized over a leading batch axis; a single ``(4,)`` vector returns a
    ``(2,)`` result. The returned pair is ``(x1, x2)`` exactly as the notebook
    defines it -- ``x1`` is the sine branch and carries the arm height, ``x2`` is
    the cosine branch. Note that the notebook *plots* these reversed; see
    :func:`joint_positions`.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    single = parameters.ndim == 1
    parameters = np.atleast_2d(parameters)

    height = parameters[:, 0]
    a1 = parameters[:, 1]
    a2 = parameters[:, 1] + parameters[:, 2]
    a3 = parameters[:, 1] + parameters[:, 2] + parameters[:, 3]

    x1 = L1 * np.sin(a1) + L2 * np.sin(a2) + L3 * np.sin(a3) + height
    x2 = L1 * np.cos(a1) + L2 * np.cos(a2) + L3 * np.cos(a3)

    observables = np.stack([x1, x2], axis=1)
    return observables[0] if single else observables


def joint_positions(parameters):
    """The four joint positions per configuration, in *plot* coordinates.

    The notebook draws the arm with the rail vertical, passing
    ``observables[::-1]`` to its plotting helper -- i.e. horizontal = the cosine
    branch ``x2``, vertical = the sine branch ``x1``. This function returns
    ``(batch, 4, 2)`` in that same ``(horizontal, vertical)`` order, so the last
    joint equals ``forward(parameters)[..., ::-1]`` by construction. That
    identity is asserted in ``tests/test_simulator.py``.
    """
    parameters = np.atleast_2d(np.asarray(parameters, dtype=np.float64))
    height = parameters[:, 0]
    a1 = parameters[:, 1]
    a2 = parameters[:, 1] + parameters[:, 2]
    a3 = parameters[:, 1] + parameters[:, 2] + parameters[:, 3]

    horizontal = np.zeros((parameters.shape[0], 4))
    vertical = np.zeros((parameters.shape[0], 4))

    # Shoulder: on the rail, displaced by the height parameter.
    vertical[:, 0] = height
    for index, (angle, length) in enumerate(
        ((a1, L1), (a2, L2), (a3, L3)), start=1
    ):
        horizontal[:, index] = horizontal[:, index - 1] + length * np.cos(angle)
        vertical[:, index] = vertical[:, index - 1] + length * np.sin(angle)
    return np.stack([horizontal, vertical], axis=2)


def simulate(count, rng):
    """``count`` prior-predictive draws as ``(parameters, observables)``."""
    parameters = sample_prior(count, rng)
    return parameters, forward(parameters)


def sample_consistent_with(target, count, rng, tolerance=0.01, max_draws=20_000_000):
    """Prior draws whose end effector lands within ``tolerance`` of ``target``.

    Plain rejection sampling against the prior, which is exactly
    ``p(parameters | observables = target)`` up to the width of the acceptance
    ball -- no trained model and no approximate inference is involved. That is
    the point: the non-identifiability the illustration shows is a property of
    the forward model, not of any estimator, and it must be demonstrated without
    one.

    ``target`` is given in observation order ``(x1, x2)``.
    """
    target = np.asarray(target, dtype=np.float64).reshape(2)
    accepted = []
    drawn = 0
    block = 200_000
    while sum(len(item) for item in accepted) < count:
        if drawn >= max_draws:
            raise RuntimeError(
                f"Only {sum(len(i) for i in accepted)} of {count} draws landed "
                f"within {tolerance} of {target} after {drawn} attempts; widen "
                "the tolerance or pick a target nearer the prior's mass."
            )
        parameters = sample_prior(block, rng)
        distance = np.linalg.norm(forward(parameters) - target, axis=1)
        accepted.append(parameters[distance < tolerance])
        drawn += block
    return np.concatenate(accepted)[:count], drawn
