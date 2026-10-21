"""``sign_model`` with the data standardized -- the same problem, rescaled.

The question this exists to answer is whether COMPASS learns the row score
better when the data is normalized, which the sampler's own documentation
implicitly assumes: ``MultiObsSampler``'s default prior is ``N(0, 1)``, described
as correct "if the model was trained on parameters standardized to zero mean and
unit variance". ``sign_model`` is not standardized -- it ships
``sigma_g = 1.5``, and the marginal standard deviations are 1.50 (g), 1.58 (l)
and 1.60 (y).

Why a rescaled *model* rather than a normalization *step*
---------------------------------------------------------
Dividing every coordinate by one constant ``c`` maps this model exactly onto
another member of the same family, with ``(sigma_g, tau, sigma_y) / c``. That is
not an approximation: ``g/c ~ N(0, (sigma_g/c)^2)``, ``l/c | g/c ~ N(g/c,
(tau/c)^2)`` and ``y/c | l/c ~ N(+-l/c, (sigma_y/c)^2)``. So "train on
standardized data" and "train on this module" are the same experiment, and this
way every exact reference, the closed-form row score and every metric come along
for free with no rescaling code to get wrong.

A single global constant is the only rescaling that stays in the family. Per
coordinate z-scoring would divide ``l`` and ``y`` by different numbers, which
changes the emission gain ``y = s l + eps`` into ``y' = (a/b) s l' + eps'`` --
a different model, not a normalized one.

``c = sigma_g = 1.5`` is chosen so the shared parameter has unit variance, which
is exactly the condition the sampler's default prior assumes. The other two
coordinates land at 1.054 and 1.068, so this is a genuine standardization of all
three.

What is and is not comparable
-----------------------------
Relative RMS score error is scale free -- both the error and the exact score
scale by ``c`` -- so the two networks can be compared directly, *provided they
are graded at corresponding noise levels*: a level ``lam`` in the original
problem is ``lam / c`` here, because the diffusion schedule is fixed while the
data shrank. ``training_variants.py`` does that rescaling. ``W1 / sigma`` is
likewise scale free and directly comparable.
"""
from __future__ import annotations

import math

import numpy as np  # noqa: F401  (kept so the module mirrors sign_model's API)
import torch  # noqa: F401

import sign_model

# The standardizing constant: the shared parameter's own standard deviation.
SCALE = sign_model.SIGMA_G

SIGMA_G = sign_model.SIGMA_G / SCALE
TAU = sign_model.TAU / SCALE
SIGMA_Y = sign_model.SIGMA_Y / SCALE

MU_G = 0.0
LOCAL_PRIOR_MEAN = MU_G
LOCAL_PRIOR_STD = math.sqrt(SIGMA_G**2 + TAU**2)

NAME = "sign_normalized"
TITLE = "Sign-ambiguous hierarchy, standardized data"

NODES = sign_model.NODES
GLOBAL_INDEX = sign_model.GLOBAL_INDEX
LOCAL_INDEX = sign_model.LOCAL_INDEX
OBSERVED_INDEX = sign_model.OBSERVED_INDEX
CONDITION_MASK = sign_model.CONDITION_MASK
HIERARCHY = sign_model.HIERARCHY


def _rescaled(function, *args, **kwargs):
    """Call a ``sign_model`` function with this module's constants in force.

    ``sign_model``'s functions read its module-level constants, so the scaled
    variant is obtained by swapping them for the duration of the call rather
    than by duplicating three hundred lines of algebra. Restored in a ``finally``
    so an exception cannot leave the original module rescaled.
    """
    saved = (sign_model.SIGMA_G, sign_model.TAU, sign_model.SIGMA_Y)
    try:
        sign_model.SIGMA_G, sign_model.TAU, sign_model.SIGMA_Y = (
            SIGMA_G, TAU, SIGMA_Y
        )
        return function(*args, **kwargs)
    finally:
        sign_model.SIGMA_G, sign_model.TAU, sign_model.SIGMA_Y = saved


def simulate(count, generator=None):
    return _rescaled(sign_model.simulate, count, generator)


def observations(count, seed):
    """The *same* dataset as ``sign_model``, in standardized units.

    Drawing fresh standardized data with the same seed would give a different
    dataset (the generator is consumed identically but the constants differ), and
    then the two networks would be graded on different problems. Scaling the
    original dataset keeps the comparison exact: this is literally
    ``sign_model``'s dataset divided by ``SCALE``.
    """
    truth, local, discrete, y = sign_model.observations(count, seed)
    return truth / SCALE, local / SCALE, discrete, y / SCALE


def sign_configurations(n):
    return sign_model.sign_configurations(n)


def coherent_components(y):
    return _rescaled(sign_model.coherent_components, y)


def mixture(y):
    return _rescaled(sign_model.mixture, y)


def row_mixture(value):
    return _rescaled(sign_model.row_mixture, value)


def conditional_local_mixture(g, value):
    return _rescaled(sign_model.conditional_local_mixture, g, value)


def row_components(obs):
    return _rescaled(sign_model.row_components, obs)


def conditional_local_components(g, obs):
    return _rescaled(sign_model.conditional_local_components, g, obs)
