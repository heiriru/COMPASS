"""EDM-style input scaling for the COMPASS trunk (benchmark bridge only).

COMPASS feeds the noisy latent values to the network unscaled: under the VESDE
a latent token at time t has scale sqrt(1 + sigma(t)^2) (z-scored data plus
noise), up to ~10x an observed token at sigma_max. With unmasked attention those
large, mostly-noise tokens draw ~40% of the latent queries' attention at high
noise (diagnosed on Two Moons 10k). Here every latent coordinate is multiplied
by c_in(t) = 1 / sqrt(1 + sigma(t)^2) before the trunk; observed coordinates are
untouched. No parameters are added and the output scaling (score = out / sigma)
is unchanged.

``install(sigma)`` wraps whatever ``compass.ScoreBasedInferenceModel.
ConditionTransformer`` currently is (the legacy trunk, or a variant installed
just before it, e.g. full attention), so it composes with the other bridge
variants. It touches nothing in ``src/compass``; a config without
``input_scaling: true`` never calls it.
"""

import importlib

import torch

from compass.SDE import VESDE


def install(sigma=25.0):
    module = importlib.import_module("compass.ScoreBasedInferenceModel")
    base = module.ConditionTransformer
    sde = VESDE(sigma=float(sigma))

    class InputScaled(base):
        def forward(self, x, t, c, return_attn_weights=False):
            tt = t.reshape(-1, 1) if torch.is_tensor(t) else torch.full((x.shape[0], 1), float(t), device=x.device)
            std = sde.marginal_prob_std(tt.float()).to(x.device, x.dtype).reshape(-1, 1)
            c_in = torch.rsqrt(1.0 + std ** 2)
            cf = c.to(x.dtype)
            x = x * (cf + (1.0 - cf) * c_in)
            return super().forward(x, t, c, return_attn_weights=return_attn_weights)

    InputScaled.__name__ = f"InputScaled{base.__name__}"
    module.ConditionTransformer = InputScaled
