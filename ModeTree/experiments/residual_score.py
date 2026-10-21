"""Benchmark-independent score wrappers: residual parameterization and preconditioning.

Both are wrappers around a stock COMPASS network rather than edits to it, so
the backbone, optimizer, mask menu and denoising loss stay exactly what the
baseline uses. Neither holds a trainable parameter of its own. They are kept
here, apart from either benchmark, because the Gaussian mixture and inverse
kinematics differ only in what supplies the baseline score -- an exact mixture
marginal in one case, an independent Gaussian prior in the other -- and nothing
about the wrappers themselves should be duplicated between them.

See ``gmresidual`` for the derivation of the raw-convention algebra
(``raw = sigma * s_base + backbone``) and of the two preconditioners.
"""

from __future__ import annotations

import math

import torch


class ResidualScore(torch.nn.Module):
    """``raw = sigma * s_base + backbone``; the backbone learns the residual.

    ``state_dict`` delegates to the backbone so ``SBIm.save``/``SBIm.load`` write
    and read an ordinary checkpoint: the baseline is a closed form, not a
    parameter, and nothing about the saved file should record that a wrapper was
    used at training time.
    """

    def __init__(self, backbone, baseline):
        super().__init__()
        self.backbone = backbone
        # not a submodule: its buffers must stay out of the backbone's
        # state_dict, and it holds nothing to train
        object.__setattr__(self, "baseline", baseline)

    def forward(self, x, t, c, return_attn_weights=False, **kwargs):
        out = self.backbone(x=x, t=t, c=c,
                            return_attn_weights=return_attn_weights, **kwargs)
        attention = None
        if isinstance(out, tuple):
            out, attention = out
        result = out + self.baseline(x, t, c)
        return (result, attention) if return_attn_weights else result

    def state_dict(self, *args, **kwargs):
        return self.backbone.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.backbone.load_state_dict(*args, **kwargs)


class PreconditionedScore(ResidualScore):
    """:class:`ResidualScore` plus the two derived preconditioners above."""

    def __init__(self, backbone, baseline, sde, sigma_data, lambda_range,
                 embedding_span=1000.0):
        super().__init__(backbone, baseline)
        self.sde = sde
        self.register_buffer("sigma_data",
                             torch.as_tensor(sigma_data, dtype=torch.float32))
        self.log_lambda_min = math.log(float(lambda_range[0]))
        self.log_lambda_max = math.log(float(lambda_range[1]))
        self.embedding_span = float(embedding_span)

    def _c_noise(self, t):
        lam = self.sde.lambda_t(t).clamp_min(1e-12)
        unit = (torch.log(lam) - self.log_lambda_min) / (
            self.log_lambda_max - self.log_lambda_min)
        return self.embedding_span * unit

    def _c_in(self, t, c):
        lam = self.sde.lambda_t(t).to(c.device).reshape(-1, 1)
        data = self.sigma_data.to(c.device).reshape(1, -1)
        # conditioned coordinates carry clean data and are never noised
        variance = (1.0 - c) * lam ** 2
        return 1.0 / torch.sqrt(variance + data ** 2)

    def forward(self, x, t, c, return_attn_weights=False, **kwargs):
        out = self.backbone(x=(x * self._c_in(t, c)).to(x.dtype),
                            t=self._c_noise(t), c=c,
                            return_attn_weights=return_attn_weights, **kwargs)
        attention = None
        if isinstance(out, tuple):
            out, attention = out
        result = out + self.baseline(x, t, c)
        return (result, attention) if return_attn_weights else result


