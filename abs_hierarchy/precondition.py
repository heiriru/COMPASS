"""Ablation wrappers for the low-noise score: time remap, noise floor, residual output.

Each change is switched on separately so its effect can be measured alone
(the ModeTree version, ``ModeTree/experiments/residual_score.py``, applies them
together):

``time``      The network sees ``c_noise = span * (log lam - log lam_min) /
              (log lam_max - log lam_min)`` instead of the raw diffusion time.
              ``TimestepEmbedder`` builds ``cos/sin(t * f)`` with ``f <= 1``, so
              for ``t`` in ``(1e-3, 1]`` the composition band (``lam`` 0.03-0.3 is
              ``t`` 0.001-0.04) gives arguments below 0.04: every cos is 1 and the
              embedding is nearly constant exactly where the score must be sharp.
``eps``       The training floor: ``Trainer.eps`` (stock 1e-3, ``lam`` = 0.0317).
``residual``  ``raw = lam * s_prior + backbone`` on the latent coordinates, where
              ``s_prior`` is the diffused score of the Gaussian moment match of
              each latent's prior marginal (exact for g, approximate for l). The
              DSM loss is unchanged, so the backbone learns the posterior minus
              prior.

Nothing here has a trainable parameter and ``state_dict`` delegates to the
backbone, so checkpoints stay ordinary; ``precondition.json`` next to the
checkpoint records the switches and ``load`` re-applies them.
"""

import json
import math
from pathlib import Path

import torch

SIDECAR = "precondition.json"
STOCK_EPS = 1e-3


class Wrapped(torch.nn.Module):
    def __init__(self, backbone, sde, time=False, residual=False, prior=None,
                 lambda_range=None, span=1000.0):
        super().__init__()
        self.backbone, self.sde = backbone, sde
        self.time, self.residual, self.span = bool(time), bool(residual), float(span)
        if self.time:
            self.log_lo, self.log_hi = (math.log(v) for v in lambda_range)
        if self.residual:
            means, stds = prior
            self.register_buffer("prior_mean", torch.tensor(means, dtype=torch.float32))
            self.register_buffer("prior_var", torch.tensor(stds, dtype=torch.float32) ** 2)

    def _c_noise(self, t):
        lam = self.sde.lambda_t(t).clamp_min(1e-12)
        return self.span * (torch.log(lam) - self.log_lo) / (self.log_hi - self.log_lo)

    def _baseline(self, x, t, c):
        """``lam * s_prior`` on latent coordinates, 0 on conditioned ones."""
        lam = self.sde.lambda_t(t).to(x.device).reshape(-1, 1)
        k = self.prior_mean.numel()
        score = -(x[:, :k] - self.prior_mean) / (self.prior_var + lam ** 2)
        out = torch.zeros_like(x)
        latent = (c < 0.5).to(x.dtype)
        if latent.dim() == 1:
            latent = latent.expand_as(x)
        out[:, :k] = score * lam * latent[:, :k]
        return out

    def forward(self, x, t, c, return_attn_weights=False, **kwargs):
        out = self.backbone(x=x, t=self._c_noise(t) if self.time else t, c=c,
                            return_attn_weights=return_attn_weights, **kwargs)
        attention = None
        if isinstance(out, tuple):
            out, attention = out
        if self.residual:
            out = out + self._baseline(x, t, c).to(out.dtype)
        return (out, attention) if return_attn_weights else out

    def state_dict(self, *args, **kwargs):
        return self.backbone.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.backbone.load_state_dict(*args, **kwargs)

    def __getattr__(self, name):
        # architecture_payload() and friends read trunk attributes off .model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("backbone"), name)


def attach(network, model, time=False, eps=STOCK_EPS, residual=False, prior=None):
    """Wrap ``network.model`` in place; set the trainer's floor. Returns the config.

    ``prior=(means, stds)`` over the leading latent coordinates overrides the
    (g, l) moments read off ``model`` (the IK benchmark passes its 4 latents).
    """
    config = {"time": bool(time), "eps": float(eps), "residual": bool(residual)}
    if prior is not None:
        config["prior"] = [list(map(float, prior[0])), list(map(float, prior[1]))]
    if not (time or residual):
        network.trainer.eps = float(eps)
        return config
    device = next(network.model.parameters()).device
    lam = lambda s: float(network.sde.lambda_t(torch.tensor([s])))  # noqa: E731
    network.model = Wrapped(
        network.model, network.sde, time=time, residual=residual,
        prior=prior if prior is not None else
        ([model.global_prior_mean, model.local_prior_mean],
         [model.global_prior_std, model.local_prior_std]),
        lambda_range=(lam(eps), lam(1.0)),
    ).to(device)
    network.trainer.eps = float(eps)
    return config


def save(config, directory):
    Path(directory, SIDECAR).write_text(json.dumps(config, indent=2))


def load(network, directory, model):
    """Re-apply a saved config to a freshly loaded network (no-op without a sidecar)."""
    path = Path(directory, SIDECAR)
    if not path.exists():
        return None
    config = json.loads(path.read_text())
    attach(network, model, **config)
    return config
