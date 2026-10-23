"""Conservative score parameterization: the score is the gradient of a scalar energy.

A plain score network's Jacobian is not a Hessian (measured 30% antisymmetric on the
IK checkpoint), which hurts the Jacobian compositions and the PF-ODE likelihood. Here
the network ``r`` (raw COMPASS convention, ``r ~ sigma * score``) only parameterizes a
denoiser ``d = x + sigma r``, and the score is the gradient of

    E(x) = - || (1 - c) * (x - d(x)) ||^2 / (2 sigma^2)          (Salimans & Ho 2021)

over the latent coordinates (``c`` the condition mask). Then ``grad s = Hess E`` is
symmetric by construction, and wherever ``d`` is locally flat (``J_d = 0``) the score
is exactly the plain one, ``s = (d - x) / sigma^2 = r / sigma``. Cost: one extra
backward pass at inference, a double backward in training (~2-3x).

Use :func:`enable_conservative` on a freshly built SBIm before training, and
:func:`load_conservative` to rebuild a trained one (the checkpoint stores ``base.*``
keys). Remove by deleting this module and its test.
"""
import torch
import torch.nn as nn


class ConservativeScore(nn.Module):
    def __init__(self, base, sde):
        super().__init__()
        self.base = base
        self.sde = sde

    # rows per energy-gradient pass at inference (bounds the backward-graph memory)
    INFERENCE_CHUNK = 16384

    def forward(self, x, t, c, return_attn_weights=False, **kwargs):
        create_graph = torch.is_grad_enabled() and (self.training or x.requires_grad)
        if not create_graph and x.shape[0] > self.INFERENCE_CHUNK:
            step = self.INFERENCE_CHUNK
            t_rows = t.expand(x.shape[0], *t.shape[1:]) if t.dim() and t.shape[0] == x.shape[0] else None
            c_rows = c if c.dim() > 1 and c.shape[0] == x.shape[0] else None
            out = torch.cat([self._forward(x[i:i + step],
                                           t_rows[i:i + step] if t_rows is not None else t,
                                           c_rows[i:i + step] if c_rows is not None else c,
                                           **kwargs) for i in range(0, x.shape[0], step)])
        else:
            out = self._forward(x, t, c, **kwargs)
        return (out, None) if return_attn_weights else out

    def _forward(self, x, t, c, **kwargs):
        sigma = self.sde.marginal_prob_std(t).to(x.device)
        sigma = sigma.reshape(-1, 1) if sigma.numel() > 1 else sigma.reshape(1, 1)
        # a graph through the score is needed when training, or when the caller
        # differentiates the score itself (x.requires_grad)
        create_graph = torch.is_grad_enabled() and (self.training or x.requires_grad)
        with torch.enable_grad(), _attention_backend(create_graph):
            state = x if create_graph else x.detach()
            if not state.requires_grad:
                state = state.requires_grad_(True)
            raw = self.base(x=state, t=t, c=c, **kwargs)
            if isinstance(raw, tuple):
                raw = raw[0]
            latent = 1.0 - c.to(raw.dtype)
            # x - d = -sigma r, so E = -|| (1-c) r ||^2 / 2 with d = x + sigma r
            energy = -0.5 * ((latent * raw) ** 2).sum()
            grad, = torch.autograd.grad(energy, state, create_graph=create_graph)
        out = sigma * grad * latent                     # raw convention: sigma * score
        if not create_graph:
            out = out.detach()
        return out


def _attention_backend(double_backward):
    """The fused attention kernels have no double-backward; use the math backend then."""
    from contextlib import nullcontext
    if not double_backward:
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:                                    # pragma: no cover
        return nullcontext()
    return sdpa_kernel([SDPBackend.MATH])


def enable_conservative(sbim):
    """Wrap ``sbim.model`` in place; returns ``sbim``."""
    if not isinstance(sbim.model, ConservativeScore):
        sbim.model = ConservativeScore(sbim.model, sbim.sde)
    return sbim


def load_conservative(sbim, checkpoint, device="cpu"):
    """Wrap ``sbim`` and load a checkpoint saved from a wrapped model."""
    enable_conservative(sbim)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    sbim.model.load_state_dict(state.get("model_state_dict", state))
    sbim.model.to(device)
    return sbim
