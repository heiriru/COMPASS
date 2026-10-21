"""Shared network recipe for the Arruda paper benchmarks.

The stock COMPASS network cannot resolve the small noise scales these
benchmarks need. Measured on the Gaussian toy (plain h64d4, eps=1e-5, 200k
simulations): relative score error 10.1 at lambda=0.003, 2.4 at 0.01 and a
slope of 0.38 at 0.1, while being 2-15% accurate for lambda >= 0.3. Every
composition rule then failed by 1.7-2.8 posterior sd at only J=10. The cause is
the time embedding (``cos/sin(t * freqs)`` with ``freqs <= 1`` is nearly
constant over t in (1e-5, 0.1]) together with the plain epsilon-style output.

The fix is the one measured in ModeTree/artifacts/gm_residual/REPORT.md:
``PreconditionedScore`` -- a closed-form Gaussian prior score as the residual
baseline, EDM input scaling, and a log-lambda time coordinate. The wrapper has no
parameters of its own and ``SBIm.save`` writes an ordinary backbone checkpoint,
so ``build`` must re-attach it before loading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ModeTree" / "experiments"))
from residual_score import PreconditionedScore  # noqa: E402


class GaussianPriorScore(torch.nn.Module):
    """``sigma(t) * grad log p_t`` of an independent Gaussian prior, raw convention.

    Zero on coordinates without a closed-form Gaussian marginal and on
    conditioned coordinates.
    """

    def __init__(self, sde, prior_std, latent_indices, nodes):
        super().__init__()
        self.sde = sde
        variance = torch.zeros(nodes, dtype=torch.float64)
        available = torch.zeros(nodes, dtype=torch.float64)
        for position, index in enumerate(latent_indices):
            variance[index] = float(prior_std[position]) ** 2
            available[index] = 1.0
        self.register_buffer("prior_variance", variance)
        self.register_buffer("available", available)

    def forward(self, x, t, c, return_attn_weights=False, **unused):
        lam = self.sde.lambda_t(t).to(x.device).reshape(-1, 1).to(torch.float64)
        variance = self.prior_variance.to(x.device).reshape(1, -1) + lam ** 2
        active = self.available.to(x.device).reshape(1, -1) * (1.0 - c.to(torch.float64))
        score = -active * x.to(torch.float64) / variance
        scale = self.sde.marginal_prob_std(t).to(x.device).reshape(-1, 1)
        result = (score * scale).to(x.dtype)
        return (result, None) if return_attn_weights else result


def build(nodes, sde_kwargs, network, device, eps, prior_std, latent_indices,
          sigma_data, checkpoint=None):
    """A preconditioned SBIm; loads the backbone from ``checkpoint`` if given."""
    from compass import ScoreBasedInferenceModel as SBIm

    model = SBIm(nodes_size=nodes, device=device, **sde_kwargs, **network)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device)
        model.model.load_state_dict(state.get("model_state_dict", state))
    baseline = GaussianPriorScore(model.sde, prior_std, latent_indices, nodes)
    lam_min = float(model.sde.lambda_t(torch.tensor([eps])))
    lam_max = float(model.sde.lambda_t(torch.tensor([1.0])))
    model.model = PreconditionedScore(model.model, baseline.to(device), model.sde,
                                      sigma_data, (lam_min, lam_max)).to(device)
    return model


def floor_training_eps(eps):
    """Context manager: ``Trainer`` samples t from ``(eps, 1]`` instead of 1e-3."""
    import contextlib
    from compass.Trainer import Trainer

    @contextlib.contextmanager
    def patched():
        original = Trainer._sample_timesteps

        def floored(self, batch_size):
            self.eps = eps
            return original(self, batch_size)
        Trainer._sample_timesteps = floored
        try:
            yield
        finally:
            Trainer._sample_timesteps = original
    return patched()


def sample_chunked(sample, observations, num_samples, max_rows, **kwargs):
    """Call ``MultiObsSampler.sample`` on independent chunks of draws, no autograd.

    Draws are independent particles under every rule here, so splitting them
    changes nothing statistically; it only bounds memory. Rules with a pilot
    (``gauss``, ``full_gaussian``, ``gauss_hierarchical``) re-estimate it per
    chunk. ``no_grad`` is safe for ``gauss_jacobian`` (forward-mode JVPs need no
    graph) and removes the backward activations the JVP would otherwise keep.
    """
    chunk = max(1, min(num_samples, int(max_rows // max(observations, 1))))
    blocks = []
    with torch.no_grad():
        for start in range(0, num_samples, chunk):
            size = min(chunk, num_samples - start)
            blocks.append(sample(num_samples=size, **kwargs).detach().cpu())
    return torch.cat(blocks, dim=1)


def freeze(model):
    """Inference only: parameters stop requiring grad.

    ``MultiObsSampler``'s Jacobian rules run their forward-mode JVPs under
    ``torch.enable_grad()``; with trainable parameters every JVP also records a
    full autograd graph (measured 4.4 GB at 2,000 rows vs 0.17 GB for the
    uncorrected rule, h64d4). Frozen parameters make that graph empty.
    """
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    return model
