"""PyTorch ways of taking the per-row latent score Jacobian.

Each variant maps a flat state ``z`` of shape ``(batch, 100)`` to the
``(batch, 10, 10)`` Jacobian of the theta scores with respect to the theta
coordinates, at fixed ``t`` and condition mask (theta latent, trials observed).
The score is the one ``MultiObsSampler._raw_row_scores`` returns for the VESDE:
``model(z, t, c) / std(t)``.

- ``jvp_loop``: what ``gauss_jacobian`` does now -- 10 sequential
  ``torch.func.jvp`` calls on the MATH attention backend.
- ``vmap_jvp``: the same 10 JVPs batched with ``torch.func.vmap``, i.e. what
  ``jax.jacfwd`` does.
- ``fd``: what ``gauss_jacobian_FD`` does -- the base point and 10 copies, each
  shifted by ``h = 1e-2 * lambda`` in one coordinate, stacked into one no-grad
  call on the fused attention kernel.
- ``diag_jvp`` / ``diag_fd``: one tangent / one shifted copy moving *all* 10
  coordinates at once. They return the row sums of the Jacobian, which equal
  its diagonal only when the Jacobian is diagonal; ``run_torch`` measures
  whether it is before they are compared on accuracy.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import FD_STEP, NODES, THETA_DIM, vesde_std  # noqa: E402


def load_network(path, device):
    from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel
    model = ScoreBasedInferenceModel.load(str(path), device=device, arch="tokenized")
    network = model.model.to(device)
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def make_inputs(observations, draws, lam, generator):
    """``draws`` noisy copies of each row's normalized parameters, with the row's
    observed trials; theta is latent (mask 0) and the trials observed (mask 1)."""
    theta, obs = observations["theta"], observations["obs"]
    rows = theta.shape[0]
    noise = torch.randn(rows, draws, THETA_DIM, generator=generator)
    z_theta = theta[:, None, :] + lam * noise
    z = torch.cat((z_theta, obs[:, None, :].expand(rows, draws, -1)), dim=-1)
    z = z.reshape(rows * draws, NODES)
    c = torch.zeros_like(z)
    c[:, THETA_DIM:] = 1.0
    return z, c


def score_fn(network, t, c):
    t_tensor = torch.tensor([t], dtype=c.dtype, device=c.device)
    scale = vesde_std(t)

    def f(z):
        return network(x=z, t=t_tensor, c=c) / scale
    return f


def math_attention():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.MATH])


def jvp_loop(f, z, lam):
    from torch.func import jvp
    columns = []
    with math_attention():
        for k in range(THETA_DIM):
            tangent = torch.zeros_like(z)
            tangent[:, k] = 1.0
            _, column = jvp(f, (z,), (tangent,))
            columns.append(column[:, :THETA_DIM])
    return torch.stack(columns, dim=-1)


def _basis(z):
    basis = torch.zeros(THETA_DIM, *z.shape, dtype=z.dtype, device=z.device)
    basis[torch.arange(THETA_DIM), :, torch.arange(THETA_DIM)] = 1.0
    return basis


def vmap_jvp(f, z, lam):
    from torch.func import jvp, vmap
    with math_attention():
        columns = vmap(lambda v: jvp(f, (z,), (v,))[1])(_basis(z))
    return columns[..., :THETA_DIM].permute(1, 2, 0)


def fd(f, z, lam):
    step = FD_STEP * lam
    copies = z.unsqueeze(0) + step * _basis(z)
    stacked = torch.cat((z.unsqueeze(0), copies), dim=0).reshape(-1, NODES)
    # c is closed over at batch size B; tile it for the stacked call.
    with torch.no_grad():
        scores = f.stacked(stacked, THETA_DIM + 1)
    scores = scores.reshape(THETA_DIM + 1, *z.shape)[..., :THETA_DIM]
    return ((scores[1:] - scores[:1]) / step).permute(1, 2, 0)


def diag_jvp(f, z, lam):
    from torch.func import jvp
    tangent = torch.zeros_like(z)
    tangent[:, :THETA_DIM] = 1.0
    with math_attention():
        _, column = jvp(f, (z,), (tangent,))
    return torch.diag_embed(column[:, :THETA_DIM])


def diag_fd(f, z, lam):
    step = FD_STEP * lam
    shifted = z.clone()
    shifted[:, :THETA_DIM] += step
    with torch.no_grad():
        scores = f.stacked(torch.cat((z, shifted), dim=0), 2)
    scores = scores.reshape(2, *z.shape)[..., :THETA_DIM]
    return torch.diag_embed((scores[1] - scores[0]) / step)


VARIANTS = {
    "jvp_loop": jvp_loop,
    "vmap_jvp": vmap_jvp,
    "fd": fd,
    "diag_jvp": diag_jvp,
    "diag_fd": diag_fd,
}


class Score:
    """``f(z)`` plus ``f.stacked(z, copies)`` for the stacked FD calls, with an
    optional ``torch.compile`` of the network call."""

    def __init__(self, network, t, c, compile_mode=None):
        self.network = network
        self.t = torch.tensor([t], dtype=c.dtype, device=c.device)
        self.scale = vesde_std(t)
        self.c = c
        self._stacked_c = {}
        self.forward = (
            torch.compile(self._forward, mode=compile_mode, dynamic=False)
            if compile_mode else self._forward
        )

    def _forward(self, z, c):
        return self.network(x=z, t=self.t, c=c) / self.scale

    def __call__(self, z):
        return self.forward(z, self.c)

    def stacked(self, z, copies):
        if copies not in self._stacked_c:
            self._stacked_c[copies] = self.c.repeat(copies, 1)
        return self.forward(z, self._stacked_c[copies])
