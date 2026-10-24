"""A covariance output on the score network itself (opt-in).

The Jacobian composition (``gauss_jacobian`` / ``gauss_jacobian_FD``) needs every row's
backward precision ``Lambda_j = Cov(theta_0 | y, x_j)^-1``. It differentiates the score
for it (D extra network copies, and a curvature that nothing in DSM trains), and
``gauss_hierarchical`` estimates a constant one from a pilot run. This module makes the
network *predict* it, from the same trunk that predicts the score:

* **Head.** The trunk's last token states (the input of ``final_layer``, captured by a
  forward pre-hook, so every trunk works: legacy, tokenized, latent attention, wrapped
  by ``PreconditionedScore`` or not) are read out per node into ``rank + 1`` features
  ``(v_i, a_i)``. The lambda-free information over any node subset is

      K = S^-1 ( V V^T + diag(softplus(a)) ) S^-1,        S = diag(node scales),

  symmetric positive definite by construction (the production curvature clamp
  ``H >= 0`` built in), and any SPD matrix once ``rank >= D``. ``Lambda = K + lam^-2 I``;
  for a Gaussian posterior ``N(mu, C)`` the target is ``K = C^-1`` at every ``(y, lam)``.
* **Objective** (second-order DSM). With ``y = theta_0 + lam eps`` and the network's own
  Tweedie mean ``D = y + lam^2 s`` (detached), the Gaussian NLL of ``r = theta_0 - D``
  relative to ``K = 0`` is

      L = 1/2 r^T K r - 1/2 log det(I + lam^2 K),

  which is zero at ``K = 0``, stable in ``lam``, and minimised by ``K = Cov(theta_0|y,x)^-1
  - lam^-2 I``. It is added to the DSM loss on training passes only (the validation loss
  stays the plain DSM loss), via ``Trainer.covariance_objective``.
* **Sampler.** :class:`NetworkCovarianceHead` has the interface of the post-hoc
  ``CovarianceHead``, so ``MultiObsSampler.set_covariance_head`` uses it in place of the
  score Jacobian: one forward pass per refresh instead of D + 1.

Usage::

    enable_covariance_output(sbim, rank=8, node_scale=train_std)   # before training/loading
    sbim.train(..., covariance_weight=1.0)                           # joint training
    sampler.set_covariance_head(NetworkCovarianceHead(sbim, latent_nodes, observed_nodes))

Nothing changes unless ``enable_covariance_output`` is called.
"""
import math

import torch
import torch.nn as nn

from .schema import Atomic


def _backbone(model):
    """The innermost module that owns ``final_layer`` (unwraps PreconditionedScore etc.)."""
    seen = set()
    stack = [model]
    while stack:
        module = stack.pop(0)
        if id(module) in seen:
            continue
        seen.add(id(module))
        # a registered child, not an attribute a wrapper forwards (PreconditionedScore and
        # the precondition Wrapped forward __getattr__ to their backbone; their state_dict
        # is the backbone's, so the output must live on the backbone itself)
        if "final_layer" in module._modules:
            return module
        stack.extend(module.children())
    raise ValueError("no submodule with a final_layer: cannot attach a covariance output")


def _time_size(final_layer):
    for module in final_layer.adaLN_modulation:
        if isinstance(module, nn.Linear):
            return module.in_features
    raise ValueError("final_layer has no adaLN Linear")


class CovarianceOutput(nn.Module):
    """Per-node ``rank + 1`` readout of the captured trunk tokens -> information K."""

    def __init__(self, backbone, rank, node_scale):
        super().__init__()
        nodes = int(backbone.final_layer.nodes_size) if hasattr(backbone.final_layer, "nodes_size") \
            else int(backbone.nodes_size)
        hidden = int(backbone.final_layer.hidden_size)
        schema = getattr(backbone, "schema", None)
        legacy = not hasattr(backbone, "schema") or getattr(backbone, "trunk", "shared") == "position"
        # node -> (token index, readout id). Atomic nodes get their own readout; the slots of
        # an IID block share one per slot, so exchangeability is kept.
        node_token, readout = [], []
        if legacy:
            node_token = list(range(nodes))
            readout = list(range(nodes))
        else:
            next_id = 0
            for index, block in enumerate(schema.blocks):
                start, stop = schema.token_slices[index]
                if block.kind == Atomic.kind:
                    for k in range(stop - start):
                        node_token.append(start + k)
                        readout.append(next_id)
                        next_id += 1
                else:
                    item_dim = block.item_dim
                    for k in range(stop - start):
                        for slot in range(item_dim):
                            node_token.append(start + k)
                            readout.append(next_id + slot)
                    next_id += item_dim
        if len(node_token) != nodes:
            raise ValueError(f"node map covers {len(node_token)} of {nodes} nodes")
        self.rank = int(rank)
        self.nodes = nodes
        self.register_buffer("node_token", torch.tensor(node_token, dtype=torch.long))
        self.register_buffer("readout_id", torch.tensor(readout, dtype=torch.long))
        scale = torch.ones(nodes) if node_scale is None else torch.as_tensor(node_scale, dtype=torch.float32).reshape(-1)
        if scale.numel() != nodes:
            raise ValueError(f"node_scale has {scale.numel()} entries for {nodes} nodes")
        self.register_buffer("node_scale", scale.clamp_min(1e-6))
        readouts = max(readout) + 1
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(_time_size(backbone.final_layer), 2 * hidden))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        self.weight = nn.Parameter(torch.zeros(readouts, hidden, self.rank + 1))
        self.bias = nn.Parameter(torch.zeros(readouts, self.rank + 1))
        nn.init.normal_(self.weight, std=0.02 / math.sqrt(hidden))
        self._captured = None

    def capture(self, module, inputs):
        self._captured = (inputs[0], inputs[1])

    def features(self):
        if self._captured is None:
            raise RuntimeError("no captured trunk state: run the network forward first")
        tokens, t = self._captured
        if t.dim() == 3:                       # legacy embeds time per node
            t = t.mean(1)
        shift, scale = self.adaLN(t).unsqueeze(1).chunk(2, dim=-1)
        h = self.norm(tokens[:, self.node_token]) * (1 + scale) + shift
        h = nn.functional.silu(h)
        w = self.weight[self.readout_id]       # (nodes, hidden, rank + 1)
        return torch.einsum("bnh,nhk->bnk", h, w) + self.bias[self.readout_id]

    def information(self, nodes=None):
        """``K`` over ``nodes`` (all by default) for the last forward pass: ``(B, D, D)``."""
        f = self.features()
        index = torch.arange(self.nodes, device=f.device) if nodes is None else \
            torch.as_tensor(nodes, dtype=torch.long, device=f.device)
        f = f[:, index].to(torch.float32)
        v, a = f[..., :self.rank], f[..., self.rank]
        s = self.node_scale[index]
        K = v @ v.mT + torch.diag_embed(nn.functional.softplus(a + 0.5) + 1e-4)
        return K / s.reshape(1, -1, 1) / s.reshape(1, 1, -1)


def enable_covariance_output(sbim, rank=8, node_scale=None):
    """Attach a :class:`CovarianceOutput` to ``sbim``'s trunk (idempotent); returns it.

    Call before training, and before ``load_state_dict`` of a checkpoint trained with it.
    ``node_scale``: per-node scale (e.g. the training data std) setting K's units."""
    backbone = _backbone(sbim.model)
    head = getattr(backbone, "covariance_output", None)
    if head is None:
        head = CovarianceOutput(backbone, rank, node_scale).to(
            next(backbone.parameters()).device)
        backbone.covariance_output = head
        backbone.final_layer.register_forward_pre_hook(head.capture)
    sbim.covariance_output = head
    return head


def load_with_covariance_output(sbim, checkpoint, rank=8, device="cpu"):
    """Attach the output and load a checkpoint that was trained with it."""
    enable_covariance_output(sbim, rank=rank)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    sbim.model.load_state_dict(state.get("model_state_dict", state))
    sbim.model.to(device)
    return sbim


class CovarianceObjective:
    """Second-order DSM term for ``Trainer.covariance_objective`` (training passes only).

    ``weight`` scales the batch-summed relative NLL against the batch-summed DSM loss.
    ``detach_trunk=True`` trains only the readout (the score training is then unchanged)."""

    def __init__(self, head, weight=1.0, detach_trunk=False):
        self.head = head
        self.weight = float(weight)
        self.detach_trunk = bool(detach_trunk)

    def __call__(self, trainer, data, x_t, timesteps, score, condition_mask):
        sde = trainer.sde
        alpha = sde.alpha_t(timesteps).reshape(-1, 1).to(x_t.device)
        sigma = sde.marginal_prob_std(timesteps).reshape(-1, 1).to(x_t.device)
        lam = (sigma / alpha).reshape(-1)
        latent = (1 - condition_mask).to(x_t.dtype)
        active = torch.where(latent.sum(0) > 0)[0]          # nodes latent anywhere in the batch
        if active.numel() == 0:
            return x_t.new_zeros(())
        if self.detach_trunk:
            tokens, t = self.head._captured
            self.head._captured = (tokens.detach(), t.detach())
        K = self.head.information(active).to(torch.float64)
        m = latent[:, active].to(torch.float64)
        K = K * m.unsqueeze(-1) * m.unsqueeze(-2)
        y = x_t[:, active] / alpha
        theta0 = data[:, active]
        # score is d log p / d x; in y-coordinates s_y = alpha s_x, and D = y + lam^2 s_y
        denoised = y + (lam ** 2).reshape(-1, 1) * alpha * score[:, active].detach()
        r = ((theta0 - denoised) * m).to(torch.float64)
        quad = 0.5 * (r.unsqueeze(-2) @ K @ r.unsqueeze(-1)).reshape(-1)
        eye = torch.eye(active.numel(), dtype=torch.float64, device=K.device)
        lam2 = (lam.to(torch.float64) ** 2).reshape(-1, 1, 1)
        logdet = torch.linalg.cholesky(eye + lam2 * K).diagonal(dim1=-2, dim2=-1).log().sum(-1) * 2
        return self.weight * (quad - 0.5 * logdet).sum().to(x_t.dtype)


class NetworkCovarianceHead:
    """Sampler adapter: ``precision(y, x, lam)`` from the network's covariance output.

    Same interface as ``compass.experimental.covariance_head.CovarianceHead``, so
    ``MultiObsSampler.set_covariance_head(NetworkCovarianceHead(...))`` replaces the score
    Jacobian (and any pilot covariance) with one network forward pass per refresh."""

    def __init__(self, sbim, latent_nodes, observed_nodes):
        self.sbim = sbim
        self.head = sbim.covariance_output
        self.latent_nodes = [int(i) for i in latent_nodes]
        self.observed_nodes = [int(i) for i in observed_nodes]
        self.calls = 0

    def to(self, device):
        return self

    @torch.no_grad()
    def information(self, y, x, lam):
        sde = self.sbim.sde
        nodes = self.head.nodes
        lam = torch.as_tensor(lam, dtype=torch.float32, device=y.device).reshape(-1)
        t = sde.time_of_lambda(lam).reshape(-1, 1)
        alpha = sde.alpha_t(t).to(y.device)
        state = torch.zeros(y.shape[0], nodes, device=y.device)
        state[:, self.latent_nodes] = alpha * y.float()
        state[:, self.observed_nodes] = x.float()
        mask = torch.zeros(nodes, device=y.device)
        mask[self.observed_nodes] = 1.0
        self.sbim.model(x=state, t=t, c=mask.expand(y.shape[0], -1))
        self.calls += 1
        return self.head.information(self.latent_nodes).to(torch.float64)

    def precision(self, y, x, lam):
        K = self.information(y, x, lam)
        eye = torch.eye(K.shape[-1], dtype=torch.float64, device=K.device)
        lam = torch.as_tensor(lam, dtype=torch.float64, device=K.device).reshape(-1, 1, 1)
        return K + eye / lam ** 2
