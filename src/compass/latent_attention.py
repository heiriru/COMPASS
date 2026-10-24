"""Latent-to-latent attention for the shared tokenized trunk.

The production ``TokenizedTransformer`` (``trunk="shared"``) builds its
attention with ``key_padding_mask = 1 - c``: only *conditioned* tokens are keys.
A latent token therefore never sees another latent token, and the token-wise
head adds no mixing either, so every latent output is a function of its own
noisy value, the conditioned tokens and ``t`` alone. The network's joint
conditional is then a product of one-dimensional conditionals (mean-field over
latents). ``p(g, l | x)`` factorises into ``prod_k p(g_k | x) prod_m p(l_m | x)``.

This module adds a per-pair attention rule on top of the same parameters.
Nothing is added to the state dict, so a production checkpoint loads into
either variant, and the production class and files stay untouched:

``"none"``
    Production behaviour (only conditioned keys). Kept so a converted model
    with ``mode="none"`` can be checked against the unconverted one.
``"global"``
    Latent group-level tokens also attend to latent group-level tokens. The
    ``g`` output under ``p(g,l|x)`` then targets the *joint* marginal score of
    ``p_t(g | x)``. It is still marginal over ``l``, because ``g`` never sees a
    latent ``l``.
``"hierarchical"``
    Role order ``g > l > x``. A latent token attends to latent tokens of its own
    role or of a role above it: ``g`` sees ``g``; ``l`` sees ``g, l``; ``x``
    sees ``g, l``. ``p(g,l|x)`` is then represented as
    ``p_t(g|x) p_t(l|g,x)``, with the ``g`` output still exactly marginal.
``"full"``
    Every latent token attends to every latent token. The outputs are the
    joint score of ``p_t(g, l | x)``. The ``g`` output is *conditional* on the
    current ``l``, so a marginal needs the Schur reduction of
    ``compass.experimental.marginal_schur``.

In every mode, items of one ``IID`` block never attend each other while
latent. Exchangeable trials are conditionally independent given the
parameters, and the likelihood ``p(x | g, l)`` keeps that factorisation.

Measured (docs/arruda_schur_progress.md): ``"hierarchical"`` is the best trunk on
partial pooling, inverse kinematics and the folded |g| hierarchies; on partial
pooling ``hierarchical`` + gauss_jacobian_FD gives mean r 0.869 against 0.820 for
``"none"`` under the same rule. The exception is the Gaussian-mixture benchmark, which is
network-limited under every mode.

Usage::

    model = ScoreBasedInferenceModel(theta_size=..., x_spec=..., arch="tokenized",
                                     latent_attention="hierarchical",
                                     latent_roles=(global_size, local_size))
    model.train(...)
    model.save(directory)        # the mode and roles are part of the checkpoint
    model = ScoreBasedInferenceModel.load(directory + "/Model.pt")

``latent_roles`` is ``(global_size, local_size)`` for the flat
``[g, l, x]`` layout (:func:`compass.hierarchical_roles`) or an explicit
``{"g": [...], "l": [...], "x": [...]}`` map of node indices.

The older sidecar workflow still works for checkpoints written by the
experimental module::

    enable_latent_attention(model, "full", roles={"g": [...], "l": [...], "x": [...]})
    save_latent_attention(model, directory)       # next to model.save(directory)
    model = ScoreBasedInferenceModel.load(directory)
    load_latent_attention(model, directory)
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from .ConditionTransformer import modulate
from .TokenizedTransformer import TokenizedTransformer
from .schema import IID, hierarchical_roles

MODES = ("none", "global", "hierarchical", "full")
ROLE_CODES = {"g": 0, "l": 1, "x": 2}
SIDECAR = "latent_attention.json"


def _token_roles(schema, roles):
    """Role code and IID-block id per token.

    A token takes a role only if every scalar it carries has that role. A
    token straddling two roles would make the rule ambiguous, so it raises.
    """
    nodes = schema.nodes_size
    code = torch.full((nodes,), -1, dtype=torch.long)
    for name, indices in roles.items():
        if name not in ROLE_CODES:
            raise ValueError(f"Unknown role {name!r}; roles are g, l and x.")
        code[list(indices)] = ROLE_CODES[name]
    if torch.any(code < 0):
        missing = torch.where(code < 0)[0].tolist()
        raise ValueError(f"roles must partition all nodes; missing {missing}.")

    per_role = []
    for value in range(3):
        flags = schema.token_mask((code == value).float().unsqueeze(0))[0]
        per_role.append(flags)
    per_role = torch.stack(per_role)                       # (3, tokens)
    if torch.any(per_role.sum(0) != 1):
        raise ValueError("A token mixes roles; the latent attention rule is undefined.")
    token_role = per_role.argmax(0)

    iid_block = torch.full((schema.token_count,), -1, dtype=torch.long)
    for index, block in enumerate(schema.blocks):
        if block.kind == IID.kind:
            start, stop = schema.token_slices[index]
            iid_block[start:stop] = index
    return token_role, iid_block


def latent_edges(token_role, iid_block, mode):
    """``(tokens, tokens)`` bool, True where latent query i may attend latent key j."""
    if mode not in MODES:
        raise ValueError(f"Unknown latent attention mode {mode!r}; choose from {MODES}.")
    ri = token_role.unsqueeze(1)
    rj = token_role.unsqueeze(0)
    if mode == "none":
        edges = torch.zeros(ri.shape[0], rj.shape[1], dtype=torch.bool)
    elif mode == "global":
        edges = (ri == 0) & (rj == 0)
    elif mode == "hierarchical":
        # rank(j) <= rank(i), and x never attends latent x.
        edges = (rj <= ri) & ~((ri == 2) & (rj == 2))
    else:
        edges = torch.ones(ri.shape[0], rj.shape[1], dtype=torch.bool)
    same_iid = (iid_block.unsqueeze(1) >= 0) & (
        iid_block.unsqueeze(1) == iid_block.unsqueeze(0)
    )
    # Items of one IID block stay conditionally independent. A token attending
    # itself is harmless (its own value already enters via the residual) and
    # is kept for every latent pair the mode allows.
    diagonal = torch.eye(ri.shape[0], dtype=torch.bool)
    return edges & ~(same_iid & ~diagonal)


class LatentAttentionTransformer(TokenizedTransformer):
    """``TokenizedTransformer`` with a role-aware latent-to-latent attention rule.

    Instances are made by :func:`enable_latent_attention`, which switches the
    class of an existing ``trunk="shared"`` network in place. The parameters are
    untouched and no state-dict key is added.
    """

    def _attention_mask(self, token_mask):
        """Additive-free bool mask ``(batch*heads, tokens, tokens)``, True = blocked."""
        conditioned = token_mask.bool()                      # (B, T)
        latent = ~conditioned
        edges = self._latent_edges.to(token_mask.device)     # (T, T)
        allowed = conditioned.unsqueeze(1) | (
            latent.unsqueeze(2) & latent.unsqueeze(1) & edges.unsqueeze(0)
        )
        # Conditioned queries keep attending to conditioned keys only, exactly
        # as in production: their outputs carry no loss, but they feed later
        # layers, and a conditioned token reading latent values would let
        # information leak between latents through a detour.
        allowed = torch.where(
            conditioned.unsqueeze(2), conditioned.unsqueeze(1).expand_as(allowed),
            allowed,
        )
        blocked = ~allowed
        return blocked.repeat_interleave(self.num_heads, dim=0)

    def _block_forward(self, block, x, attn_mask, t, return_attn_weights=False):
        modulation = block.adaLN_modulation(t).reshape(-1, 1, 6 * block.hidden_size)
        if block.token_offset is not None:
            modulation = modulation + block.token_offset
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation.chunk(6, dim=-1)
        x_norm = modulate(block.norm1(x), shift_msa, scale_msa)
        q, k, v = x_norm.repeat(1, 1, 3).chunk(3, dim=-1)
        # add_bias_kv=True appends one always-visible bias key, so a query
        # whose every real key is blocked still has a valid softmax.
        attn_output, attn_weights = block.attn(
            q, k, v, need_weights=return_attn_weights, attn_mask=attn_mask,
        )
        x = x + gate_msa * attn_output
        x = x + gate_mlp * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
        if return_attn_weights:
            return x, attn_weights
        return x

    def forward(self, x, t, c, return_attn_weights=False):
        tokens = self.x_embedder(x)
        token_mask = self.schema.token_mask(c)
        t = self.t_embedder(t)
        attn_mask = self._attention_mask(token_mask)
        if return_attn_weights:
            weights = []
            for block in self.blocks:
                tokens, block_weights = self._block_forward(block, tokens, attn_mask, t, True)
                weights.append(block_weights.mean(0))
            return self.final_layer(tokens, t), torch.stack(weights, dim=0)
        for block in self.blocks:
            tokens = self._block_forward(block, tokens, attn_mask, t)
        return self.final_layer(tokens, t)


def resolve_roles(schema, roles):
    """``roles`` as ``{"g": [...], "l": [...], "x": [...]}``; a ``(global_size,
    local_size)`` pair is expanded with :func:`compass.hierarchical_roles`."""
    if isinstance(roles, dict):
        return {name: [int(i) for i in indices] for name, indices in roles.items()}
    if isinstance(roles, (tuple, list)) and len(roles) == 2 and all(
            isinstance(v, int) for v in roles):
        return {name: [int(i) for i in indices]
                for name, indices in hierarchical_roles(schema, *roles).items()}
    raise ValueError(
        "latent roles must be (global_size, local_size) or a dict of g/l/x node "
        f"indices; got {roles!r}."
    )


def enable_latent_attention(model, mode, roles):
    """Switch ``model.model`` (a shared-trunk ``TokenizedTransformer``) to ``mode``.

    ``model`` is a ``ScoreBasedInferenceModel``. ``roles`` maps ``g``, ``l`` and
    ``x`` to node indices (``compass.hierarchical_roles`` builds it), or is a
    ``(global_size, local_size)`` pair. Returns the model for chaining.
    """
    network = model.model
    if not isinstance(network, TokenizedTransformer) or network.trunk != "shared":
        raise TypeError("Latent attention needs a TokenizedTransformer with trunk='shared'.")
    roles = resolve_roles(network.schema, roles)
    token_role, iid_block = _token_roles(network.schema, roles)
    network.__class__ = LatentAttentionTransformer
    network._latent_edges = latent_edges(token_role, iid_block, mode)
    network.latent_attention = mode
    network.latent_roles = roles
    model.latent_attention = mode
    return model


def latent_attention_payload(model):
    network = model.model
    return {
        "mode": getattr(network, "latent_attention", "none"),
        "roles": getattr(network, "latent_roles", None),
    }


def save_latent_attention(model, directory):
    """Write the sidecar that :func:`load_latent_attention` needs next to a checkpoint."""
    path = Path(directory) / SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(latent_attention_payload(model), indent=2))
    return path


def load_latent_attention(model, directory, required=True):
    """Re-apply the latent attention rule recorded next to a checkpoint."""
    path = Path(directory) / SIDECAR
    if not path.exists():
        if required:
            raise FileNotFoundError(f"No {SIDECAR} in {directory}.")
        return model
    payload = json.loads(path.read_text())
    if payload["mode"] == "none" and payload["roles"] is None:
        return model
    return enable_latent_attention(model, payload["mode"], payload["roles"])
