"""Inference-only JAX port of the trial-token TokenizedTransformer (shared trunk).

Covers exactly the configuration of the H64/d2 checkpoint: schema
``[Atomic("theta", 10), IID("trials", item_dim=3, count=30)]``, shared trunk,
no token offset, shared (not per-position) head, time_embedding_size 256. It
reads the PyTorch ``state_dict`` written by ``export_reference.py``, so there is
no training and no second source of weights.

Mirrors ``compass.TokenizedTransformer`` line by line; in particular
``nn.MultiheadAttention(add_bias_kv=True)`` appends the learned key/value
*after* the input projection, and the key-padding mask ``1 - c`` makes only
conditioned (observed) tokens -- plus that bias slot -- attendable.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from common import FD_STEP, ITEM_DIM, NODES, THETA_DIM, TRIALS

HIDDEN, HEADS, DEPTH = 64, 8, 2
FREQUENCY_SIZE = 256


def load_params(path, dtype=jnp.float32):
    raw = np.load(path)
    return {key: jnp.asarray(raw[key], dtype=dtype) for key in raw.files
            if raw[key].size}


def _linear(params, prefix, x):
    return x @ params[f"{prefix}.weight"].T + params[f"{prefix}.bias"]


def _layer_norm(x, eps=1e-6):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


def time_embedding(params, t):
    half = FREQUENCY_SIZE // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    args = t * freqs.astype(params["t_embedder.mlp.0.weight"].dtype)
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)])
    hidden = jax.nn.silu(_linear(params, "t_embedder.mlp.0", embedding))
    return _linear(params, "t_embedder.mlp.2", hidden)


def _attention(params, prefix, x, keep):
    """``x``: (B, T, H); ``keep``: (B, T) True where a token may be a key."""
    batch, tokens, _ = x.shape
    weight, bias = params[f"{prefix}.in_proj_weight"], params[f"{prefix}.in_proj_bias"]
    q = x @ weight[:HIDDEN].T + bias[:HIDDEN]
    k = x @ weight[HIDDEN:2 * HIDDEN].T + bias[HIDDEN:2 * HIDDEN]
    v = x @ weight[2 * HIDDEN:].T + bias[2 * HIDDEN:]
    bias_k = jnp.broadcast_to(params[f"{prefix}.bias_k"][0], (batch, 1, HIDDEN))
    bias_v = jnp.broadcast_to(params[f"{prefix}.bias_v"][0], (batch, 1, HIDDEN))
    k = jnp.concatenate([k, bias_k], axis=1)
    v = jnp.concatenate([v, bias_v], axis=1)
    keep = jnp.concatenate([keep, jnp.ones((batch, 1), bool)], axis=1)

    head_dim = HIDDEN // HEADS
    split = lambda a: a.reshape(batch, -1, HEADS, head_dim)
    logits = jnp.einsum("bqhd,bkhd->bhqk", split(q), split(k)) / math.sqrt(head_dim)
    logits = jnp.where(keep[:, None, None, :], logits, -jnp.inf)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, split(v)).reshape(batch, tokens, HIDDEN)
    return _linear(params, f"{prefix}.out_proj", out)


def _block(params, index, x, keep, t_emb):
    prefix = f"blocks.{index}"
    modulation = _linear(params, f"{prefix}.adaLN_modulation.1", jax.nn.silu(t_emb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6)
    x_norm = _modulate(_layer_norm(x), shift_msa, scale_msa)
    x = x + gate_msa * _attention(params, f"{prefix}.attn", x_norm, keep)
    hidden = _modulate(_layer_norm(x), shift_mlp, scale_mlp)
    hidden = jax.nn.gelu(_linear(params, f"{prefix}.mlp.fc1", hidden), approximate=True)
    return x + gate_mlp * _linear(params, f"{prefix}.mlp.fc2", hidden)


def network(params, x, t, c):
    """Flat ``(B, 100)`` state, scalar ``t``, flat ``(B, 100)`` mask -> ``(B, 100)``."""
    batch = x.shape[0]
    theta_tokens = (x[:, :THETA_DIM, None] * params["x_embedder.scales.0"]
                    + params["x_embedder.offsets.0"])
    trials = x[:, THETA_DIM:].reshape(batch, TRIALS, ITEM_DIM)
    trial_tokens = _linear(params, "x_embedder.item_embedders.1", trials)
    tokens = jnp.concatenate([theta_tokens, trial_tokens], axis=1)

    trial_mask = c[:, THETA_DIM:].reshape(batch, TRIALS, ITEM_DIM).min(-1)
    keep = jnp.concatenate([c[:, :THETA_DIM], trial_mask], axis=1) > 0.5

    t_emb = time_embedding(params, t)
    for index in range(DEPTH):
        tokens = _block(params, index, tokens, keep, t_emb)

    shift, scale = jnp.split(
        _linear(params, "final_layer.adaLN_modulation.1", jax.nn.silu(t_emb)), 2)
    tokens = jax.nn.silu(_modulate(_layer_norm(tokens), shift, scale))
    theta_out = _linear(params, "final_layer.head.heads.0", tokens[:, :THETA_DIM])[..., 0]
    trial_out = _linear(params, "final_layer.head.heads.1", tokens[:, THETA_DIM:])
    return jnp.concatenate([theta_out, trial_out.reshape(batch, NODES - THETA_DIM)], axis=1)


def make_score(params, t, scale, c):
    """``z -> model(z, t, c) / std(t)``, the raw per-row score (VESDE, alpha=1)."""
    def score(z):
        return network(params, z, t, c) / scale
    score.lam = scale          # VESDE: lambda_t = std(t)
    return score


# ----- Jacobian variants: z (B, 100) -> (B, 10, 10) ------------------------

def _basis(z):
    eye = jnp.eye(THETA_DIM, NODES, dtype=z.dtype)            # (10, 100)
    return jnp.broadcast_to(eye[:, None, :], (THETA_DIM, *z.shape))


def jacfwd_vmap(score, z):
    """Exact Jacobian: 10 JVPs batched by vmap (what jax.jacfwd does)."""
    columns = jax.vmap(lambda v: jax.jvp(score, (z,), (v,))[1])(_basis(z))
    return jnp.transpose(columns[..., :THETA_DIM], (1, 2, 0))


def jvp_scan(score, z):
    """Exact Jacobian: 10 JVPs one after another inside the compiled program."""
    def step(_, v):
        return None, jax.jvp(score, (z,), (v,))[1][:, :THETA_DIM]
    _, columns = jax.lax.scan(step, None, _basis(z))
    return jnp.transpose(columns, (1, 2, 0))


def diag_jvp(score, z):
    """One JVP with every theta coordinate moved: the Jacobian's row sums."""
    tangent = jnp.zeros_like(z).at[:, :THETA_DIM].set(1.0)
    column = jax.jvp(score, (z,), (tangent,))[1][:, :THETA_DIM]
    return jax.vmap(jnp.diag)(column)


def fd(score, z):
    """Forward differences, base point and 10 shifted copies in one call."""
    step = FD_STEP * score.lam
    copies = jnp.concatenate([z[None], z[None] + step * _basis(z)])
    scores = jax.vmap(score)(copies)[..., :THETA_DIM]
    return jnp.transpose((scores[1:] - scores[:1]) / step, (1, 2, 0))


def diag_fd(score, z):
    """One shifted copy moving every theta coordinate: FD row sums."""
    step = FD_STEP * score.lam
    shifted = z.at[:, :THETA_DIM].add(step)
    scores = jax.vmap(score)(jnp.stack([z, shifted]))[..., :THETA_DIM]
    return jax.vmap(jnp.diag)((scores[1] - scores[0]) / step)


VARIANTS = {"jacfwd_vmap": jacfwd_vmap, "jvp_scan": jvp_scan, "diag_jvp": diag_jvp,
            "fd": fd, "diag_fd": diag_fd}
