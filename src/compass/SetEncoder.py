"""Permutation-invariant set encoder: ``(batch, set, item_dim) -> (batch, summary_dim)``.

A PyTorch reimplementation of the architecture the reference implementation uses
as its summary network (`bf.networks.SetTransformer`, Lee et al. 2019), adapted
rather than copied -- bayesflow is Keras and COMPASS is torch, so nothing can be
transferred literally.

Reference defaults, reproduced here: two set-attention blocks at width 64 with 4
heads, pooling by multi-head attention with 4 learned seeds, then a linear
projection to ``summary_dim=16``. Pre-norm residual blocks, SwiGLU feed-forward
with the 2/3 width correction, no biases in the projections.

Three deliberate departures from the reference, each for a reason COMPASS cares
about:

* **RMSNorm, not LayerNorm**, matching the reference's `layer_norm=True` path,
  which is RMS in that codebase. Kept because it is what the published numbers
  were produced with.
* **An optional set-size feature.** The reference encodes a fixed 30 trials, so
  it never needs to know the count. COMPASS is meant to survive a varying trial
  count, and a permutation-invariant summary that cannot see ``n`` cannot
  express how sharply the local parameters are identified. ``include_log_count``
  appends ``log n`` to the pooled vector before the output projection.
* **Masking support.** Ragged sets need a key-padding mask; the reference has no
  use for one.

The encoder is deliberately free of any COMPASS-specific coupling: it maps a set
of items to a fixed vector and knows nothing about schemas, diffusion or
condition masks. That is what makes it reusable as the single frozen encoder
shared across simulation models.
"""

import math

import torch
import torch.nn as nn

GLU_VARIANTS = ("swiglu", "geglu")


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, as the reference's ``layer_norm=True`` uses."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class GLUFeedForward(nn.Module):
    """Gated feed-forward with the 2/3 width correction the reference applies.

    A GLU layer uses two projections where a plain FFN uses one, so the hidden
    width is scaled by 2/3 to keep the parameter count comparable to a
    non-gated FFN at the same expansion factor.
    """

    def __init__(self, width, expansion_factor=4.0, variant="swiglu",
                 dropout=0.05, bias=False):
        super().__init__()
        if variant not in GLU_VARIANTS:
            raise ValueError(f"Unknown glu variant {variant!r}; choose from {GLU_VARIANTS}.")
        hidden = int(2 * expansion_factor * width / 3)
        self.gate = nn.Linear(width, hidden, bias=bias)
        self.value = nn.Linear(width, hidden, bias=bias)
        self.out = nn.Linear(hidden, width, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU() if variant == "swiglu" else nn.GELU()

    def forward(self, x):
        return self.dropout(self.out(self.activation(self.gate(x)) * self.value(x)))


class AttentionBlock(nn.Module):
    """Pre-norm multi-head attention followed by a pre-norm gated feed-forward.

    Serves as both the set-attention block (self-attention, ``kv is None``) and
    the pooling block's cross-attention (seeds attending to the set).
    """

    def __init__(self, width, num_heads, expansion_factor=4.0, variant="swiglu",
                 dropout=0.05, bias=False):
        super().__init__()
        self.norm_q = RMSNorm(width)
        self.norm_kv = RMSNorm(width)
        self.attention = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, bias=bias, batch_first=True,
        )
        self.norm_ffn = RMSNorm(width)
        self.ffn = GLUFeedForward(width, expansion_factor, variant, dropout, bias)

    def forward(self, x, kv=None, key_padding_mask=None):
        query = self.norm_q(x)
        keys = query if kv is None else self.norm_kv(kv)
        attended = self.attention(
            query, keys, keys, need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]
        x = x + attended
        return x + self.ffn(self.norm_ffn(x))


class SetEncoder(nn.Module):
    """Set transformer encoder producing one fixed-width summary per set."""

    def __init__(self, item_dim, summary_dim=16, width=64, depth=2, num_heads=4,
                 num_seeds=4, dropout=0.05, expansion_factor=4.0,
                 glu_variant="swiglu", include_log_count=True):
        super().__init__()
        self.item_dim = item_dim
        self.summary_dim = summary_dim
        self.width = width
        self.num_seeds = num_seeds
        self.include_log_count = bool(include_log_count)

        self.embed = nn.Linear(item_dim, width)
        block = lambda: AttentionBlock(width, num_heads, expansion_factor,
                                       glu_variant, dropout)
        self.blocks = nn.ModuleList([block() for _ in range(depth)])
        # rFF before pooling, as in the paper's PMA.
        self.pre_pool = GLUFeedForward(width, expansion_factor, glu_variant, dropout)
        self.norm_pre_pool = RMSNorm(width)
        self.seeds = nn.Parameter(torch.empty(num_seeds, width))
        nn.init.xavier_uniform_(self.seeds)
        self.pool = AttentionBlock(width, num_heads, expansion_factor,
                                   glu_variant, dropout)
        # The pooled vector is normalized before projection. Without this the
        # summary has no scale anchor at all: a predictive objective is happy to
        # grow |z| without bound (measured: |z| reaching 7e4 after 11 epochs),
        # which is numerically unhealthy even though the downstream
        # z-normalization would rescale it.
        self.norm_pooled = RMSNorm(num_seeds * width)
        self.project = nn.Linear(
            num_seeds * width + (1 if self.include_log_count else 0), summary_dim,
        )

    def forward(self, items, key_padding_mask=None):
        """``(batch, set, item_dim)`` -> ``(batch, summary_dim)``.

        ``key_padding_mask`` is True where an item is padding, matching torch's
        convention and COMPASS's ``key_padding_mask = 1 - c``.
        """
        if items.shape[-1] != self.item_dim:
            raise ValueError(
                f"SetEncoder({self.item_dim}) got items of width {items.shape[-1]}."
            )
        x = self.embed(items)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.norm_pre_pool(x)
        x = x + self.pre_pool(x)
        seeds = self.seeds.unsqueeze(0).expand(x.shape[0], -1, -1)
        pooled = self.pool(seeds, kv=x, key_padding_mask=key_padding_mask)
        flat = self.norm_pooled(pooled.reshape(pooled.shape[0], -1))
        if self.include_log_count:
            if key_padding_mask is None:
                count = torch.full((x.shape[0], 1), float(x.shape[1]), device=x.device)
            else:
                count = (~key_padding_mask).sum(dim=-1, keepdim=True).float()
            flat = torch.cat([flat, torch.log(count.clamp_min(1.0))], dim=-1)
        return self.project(flat)
