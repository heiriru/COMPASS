"""Schema-driven score network with a token-count-agnostic trunk.

The legacy :class:`~compass.ConditionTransformer.ConditionTransformer` spends
almost all of its parameters on quantities that are replicated per node: the
per-node adaLN modulation ``Linear(256, 6*N*H)`` in every block and the
``(N*H, N)`` matmul in the final layer. At the partial-pooling layout
(``nodes_size=100``) that is 99.3% of a 5,957,984-parameter model, and it is a
function of the diffusion time alone.

This module keeps the same external contract -- ``forward(x, t, c)`` maps a flat
``(batch, nodes_size)`` diffusion state to a flat ``(batch, nodes_size)`` output,
with ``key_padding_mask = 1 - c`` -- and offers two trunks:

``trunk="position"``
    Byte-for-byte the legacy modules, driven by a :class:`NodeSchema` instead of
    a bare ``nodes_size``. Requires an all-atomic schema. Its state dict is
    key-compatible with a legacy checkpoint, which is what makes the Stage-1
    "same model, new API" equivalence testable rather than asserted.

``trunk="shared"``
    ``LayerNorm(H)`` instead of ``LayerNorm((N, H))``, one ``Linear(256, 6H)``
    modulation shared by every token, and a token-wise output head instead of
    the flattened matmul. The parameter count no longer depends on the number of
    tokens.

Tokenization itself is the schema's business: an ``Atomic`` block contributes one
token per scalar, an ``IID`` block one token per item through a single shared
item embedder with no positional embedding.
"""

import math
import torch
import torch.nn as nn

from .ConditionTransformer import (
    FinalLayer, InputEmbedder, Mlp, TimestepEmbedder, TransformerBlock, modulate,
)
from .schema import Atomic, NodeSchema

TRUNKS = ("shared", "position")


#############################################
# ----- Schema-driven input embedding -----
#############################################

class SchemaEmbedder(nn.Module):
    """Flat ``(batch, nodes_size)`` -> ``(batch, tokens, hidden_size)``.

    Atomic blocks get a per-position *affine* embedding ``x_n * w_n + b_n``. The
    affine part is not decoration: under ``LayerNorm(H)`` a purely
    multiplicative embedding ``x_n * w_n`` normalizes to ``sign(x_n)`` times a
    fixed direction, which would discard every token's magnitude. The legacy
    trunk could get away with the multiplicative form only because its LayerNorm
    ran over ``(N, H)`` jointly.

    IID blocks get one shared ``Linear(item_dim, hidden_size)``, applied
    identically to every item and with no positional embedding, so permuting
    items permutes their tokens.
    """

    def __init__(self, schema, hidden_size):
        super().__init__()
        self.schema = schema
        self.hidden_size = hidden_size
        self.scales = nn.ParameterList()
        self.offsets = nn.ParameterList()
        self.item_embedders = nn.ModuleList()
        for block in schema.blocks:
            if block.kind == Atomic.kind:
                self.scales.append(nn.Parameter(torch.empty(block.dim, hidden_size)))
                self.offsets.append(nn.Parameter(torch.empty(block.dim, hidden_size)))
                self.item_embedders.append(nn.Identity())
            else:
                self.scales.append(nn.Parameter(torch.empty(0)))
                self.offsets.append(nn.Parameter(torch.empty(0)))
                self.item_embedders.append(nn.Linear(block.item_dim, hidden_size))

    def forward(self, x):
        tokens = []
        for index, block in enumerate(self.schema.blocks):
            view = self.schema.block_view(x, index)
            if block.kind == Atomic.kind:
                tokens.append(view * self.scales[index] + self.offsets[index])
            else:
                tokens.append(self.item_embedders[index](view))
        return torch.cat(tokens, dim=-2)


class SchemaHead(nn.Module):
    """``(batch, tokens, hidden_size)`` -> flat ``(batch, nodes_size)``.

    One shared head per block: a scalar head for an atomic block, a shared
    ``hidden_size -> item_dim`` head for an IID block, so the output stays the
    raw per-item variables and the model keeps representing ``p(theta, X)``
    rather than a summary.

    ``per_position=True`` gives every scalar of an *atomic* block its own
    readout ``w_n . h_n + b_n`` instead of one ``w`` shared across the block.
    With a single shared head the score for every parameter is read off the same
    direction in hidden space, so parameters that need different readouts --
    a group mean and a group log-scale, say -- compete for one vector; the
    symptom is posteriors that are individually too wide. The cost is
    ``dim * hidden_size`` per atomic block (650 parameters for the 10 theta
    dimensions at ``hidden_size=64``), and IID blocks are deliberately left
    shared because a per-item head would destroy exchangeability.
    """

    def __init__(self, schema, hidden_size, per_position=False):
        super().__init__()
        self.schema = schema
        self.hidden_size = hidden_size
        self.per_position = bool(per_position)
        # Only the parameters this mode actually uses are registered. Padding
        # the unused mode with empty placeholders would add state-dict keys and
        # make every checkpoint written before this option unloadable.
        self.heads = nn.ModuleList()
        self.position_weights = nn.ParameterList()
        self.position_biases = nn.ParameterList()
        self.position_index = {}
        for index, block in enumerate(schema.blocks):
            if self.per_position and block.kind == Atomic.kind:
                self.heads.append(nn.Identity())
                self.position_index[index] = len(self.position_weights)
                self.position_weights.append(
                    nn.Parameter(torch.zeros(block.dim, hidden_size))
                )
                self.position_biases.append(nn.Parameter(torch.zeros(block.dim)))
            else:
                self.heads.append(nn.Linear(
                    hidden_size,
                    1 if block.kind == Atomic.kind else block.item_dim,
                ))

    def weight_for(self, block_index):
        """The per-position readout matrix of an atomic block, or None."""
        slot = self.position_index.get(block_index)
        return None if slot is None else self.position_weights[slot]

    def forward(self, tokens):
        outputs = []
        for index, block in enumerate(self.schema.blocks):
            start, stop = self.schema.token_slices[index]
            view = tokens[..., start:stop, :]
            slot = self.position_index.get(index)
            if slot is not None:
                chunk = (
                    (view * self.position_weights[slot]).sum(dim=-1)
                    + self.position_biases[slot]
                )
                outputs.append(chunk)
                continue
            chunk = self.heads[index](view)
            outputs.append(chunk.reshape(*chunk.shape[:-2], block.width))
        return torch.cat(outputs, dim=-1)


#############################################
# ----- Token-count-agnostic block -----
#############################################

class SharedTransformerBlock(nn.Module):
    """adaLN-Zero block whose modulation is shared across tokens."""

    def __init__(self, hidden_size, num_heads, token_count, mlp_ratio=4.0,
                 time_embedding_size=256, token_offset=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_count = token_count
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, add_bias_kv=True,
            batch_first=True, **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embedding_size, 6 * hidden_size, bias=True),
        )
        # Stage-2 ablation: a small per-token offset on the shared modulation.
        # Off by default; it reintroduces token-count-dependent parameters.
        self.token_offset = (
            nn.Parameter(torch.zeros(token_count, 6 * hidden_size))
            if token_offset else None
        )

    def forward(self, x, c, t, return_attn_weights=False):
        modulation = self.adaLN_modulation(t).reshape(-1, 1, 6 * self.hidden_size)
        if self.token_offset is not None:
            modulation = modulation + self.token_offset
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation.chunk(6, dim=-1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = x_norm.repeat(1, 1, 3).chunk(3, dim=-1)

        # Same convention as the legacy trunk: only conditioned tokens are keys.
        key_padding_mask = (1 - c).type(torch.bool)

        if return_attn_weights:
            attn_output, attn_weights = self.attn(
                q, k, v, need_weights=True, key_padding_mask=key_padding_mask,
            )
        else:
            attn_output = self.attn(
                q, k, v, need_weights=False, key_padding_mask=key_padding_mask,
            )[0]

        x = x + gate_msa * attn_output
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        if return_attn_weights:
            return x, attn_weights
        return x


class SharedFinalLayer(nn.Module):
    """Token-wise output head with a shared adaLN modulation."""

    def __init__(self, schema, hidden_size, time_embedding_size=256,
                 per_position_head=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embedding_size, 2 * hidden_size, bias=True),
        )
        self.head = SchemaHead(schema, hidden_size, per_position=per_position_head)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).reshape(
            -1, 1, 2 * self.hidden_size
        ).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = nn.functional.silu(x)
        return self.head(x)


#############################################
# ----- The network -----
#############################################

class TokenizedTransformer(nn.Module):
    """Score network driven by a :class:`NodeSchema`."""

    def __init__(self, schema, hidden_size=64, time_embedding_size=256, depth=6,
                 num_heads=16, mlp_ratio=4.0, trunk="shared", token_offset=False,
                 per_position_head=False):
        super().__init__()
        if trunk not in TRUNKS:
            raise ValueError(f"Unknown trunk {trunk!r}; choose one of {TRUNKS}.")
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}."
            )
        if trunk == "position" and not schema.is_atomic:
            raise ValueError(
                "trunk='position' reproduces the legacy per-node parameterization "
                "and is only defined for an all-atomic schema; this schema has "
                f"{schema.token_count} tokens for {schema.nodes_size} nodes."
            )
        if trunk == "position" and token_offset:
            raise ValueError("token_offset is a trunk='shared' ablation.")
        if trunk == "position" and per_position_head:
            raise ValueError(
                "per_position_head is a trunk='shared' option; the position "
                "trunk is already per-node throughout."
            )
        if trunk == "position" and time_embedding_size != 256:
            raise ValueError(
                "trunk='position' reproduces the legacy network bit-for-bit "
                "and is fixed at time_embedding_size=256."
            )

        self.schema = schema
        self.nodes_size = schema.nodes_size
        self.token_count = schema.token_count
        self.num_heads = num_heads
        self.time_embedding_size = time_embedding_size
        self.trunk = trunk
        self.token_offset = bool(token_offset)
        self.per_position_head = bool(per_position_head)

        self.t_embedder = TimestepEmbedder(
            schema.token_count, hidden_size, mlp_ratio=mlp_ratio,
            frequency_embedding_size=time_embedding_size,
        )
        if trunk == "position":
            self.x_embedder = InputEmbedder(
                nodes_size=schema.nodes_size, hidden_size=hidden_size,
            )
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_size, num_heads, schema.nodes_size,
                                 mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ])
            self.final_layer = FinalLayer(hidden_size, schema.nodes_size)
        else:
            self.x_embedder = SchemaEmbedder(schema, hidden_size)
            self.blocks = nn.ModuleList([
                SharedTransformerBlock(hidden_size, num_heads, schema.token_count,
                                       mlp_ratio=mlp_ratio,
                                       time_embedding_size=time_embedding_size,
                                       token_offset=token_offset)
                for _ in range(depth)
            ])
            self.final_layer = SharedFinalLayer(
                schema, hidden_size, time_embedding_size=time_embedding_size,
                per_position_head=self.per_position_head,
            )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.trunk == "shared":
            # The token embedding must not be degenerate under LayerNorm(H): a
            # constant direction (the legacy all-ones initialization) normalizes
            # to the same vector for every token and every value.
            for index, block in enumerate(self.schema.blocks):
                if block.kind == Atomic.kind:
                    nn.init.normal_(self.x_embedder.scales[index], std=1.0)
                    nn.init.normal_(self.x_embedder.offsets[index], std=1.0)
                else:
                    nn.init.normal_(self.x_embedder.item_embedders[index].bias, std=1.0)

        if self.trunk == "shared" and self.final_layer.head.per_position:
            # Match the shared nn.Linear(hidden_size, 1) head this replaces:
            # xavier_uniform on a (1, H) weight is U(-b, b) with
            # b = sqrt(6 / (H + 1)). Zero-initializing instead would make the
            # whole atomic score identically zero at step 0.
            head = self.final_layer.head
            bound = math.sqrt(6.0 / (head.hidden_size + 1))
            for slot in head.position_index.values():
                nn.init.uniform_(head.position_weights[slot], -bound, bound)
                nn.init.constant_(head.position_biases[slot], 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, c, return_attn_weights=False):
        """Flat ``(batch, nodes_size)`` in, flat ``(batch, nodes_size)`` out."""
        if self.trunk == "position":
            tokens = self.x_embedder(x)
            token_mask = c
        else:
            tokens = self.x_embedder(x)
            token_mask = self.schema.token_mask(c)
        t = self.t_embedder(t)

        if return_attn_weights:
            all_attn_weights = []
            for block in self.blocks:
                tokens, attn_weights = block(tokens, token_mask, t, True)
                all_attn_weights.append(attn_weights.mean(0))
            out = self.final_layer(tokens, t)
            return out, torch.stack(all_attn_weights, dim=0)

        for block in self.blocks:
            tokens = block(tokens, token_mask, t, False)
        out = self.final_layer(tokens, t)
        if self.trunk == "position":
            return out
        return out


def parameter_breakdown(model):
    """Split a trunk's parameters into the categories the ladder reports on."""
    total = sum(p.numel() for p in model.parameters())
    modulation = sum(
        p.numel() for block in model.blocks for p in block.adaLN_modulation.parameters()
    )
    offsets = sum(
        block.token_offset.numel() for block in model.blocks
        if getattr(block, "token_offset", None) is not None
    )
    final = sum(p.numel() for p in model.final_layer.parameters())
    embedding = sum(p.numel() for p in model.x_embedder.parameters())
    time = sum(p.numel() for p in model.t_embedder.parameters())
    return {
        "total": total,
        "block_adaLN": modulation,
        "token_offsets": offsets,
        "final_layer": final,
        "input_embedding": embedding,
        "time_embedding": time,
        "attention_and_mlp": total - modulation - offsets - final - embedding - time,
    }
