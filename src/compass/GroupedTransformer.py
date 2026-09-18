"""Two-level score network for a :class:`~compass.schema.Group` schema.

``TokenizedTransformer`` reads one flat token sequence. A ``Group`` schema is
nested -- ``[globals, (l_1, x_1), (l_2, x_2), ...]`` -- and the point of the
nesting is to let the group-level parameters see several subjects at once
without flattening every subject's trials into one undifferentiated context.
This module is that architecture:

1. **within-subject**, weights shared across repeats: each repeat's own tokens
   (3 local scalars + 30 trial tokens for partial pooling) go through
   ``depth_within`` shared blocks and are then pooled to one vector ``h_j``;
2. **across-subject**: the global tokens and the ``K`` subject vectors --
   ``G + K`` tokens in all -- go through ``depth_across`` shared blocks. This is
   the only place the globals see more than one subject;
3. **readout**, weights shared across repeats: each updated ``h'_j`` is
   broadcast back onto repeat ``j``'s own tokens, optionally followed by
   ``depth_readout`` more shared blocks, and the existing shared heads emit the
   raw per-scalar score.

Attention cost is ``O(K * inner_tokens^2) + O((G + K)^2)`` rather than
``O((G + K * inner_tokens)^2)``: at K=20 for partial pooling that is ~22.5k
attention entries against ~445k. The outer term is still quadratic, but in ``K``
rather than in ``33K``.

**Symmetry.** Sharing the within and readout weights across repeats is what keeps
the inner ``IID`` guarantee intact: permuting trials inside a repeat permutes
exactly that repeat's tokens. Nothing here promises anything about permuting the
repeats, and ``slot="learned"`` is the default precisely so the implementation
does not accidentally promise more than the schema does -- without a slot term
the across-subject trunk would be permutation-equivariant whether or not anyone
wanted it.

Everything is built from ``TokenizedTransformer``'s own pieces -- the same
``SchemaEmbedder``, ``SchemaHead``, ``SharedTransformerBlock`` and
``TimestepEmbedder`` -- so a block here is the same block the 463,364-parameter
ladder winner uses. That file is imported, never modified.

The external contract is unchanged: flat ``(batch, nodes_size)`` in, flat
``(batch, nodes_size)`` out, so ``Sampler``, ``PFODE`` and ``MultiObsSampler``
need no changes.
"""

import math

import torch
import torch.nn as nn

from .ConditionTransformer import TimestepEmbedder, modulate
from .schema import Atomic, Group, NodeSchema
from .TokenizedTransformer import (
    SchemaEmbedder, SchemaHead, SharedTransformerBlock,
)

POOLINGS = Group.POOLINGS
SLOTS = Group.SLOTS


def _per_repeat(modulation, count):
    """Match a per-example modulation to the flattened ``batch * count`` rows.

    Callers are not consistent about the batch of ``t``: ``MultiObsSampler`` and
    the MAP routines pass a single diffusion time for a whole block of rows and
    rely on it broadcasting, exactly as it does in the flat trunk. A leading
    dimension of 1 must therefore be left alone; anything else is per example and
    has to be expanded once per repeat.
    """
    if modulation.shape[0] == 1:
        return modulation
    return modulation.repeat_interleave(count, dim=0)


def _split_schema(schema):
    """Validate the nested layout and return ``(globals_schema, index, group)``.

    Deliberately strict: exactly one ``Group``, last, with only atomic blocks
    before it. That is the partial-pooling layout and the only one the roles
    helpers, the benchmark adapter and this trunk are written for. A more
    permissive check would accept schemas that slice correctly and train wrongly.
    """
    groups = [i for i, block in enumerate(schema.blocks) if block.kind == Group.kind]
    if len(groups) != 1:
        raise ValueError(
            f"GroupedTransformer needs exactly one Group block, found {len(groups)}."
        )
    index = groups[0]
    if index != len(schema.blocks) - 1:
        raise ValueError(
            "The Group block must be last; the group-level parameters are "
            f"everything before it, but block {index} of {len(schema.blocks)} is "
            "the Group."
        )
    leading = schema.blocks[:index]
    if not leading:
        raise ValueError(
            "GroupedTransformer needs at least one block of group-level "
            "parameters before the Group block."
        )
    if any(block.kind != Atomic.kind for block in leading):
        raise ValueError(
            "The blocks before the Group must all be Atomic; got "
            f"{[block.kind for block in leading]}."
        )
    return NodeSchema(list(leading)), index, schema.blocks[index]


class SubjectPool(nn.Module):
    """``(rows, tokens, H)`` -> ``(rows, H)``: one vector per repeat.

    ``"mean"`` averages over the repeat's tokens, including the latent ones --
    a latent token still carries the current diffusion state, which is exactly
    what the group-level parameters need to see. ``"pma"`` replaces the average
    with a single learned query attending over the same tokens, under the same
    key-padding convention the trunk uses everywhere else.
    """

    def __init__(self, hidden_size, num_heads, pool="mean"):
        super().__init__()
        if pool not in POOLINGS:
            raise ValueError(f"Unknown pool {pool!r}; choose one of {POOLINGS}.")
        self.pool = pool
        self.hidden_size = hidden_size
        if pool == "pma":
            self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.attn = nn.MultiheadAttention(
                hidden_size, num_heads=num_heads, add_bias_kv=True, batch_first=True,
            )

    def forward(self, tokens, key_padding_mask):
        if self.pool == "mean":
            return tokens.mean(dim=-2)
        query = self.query.expand(tokens.shape[0], 1, self.hidden_size)
        pooled = self.attn(
            query, tokens, tokens, need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]
        return pooled[:, 0]


class GroupedFinalLayer(nn.Module):
    """Shared adaLN modulation, then one head for the globals and one per repeat.

    Two ``SchemaHead`` instances rather than one: the inner head is applied to
    every repeat, so its weights are shared across subjects, which is what keeps
    the repeats interchangeable at readout and the trials exchangeable inside
    one. A single head over the flattened schema could not express that.
    """

    def __init__(self, globals_schema, inner_schema, hidden_size,
                 time_embedding_size=256, per_position_head=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embedding_size, 2 * hidden_size, bias=True),
        )
        self.global_head = SchemaHead(globals_schema, hidden_size,
                                      per_position=per_position_head)
        self.inner_head = SchemaHead(inner_schema, hidden_size,
                                     per_position=per_position_head)

    def _apply_norm(self, tokens, shift, scale):
        return nn.functional.silu(modulate(self.norm_final(tokens), shift, scale))

    def forward(self, global_tokens, repeat_tokens, t, batch, count):
        shift, scale = self.adaLN_modulation(t).reshape(
            -1, 1, 2 * self.hidden_size
        ).chunk(2, dim=-1)
        globals_out = self.global_head(self._apply_norm(global_tokens, shift, scale))
        shift_r = _per_repeat(shift, count)
        scale_r = _per_repeat(scale, count)
        repeats_out = self.inner_head(self._apply_norm(repeat_tokens, shift_r, scale_r))
        return torch.cat(
            [globals_out, repeats_out.reshape(batch, -1)], dim=-1
        )


class GroupedTransformer(nn.Module):
    """Score network for a nested ``[globals, Group(repeats)]`` schema."""

    def __init__(self, schema, hidden_size=64, time_embedding_size=256,
                 depth_within=2, depth_across=2, depth_readout=0, num_heads=8,
                 mlp_ratio=4.0, per_position_head=False):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}."
            )
        if depth_within < 1 or depth_across < 1:
            raise ValueError(
                "depth_within and depth_across must both be >= 1: with either at "
                "zero the globals never see the subjects, or the subjects never "
                "see their own trials, which is the whole point of this trunk."
            )
        if depth_readout < 0:
            raise ValueError("depth_readout must be >= 0.")

        globals_schema, group_index, group = _split_schema(schema)
        self.schema = schema
        self.globals_schema = globals_schema
        self.inner_schema = group.inner
        self.group_index = group_index
        self.count = group.count
        self.pool_kind = group.pool
        self.slot_kind = group.slot

        self.nodes_size = schema.nodes_size
        self.token_count = schema.token_count
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.time_embedding_size = time_embedding_size
        self.depth_within = depth_within
        self.depth_across = depth_across
        self.depth_readout = depth_readout
        self.per_position_head = bool(per_position_head)

        self.global_tokens = globals_schema.token_count
        self.inner_tokens = group.inner.token_count
        self.inner_nodes = group.inner.nodes_size

        self.t_embedder = TimestepEmbedder(
            schema.token_count, hidden_size, mlp_ratio=mlp_ratio,
            frequency_embedding_size=time_embedding_size,
        )
        # One embedder per level; the inner one is shared across all repeats.
        self.global_embedder = SchemaEmbedder(globals_schema, hidden_size)
        self.inner_embedder = SchemaEmbedder(group.inner, hidden_size)

        def stack(depth, tokens):
            return nn.ModuleList([
                SharedTransformerBlock(
                    hidden_size, num_heads, tokens, mlp_ratio=mlp_ratio,
                    time_embedding_size=time_embedding_size,
                )
                for _ in range(depth)
            ])

        self.within_blocks = stack(depth_within, self.inner_tokens)
        self.across_blocks = stack(depth_across, self.global_tokens + self.count)
        self.readout_blocks = stack(depth_readout, self.inner_tokens)
        self.pool = SubjectPool(hidden_size, num_heads, pool=group.pool)

        if group.slot == "learned":
            self.slot_embedding = nn.Parameter(torch.zeros(self.count, hidden_size))
        elif group.slot == "sinusoidal":
            self.register_buffer(
                "slot_embedding",
                TimestepEmbedder.timestep_embedding(
                    torch.arange(self.count, dtype=torch.float32).unsqueeze(-1),
                    hidden_size,
                ),
                persistent=False,
            )
        else:
            self.slot_embedding = None

        self.final_layer = GroupedFinalLayer(
            globals_schema, group.inner, hidden_size,
            time_embedding_size=time_embedding_size,
            per_position_head=self.per_position_head,
        )
        self.initialize_weights()

    # ----- initialization ------------------------------------------------

    def initialize_weights(self):
        """Mirrors ``TokenizedTransformer.initialize_weights`` exactly.

        The blocks are the same blocks, so they must be initialized the same way
        or a Group arm would differ from the ladder winner for a reason that has
        nothing to do with the representation.
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # A constant token embedding normalizes to the same vector for every
        # token under LayerNorm(H); break it, exactly as the flat trunk does.
        for embedder, schema in (
            (self.global_embedder, self.globals_schema),
            (self.inner_embedder, self.inner_schema),
        ):
            for index, block in enumerate(schema.blocks):
                if block.kind == Atomic.kind:
                    nn.init.normal_(embedder.scales[index], std=1.0)
                    nn.init.normal_(embedder.offsets[index], std=1.0)
                else:
                    nn.init.normal_(embedder.item_embedders[index].bias, std=1.0)

        for head in (self.final_layer.global_head, self.final_layer.inner_head):
            if not head.per_position:
                continue
            # Match the shared nn.Linear(hidden_size, 1) this replaces: xavier on
            # a (1, H) weight is U(-b, b) with b = sqrt(6 / (H + 1)). Zeros would
            # make the whole atomic score identically zero at step 0.
            bound = math.sqrt(6.0 / (head.hidden_size + 1))
            for slot in head.position_index.values():
                nn.init.uniform_(head.position_weights[slot], -bound, bound)
                nn.init.constant_(head.position_biases[slot], 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        if self.slot_kind == "learned":
            # Not zero: a zero slot table would leave the across-subject trunk
            # permutation-equivariant at initialization, i.e. silently asserting
            # a subject symmetry the schema explicitly declines to promise.
            nn.init.normal_(self.slot_embedding, std=0.02)
        if self.pool_kind == "pma":
            nn.init.normal_(self.pool.query, std=0.02)

    @property
    def blocks(self):
        """Every transformer block, in depth order. Used by the initializer and
        by ``parameter_breakdown``."""
        return list(self.within_blocks) + list(self.across_blocks) + \
            list(self.readout_blocks)

    # ----- forward -------------------------------------------------------

    def _subject_key_mask(self, inner_mask):
        """A subject token is an attention key if *any* of its tokens is one.

        ``min`` (the within-repeat convention, where a grouped token is a key
        only if every scalar it carries is conditioned) would be wrong here and
        catastrophically so: under the deployed posterior mask every subject has
        latent locals, so ``min`` is zero for every subject and the group-level
        parameters would attend to nothing at all.
        """
        return inner_mask.max(dim=-1).values

    def forward(self, x, t, c, return_attn_weights=False):
        """Flat ``(batch, nodes_size)`` in, flat ``(batch, nodes_size)`` out."""
        batch = x.shape[0]
        count, rows = self.count, batch * self.count

        token_mask = self.schema.token_mask(c)
        global_mask = token_mask[..., :self.global_tokens]
        inner_mask = token_mask[..., self.global_tokens:].reshape(
            rows, self.inner_tokens
        )

        global_nodes = self.globals_schema.nodes_size
        global_tokens = self.global_embedder(x[..., :global_nodes])
        repeats = x[..., global_nodes:].reshape(rows, self.inner_nodes)
        repeat_tokens = self.inner_embedder(repeats)

        t = self.t_embedder(t)
        t_repeat = _per_repeat(t, count)

        for block in self.within_blocks:
            repeat_tokens = block(repeat_tokens, inner_mask, t_repeat, False)

        pooled = self.pool(repeat_tokens, (1 - inner_mask).type(torch.bool))
        subject_tokens = pooled.reshape(batch, count, self.hidden_size)
        if self.slot_embedding is not None:
            subject_tokens = subject_tokens + self.slot_embedding

        outer = torch.cat([global_tokens, subject_tokens], dim=-2)
        outer_mask = torch.cat(
            [global_mask, self._subject_key_mask(inner_mask).reshape(batch, count)],
            dim=-1,
        )

        attn = []
        for block in self.across_blocks:
            if return_attn_weights:
                outer, weights = block(outer, outer_mask, t, True)
                attn.append(weights.mean(0))
            else:
                outer = block(outer, outer_mask, t, False)

        global_tokens = outer[..., :self.global_tokens, :]
        subject_tokens = outer[..., self.global_tokens:, :]

        # Broadcast each updated subject vector back onto its own tokens. An
        # additive broadcast keeps the readout parameter-free; the alternative
        # (concatenate and project) costs 2H^2 and is not obviously better.
        repeat_tokens = repeat_tokens + subject_tokens.reshape(rows, 1, self.hidden_size)
        for block in self.readout_blocks:
            repeat_tokens = block(repeat_tokens, inner_mask, t_repeat, False)

        out = self.final_layer(global_tokens, repeat_tokens, t, batch, count)
        if return_attn_weights:
            return out, torch.stack(attn, dim=0)
        return out


def grouped_parameter_breakdown(model):
    """Split a grouped trunk's parameters into reportable categories."""
    total = sum(p.numel() for p in model.parameters())

    def over(modules):
        return sum(p.numel() for module in modules for p in module.parameters())

    modulation = sum(
        p.numel() for block in model.blocks for p in block.adaLN_modulation.parameters()
    )
    breakdown = {
        "total": total,
        "within": over(model.within_blocks),
        "across": over(model.across_blocks),
        "readout": over(model.readout_blocks),
        "block_adaLN": modulation,
        "pool": over([model.pool]),
        "slot": (
            model.slot_embedding.numel()
            if isinstance(getattr(model, "slot_embedding", None), nn.Parameter) else 0
        ),
        "final_layer": over([model.final_layer]),
        "input_embedding": over([model.global_embedder, model.inner_embedder]),
        "time_embedding": over([model.t_embedder]),
    }
    accounted = (
        breakdown["within"] + breakdown["across"] + breakdown["readout"]
        + breakdown["pool"] + breakdown["slot"] + breakdown["final_layer"]
        + breakdown["input_embedding"] + breakdown["time_embedding"]
    )
    breakdown["unaccounted"] = total - accounted
    return breakdown
