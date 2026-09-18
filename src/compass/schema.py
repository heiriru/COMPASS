"""Node schemas: how a flat (theta, x) vector is grouped into transformer tokens.

The diffusion state stays what it has always been -- one flat vector of
``nodes_size`` real numbers -- so every sampler, the PF-ODE and the
multi-observation compositional machinery are untouched. A schema only says how
that flat vector is *read* by the score network:

* :class:`Atomic` -- ``dim`` independent scalars, one token each. This is what
  COMPASS has always done, and a schema made of a single ``Atomic`` block
  reproduces the legacy tokenization exactly.
* :class:`IID` -- ``count`` exchangeable items of ``item_dim`` scalars each,
  laid out item-major in the flat vector, embedded by one shared item embedder
  into ``count`` tokens with no positional embedding.
* :class:`Group` -- ``count`` *structural* repeats of an inner block list, laid
  out repeat-major. This is nesting, not exchangeability: repeat ``j`` owns its
  sub-blocks and that association is never flattened away, but nothing about
  permuting the repeats themselves is promised. It is what lets one COMPASS
  factor hold several subjects at once, each with its own locals and its own
  exchangeable trials.

A schema also owns two things that follow from the grouping and must not be
re-derived elsewhere:

* :meth:`NodeSchema.token_mask` -- the attention key-padding convention. COMPASS
  uses ``key_padding_mask = 1 - c``: only *conditioned* tokens are keys. A
  grouped token is conditioned only if every scalar it carries is conditioned.
* :class:`GroupedBernoulli` -- the training-time condition mask. For an IID
  block the Bernoulli draw is per item, not per scalar, so an item is observed
  or latent as a whole. The marginal rate of observed *scalars* is unchanged,
  which is what makes a grouped model comparable to the legacy one.
"""

import torch


class Atomic:
    """``dim`` scalar nodes, one token per scalar."""

    kind = "atomic"

    def __init__(self, name, dim):
        dim = int(dim)
        if dim < 1:
            raise ValueError(f"Atomic({name!r}) needs dim >= 1, got {dim}.")
        self.name = str(name)
        self.dim = dim

    @property
    def width(self):
        """Scalars this block occupies in the flat node vector."""
        return self.dim

    @property
    def tokens(self):
        """Transformer tokens this block contributes."""
        return self.dim

    @property
    def item_dim(self):
        return 1

    def to_dict(self):
        return {"type": self.kind, "name": self.name, "dim": self.dim}

    def __repr__(self):
        return f"Atomic({self.name!r}, dim={self.dim})"

    def __eq__(self, other):
        return isinstance(other, Atomic) and self.to_dict() == other.to_dict()


class IID:
    """``count`` exchangeable items of ``item_dim`` scalars, item-major.

    The flat layout is ``[item_0 (item_dim scalars), item_1, ...]`` -- exactly
    the layout ``Partial_Pooling/schema.py:transformed_observations`` already
    produces for trial-major ``(choice, log_rt, censored)`` features, so no data
    is reshaped anywhere to use this.
    """

    kind = "iid"
    REPRESENTATIONS = ("trial_tokens",)

    def __init__(self, name, item_dim, count, representation="trial_tokens"):
        item_dim, count = int(item_dim), int(count)
        if item_dim < 1 or count < 1:
            raise ValueError(
                f"IID({name!r}) needs item_dim >= 1 and count >= 1, "
                f"got item_dim={item_dim}, count={count}."
            )
        if representation not in self.REPRESENTATIONS:
            raise ValueError(
                f"IID({name!r}): representation {representation!r} is not "
                f"implemented; available: {', '.join(self.REPRESENTATIONS)}."
            )
        self.name = str(name)
        self.item_dim = item_dim
        self.count = count
        self.representation = representation

    @property
    def width(self):
        return self.item_dim * self.count

    @property
    def tokens(self):
        return self.count

    def to_dict(self):
        return {
            "type": self.kind, "name": self.name, "item_dim": self.item_dim,
            "count": self.count, "representation": self.representation,
        }

    def __repr__(self):
        return (
            f"IID({self.name!r}, item_dim={self.item_dim}, count={self.count}, "
            f"representation={self.representation!r})"
        )

    def __eq__(self, other):
        return isinstance(other, IID) and self.to_dict() == other.to_dict()


class Group:
    """``count`` structural repeats of an inner block list, repeat-major.

    The flat layout is ``[repeat_0(all sub-blocks), repeat_1, ...]``, and each
    repeat is itself an ordinary :class:`NodeSchema`, so a repeat may mix block
    kinds -- for partial pooling, ``Atomic("local", 3)`` for the subject's own
    parameters followed by ``IID("trials", 3, 30)`` for its exchangeable trials.

    **What symmetry this asserts, and what it does not.** The inner ``IID`` block
    keeps its own guarantee: permuting trials within one repeat permutes exactly
    that repeat's trial tokens. ``Group`` itself asserts only that the *nesting*
    is preserved -- repeat ``j``'s locals stay attached to repeat ``j``'s trials.
    It promises **no** invariance or equivariance under permuting the repeats.
    That is deliberate: with weights shared across repeats and no slot term the
    across-repeat trunk would be permutation-equivariant whether or not anyone
    asked for it, so ``slot="learned"`` is the default and makes the contract the
    architecture actually implements match the one documented here.

    ``pool`` and ``slot`` are recorded here rather than on the network for the
    same reason ``IID.representation`` is: they are restored with the schema when
    a checkpoint is loaded, so a model cannot come back with a different
    architecture than it was trained with.
    """

    kind = "group"
    POOLINGS = ("mean", "pma")
    SLOTS = ("learned", "sinusoidal", "none")

    def __init__(self, name, count, blocks, pool="mean", slot="learned"):
        count = int(count)
        if count < 1:
            raise ValueError(f"Group({name!r}) needs count >= 1, got {count}.")
        if pool not in self.POOLINGS:
            raise ValueError(
                f"Group({name!r}): pool {pool!r} is not implemented; "
                f"available: {', '.join(self.POOLINGS)}."
            )
        if slot not in self.SLOTS:
            raise ValueError(
                f"Group({name!r}): slot {slot!r} is not implemented; "
                f"available: {', '.join(self.SLOTS)}."
            )
        blocks = list(blocks)
        if not blocks:
            raise ValueError(f"Group({name!r}) needs at least one inner block.")
        if any(block.kind == Group.kind for block in blocks):
            # Nothing in the layout maths forbids it, but every consumer of a
            # schema (the trunk, the roles helpers, the benchmark adapter) is
            # written for exactly two levels. Refuse loudly rather than produce
            # a schema that slices correctly and trains wrongly.
            raise ValueError(
                f"Group({name!r}): nesting a Group inside a Group is not "
                "supported; the trunk is two-level by construction."
            )
        self.name = str(name)
        self.count = count
        self.pool = pool
        self.slot = slot
        # One repeat, as an ordinary schema. Every recursive operation below
        # delegates to it, so heterogeneous repeats need no special handling.
        self.inner = NodeSchema(blocks)

    @property
    def blocks(self):
        return self.inner.blocks

    @property
    def width(self):
        return self.count * self.inner.nodes_size

    @property
    def tokens(self):
        return self.count * self.inner.token_count

    @property
    def item_dim(self):
        raise TypeError(
            f"Group({self.name!r}) has no single item_dim: its repeats carry "
            f"{self.inner.nodes_size} scalars across {self.inner.token_count} "
            "tokens of possibly different widths. Consumers that need a uniform "
            "item width must handle Group explicitly."
        )

    def to_dict(self):
        return {
            "type": self.kind, "name": self.name, "count": self.count,
            "pool": self.pool, "slot": self.slot,
            "blocks": [block.to_dict() for block in self.inner.blocks],
        }

    def __repr__(self):
        return (
            f"Group({self.name!r}, count={self.count}, pool={self.pool!r}, "
            f"slot={self.slot!r}, blocks={self.inner.blocks})"
        )

    def __eq__(self, other):
        return isinstance(other, Group) and self.to_dict() == other.to_dict()


def block_from_dict(payload):
    kind = payload["type"]
    if kind == Atomic.kind:
        return Atomic(payload["name"], payload["dim"])
    if kind == IID.kind:
        return IID(
            payload["name"], payload["item_dim"], payload["count"],
            payload.get("representation", "trial_tokens"),
        )
    if kind == Group.kind:
        return Group(
            payload["name"], payload["count"],
            [block_from_dict(entry) for entry in payload["blocks"]],
            pool=payload.get("pool", "mean"),
            slot=payload.get("slot", "learned"),
        )
    raise ValueError(f"Unknown schema block type {kind!r}.")


class NodeSchema:
    """An ordered list of blocks covering the whole flat node vector."""

    def __init__(self, blocks):
        blocks = list(blocks)
        if not blocks:
            raise ValueError("A NodeSchema needs at least one block.")
        names = [block.name for block in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate schema block names: {names}.")
        self.blocks = blocks

        self.node_slices = []
        self.token_slices = []
        node, token = 0, 0
        for block in blocks:
            self.node_slices.append((node, node + block.width))
            self.token_slices.append((token, token + block.tokens))
            node += block.width
            token += block.tokens
        self.nodes_size = node
        self.token_count = token

    # ----- construction -------------------------------------------------

    @classmethod
    def legacy(cls, nodes_size):
        """One scalar token per node -- the tokenization COMPASS has always used."""
        return cls([Atomic("nodes", nodes_size)])

    @classmethod
    def from_spec(cls, theta_size, x_spec):
        """``theta`` as an atomic block, followed by the observation blocks."""
        if x_spec is None:
            raise ValueError("from_spec needs an x_spec.")
        if isinstance(x_spec, (Atomic, IID, Group)):
            x_spec = [x_spec]
        return cls([Atomic("theta", theta_size), *x_spec])

    # ----- serialization ------------------------------------------------

    def to_dict(self):
        return {"blocks": [block.to_dict() for block in self.blocks]}

    @classmethod
    def from_dict(cls, payload):
        return cls([block_from_dict(entry) for entry in payload["blocks"]])

    # ----- derived quantities -------------------------------------------

    @property
    def is_atomic(self):
        """True when every block is atomic, i.e. one token per scalar.

        A :class:`Group` of purely atomic sub-blocks also happens to have one
        token per scalar, but it is deliberately **not** reported as atomic: the
        repeat structure is real, the two-level trunk keys off it, and
        ``trunk="position"`` (which claims bit-for-bit legacy equivalence) must
        never accept it.
        """
        return all(block.kind == Atomic.kind for block in self.blocks)

    @property
    def has_group(self):
        """True when any block is a :class:`Group`, i.e. the schema is nested."""
        return any(block.kind == Group.kind for block in self.blocks)

    def block_view(self, x, index):
        """``(batch, count, item_dim)`` view of one block of a flat vector."""
        block = self.blocks[index]
        if block.kind == Group.kind:
            raise TypeError(
                f"block_view is undefined for Group({block.name!r}): its repeats "
                "hold tokens of different widths. Use repeat_view(x, index), "
                "which returns (..., count, inner_nodes_size), and then index "
                "into the repeat with the Group's own inner schema."
            )
        start, stop = self.node_slices[index]
        chunk = x[..., start:stop]
        return chunk.reshape(*chunk.shape[:-1], block.tokens, block.width // block.tokens)

    def repeat_view(self, x, index):
        """``(..., count, inner_nodes_size)`` view of one :class:`Group` block."""
        block = self.blocks[index]
        if block.kind != Group.kind:
            raise TypeError(
                f"repeat_view is only defined for Group blocks; block {index} is "
                f"{block.kind!r}. Use block_view for it."
            )
        start, stop = self.node_slices[index]
        chunk = x[..., start:stop]
        return chunk.reshape(*chunk.shape[:-1], block.count, block.inner.nodes_size)

    def repeat_token_view(self, tokens, index):
        """``(..., count, inner_token_count)`` view of one :class:`Group`'s tokens."""
        block = self.blocks[index]
        if block.kind != Group.kind:
            raise TypeError(
                f"repeat_token_view is only defined for Group blocks; block "
                f"{index} is {block.kind!r}."
            )
        start, stop = self.token_slices[index]
        chunk = tokens[..., start:stop]
        return chunk.reshape(*chunk.shape[:-1], block.count, block.inner.token_count)

    def token_mask(self, condition_mask):
        """Per-token conditioning flags under COMPASS' key-padding convention.

        A grouped token is a valid attention key only if *every* scalar it
        carries is conditioned; a partially observed item is treated as latent.
        """
        if self.is_atomic:
            return condition_mask
        parts = []
        for index, block in enumerate(self.blocks):
            if block.kind == Group.kind:
                # Delegate to the repeat's own schema, so a repeat that mixes
                # atomic locals with IID trials is handled by the same code that
                # handles them at the top level.
                view = self.repeat_view(condition_mask, index)
                inner = block.inner.token_mask(view)
                parts.append(inner.reshape(*inner.shape[:-2], block.tokens))
                continue
            view = self.block_view(condition_mask, index)
            parts.append(view.min(dim=-1).values if view.shape[-1] > 1 else view[..., 0])
        return torch.cat(parts, dim=-1)

    def expand_token_values(self, token_values):
        """Scatter one value per token back onto the flat node vector."""
        if self.is_atomic:
            return token_values
        parts = []
        for index, block in enumerate(self.blocks):
            if block.kind == Group.kind:
                view = self.repeat_token_view(token_values, index)
                inner = block.inner.expand_token_values(view)
                parts.append(inner.reshape(*inner.shape[:-2], block.width))
                continue
            start, stop = self.token_slices[index]
            chunk = token_values[..., start:stop]
            item_dim = block.width // block.tokens
            if item_dim > 1:
                chunk = chunk.unsqueeze(-1).expand(*chunk.shape, item_dim)
                chunk = chunk.reshape(*chunk.shape[:-2], block.width)
            parts.append(chunk)
        return torch.cat(parts, dim=-1)

    def __repr__(self):
        return (
            f"NodeSchema(nodes_size={self.nodes_size}, "
            f"tokens={self.token_count}, blocks={self.blocks})"
        )

    def __eq__(self, other):
        return isinstance(other, NodeSchema) and self.to_dict() == other.to_dict()


class GroupedBernoulli:
    """Training-time condition-mask sampler that respects the schema's grouping.

    Drop-in replacement for the ``torch.distributions.Bernoulli(p)`` the trainer
    used to hold: ``sample(shape)`` returns a mask whose trailing dimension is
    ``nodes_size``. For an all-atomic schema it *is* that Bernoulli, drawn with
    the same call, so legacy training is unchanged bit-for-bit. For a schema with
    IID blocks the draw is per item and then broadcast across the item's scalars,
    so an item is observed or latent as a whole while the marginal rate of
    observed scalars stays ``p``.
    """

    def __init__(self, schema, probability=0.33):
        self.schema = schema
        self.probability = float(probability)
        self._bernoulli = torch.distributions.bernoulli.Bernoulli(self.probability)

    def sample(self, shape):
        shape = torch.Size(shape)
        if shape[-1:] != torch.Size([self.schema.nodes_size]):
            raise ValueError(
                f"GroupedBernoulli expects a trailing dimension of "
                f"{self.schema.nodes_size}, got {tuple(shape)}."
            )
        if self.schema.is_atomic:
            return self._bernoulli.sample(shape)
        tokens = self._bernoulli.sample(shape[:-1] + torch.Size([self.schema.token_count]))
        return self.schema.expand_token_values(tokens)


#############################################
# ----- Structured conditioning menus -----
#############################################

# The six non-degenerate conditionals over a (global, local, observation)
# partition. Each entry is (name, roles that are *conditioned*); everything not
# listed is latent and therefore carries score-matching loss.
#
# The two excluded combinations are excluded for a reason: "" conditions nothing
# (the joint prior p(g,l,x), which no COMPASS sampler ever evaluates) and "glx"
# conditions everything (no latent coordinate, so the loss is identically zero).
SIX_CONDITIONALS = (
    ("p(g,l|x)", "x"),    # the deployed posterior: both parameter levels latent
    ("p(g|l,x)", "lx"),   # group level given subject level
    ("p(l|g,x)", "gx"),   # subject level given group level -- the conditional
                          #   hierarchical score ascent / newton_map_estimate use
    ("p(x|g,l)", "gl"),   # the likelihood
    ("p(x,l|g)", "g"),    # prior predictive within a group draw
    ("p(x,g|l)", "l"),    # observations and group level given subject level
)


class ConditionalMenu:
    """Training-time condition-mask sampler over an explicit menu of conditionals.

    ``GroupedBernoulli`` draws each node's conditioning flag independently, which
    spreads the training signal over all ``2**tokens`` conditionals. For a
    partial-pooling problem that is not where the mass is needed: the masks the
    samplers actually evaluate are a handful of *structured* ones in which whole
    roles -- group-level parameters, subject-level parameters, observations --
    are conditioned or latent together. At ``nodes_size=100`` with 30 trial
    tokens the deployed posterior mask has Bernoulli probability
    ``0.67**10 * 0.33**30 ~ 6.5e-17``, i.e. it is never drawn.

    This sampler draws from a named menu instead, optionally mixed with the
    Bernoulli draw so the general conditional capability is not lost.

    Args:
        schema: the model's :class:`NodeSchema`. Used to check that every menu
            mask is constant within each grouped token -- a mask that split an
            IID item would be unrepresentable under
            :meth:`NodeSchema.token_mask`.
        roles: ordered mapping ``{role_name: node indices}``. The roles must
            partition ``range(schema.nodes_size)``; a node left out of every
            role would silently never be conditioned.
        patterns: sequence of ``(name, conditioned_roles)`` pairs, where
            ``conditioned_roles`` is an iterable of role names (a string of
            single-character role names is accepted).
        weights: relative probabilities of the patterns; uniform by default.
        bernoulli_fraction: probability that a draw comes from
            :class:`GroupedBernoulli` instead of the menu. ``0.0`` trains on the
            menu alone.
        bernoulli_probability: the Bernoulli rate used for that fraction.
    """

    def __init__(self, schema, roles, patterns, weights=None,
                 bernoulli_fraction=0.0, bernoulli_probability=0.33):
        self.schema = schema
        self.roles = {str(name): torch.as_tensor(sorted(index), dtype=torch.long)
                      for name, index in dict(roles).items()}

        covered = torch.cat(list(self.roles.values())) if self.roles else torch.empty(0)
        expected = torch.arange(schema.nodes_size)
        if not torch.equal(torch.sort(covered).values, expected):
            raise ValueError(
                "roles must partition range(nodes_size)="
                f"{schema.nodes_size}; got {covered.numel()} indices covering "
                f"{torch.unique(covered).numel()} distinct nodes."
            )

        self.names = []
        masks = []
        for name, conditioned in patterns:
            conditioned = list(conditioned)
            unknown = set(conditioned) - set(self.roles)
            if unknown:
                raise ValueError(
                    f"Pattern {name!r} names unknown roles {sorted(unknown)}; "
                    f"available: {sorted(self.roles)}."
                )
            mask = torch.zeros(schema.nodes_size)
            for role in conditioned:
                mask[self.roles[role]] = 1.0
            if not mask.any():
                raise ValueError(
                    f"Pattern {name!r} conditions nothing; that is the joint "
                    "prior, which no sampler evaluates."
                )
            if mask.all():
                raise ValueError(
                    f"Pattern {name!r} conditions every node, so it has no "
                    "latent coordinate and contributes no loss."
                )
            # A grouped token is a key only if all of its scalars are
            # conditioned, so a mask that splits an item is not representable.
            token = schema.token_mask(mask)
            if not torch.equal(schema.expand_token_values(token), mask):
                raise ValueError(
                    f"Pattern {name!r} splits a grouped token: the schema's "
                    "items must be conditioned or latent as a whole."
                )
            self.names.append(str(name))
            masks.append(mask)

        self.masks = torch.stack(masks)
        if weights is None:
            weights = torch.ones(len(masks))
        else:
            weights = torch.as_tensor(list(weights), dtype=torch.float32)
            if weights.shape != (len(masks),):
                raise ValueError(
                    f"weights must have one entry per pattern ({len(masks)}); "
                    f"got {tuple(weights.shape)}."
                )
            if (weights <= 0).any():
                raise ValueError("Pattern weights must be positive.")
        self.weights = weights / weights.sum()

        self.bernoulli_fraction = float(bernoulli_fraction)
        if not 0.0 <= self.bernoulli_fraction <= 1.0:
            raise ValueError(
                "bernoulli_fraction must be in [0, 1]; got "
                f"{self.bernoulli_fraction}."
            )
        self.bernoulli = GroupedBernoulli(schema, bernoulli_probability)
        self._cumulative = None      # built lazily by `sample`

    def sample(self, shape):
        shape = torch.Size(shape)
        if shape[-1:] != torch.Size([self.schema.nodes_size]):
            raise ValueError(
                f"ConditionalMenu expects a trailing dimension of "
                f"{self.schema.nodes_size}, got {tuple(shape)}."
            )
        lead = shape[:-1]
        count = 1
        for size in lead:
            count *= size

        # `torch.multinomial` carries a large fixed overhead relative to the
        # work here, and `Trainer.__getitem__` calls this once per *example*
        # (with shape (nodes,)), not once per batch -- so that overhead is paid
        # `batch_size` times per step. Measured on a 3-node menu: 90.7 us per
        # example against 17.3 us for GroupedBernoulli, i.e. 93 ms versus 18 ms
        # of pure mask-drawing per 1024-row step. Inverse-CDF sampling off a
        # cached cumulative weight vector is exact and ~an order of magnitude
        # cheaper at these sizes.
        if self._cumulative is None:
            self._cumulative = torch.cumsum(self.weights, dim=0)
        index = torch.searchsorted(
            self._cumulative, torch.rand(count) * self._cumulative[-1]
        ).clamp_(max=len(self.weights) - 1)
        out = self.masks[index]
        if self.bernoulli_fraction > 0.0:
            use_bernoulli = torch.rand(count) < self.bernoulli_fraction
            drawn = self.bernoulli.sample(
                torch.Size([count, self.schema.nodes_size])
            )
            out = torch.where(use_bernoulli.unsqueeze(-1), drawn, out)
        return out.reshape(*lead, self.schema.nodes_size)

    def describe(self):
        """Human-readable menu, for training logs and manifests."""
        rows = [
            f"{name}: p={float(weight):.3f} conditioned={int(mask.sum())}/"
            f"{self.schema.nodes_size} nodes"
            for name, weight, mask in zip(self.names, self.weights, self.masks)
        ]
        if self.bernoulli_fraction:
            rows.append(
                f"Bernoulli({self.bernoulli.probability}): "
                f"p={self.bernoulli_fraction:.3f}"
            )
        return "\n".join(rows)

    def __repr__(self):
        return (
            f"ConditionalMenu(patterns={self.names}, "
            f"bernoulli_fraction={self.bernoulli_fraction})"
        )


def hierarchical_roles(schema, global_size, local_size):
    """``{'g': ..., 'l': ..., 'x': ...}`` for a flat ``[globals, locals, x]`` vector.

    This is the layout COMPASS' partial-pooling benchmark already uses: the
    first ``global_size`` nodes are the group-level parameters, the next
    ``local_size`` are that subject's parameters, and everything after is the
    subject's observations.

    When the schema is nested this delegates to
    :func:`nested_hierarchical_roles` automatically. That is deliberate: every
    caller that builds a conditioning menu goes through here, and a caller that
    forgot to switch would silently get a menu whose ``l`` role covered one
    subject's parameters and 19 subjects' trials.
    """
    if schema.has_group:
        return nested_hierarchical_roles(schema, global_size, local_size)
    global_size, local_size = int(global_size), int(local_size)
    if global_size < 1 or local_size < 1:
        raise ValueError("global_size and local_size must both be >= 1.")
    if global_size + local_size >= schema.nodes_size:
        raise ValueError(
            f"global_size + local_size = {global_size + local_size} leaves no "
            f"observation nodes in a {schema.nodes_size}-node vector."
        )
    return {
        "g": list(range(global_size)),
        "l": list(range(global_size, global_size + local_size)),
        "x": list(range(global_size + local_size, schema.nodes_size)),
    }


def nested_hierarchical_roles(schema, global_size, local_size):
    """``{'g': ..., 'l': ..., 'x': ...}`` for a ``[globals, Group(subjects)]`` vector.

    Under a :class:`Group` the layout is ``[g, (l_1, x_1), (l_2, x_2), ...]``
    rather than flat ``[g, l, x]``, so ``l`` is the union of every repeat's
    first ``local_size`` scalars and ``x`` is the union of the remainders. The
    per-repeat convention is exactly the flat one applied inside each repeat,
    which is what keeps :data:`SIX_CONDITIONALS` meaning the same six
    conditionals it always meant.

    The three roles still partition the whole node vector, which is what
    :class:`ConditionalMenu` requires.
    """
    global_size, local_size = int(global_size), int(local_size)
    if global_size < 1 or local_size < 1:
        raise ValueError("global_size and local_size must both be >= 1.")

    groups = [i for i, block in enumerate(schema.blocks) if block.kind == Group.kind]
    if len(groups) != 1:
        raise ValueError(
            f"nested_hierarchical_roles expects exactly one Group block, found "
            f"{len(groups)}."
        )
    index = groups[0]
    group = schema.blocks[index]
    start, _ = schema.node_slices[index]
    if start != global_size:
        raise ValueError(
            f"Expected the {global_size} group-level nodes to be everything "
            f"before the Group block, but it starts at node {start}."
        )
    inner_width = group.inner.nodes_size
    if local_size >= inner_width:
        raise ValueError(
            f"local_size={local_size} leaves no observation nodes in a repeat of "
            f"{inner_width} scalars."
        )
    # The split must land on an inner block boundary, or a menu pattern would
    # cut a token in half and ConditionalMenu would reject it later with a much
    # less informative message.
    boundaries = {0} | {stop for _, stop in group.inner.node_slices}
    if local_size not in boundaries:
        raise ValueError(
            f"local_size={local_size} does not fall on an inner block boundary "
            f"of Group({group.name!r}); boundaries are {sorted(boundaries)}."
        )

    locals_, observations = [], []
    for repeat in range(group.count):
        base = start + repeat * inner_width
        locals_.extend(range(base, base + local_size))
        observations.extend(range(base + local_size, base + inner_width))
    return {
        "g": list(range(global_size)),
        "l": locals_,
        "x": observations,
    }


def six_conditional_menu(schema, global_size, local_size, weights=None,
                         bernoulli_fraction=0.0, bernoulli_probability=0.33):
    """The :data:`SIX_CONDITIONALS` menu over a hierarchical node layout."""
    return ConditionalMenu(
        schema, hierarchical_roles(schema, global_size, local_size),
        SIX_CONDITIONALS, weights=weights,
        bernoulli_fraction=bernoulli_fraction,
        bernoulli_probability=bernoulli_probability,
    )
