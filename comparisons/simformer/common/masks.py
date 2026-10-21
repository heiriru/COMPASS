"""The Simformer training condition-mask law, as a COMPASS mask sampler.

Sec. 3.1 / A2.1: "At every training batch, we selected uniformly at random a
mask corresponding to the joint, the posterior, the likelihood or two random
masks. The random masks were drawn from a Bernoulli distribution with p = 0.3
and p = 7."  (0.7; the released code confirms.)

Why this is not ``compass.ConditionalMenu``
-------------------------------------------
Two reasons, both structural:

1. Their menu contains the **joint** mask ``M_C = 0`` -- nothing conditioned.
   ``ConditionalMenu`` raises on such a pattern ("conditions nothing; that is
   the joint prior, which no sampler evaluates"), because in COMPASS's
   key-padding convention that mask leaves the attention with *no keys at all*.
   It runs only because ``nn.MultiheadAttention`` was built with
   ``add_bias_kv=True``. Keeping it is what makes this arm apples-to-apples;
   dropping it would be a different training law.
2. Their two "random masks" are per-node Bernoulli draws at two different rates,
   which a fixed pattern menu cannot express.

A per-batch draw, not a per-row draw
------------------------------------
The paper selects one mask **per training batch**. COMPASS's trainer calls
``mask_sampler.sample(shape)`` with ``shape = (batch, nodes_size)`` and would
otherwise get an independent mask per row. ``per_batch=True`` (the default here)
reproduces the paper: one pattern is drawn and broadcast over the batch.
``per_batch=False`` is the COMPASS-native variant and is the interesting
ablation -- it lowers the gradient variance of the mask law without changing its
marginal, and the paper never tested it.
"""

import torch


class SimformerMaskSampler:
    """Uniform five-way draw over {joint, posterior, likelihood, Bern(.3), Bern(.7)}.

    Args:
        schema: the model's ``compass.NodeSchema``.
        theta_dim: number of leading nodes that are parameters; the rest are data.
        weights: relative pattern probabilities. ``None`` is the paper's uniform
            draw. Supplying weights is how the COMPASS-side reweighting question
            (``docs/followup_results.md`` section 1) gets asked of *their* law.
        per_batch: draw one mask per call and broadcast it (the paper) rather
            than one per row.
        bernoulli_rates: the two random-mask rates.
        generator: optional ``torch.Generator`` for reproducible draws.
    """

    PATTERNS = ("joint", "posterior", "likelihood", "bernoulli_low", "bernoulli_high")

    def __init__(self, schema, theta_dim, weights=None, per_batch=True,
                 bernoulli_rates=(0.3, 0.7), generator=None):
        self.schema = schema
        self.theta_dim = int(theta_dim)
        self.nodes_size = schema.nodes_size
        if not 0 < self.theta_dim < self.nodes_size:
            raise ValueError(
                f"theta_dim must be strictly inside (0, {self.nodes_size}); "
                f"got {theta_dim}."
            )
        if weights is None:
            weights = [1.0] * len(self.PATTERNS)
        if len(weights) != len(self.PATTERNS):
            raise ValueError(
                f"weights must have one entry per pattern {self.PATTERNS}; "
                f"got {len(weights)}."
            )
        w = torch.as_tensor(weights, dtype=torch.float)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative with a positive sum.")
        self.weights = w / w.sum()
        self.per_batch = bool(per_batch)
        self.bernoulli_rates = tuple(float(r) for r in bernoulli_rates)
        self.generator = generator

        posterior = torch.zeros(self.nodes_size)
        posterior[self.theta_dim:] = 1.0          # data conditioned, theta latent
        likelihood = torch.zeros(self.nodes_size)
        likelihood[: self.theta_dim] = 1.0        # theta conditioned, data latent
        self._fixed = {
            "joint": torch.zeros(self.nodes_size),
            "posterior": posterior,
            "likelihood": likelihood,
        }

    def describe(self):
        return {name: float(p) for name, p in zip(self.PATTERNS, self.weights)}

    def _bernoulli(self, shape, rate):
        draw = torch.rand(shape, generator=self.generator)
        return (draw < rate).to(torch.float)

    def _one(self, lead_shape):
        """One mask pattern, shaped ``lead_shape + (nodes_size,)``."""
        index = int(torch.multinomial(self.weights, 1, generator=self.generator))
        name = self.PATTERNS[index]
        if name in self._fixed:
            row = self._fixed[name]
        else:
            # One Bernoulli draw for the whole batch, not one per row: the paper
            # selects *a mask* per training batch, and a per-row draw would be a
            # different (lower-variance) law wearing the same name.
            rate = self.bernoulli_rates[0 if name == "bernoulli_low" else 1]
            if self.schema.is_atomic:
                row = self._bernoulli((self.nodes_size,), rate)
            else:
                tokens = self._bernoulli((self.schema.token_count,), rate)
                row = self.schema.expand_token_values(tokens)
        return row.expand(*lead_shape, self.nodes_size).clone()

    def sample(self, shape):
        shape = torch.Size(shape)
        if shape[-1:] != torch.Size([self.nodes_size]):
            raise ValueError(
                f"Expected a trailing dimension of {self.nodes_size}, "
                f"got {tuple(shape)}."
            )
        lead = tuple(shape[:-1])
        if self.per_batch or not lead:
            return self._one(lead)
        rows = [self._one(()) for _ in range(int(torch.tensor(lead).prod()))]
        return torch.stack(rows).reshape(shape)


def posterior_mask(nodes_size, theta_dim):
    """The deployed mask ``p(theta | x)``: data conditioned, parameters latent."""
    mask = torch.zeros(nodes_size)
    mask[theta_dim:] = 1.0
    return mask


def likelihood_mask(nodes_size, theta_dim):
    """``p(x | theta)``."""
    mask = torch.zeros(nodes_size)
    mask[:theta_dim] = 1.0
    return mask


def random_conditional_masks(nodes_size, count, generator=None, min_latent=1,
                             min_conditioned=1):
    """``count`` distinct random condition masks, for the all-conditional tasks.

    Fig. 4b scores "100 randomly selected conditional or full joint
    distributions". The paper does not state the distribution over masks; a
    uniform draw over the non-degenerate subsets is the neutral reading and is
    what this returns. Whatever is used, it must be *frozen and shared* across
    every arm -- a different mask set per arm is not a comparison.
    """
    if nodes_size < min_latent + min_conditioned:
        raise ValueError("nodes_size too small for the requested margins.")
    seen, masks = set(), []
    guard = 0
    while len(masks) < count:
        guard += 1
        if guard > 1000 * count:
            raise RuntimeError(
                f"Could only find {len(masks)} distinct masks out of {count}."
            )
        mask = (torch.rand(nodes_size, generator=generator) < 0.5).to(torch.float)
        n_cond = int(mask.sum())
        if n_cond < min_conditioned or nodes_size - n_cond < min_latent:
            continue
        key = tuple(mask.tolist())
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
    return torch.stack(masks)
