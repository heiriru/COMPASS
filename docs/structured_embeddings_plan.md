# Structured embeddings for COMPASS

Plan for replacing the single "one scalar dim = one token" embedder with a
declarative schema of typed data blocks: atomic (legacy), i.i.d. trials,
ordered sequences, and unordered identified sets.

## 1. Where we stand

`ConditionTransformer` treats the joint vector `(theta, x)` as `nodes_size`
tokens, one per scalar dimension:

- `InputEmbedder` (`ConditionTransformer.py:27`): `x.unsqueeze(-1) * P` with
  `P: (1, N, H)` — a purely multiplicative per-node embedding, no bias, no MLP.
- `TransformerBlock` (`:127`): `LayerNorm((N, H))`, and adaLN producing
  `6 * N * H` modulation values from the time embedding — i.e. every node has
  its own shift/scale/gate.
- `FinalLayer` (`:186`): flattens `(N, H)` and multiplies by a
  `(N*H, N)` parameter matrix — O(N²H) parameters.
- Attention masking (`:157`): `key_padding_mask = (1 - c)`, so **only
  conditioned tokens are ever used as keys**. Latent tokens read from observed
  tokens; latent–latent attention is off.

Consequences that constrain the design:

1. **The token count is baked into the parameter shapes** (LayerNorm shape,
   adaLN output width, final-layer matrix). Variable-length data is impossible
   without touching all three.
2. **Any dimension can be latent or conditioned**, which is why data and
   parameters live in one vector. Compressing a block of x into summary tokens
   forfeits that flexibility for that block — you cannot diffuse a dimension you
   only ever see through a pooled encoder.
3. Everything downstream (`Sampler`, `PFODE`, `MultiObsSampler`,
   `ModelTransfuser`, `Trainer`) assumes a flat `nodes_size` state vector — but
   there are only **9 call sites** into the network
   (`Trainer.py:327`, `Sampler.py:285/291/299`, `PFODE.py:170/176`,
   `MultiObsSampler.py:1430/3135/3141`), so the surgery is contained.

## 2. Key design decision: split the node vector

Introduce an explicit boundary between

- the **diffusion state**: latent-capable atomic dimensions (all of `theta`,
  plus any x block declared atomic) — a flat vector, exactly as today; and
- the **context**: tokens produced by structured encoders from condition-only
  data blocks.

```
score(x_latent, t, condition_mask, context)
```

This buys three things at once:

- Structured encoders run **once per observation** and are cached across all
  diffusion timesteps, all corrector steps and all posterior samples (today the
  embedder is re-run every step) — a large speedup for anything expensive like a
  sequence encoder.
- `Sampler`, `PFODE` and `MultiObsSampler` keep operating on a flat state of
  size `n_latent_dims` and only need to thread an opaque `context` through.
- Jacobian-based composition (`gauss_jacobian`) differentiates only w.r.t. the
  latent state; a constant context changes nothing in that math.

Context tokens carry `c = 1` in the existing key-padding convention, so they are
attended to with no change to masking semantics. A **missing block gets `c = 0`
and is ignored for free** — which also gives missing-modality support and makes
the unconditional branch of classifier-free guidance (`cfg_alpha`, `Sampler.py:299`)
correct: zeroing `c` must zero the context validity flags too.

## 3. API

```python
from compass import ScoreBasedInferenceModel
from compass.embeddings import Atomic, IID, Sequence, Set

model = ScoreBasedInferenceModel(
    theta_size=10,
    x_spec=[
        Atomic("summaries", dim=4),                     # today's behaviour
        IID("trials", dim=3, tokens=8),                 # exchangeable replicates
        Sequence("lfp", dim=2, tokens=8, patch=8),      # ordered
        Set("areas", dim=16, items=8),                  # unordered, identified
    ],
    hidden_size=128, depth=6, num_heads=8,
)

model.train(theta=theta, x={"summaries": ..., "trials": ..., "lfp": ..., "areas": ...})
samples = model.sample(x={"trials": ..., "areas": ...}, num_samples=1000)   # lfp missing -> masked
```

Expected shapes (`B` = batch):

| Block | input | tokens emitted | variable length |
|---|---|---|---|
| `Atomic(dim=d)` | `(B, d)` | `d` | no |
| `IID(dim=d, tokens=k)` | `(B, n, d)` (+ optional mask) | `k` | yes, over `n` |
| `Sequence(dim=d, tokens=k)` | `(B, T, d)` (+ optional mask) | `k` | yes, over `T` |
| `Set(dim=d, items=m)` | `(B, m, d)` | `m` (or `k` if `tokens=k`) | no (fixed membership) |

Backwards compatibility, non-negotiable:

- `ScoreBasedInferenceModel(nodes_size=14)` keeps working and is internally
  rewritten to `theta_size = 14 - x_dim`, `x_spec=[Atomic("x", dim=x_dim)]`
  with `arch="legacy"`.
- `train(theta, x)` / `sample(x=tensor, condition_mask=...)` with plain tensors
  keep working; a dict for `x` selects the new path.
- Checkpoints gain `schema` and `arch_version`; `load` dispatches on them and
  reconstructs legacy models bit-exactly.

## 4. The four embedders

All of them emit `(B, k, H)` plus a validity flag per token, and add a learned
**block-identity embedding** so the trunk can tell blocks apart.

### 4.1 `Atomic` — keep as is

Unchanged `InputEmbedder`. Used for `theta` always, and for any x dimensions the
user wants to remain latent-capable (missing-data imputation, posterior
predictive, `log_prob(x | theta)`).

### 4.2 `IID` — exchangeable trials (partial pooling)

Shared per-trial MLP `d -> H`, then **masked mean pooling**, then `k` learned
query tokens via PMA (Set Transformer multihead attention with learned seeds).
No positional encoding.

Two details that matter statistically:

- Mean pooling of a learned per-trial feature map is the right inductive bias:
  for an exponential family the posterior depends on the data only through the
  average sufficient statistic.
- **Mean pooling alone throws away `n`**, and posterior contraction is driven by
  `n`. Concatenate `log n` (and optionally the masked second moment) to the
  pooled vector before the PMA so the network can represent `1/sqrt(n)`
  sharpening.
- Train with `n` resampled per batch (`n ~ U{1..N_max}`) or the encoder will not
  generalise across trial counts.

Relation to `MultiObsSampler`: the IID encoder amortises pooling *inside* the
network for the trials of one unit; score composition pools *across* units. They
are complementary — per-unit contexts feed the composed score over the shared
coordinates — and the plan should not let the encoder quietly replace the
compositional path for the tall-data regime it was built for.

### 4.3 `Sequence` — time series / ordered sets

Per-step linear `d -> H` + sinusoidal or rotary positional encoding, 2 small
transformer encoder layers, then PMA to `k` tokens. For long series, a
`patch=p` stride-`p` Conv1d front-end (or a dilated TCN) keeps the cost down;
padding mask supports variable `T`. Optionally expose `causal=True` for
generative/filtering use.

### 4.4 `Set` — unordered but identified (brain areas)

Shared per-item MLP `d -> H` plus a learned `nn.Embedding(m, H)` item identity,
emitting **one token per item** by default and letting the main trunk do the
inter-item attention (a transformer without positional encoding is already
permutation-equivariant). Pool to `k` tokens only when `m` is large.

The distinction from `IID` is deliberate: `IID` is anonymous, count-carrying and
variable-length; `Set` is heterogeneous, identified and fixed-membership.
`Set(..., identity=False)` degrades to an anonymous set encoder.

### 4.5 Nesting (later phase)

Encoders compose: `IID("subjects", inner=Sequence(...))`, or
`Set("areas", inner=Sequence("time", ...))` for area × time recordings. The
inner encoder runs on the flattened item axis, its tokens are pooled per item,
and the outer encoder proceeds as usual. Design the base class for this now;
implement it last.

## 5. Trunk changes required (node-count agnosticism)

These are the blockers for anything variable-length, and they change parameter
shapes — so they go behind `arch="tokenized"` while `arch="legacy"` stays
bit-exact:

- `LayerNorm((N, H))` -> `LayerNorm(H)`.
- adaLN: one shared `Linear(t_dim, 6H)` (standard DiT) plus an optional
  per-token learned offset for the atomic latent tokens, instead of
  `Linear(t_dim, 6*N*H)`.
- `FinalLayer`: shared `Linear(H, 1)` head (optionally with a per-token scale)
  applied to the **latent atomic tokens only**, replacing the `(N*H, N)` matrix.
  This alone removes the O(N²H) parameter growth.

Worth deciding at the same time (currently latent tokens are masked out as
keys): whether latent–latent attention should be enabled under the new arch.
It is a separate change and should be a separate flag and a separate experiment.

## 6. Training changes

- Dataset/collate accepts a dict of blocks, pads ragged `n`/`T`, and carries
  per-block masks.
- The Bernoulli(0.33) condition mask (`Trainer._prepare_data`) applies only to
  atomic dims. Structured blocks get **block-level dropout** (e.g. p = 0.1
  fully-missing) so missing modalities and the CFG unconditional branch are
  trained.
- Loss (`Trainer.loss_fn`) is unchanged: it is already restricted to latent
  dims, and structured blocks contribute none.

## 7. Phasing

0. **Refactor, no new behaviour.** Split diffusion state from context, add the
   `context=` kwarg through the 9 call sites, keep the legacy schema bit-exact.
   All existing tests must pass unchanged.
1. **Schema + `Atomic` + `Set`.** Smallest new surface; one token per item needs
   no pooling machinery.
2. **`IID`** + variable-`n` training + the contraction tests below.
3. **`Sequence`** (+ patching, long-series cost control).
4. **Nesting**, e.g. subjects × areas × time.

## 8. Tests

- Legacy checkpoint loads and reproduces stored outputs bit-exactly.
- Permutation invariance: shuffling trials (`IID`) or items (`Set`) leaves the
  score unchanged to float tolerance.
- Variable `n`/`T`: padding to different lengths gives identical results.
- Conjugate-Gaussian analytic check (in the style of
  `tests/test_multiobs_analytic.py`): with an `IID` block the posterior standard
  deviation must track `1/sqrt(n)` across `n` unseen in training.
- Missing-block masking: dropping a block changes the posterior but does not
  NaN, and gradients do not flow from absent blocks.
- Integration with `MultiObsSampler` + `gauss_jacobian` and with
  `newton_map_estimate`, per-observation contexts held constant.
- Attention memory: extend `tests/test_memory_bounded_attention.py` — context
  tokens lengthen the sequence and the trunk cost is quadratic in it.

## 9. Risks

- `MultiObsSampler` is ~3.9k lines with 125 `condition_mask` references; the
  flat-state boundary in §2 is what keeps it out of scope, and it must be
  verified rather than assumed.
- Checkpoint compatibility is the most likely source of silent breakage — pin it
  with a stored-output regression test in phase 0, before any new block exists.
- Sequence length: every context token is another key for every trunk block.
  Prefer small `k` and patching over dumping raw steps into the trunk.
- Encoder capacity is not free: a pooled block can only ever be conditioning,
  so a block declared `IID` can no longer be imputed or evaluated under
  `log_prob(x | theta)`. Make that explicit at the API level.
