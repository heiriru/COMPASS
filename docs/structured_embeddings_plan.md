# IID (partial-pooling) embeddings for COMPASS — experimental ladder

## 0. Scope

Exactly **two** observation representations are in scope. Everything else
(ordered sequences, unordered identified sets, nesting) is out of scope and is
not to be built, stubbed, or designed around.

1. **Legacy** — COMPASS exactly as it is today: one token per scalar dimension,
   position-specific parameters, `nodes_size`-shaped everything.
2. **IID** — exchangeable trials within one observation unit, in three
   progressively more aggressive representations that are climbed **in order**,
   each only if the previous one is insufficient:
   `trial_tokens` -> `fixed_summary` -> `predictive_summary`.

The ladder below is an experimental protocol, not a refactor plan: every stage
is a measurement that can fail independently without contaminating the stage
before it. Do not skip ahead — in particular, do not build any summary
representation until `trial_tokens` has been measured against the 5M legacy
model, because it may already win, in which case summaries become an optional
scalability feature rather than a fix.

## 1. Where we stand (measured, not estimated)

Partial-pooling layout today: `nodes_size = 100` = 7 shared + 3 local `theta`
dims + 30 trials x 3 features flattened into 90 scalar nodes
(`Partial_Pooling/schema.py:transformed_observations`, `config.py` presets).

Actual parameter breakdown of `ConditionTransformer` at that layout:

| preset | H | depth | total | block adaLN | final layer | everything else |
|---|---|---|---|---|---|---|
| `compact` | 4 | 1 | 867,324 | 616,800 (71%) | 245,600 | 4,924 |
| `small` | 16 | 2 | **5,957,984** | 4,934,400 (83%) | 982,400 | 41,184 |
| `large` | 128 | 6 | 127,748,608 | 118,425,600 (93%) | 7,859,200 | 1,463,808 |

For the 5.96M `small` model, **99.3% of the parameters are position-specific**:
`adaLN_modulation: Linear(256, 6*N*H)` in each block plus the
`(N*H, N)` matrix in `FinalLayer`. The entire data-dependent pathway (attention,
MLP, input embedding) is **8,096 parameters**. The `config.py` docstring for
`compact` already records the consequence: at `nodes_size=100`, depth=2 at any
small hidden size blows past 1M purely on modulation cost.

That is the headline: the current parameter budget is spent almost entirely on
a function of the diffusion time `t` that is replicated per node.

Two further structural facts that shape the ladder:

- `key_padding_mask = (1 - c)` (`ConditionTransformer.py:157`) — only conditioned
  tokens are used as attention keys. Whatever tokenization is chosen must
  produce a mask under that same convention.
- There are only **9 call sites** into the network (`Trainer.py:327`,
  `Sampler.py:285/291/299`, `PFODE.py:170/176`,
  `MultiObsSampler.py:1430/3135/3141`), so the blast radius of an architecture
  change is small — provided the diffusion state stays a flat vector.

## 2. Stage 0 — Freeze the current state

Before touching anything.

- Preserve the existing checkpoints (`Partial_Pooling/artifacts/checkpoints/small/`
  = 5,957,984 params; the sub-1M `compact` run) together with the exact test
  simulations, normalization constants, seeds, parameter counts, training
  curves, sampling metrics, and PFODE/likelihood metrics where available.
- Store model outputs for a fixed set of example inputs (score at several fixed
  `(x, t, condition_mask)` triples, plus fixed-seed posterior draws).
- Add a regression test asserting that `arch="legacy"` reproduces those stored
  outputs bit-for-bit, or within an extremely tight float tolerance.
- Freeze the benchmark itself so later stages are compared on identical data.

Exit criterion: the regression test passes on a clean checkout and is wired
into `tests/`.

## 3. Stage 1 — New API, unchanged statistics

Add the schema-based constructor while changing nothing about the
representation:

```python
ScoreBasedInferenceModel(theta_size=..., x_spec=[Atomic("x", dim=...)], arch="tokenized")
ScoreBasedInferenceModel(nodes_size=..., arch="legacy")   # unchanged, still default for this signature
```

- Checkpoints store `arch_version` and the schema; `load` dispatches on them.
- **A legacy model must never silently enter the new implementation.** Loading a
  checkpoint without `arch_version` yields `arch="legacy"`; requesting
  `arch="tokenized"` for such a checkpoint is an error, not a conversion.
- At this stage `arch="tokenized"` with a single `Atomic` block is
  mathematically the same model as legacy — the point is that the plumbing and
  the regression test are in place before the trunk changes.

## 4. Stage 2 — Token-count-agnostic trunk, still raw scalar tokens

This isolates the parameter-efficiency question from the IID question. Three
changes, no permutation invariance, no summary network:

- `LayerNorm((N, H))` -> `LayerNorm(H)`
- `adaLN: Linear(256, 6*N*H)` -> `Linear(256, 6*H)`, shared across tokens
- `FinalLayer`'s `(N*H, N)` matmul -> token-wise output head

Projected budget after this change (mlp_ratio=4, time-embedding width 256),
now **independent of the token count**:

| H | depth=2 | depth=3 | depth=4 | depth=6 |
|---|---|---|---|---|
| 16 | 99k | 127k | 155k | 210k |
| 32 | 211k | 273k | 335k | 459k |
| 48 | 339k | 441k | 543k | 747k |
| 64 | 480k | 628k | 777k | 1.07M |
| 96 | 806k | 1.07M | 1.33M | 1.84M |
| 128 | 1.19M | 1.58M | 1.98M | 2.77M |

Measure how much performance is retained by removing position-specific
parameters alone. This is the stage that decides whether the sub-1M target is
about capacity at all, or only about where the capacity was being spent.

Open question to settle empirically here: whether a small per-token learned
offset on the shared modulation buys anything back. Test it as an ablation, not
as a default.

## 5. Stage 3 — One token per complete trial (lossless)

The reference IID architecture, and the one to test before any compression.

For partial pooling, `x_i = (choice_i, log RT_i, censored_i)`. One shared trial
embedder `phi: R^3 -> R^H`, identical for every trial, **no positional
embedding**. Thirty trials become thirty tokens: 90 scalar trial tokens -> 30
trial tokens, and the node vector goes from `7 + 3 + 90 = 100` tokens to
`7 + 3 + 30 = 40`.

Crucially, **the diffusion state stays the three raw variables per trial**. A
shared trial output head `H -> 3` emits `(s_choice_i, s_logRT_i, s_censor_i)`.
So this still models the raw joint `p(theta, X)` and still supports both
`p(theta | X)` and raw `p(X | theta)`. Nothing is thrown away; the three
features of a trial simply stay together, so the model knows which RT belonged
to which response, and permuting whole trials permutes their representations.

What this stage does and does not buy:

- It buys **permutation symmetry over trials** and a 2.5x reduction in sequence
  length (attention cost is quadratic in it).
- After Stage 2 it does **not** buy much parameter reduction — the trunk is
  already token-count independent. Do not conflate the two effects when
  reporting results; the token reduction is an inductive-bias and compute
  argument, and it should be measured as such.

Masking decision to pin down explicitly: with grouped tokens a trial is
**observed or latent as a whole** (`condition_mask` becomes group-level for IID
blocks, and `Trainer`'s Bernoulli(0.33) is drawn per trial rather than per
scalar). Partially-observed trials are possible only by feeding the per-dim mask
into `phi` as an extra input channel — treat that as an optional extension, and
only if the benchmark actually needs it.

## 6. Stage 4 — Parameter-budget search on the lossless IID architecture

Search before compressing. Sweep `H in {16, 32, 48, 64}` and `depth in {2, 3, 4}`,
discarding any configuration with >= 1M trainable parameters. Deliberately
include very small budgets — 250k, 400k, 600k, 800k — because the correct
inductive bias may make the old parameter count unnecessary.

The comparison table:

| Model | Representation | Parameter target | What it tests |
|---|---|---|---|
| Legacy-small | current | ~500k–870k | current floor |
| Legacy-large | current | ~5.96M | target to beat |
| Tokenized-raw | 100 scalar tokens | <1M | efficient trunk only |
| IID-trial | 30 trial tokens | <1M | efficient trunk + correct symmetry |
| IID-trial-small | 30 trial tokens | as low as possible | actual efficiency limit |

**Do not proceed to Stage 5 until it is known whether IID-trial already beats
the 5.96M legacy model.**

## 7. Stage 5 — Fixed-dimensional IID summaries

Only if Stages 3–4 leave a gap. This changes the problem from raw `X` to
`z = E(X)`, and COMPASS learns `p(theta, z)`.

Start with the most model-independent version: a **fixed** permutation-invariant
distribution embedding

    z = (1/N) sum_i phi_fixed(x_i),   with log N appended explicitly.

Random Fourier / kernel features or another rich fixed feature bank are
attractive precisely because there is no encoder to train and the representation
can stay permanently fixed as new simulation models are added. Test
`D in {8, 16, 32, 64}`.

```python
IID(name="trials", item_dim=3, representation="fixed_summary",
    summary_dim=32, include_log_count=True)
```

COMPASS then sees 33 stochastic observation dimensions instead of 90. Note that
this makes the encoder a **preprocessing step**, not a network component: the
node vector is just `[theta, z]` atomic tokens, so no new trunk machinery is
needed and Stage 2's trunk carries it unchanged.

Posterior inference is `p(theta | z_obs)`; the likelihood is `p(z_obs | theta)`.
**The API must report explicitly that this is a summary-space likelihood, not
raw `p(X | theta)`** — in the returned metadata and in the docstring, not just
in a comment.

## 8. Stage 6 — Learned predictive IID encoder

Only if fixed summaries lose too much information. **Do not start from an
ordinary autoencoder** — use a held-out-trial predictive objective.

Split one IID dataset `X = C u T`, encode the context trials `z = E_psi(C)`, and
train a decoder to maximize `sum_{x_i in T} log q_omega(x_i | z)`. This forces
`z` to describe the underlying trial distribution rather than memorize the exact
trials it saw.

Encoder: `x_i -> phi_psi(x_i) in R^H`, then permutation-invariant PMA /
cross-attention. Set the number of PMA queries equal to the summary dimension
(`K = D`); each query returns an `H`-vector followed immediately by a scalar
head `r_j in R^H -> z_j in R`. Externally the encoder is simply
`E_psi: R^{N x 3} -> R^D` — no intermediate `K x H` representation appears in
the public API.

After pretraining: **freeze the encoder, compute the shared `z`-normalization,
then** train COMPASS.

Encoder ablation — DeepSets vs PMA:

    DeepSets:  z = rho( (1/N) sum_i phi_psi(x_i), log N )
    PMA:       z = E_PMA( {phi_psi(x_i)}, log N )

If DeepSets performs equally well, use it: simpler and cheaper. PMA earns its
complexity only if it materially improves partial pooling, particularly on
quantities driven by tails or choice–RT interactions.

## 9. Model comparison as a separate validation axis

Keep it distinct from posterior inference.

- Lossless trial-token models: keep evaluating raw `p(X | theta)`.
- Summary models: evaluate `p(z | theta)`.
- When comparing simulation models, every model using summary likelihoods must
  use the **same frozen `E`**. They need not be trained together: models A, B,
  and a future C each get their own independent COMPASS network after the same
  frozen observation encoder is applied.

## 10. Metric contract — define before running anything

Primary metric: **posterior accuracy**, with **calibration as a hard
constraint**. Track, at minimum:

- standardized parameter error, reported **separately for global and local**
  parameters;
- posterior coverage / calibration;
- distributional distance to any available oracle or reference posterior;
- PFODE / log-density error where a reference exists;
- training stability across several seeds;
- parameter count, peak memory, inference runtime.

For model-comparison experiments additionally: correct-model selection rate,
model-posterior calibration, and log-Bayes-factor error against the best
available reference.

Success criterion for the final model:

> **better than the 5.96M legacy model on the primary partial-pooling metric,
> with no material calibration regression, at <1M parameters.**

If a 300k model achieves that, prefer it over an 800k model. Do not declare a
winner on a single posterior metric.

## 11. Expected sequence of gains

The biggest jump is expected from

    legacy -> shared tokenized trunk -> one shared token per IID trial

which attacks the two measured inefficiencies directly: position-specific
network parameters (99.3% of the `small` model) and three tokens for one
logically inseparable trial. Only after that is measured should `X -> z` be
introduced — and then the price of compression can be quantified directly
against the lossless trial-token reference at `D = 64 -> 32 -> 16 -> 8`, which
also answers empirically how small `D` can be.

## 12. API end state

One IID interface, progressively more aggressive representations:

```python
IID("trials", item_dim=3, representation="trial_tokens")                       # lossless
IID("trials", item_dim=3, representation="fixed_summary",  summary_dim=32)     # fixed compression
IID("trials", item_dim=3, representation="predictive_summary",
    summary_dim=32, pooling="pma")                                             # learned compression
```

The old tensor-based `nodes_size` API stays routed to `arch="legacy"` and
untouched.

## 13. Risks

- **Silent legacy contamination** is the main failure mode; Stage 0's stored-output
  regression test plus the hard `arch_version` dispatch in Stage 1 are what
  prevent it, and both must land before any trunk change.
- `MultiObsSampler` is ~3.9k lines with 125 `condition_mask` references. Keeping
  the diffusion state a flat raw vector through Stage 4 is what keeps it out of
  scope — verify that with the composition tests
  (`gauss_jacobian`, `newton_map_estimate`) rather than assuming it.
- Group-level masking changes what the model is trained to condition on; the
  Stage 3 result is only comparable to legacy if the marginal rate of observed
  scalars is matched.
- Stage 2 changes parameter shapes, so no Stage-2+ checkpoint can ever load into
  a legacy model or vice versa. That is intended, and the loader must say so
  with a clear error.
