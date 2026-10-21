# IID-embedding ladder — running record

Companion to `docs/structured_embeddings_plan.md`, which stays the authority on
what each stage is for. This file records what was actually built, measured and
decided, stage by stage.

All ladder arms are trained and evaluated on the **preset `small`** benchmark:
the same 400,000-simulation split, the same normalization constants and the same
100 test datasets the frozen 5,957,984-parameter baseline used. Ladder arms see
only the **first 100,000** of those simulations (`--train-subset 100000`, with
the validation split scaled to 5,000 to match), because the target is to beat
the 5.96M model while training on 100,000 samples.

---

## Stage 0 — freeze (done)

**What was frozen.**

* `tests/fixtures/legacy_arch_regression.pt` — fixed-seed construction *and*
  forward pass of `arch="legacy"` at four shapes (`tiny`, `compact`, `small`,
  `large`), over six `(x, t, condition_mask)` triples chosen to cover the mask
  regimes that matter: all-latent (which has no attention key at all and only
  survives because of `add_bias_kv`), all-conditioned, the posterior layout, the
  likelihood layout, and two Bernoulli(0.33) training masks. Depends on nothing
  but the source tree.
* `tests/fixtures/legacy_checkpoint_regression.pt` — the trained 5,957,984-parameter
  `small` checkpoint through **all four network entry points**: the raw score
  (Trainer's path), fixed-seed posterior draws (`Sampler`), PF-ODE log-densities
  (`PFODE`), and a fixed-seed `gauss_jacobian` compositional draw
  (`MultiObsSampler`).
* `Partial_Pooling/artifacts/frozen/stage0/manifest.json` — parameter counts and
  their position-specific breakdown, the checkpoint and normalization SHA-256
  digests, and a copy of the baseline recovery metrics.
* `Partial_Pooling/artifacts/frozen/stage0/baseline_metrics/` — the four
  completed 100-dataset recovery runs of the 5.96M model.

**Regression test.** `tests/test_legacy_arch_regression.py`, comparison
**bit-for-bit** (`torch.equal`), passing. Reproducibility needed one thing the
plan did not anticipate: BLAS reduction order depends on the thread count, so
both the fixture generator and the test pin `torch.set_num_threads(1)` and
`torch.use_deterministic_algorithms(True)`. Without that the checkpoint outputs
drift by ~3e-6 between a 3-thread and a default-thread process — a real effect,
but not a model change. `COMPASS_REGRESSION_TOL` relaxes the comparison to a
tolerance for a genuinely different torch or BLAS build.

**Measured parameter breakdown** (confirms the plan's table exactly):

| preset | H | depth | total | block adaLN | final layer | position-specific |
|---|---|---|---|---|---|---|
| `compact` | 4 | 1 | 867,324 | 616,800 | 245,600 | 99.4% |
| `small` | 16 | 2 | **5,957,984** | 4,934,400 | 982,400 | 99.3% |
| `large` | 128 | 6 | 127,748,608 | 118,425,600 | 7,859,200 | 98.9% |

**Frozen baseline metrics** (5.96M model, 400,000 training simulations; 100 test
datasets x 20 subjects x 256 draws x 50 steps; MAP errors standardized by each
parameter's prior standard deviation, so 1.0 means "no better than the prior"):

| inference method | global std-RMSE | local std-RMSE | global cov95 | local cov95 | sampling s |
|---|---:|---:|---:|---:|---:|
| DPM2 + gauss_jacobian, churn 2, refresh 20 | 0.8441 | 0.3361 | 0.901 | 0.947 | 7,579 |
| DPM2 + gauss_hierarchical, churn 2 | 1.0102 | 0.3544 | 0.920 | 0.941 | 3,734 |
| Langevin + F-NPSE | 0.7805 | 0.3411 | 0.786 | 0.936 | 8,293 |

**Pre-existing test failures** at the freeze, on an otherwise untouched tree:
23 failures in `test_damping_composition.py`, `test_instantaneous_divergence.py`,
`test_multiobs_analytic.py` and `test_pfode_vpsde_analytic.py` (they reference a
divergence-head API this branch does not carry), plus one pytest collection error
on `Partial_Pooling/artifact_names.py:test_data_path`, which is a function, not a
test. None are touched by the ladder; the number is recorded so a regression is
distinguishable from the inherited state. The composition tests the plan names as
the guard rail — `test_gauss_jacobian_composition`, `test_gauss_jacobian_integration`,
`test_newton_map`, `test_gauss_hierarchical_composition` — all pass.

---

## Stage 1 — schema API (done)

`src/compass/schema.py` adds `Atomic`, `IID`, `NodeSchema` and `GroupedBernoulli`;
`src/compass/TokenizedTransformer.py` adds the schema-driven network.

```python
ScoreBasedInferenceModel(nodes_size=100)                       # arch="legacy" (default)
ScoreBasedInferenceModel(nodes_size=100, arch="tokenized")     # 100 scalar tokens
ScoreBasedInferenceModel(theta_size=10,                        # 10 + 30 = 40 tokens
                         x_spec=[IID("trials", item_dim=3, count=30)])
```

**Dispatch guarantees** (`tests/test_schema_api.py`, 24 tests):

* `nodes_size=` alone still means legacy, and its checkpoints still carry **no**
  `arch_version` field — so old and new checkpoints stay distinguishable by
  absence, and every existing checkpoint keeps loading.
* Asking for `arch="tokenized"` on a checkpoint without `arch_version` raises;
  asking for `arch="legacy"` on a tokenized one raises. Never a conversion.
* `arch="tokenized", trunk="position"` with a single atomic block is proved to
  be *the same model* as legacy: same parameter count, key-compatible state
  dict, and `torch.equal` outputs after loading legacy weights into it.
* `GroupedBernoulli` on an all-atomic schema draws the **identical** mask as the
  old `Bernoulli(0.33)`, not merely one from the same distribution, so legacy
  training stays reproducible.

**Benchmark plumbing.** `Partial_Pooling/config.py` gains `arch`, `tokenization`,
`trunk`, `token_offset` and `train_subset`. All five are omitted from
`model_dict()` when at their legacy defaults, so **every existing checkpoint
signature is unchanged** (pinned by `test_ladder_contracts.py`), and none of them
enters `data_dict()`, so every arm reuses the same simulations, normalization and
test datasets. Checkpoint names gain a fragment per non-default axis
(`sde_joint-tok-trial-h16d2x2m4-sub100000-train-400000`), leaving legacy names
byte-identical.

One latent bug fixed on the way: `Trainer` writes `{name}_checkpoint.pt` while
the benchmark's canonical name is `{tag}-checkpoint.pt`, so a freshly trained
model was not loadable without a separate migration pass. `train_models` now
renames it.

---

## Stage 2 / Stage 3 — the two trunks

Both are implemented in `TokenizedTransformer` and selected by
`--tokenization {scalar,trial}`.

**Stage 2 (`scalar`)** — the three changes the plan asks for: `LayerNorm(H)`
instead of `LayerNorm((N, H))`, one `Linear(256, 6H)` modulation shared by all
tokens, and a token-wise output head instead of the `(N*H, N)` matmul.

One change the plan did not call out is *required* by the first of those: under
`LayerNorm(H)`, the legacy multiplicative embedding `x_n * w_n` normalizes to
`sign(x_n)` times a fixed direction, discarding every token's magnitude. The
embedding is therefore affine, `x_n * w_n + b_n`, with both initialized from
`N(0, 1)` (the legacy all-ones initialization would make every token identical
under a per-token LayerNorm). Cost: `N*H` extra parameters, 1,600 at H=16.

**Stage 3 (`trial`)** — `Atomic("theta", 10)` plus `IID("trials", item_dim=3,
count=30)`: one shared `Linear(3, H)` embedder, no positional embedding, 100
tokens become 40. The diffusion state stays the 90 raw trial scalars and a
shared `Linear(H, 3)` head emits them, so the model still represents the raw
joint `p(theta, X)` and still supports raw `p(X | theta)`. Nothing downstream
changed: all nine call sites take a flat vector, verified end to end in
`tests/test_tokenized_stack_integration.py` (Sampler, PFODE, `gauss_jacobian`,
`gauss_hierarchical`, F-NPSE, and `newton_map_estimate(curvature="jacobian")`,
which pushes forward-mode AD through the new embedder).

Masking, as decided in the plan: a trial token is a valid attention key only if
**all three** of its scalars are conditioned, and `Trainer` draws its
Bernoulli(0.33) **per trial**. The marginal rate of observed *scalars* is still
0.33, which is what keeps the grouped arm comparable with the legacy one.

**Parameter budgets** (`Partial_Pooling/report_ladder_budgets.py`, full grid in
`artifacts/frozen/stage0/ladder_budgets.json`). Totals no longer depend on the
token count:

| H | d=2 | d=3 | d=4 | d=6 |
|---|---:|---:|---:|---:|
| 16 | 97,604 | 125,524 | 153,444 | 209,284 |
| 32 | 207,236 | 269,220 | 331,204 | 455,172 |
| 48 | 329,156 | 431,348 | 533,540 | 737,924 |
| 64 | 463,364 | 611,908 | 760,452 | 1,057,540 |

(`tokenization="trial"`; the `scalar` grid is 1–3% larger, entirely from the
per-position input embedding.)

At H=16/depth=2 — the shape of the 5.96M legacy model — the two effects the plan
insists on separating come out as:

| | tokens | total | adaLN | final | embed | time-emb | attn+mlp |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 100 | 5,957,984 | 4,934,400 | 982,400 | 1,600 | 33,088 | 6,496 |
| tokenized `scalar` | 100 | 100,369 | 49,344 | 8,241 | 3,200 | 33,088 | 6,496 |
| tokenized `trial` | 40 | 97,604 | 49,344 | 8,292 | 384 | 33,088 | 6,496 |

**59x fewer parameters comes entirely from Stage 2.** Stage 3 buys 2.8% more
parameters — nothing — and buys instead permutation symmetry over trials and a
2.5x shorter attention sequence. The two must not be conflated when reporting.

Worth noting for Stage 4: at these budgets the *time* embedding (33,088
parameters, a `256 -> 4H -> 256` MLP) is a third of the model and does not shrink
with H.


---

## Stages 2 and 3 — measured (100 test datasets x 20 subjects x 256 draws x 50 steps)

All four models below are evaluated on the **identical** protocol, test datasets
and inference settings. Errors are MAP errors standardized by each parameter's
prior standard deviation (1.0 = no better than the prior); calibration is the
mean 95% interval coverage, nominal 0.95.

Held-out denoising loss after training (best epoch, patience 20):

| arm | parameters | tokens | best validation loss |
|---|---:|---:|---:|
| legacy, 5.96M, 100,000 sims | 5,957,984 | 100 | **136.48** |
| tokenized `trial`, 100,000 sims | 97,604 | 40 | 142.26 |
| tokenized `scalar`, 100,000 sims | 100,369 | 100 | 148.80 |

### DPM2 + gauss_jacobian (pilot-free), churn 2, refresh 20

| model | params | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 |
|---|---:|---:|---:|---:|---:|---:|
| legacy 5.96M | 5,957,984 | 400,000 | 0.8441 | 0.3361 | 0.901 | 0.947 |
| legacy 5.96M | 5,957,984 | 100,000 | **0.7996** | **0.3446** | 0.907 | 0.946 |
| tokenized `scalar` | 100,369 | 100,000 | 1.2853 | 0.5613 | 0.887 | 0.978 |
| IID `trial` | 97,604 | 100,000 | 1.0561 | 0.4481 | 0.836 | 0.979 |

### DPM2 + gauss_hierarchical, churn 2

| model | params | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 |
|---|---:|---:|---:|---:|---:|---:|
| legacy 5.96M | 5,957,984 | 400,000 | 1.0102 | 0.3544 | 0.920 | 0.941 |
| legacy 5.96M | 5,957,984 | 100,000 | 0.8250 | 0.3431 | 0.871 | 0.945 |
| tokenized `scalar` | 100,369 | 100,000 | 1.3099 | 0.5613 | 0.869 | 0.979 |
| IID `trial` | 97,604 | 100,000 | 1.0670 | 0.4490 | 0.826 | 0.981 |

### Langevin + F-NPSE

| model | params | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 |
|---|---:|---:|---:|---:|---:|---:|
| legacy 5.96M | 5,957,984 | 400,000 | 0.7805 | 0.3411 | 0.786 | 0.936 |
| legacy 5.96M | 5,957,984 | 100,000 | 0.7918 | 0.3487 | 0.774 | 0.945 |
| tokenized `scalar` | 100,369 | 100,000 | 0.9510 | 0.5604 | 0.710 | 0.982 |
| IID `trial` | 97,604 | 100,000 | 0.9081 | 0.4503 | 0.734 | 0.983 |

### What these say

1. **Training-set size is not the binding constraint.** The 5.96M model trained
   on 100,000 simulations is as accurate as the same model trained on 400,000 --
   marginally better, in fact, on two of the three samplers. So the model to beat
   is the 5.96M model *at 100,000 simulations*, and the ladder is a pure
   architecture question. This retires the concern that a 100,000-simulation
   budget would handicap the comparison.

2. **Stage 2 alone is expensive.** Removing the position-specific parameters at
   H=16/depth=2 -- a 59x parameter cut -- costs 1.5x on global error and 1.7x on
   local error. The plan asked what removing them *alone* retains; the answer at
   this budget is: not enough.

3. **Stage 3 recovers roughly half of that, for free.** At an essentially
   identical budget (97,604 vs 100,369) the trial tokenization takes local
   std-RMSE from 0.561 to 0.448 and global from 1.285 to 1.056, on every one of
   the three samplers, and cuts the denoising loss by 6.5. This is the
   inductive-bias effect the plan predicted, cleanly isolated: the two arms
   differ *only* in whether a trial's three features share a token.

4. **It is not yet enough.** The 98k IID-trial model is still 1.25x (global) and
   1.33x (local) the 5.96M model's error on the primary sampler. Per the plan's
   Stage-4 gate, that means a budget sweep -- not summaries.

5. **The calibration signature is informative.** Both small models *over*-cover
   locally (0.978-0.983 against a nominal 0.95) while *under*-covering globally
   (0.826-0.836). Over-coverage is under-confidence: the local posteriors are too
   wide, which is what a less accurate score produces. It is not a
   mis-specification of the composition rule -- the 5.96M model under the same
   rules sits at 0.94-0.95.

6. **Runtime.** The small models are *slower* per recovery run (12,800 s vs
   7,600 s for gauss_jacobian), not faster. The 100-subject compositional
   machinery, the Jacobian JVPs and the KDE/MAP stages dominate; the network
   forward pass was never the bottleneck at this scale. Parameter count buys
   memory and training cost, not inference time.

---

## Stage 4 — parameter-budget sweep on the lossless IID architecture (done)

Sixteen arms trained, all on the same 100,000 simulations, all with the same
schedule (max 500 epochs, patience 20; every arm converged in 67-80 epochs).
Training an arm costs 10-25 minutes, so the whole grid was trained and ranked on
held-out denoising loss first, and only the candidates paid for the 2-5 hour
recovery evaluation.

### Held-out denoising loss

| arm | representation | parameters | best validation loss |
|---|---|---:|---:|
| legacy H16/d2 | 100 scalar tokens, per-node | 5,957,984 | **136.48** |
| trial H48/d2 | 40 trial tokens | 329,156 | 138.12 |
| trial H32/d2 | 40 trial tokens | 207,236 | 138.33 |
| scalar H64/d4 | 100 scalar tokens | 771,521 | 138.34 |
| trial H48/d4 | 40 trial tokens | 533,540 | 138.74 |
| trial H48/d3 | 40 trial tokens | 431,348 | 138.83 |
| trial H64/d3 | 40 trial tokens | 611,908 | 138.93 |
| trial H64/d4 | 40 trial tokens | 760,452 | 139.02 |
| trial H32/d4 | 40 trial tokens | 331,204 | 139.11 |
| trial H64/d2 | 40 trial tokens | 463,364 | 139.29 |
| trial H16/d4 | 40 trial tokens | 153,444 | 139.84 |
| trial H16/d3 | 40 trial tokens | 125,524 | 139.95 |
| trial H16/d2 + offset | 40 trial tokens | 105,284 | 141.61 |
| trial H16/d2 | 40 trial tokens | 97,604 | 142.26 |
| scalar H16/d2 | 100 scalar tokens | 100,369 | 148.80 |

Three things fall out of this table.

* **Capacity saturates well below the budget.** The 760,452-parameter arm is
  *worse* than the 329,156-parameter one. At 100,000 simulations the binding
  constraint is data, not parameters, which is why the ladder's preference for
  the smaller qualifying model is not merely aesthetic here.
* **Width beats depth.** Every H is best at depth 2 or 3; depth 4 never wins its
  row.
* **Denoising loss is a good screen but not the metric.** It ranks trial H48/d2
  above trial H64/d2, while the recovery evaluation reverses them. Screen on it;
  decide on posterior recovery.

### The `token_offset` ablation — tested, not adopted

The plan's open question for Stage 2 was whether a small per-token offset on the
shared modulation buys anything back. At H=16/depth=2 it buys **0.65** of
validation loss for 7,680 token-dependent parameters. Simply widening to H=32
buys **3.9** for a comparable total. Worse, the offset gives every trial token
its own parameters and therefore **destroys the exchangeability** the trial
tokenization exists to provide -- verified, and pinned as a regression test in
`test_token_offset_ablation_destroys_exchangeability`. Not adopted.

### Recovery evaluation — the decision

Standardized MAP error against `legacy-5.96M/100k` (the fair baseline, since the
5.96M model is no worse at 100,000 simulations than at 400,000). Every arm on the
identical protocol: 100 test datasets x 20 subjects x 256 draws x 50 steps.

**DPM-Solver-2 + gauss_jacobian (pilot-free), churn 2, refresh 20**

| model | params | global std-RMSE | local std-RMSE | global cov95 | local cov95 |
|---|---:|---:|---:|---:|---:|
| legacy 5.96M | 5,957,984 | 0.7996 | 0.3446 | 0.907 | 0.946 |
| scalar H16/d2 | 100,369 | 1.2853 | 0.5613 | 0.887 | 0.978 |
| scalar H64/d4 | 771,521 | 0.8228 | 0.3926 | 0.864 | 0.984 |
| trial H16/d2 | 97,604 | 1.0561 | 0.4481 | 0.836 | 0.979 |
| trial H32/d2 | 207,236 | 0.8193 | 0.4314 | 0.810 | 0.984 |
| trial H48/d2 | 329,156 | 0.7341 | 0.3806 | 0.896 | 0.990 |
| trial H48/d3 | 431,348 | 0.7738 | 0.3565 | **0.927** | 0.990 |
| **trial H64/d2** | **463,364** | **0.7376** | **0.3149** | 0.857 | 0.981 |
| trial H48/d4 | 533,540 | 0.7132 | 0.3422 | 0.849 | 0.985 |
| trial H64/d3 | 611,908 | 0.7451 | 0.3749 | 0.843 | 0.982 |

**DPM-Solver-2 + gauss_hierarchical, churn 2**

| model | params | global std-RMSE | local std-RMSE | global cov95 | local cov95 |
|---|---:|---:|---:|---:|---:|
| legacy 5.96M | 5,957,984 | 0.8250 | 0.3431 | 0.871 | 0.945 |
| trial H16/d2 | 97,604 | 1.0670 | 0.4490 | 0.826 | 0.981 |
| trial H48/d2 | 329,156 | 0.7656 | 0.3827 | 0.886 | 0.989 |
| **trial H64/d2** | **463,364** | **0.7460** | **0.3203** | 0.853 | 0.979 |
| trial H48/d4 | 533,540 | 0.7274 | 0.3425 | 0.831 | 0.983 |

**Langevin + F-NPSE**

| model | params | global std-RMSE | local std-RMSE | global cov95 | local cov95 |
|---|---:|---:|---:|---:|---:|
| legacy 5.96M | 5,957,984 | 0.7918 | 0.3487 | 0.774 | 0.945 |
| trial H16/d2 | 97,604 | 0.9081 | 0.4503 | 0.734 | 0.983 |
| trial H48/d2 | 329,156 | 0.7692 | 0.3961 | 0.801 | 0.994 |
| **trial H64/d2** | **463,364** | **0.7512** | 0.3624 | 0.786 | 0.987 |
| trial H48/d4 | 533,540 | 0.7329 | 0.3739 | 0.794 | 0.990 |

### Verdict

**`trial H64/d2`, 463,364 parameters, 40 tokens, 100,000 training simulations.**

| | vs legacy 5.96M/100k | | |
|---|---|---|---|
| | gauss_jacobian | gauss_hierarchical | F-NPSE |
| global std-RMSE | **x0.923** | **x0.904** | **x0.949** |
| local std-RMSE | **x0.914** | **x0.934** | x1.039 |
| calibration regression | no | no | no |

Better than the 5.96M model on the global parameters under all three samplers,
and on the local parameters under two of the three (4% worse under F-NPSE), at
**12.9x fewer parameters** and with no calibration regression anywhere. Sampling
runtime is comparable or better (4,976 s vs 5,199 s on gauss_hierarchical).

Two runners-up worth recording, because they answer different questions:

* **`trial H48/d3`, 431,348 parameters** — the *best-calibrated model measured,
  the legacy baseline included*: global mean |coverage - 0.95| of 0.029 against
  the 5.96M model's 0.046, with global error still 3% better than the baseline's.
  If calibration is the binding requirement rather than a constraint, this is the
  arm to take.
* **`trial H48/d4`, 533,540 parameters** — the lowest global error of any arm
  (0.7132, x0.892), but its global coverage falls to 0.849 and it trips the
  pre-declared calibration threshold on the primary sampler. Recorded, not
  recommended.

### Resource cost, measured rather than inferred

| | legacy 5.96M | trial H64/d2 | ratio |
|---|---:|---:|---:|
| parameters | 5,957,984 | 463,364 | 12.9x |
| checkpoint on disk | 23.8 MB | 1.9 MB | 12.7x |
| forward pass, 256 x 100 nodes, 1 CPU thread | 87.7 ms | 107.8 ms | 0.81x |
| recovery run, gauss_hierarchical | 5,199 s | 4,976 s | 1.04x |
| recovery run, gauss_jacobian | 11,455 s | 12,152 s | 0.94x |

The parameter and memory savings are an order of magnitude; the **wall-clock
savings are not**, and should not be claimed. The legacy model's parameters sit
almost entirely in a `Linear(256, 6*N*H)` evaluated once per example, which is
cheap per parameter, while the tokenized model spends its smaller budget on
attention and MLP width that is evaluated per token. What the ladder buys is
capacity efficiency and model size, not inference speed.

### The one substantive caveat

Every tokenized arm **over-covers locally**: 0.98-0.99 against a nominal 0.95,
where the legacy model sits at 0.945. Over-coverage is under-confidence -- the
local posteriors come out too wide. It is within the declared tolerance and it is
the conservative direction, but it is systematic across all sixteen arms and all
three samplers, so it is a property of the architecture rather than noise. The
most likely cause is the single shared `H -> 3` trial output head: one head for
thirty trials cannot sharpen an individual trial's contribution the way ninety
per-node outputs could. A per-block-position output head for the theta tokens, or
a small mixture of shared trial heads, is the obvious next ablation. It was not
run because the success criterion was already met and the plan's Stage-4 gate is
about whether to build summaries, not about polishing the trunk.

---

## Stages 5 and 6 — not built, by the plan's own gate

The plan makes both conditional: fixed summaries "only if Stages 3-4 leave a
gap", the learned predictive encoder "only if fixed summaries lose too much
information", and it is explicit that if IID-trial already beats the 5.96M model
then "summaries become an optional scalability feature rather than a fix".

Stage 4 closed that gate: a 463,364-parameter lossless trial-token model beats
the 5,957,984-parameter legacy model on the primary metric, on all three
inference methods, at under half the 1M budget. Nothing about the partial-pooling
benchmark requires compressing `X` to `z`, and doing so would trade away raw
`p(X | theta)` for no measured gain. Neither stage was built or stubbed.

The case for building them later is a different one -- scaling to many more
trials per subject, where the attention cost and the token count, not the
accuracy, become binding. That case is untested here and should be argued on its
own evidence.
