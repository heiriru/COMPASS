# Shrinking COMPASS: testing the proposed follow-ups

Running record for the follow-up steps proposed on 2026-09-07, run under the
constraint **nothing under `src/compass/` is modified**. Everything here is
either a measurement or an out-of-src patch; each section says which.

New code, all outside the library:

| file | what |
|---|---|
| `inverse_kinematics/experiments/te_rewire.py` | makes `time_embedding_size` reachable on the legacy trunk |
| `inverse_kinematics/experiments/test_te_rewire.py` | 13 contract tests, all passing |
| `inverse_kinematics/experiments/budget_table.py` | parameter budget and data-dependent share across the te grid |
| `inverse_kinematics/experiments/train_te.py` | trains one rewired arm through `train_model.py`'s own code path |
| `inverse_kinematics/experiments/grade_te.py` | grades the sweep through `run_all_local_inference.py`'s own harness |
| `Partial_Pooling/experiments/validation_rng_test.py` | tests the validation-RNG diagnosis *and* the proposed fix |

---

## 1. The time-embedding lever on the legacy trunk — three defects, not one

The proposal named one blocker. There are three, and they compound. All three
are verified in `test_te_rewire.py`, which is written so each defect is pinned
by its own test rather than asserted in prose.

**(a) The argument is unreachable.** `ScoreBasedInferenceModel.py:136` sets
`self.time_embedding_size = 256 if arch == "legacy" else int(...)`, and the
legacy branch at `:171-175` never forwards it to `ConditionTransformer` at all.
Passing `time_embedding_size=64` to a legacy model silently yields a 256-wide
one — no error, no warning.

**(b) Reaching past it builds a model that cannot run.**
`ConditionTransformer.__init__` (`:243-246`) forwards `time_embedding_size` to
`t_embedder` only; `TransformerBlock` and `FinalLayer` keep their own `256`
defaults. The model constructs, reports a plausible parameter count, and raises
on the first forward pass:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x64 and 256x576)

This is worse than the argument being unwired, because a parameter-budget script
that never runs a forward pass would report the shrunk numbers as if they were
real.

**(c) A legacy checkpoint cannot record the width.**
`architecture_payload()` returns `{}` for `arch="legacy"` **by design** — a
legacy checkpoint must stay byte-identical to what it always was — so
`time_embedding_size` is never written however it is set, and `SBIm.load`
rebuilds at 256 and raises on the state dict. The width has to be read off the
weights instead (`te_rewire.width_of`, from the input dimension of
`blocks.0.adaLN_modulation.1.weight`). Any real fix in `src` has to decide this
deliberately: either write the field for legacy (which changes what "no
`arch_version`" means) or keep inferring it.

`te_rewire.rewire()` does the wiring after construction, on an ordinary legacy
model, so the library's own `ConditionTransformer.forward` is what is being
tested. **`te=256` is proved to be the stock model bit-for-bit** — same
parameter count, same state-dict keys, `torch.equal` outputs after loading stock
weights — so the sweep's control is a genuine control rather than a
differently-built model.

One trap the tests had to absorb: a fresh legacy model emits **exactly zero**
for every input, because adaLN-Zero zeroes every block's modulation *and*
`FinalLayer.embedding_params` is `torch.zeros(...)` and is never touched by
`_basic_init` (which only visits `nn.Linear`). Any behavioural assertion on a
freshly built model passes vacuously. `_wake()` breaks both zero-inits before
anything behavioural is asserted. This is the same class of trap
`docs/group_block_progress.md` records for the grouped trunk, and it applies to
the legacy trunk too.

### What the lever is worth, at N = 6

The diagnostic that matters here is not the total but the **data-dependent
share** — the parameters that can respond to the observation at all, as opposed
to adaLN modulation and the timestep MLP, which are functions of diffusion time
alone. Shrinking `te` removes only time-only parameters, so the share rises as
the model shrinks.

| config | te | total | time-only | data-dependent | share | vs te256 |
|---|---:|---:|---:|---:|---:|---:|
| h16d2 | 256 | 367,136 | 362,080 | 5,056 | 1.4% | 1.00x |
| h16d2 | 128 | 186,784 | 181,728 | 5,056 | 2.7% | 1.97x |
| h16d2 | 64 | 96,608 | 91,552 | 5,056 | **5.2%** | 3.80x |
| h16d2 | 32 | 51,520 | 46,464 | 5,056 | **9.8%** | 7.13x |
| h16d2 | 16 | 28,976 | 23,920 | 5,056 | 17.4% | 12.67x |
| h64d4wide | 256 | 3,163,008 | 2,828,800 | 334,208 | 10.6% | 1.00x |
| h64d4wide | 64 | 1,049,280 | 715,072 | 334,208 | 31.9% | 3.01x |
| h64d4wide | 32 | **696,992** | 362,784 | 334,208 | **48.0%** | 4.54x |

The `h64d4wide` row is the interesting one. `mask_capacity_ablation.json` shows
that arm buying real accuracy with its 334,208 data-dependent parameters
(endpoint 0.0756 vs the baseline's 0.0933, single-observation `sd` 0.786 vs
0.37). At te=32 **every one of those parameters is retained** while the model
drops from 3,163,008 to 696,992 — 4.5x — because only time-only parameters are
removed. If precision is preserved there, that is a much stronger result than
anything the tokenized trunk could offer here, and it is the arm the sweep is
really about.

**Not yet established.** The table is arithmetic, not evidence. Whether the
network *needs* the width is exactly what `te64` disproved-then-reproved on
partial pooling: 58% fewer parameters, validation loss unchanged to 0.05%, and
global NRMSE 45% worse. The sweep below is what decides it, and it is graded on
recovery, never on validation loss.

---

## 2. The validation-RNG diagnosis: confirmed, and the fix works

`docs/group_block_progress.md` inferred from a correlation (r = +0.85..0.88
between K=5 and K=20, all four arms peaking at epoch 48) that
`Trainer._run_epoch` draws the validation diffusion times and noise from the
global stream with no per-epoch reseed. `Partial_Pooling/experiments/validation_rng_test.py`
tests that directly, and tests the proposed fix, by wrapping `_run_epoch` from
outside the library so the validation pass runs under a saved-seeded-restored
RNG state — the same save/seed/restore `_fixed_validation_masks` already applies
to the masks, extended to the times and the noise.

Six runs, 60 epochs each, on the 6-node arm: two architectures (h16d2, h32d3)
and two initialization seeds, unpatched and patched.

| | up-steps | selection dip | best epoch |
|---|---:|---:|---:|
| unpatched h16d2 | 0.441 | 1.07 sd | 45/60 |
| unpatched h32d3 (different architecture) | 0.458 | 1.26 sd | 45/60 |
| unpatched h16d2, init seed 913 | 0.441 | 1.31 sd | 45/60 |
| **patched** h16d2 | **0.000** | 0.19 sd | 60/60 |
| **patched** h32d3 | **0.000** | 0.12 sd | 60/60 |
| **patched** h16d2, init seed 913 | **0.000** | 0.15 sd | 60/60 |

*up-steps* is the fraction of epochs on which validation loss **rises**;
*selection dip* is how far below its own local (running-median) trend the
selected minimum sits, in units of that run's wobble sd.

Three things follow.

* **Confirmed, and stronger than reported.** Unpatched, 44-46% of epochs move
  the wrong way and all three runs — two architectures, two seeds — take their
  minimum at *exactly* epoch 45. Patched, the curve is **monotone**: not one
  epoch in sixty rises, in any arm.
* **Early stopping was buying about 1.2 sd of luck.** The unpatched minimum sits
  1.07-1.31 wobble-sd below its own trend; patched, 0.12-0.19. That gap is the
  optimistic bias in every reported best-validation number, and it is what makes
  a screen on validation loss unreliable independently of the te64 problem.
* **A caveat the fix creates.** With a fixed validation seed the loss becomes a
  deterministic function of the model, so the curve descends monotonically and
  *best epoch = last epoch* until genuine overfitting starts. Patience then means
  something different from what it means today, and every existing patience
  setting was tuned against the noisy curve. A fix in `src` must re-tune it.

**Do not read this as reproducing the partial-pooling numbers.** It reproduces
the *mechanism* on a cheap 6-node problem; the r = 0.87 and epoch-48 figures
belong to the partial-pooling arms and were not re-run. The correlation
statistics from the first draft of this script are deliberately not quoted:
after the fix both curves are smooth monotone descents of the same task and stay
correlated at r ~ 0.96 for reasons that have nothing to do with the RNG. A
correlation between two runs is the wrong instrument here; the within-run
statistics above are the right one.

---

## 3. The inverse-kinematics te sweep — running

Six arms, all on `train_model.py`'s own path: legacy trunk, 10,000 simulations,
`mask_menu="six_weighted"`, `time_sampling="log_sigma"`, seed 7, patience 1,200,
cap 10,000 epochs. The 10k budget and the weighted menu are the current best
recipe on this benchmark, and 10k is where the model is data-saturated while
retaining mode coverage.

* `h16d2` at te ∈ {256, 128, 64, 32} — te=256 is the control.
* `h64d4wide` at te ∈ {256, 32} — the contrast that keeps all 334,208
  data-dependent parameters while cutting the model 4.5x.

Grading is `run_all_local_inference.py`'s own harness through `grade_te.py`,
which patches only the checkpoint loader. Per the lessons already paid for here:
**512 test datasets, not 128** (the n=128 coverage envelope is +-0.09 at the 50%
level), and **two test seeds**, because the earlier deficit "reproduced" only
because every run reused the same 128 datasets. Decision metrics are endpoint
error, posterior-mean RMSE, coverage against the re-simulated envelope, and the
ABC `W1/sigma` — **not** validation loss.

Results to follow.
