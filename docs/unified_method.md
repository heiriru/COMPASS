# One recipe: churn in the library, `log_sigma` training, and a mask-menu API

Three changes that move settings which were per-benchmark decisions into COMPASS
itself, so the four experiment families (`Partial_Pooling`, `inverse_kinematics`,
`tutorials/.../bimodal_distribution`, `tutorials/.../exponential_score_comparison`)
run the same sampler and the same training recipe rather than four variants of one.

Nothing here changes a composition rule. The rules (`gauss`, `gauss_hierarchical`,
`gauss_jacobian`, `fnpe`) are untouched.

---

## 1. Churn is in `compass` and is the default predictor

`src/compass/churn.py`. Reachable as `churn_eta=` on `SBIm.sample`,
`Sampler.sample` and `MultiObsSampler.sample`; default `churn_eta = 2.0`.

**Why it was worth lifting.** Churn existed in *three* copied implementations
(`Partial_Pooling/run_churn_comparison.py`, `inverse_kinematics/churn.py`,
`tutorials/.../bimodal_distribution/churn.py`, plus a fourth in
`stochastic_predictor_experiments.py`) and **zero** times in `src/compass/`. All
four were monkeypatches over `MultiObsSampler._dpm_sampler`, which is why the
single-observation `Sampler` could not churn at all — the limitation
`inverse_kinematics/README.md` records as the reason its `n = 1` probes had to go
through the multi-observation route.

**Why `eta = 2` is the default.** Corrector-free, shared-marginal W1 in units of
the posterior sigma on the exactly-solvable exponential hierarchy: 2.598 at
`eta = 0`, 0.314 at 1, 0.049 at 2, 0.034 at 4, against 0.057 for DPM2 + 2
Langevin correctors at 2.1x the score calls. With a *learned* score the optimum
moves to 2: 0.036 / 0.019 for the two Gaussian rules against 0.265 / 0.270 for
the same rules with two correctors.

**Correctness.** `tests/test_churn.py::test_library_churn_is_bit_identical_to_the_published_monkeypatch`
pins the library predictor **bit-identical** to the patch that produced every
published churn number in this repository. The K-arm, inverse-kinematics, bimodal
and noise-law tables therefore remain reproducible through the supported path.
The three benchmark-local copies are now thin shims that delegate to the library
and keep their old names, so no script's imports changed.

### Two API consequences

- `corrector_steps` now defaults to `None`, meaning *this method's own default*:
  **0** under `method="dpm"` (churn replaces the correctors) and **5** under
  `method="langevin"`, which is a corrector-only sampler and would otherwise
  become a no-op.
- Asking for churn **and** Langevin correctors together raises. They are
  alternatives, not complements, and no measurement here covers the combination.
  Pass `churn_eta=0.0` for the legacy predictor-corrector sampler.

### What this changed elsewhere

The frozen architecture fixtures (`test_legacy_arch_regression`,
`test_tokenized_arch_regression`) and `test_gauss_jacobian_integration` now pin
`churn_eta=0.0` explicitly. They exist to detect an unintended change to the
*architecture*, not to restate the sampler's current default — and they did their
job: flipping the default is exactly what they caught.

In `exponential_score_comparison/compare.py` every arm named `*_predictor_only`
or `*_correctors*` pins `churn_eta=0.0`. Without that pin the "predictor only"
control — the `eta = 0` baseline the entire initialization story rests on
(W1 2.598 sigma) — would have silently become a churned sampler and stopped
being a control.

---

## 2. Pure churn, no Langevin correctors, in the experiment sweeps

Default sweeps are now corrector-free:

| directory | before | now |
|---|---|---|
| `bimodal_distribution` | churn 8 x 2 rules, correctors 10 x 2 rules, F-NPSE | churn 2 x 2 rules, churn 8 x 2 rules, F-NPSE |
| `exponential_score_comparison` | 7 arms incl. 10- and 2-corrector | churn 2 x 2 rules + F-NPSE |
| `inverse_kinematics` | Gaussian rules already churned; probes used 10 correctors | churned everywhere |
| `Partial_Pooling` | already `--dpm-corrector-steps 0` | unchanged |

The corrector arms remain reachable by name (`--methods dpm2_gauss_jacobian_correctors`
etc.) because they are the controls that established why churn is the default,
and `RESULTS.md` cites them.

**`langevin_fnpe` stays in every sweep.** It is not a Langevin corrector bolted
onto our predictor — it is Geffner et al.'s algorithm and the external baseline
the composition rules are measured against. Removing it would remove the
comparison, not a corrector.

### The honest cost of dropping the corrector arms

On the bimodal sign benchmark **with the exact row score**, the ten-corrector
arms are the two best in the whole table: W1/sigma 0.013 (`gauss_jacobian`) and
0.020 (`gauss_hierarchical`), against 0.043–0.075 for the churned ones, which
leak 34–76 draws of 3,000 into the valley against an exact expectation of 0.075.
That is a real loss and should not be reported as a free simplification.

The reason to drop them anyway is that **the ordering reverses under a learned
score**, which is the only score that ever ships: the corrector arms go to
0.185/0.190 and 0.093/0.127 with the 552K and 5.2M networks — no better with
nine times the capacity, and worse for `gauss_jacobian` — while the churned arms
track the row score down (0.112 → 0.081 → 0.075 and 0.090 → 0.056 → 0.043). A
long Langevin chain equilibrates to the stationary law of a slightly
over-confident field; a predictor integrates that error once.

The single clearest case is `inverse_kinematics/run_bimodal.py`: correctors
*break the reflection symmetry* there, `frac>0` falling 0.503 → 0.385 → 0.240 as
correctors go 0 → 2 → 10. That probe is now churned, which the library change is
what made possible.

---

## 3. `time_sampling="log_sigma"`

Now the default in `inverse_kinematics/train_model.py` and
`exponential_score_comparison/train.py`; `bimodal_distribution` already had it.
COMPASS's own library default is still `"mixture"` and was not changed.

Measured against a closed-form row score on a 552K network: mean relative
row-score error 0.166 → 0.054, and at `lambda = 0.05` 0.440 → 0.088 — the small
network trained this way beats a 5.2M one trained the `"mixture"` way, at no
inference cost. Do **not** also standardize the data: that combination is worse
(0.085 vs 0.054).

**Checkpoint directories are tagged with the recipe** in both directories, with
the legacy `"mixture"` spelling keeping its original untagged path. Two recipes
can therefore never overwrite each other and then be compared. `--time-sampling
mixture` reproduces the existing tables exactly.

**Every checkpoint in those two directories is now stale with respect to the
default.** They were trained under `"mixture"`, so their numbers stand but a
default-settings rerun trains new ones into new directories.

---

## 4. `mask_menu`: choosing the training condition-mask law

```python
model.train(..., mask_menu="six", mask_roles=(global_size, local_size))
```

`"bernoulli"` (default, unchanged) draws each node's flag independently.
`"six"` trains on the six non-degenerate conditionals over a
`(global, local, observation)` partition — `p(g,l|x)`, `p(g|l,x)`, `p(l|g,x)`,
`p(x|g,l)`, `p(x,l|g)`, `p(x,g|l)` — **drawn uniformly, 1/6 = 16.7% each**.
`mask_weights` and `mask_bernoulli_fraction` are there when uniform is not what
is wanted. `inverse_kinematics/train_model.py` exposes it as `--mask-menu`.

The building blocks (`ConditionalMenu`, `SIX_CONDITIONALS`, `hierarchical_roles`)
already existed; what is new is that the choice is one keyword on `train` rather
than a hand-built sampler per benchmark.

### What uniform-six actually does — two effects, in opposite directions

**It fixes the sampling problem, decisively.** Probability of drawing the
deployed posterior mask `p(g,l|x)`:

| benchmark | nodes/tokens | Bernoulli(0.33) | uniform six | gain |
|---|---:|---:|---:|---:|
| bimodal / exponential row | 3 | 0.148 | 0.167 | 1.1x |
| inverse kinematics | 6 | 0.0219 | 0.167 | 7.6x |
| partial pooling (30 trial tokens) | 40 | 6.5e-17 | 0.167 | 2.6e15x |
| partial pooling (flat, 90 scalars) | 100 | 8.5e-46 | 0.167 | 2.0e44x |

The Bernoulli probability decays exponentially in the node count, so the larger
the problem the more complete the failure. At partial-pooling scale the mask
inference actually evaluates is *never drawn even once* in training.

**It creates a weighting problem, and the size of it is measurable.** Equal
*sampling* is not equal *learning*, because the six conditionals do not carry
equal loss. From the measured per-conditional validation losses
(`docs/followup_results.md`, `mask6mix` arm, 463,364 parameters):

| | share of total loss under uniform draws |
|---|---:|
| the three with observations **conditioned** (posterior side) | **12.8%** |
| the three with observations **latent** (generative side) | **87.2%** |

So a uniform menu spends roughly **seven eighths** of its gradient learning to
generate `x`. To split the gradient evenly, upweight the posterior side by about
**6.8x** — `mask_weights=[7, 7, 7, 1, 1, 1]`.

This is why `Partial_Pooling` uses `six_mixed` (75% menu, 25% Bernoulli) with
`mask_posterior_weight` rather than the uniform menu: on that benchmark uniform
`six` was measured *worse* than the control on `p(g,l|x)` (97,375 vs 95,263)
while the weighted variants were better (94,885 at pw4, 94,443 at pw8, 94,056 at
`six_mixed`).

**So the prediction for the new arms is specific and falsifiable:** uniform
`six` should help most where the Bernoulli gain is largest *and* the generative
conditionals are cheapest to learn. That is inverse kinematics (7.6x gain, only
2 observation nodes to generate) far more than the bimodal/exponential rows
(1.1x gain — Bernoulli already draws the posterior mask 15% of the time there,
so essentially nothing should change). Partial pooling has the largest gain of
all but already has the measurement, and there uniform lost to weighted.

Nothing in this section is a result yet: it is what the arithmetic and the
existing measurements imply. The arms have not been trained.

---

## Status

Implemented and tested. **No new training or inference runs have been made**, so
every number quoted above is from the existing record.

- `tests/test_churn.py` — 9 tests, all passing, including the bit-identity pin.
- `tests/test_mask_menu_api.py` — 10 tests, all passing.
- Full suite: 13 failures, all pre-existing and unrelated (`Sampler.sample()`
  `equation=`, `divergence_head_trained`, and a stale `SimpleNamespace` stub at
  `Trainer.py:297`). Before these changes the same suite failed 20; the extra 7
  were the fixtures that have since been pinned.
- `tests/test_multiobs_analytic.py` requires CUDA and was not run: every GPU on
  this host was held by another user.
