# Group block — running record

Companion to `.claude/plans/refactored-percolating-map.md`. What was built,
measured and decided; steps 1–2 of that plan's order of work.

## Step 1 — freeze the tokenized architecture (done)

`arch="legacy"` was frozen before the ladder started. Every result in
`docs/ladder_progress.md` and `docs/followup_results.md` rests on
`arch="tokenized"` instead, and nothing pinned it.

* `Partial_Pooling/freeze_tokenized.py` writes
  `tests/fixtures/tokenized_arch_regression.pt`, in two independent halves:
  five freshly constructed fixed-seed schemas (atomic-only, trial tokens,
  per-position head, te64, and the 463,364-parameter winner), and the trained
  `mask6mix` checkpoint through all four network entry points.
* `tests/test_tokenized_arch_regression.py` compares bit-for-bit.
* **The winner reassembles to exactly 463,364 parameters at 40 tokens**, which
  is asserted as its own test so the number every table quotes cannot drift.

Negative control, run and reverted: adding `+ 1e-7` inside
`SchemaEmbedder.forward` fails **7 of the 8** tests, the sampler, PF-ODE and
composition paths included. The fixture bites.

## Test-suite baseline

`pytest tests/ -q` before any change: **272 passed, 18 failed**. After the Group
work: **327 passed, 18 failed** — the same 18, plus 55 new tests.

All 18 are pre-existing and unrelated. They name APIs that exist nowhere in
`src/compass/`: `equation=` (8) and `divergence_head` /
`instantaneous_divergence` (7), both dropped by `6bbb57c Restore pre-divergence
upstream source` while `17fb465 Restore preserved tutorials and tests` kept the
tests. Plus a stale `SimpleNamespace` mock without `.schema`, a CUDA/CPU device
mismatch, and one numeric assertion (1.152 < 0.5). Detail in
`Partial_Pooling/artifacts/frozen/tokenized/test_baseline.md`.

Two consequences worth carrying forward:

* **Tier 3.4 is viable**: `test_pfode_logprob_{vesde,vesde_hutchinson,vpsde}`
  and `test_gauss_correction_analytic_precision` /
  `test_hierarchical_shared_and_local` all pass. Only the PF-ODE-*sampler*
  variants are broken.
* **`tutorials/Population_Dynamics.py` is currently broken** — its
  `COMPARISON_MODES` pass `equation=` into `compare()`. It is the plan's chosen
  `Sequence` model-comparison vehicle (step 6), so `equation=` gets restored
  first or another vehicle is chosen. Does not block steps 1–5.

`Partial_Pooling/tests/`: 96 passed. The one collection error is pre-existing —
`artifact_names.py:30` defines a helper named `test_data_path(config, paths)`
that pytest tries to collect as a test.

## Step 2 — the Group block and the two-level trunk (done)

`Group(name, count, blocks, pool="mean", slot="learned")` in `schema.py`, and
`GroupedTransformer` in its own module. `TokenizedTransformer.py` and
`ConditionTransformer.py` are **not modified** — the new trunk imports their
blocks. `NodeSchema`'s slicing is untouched; `token_mask`,
`expand_token_values` and the new `repeat_view` delegate to the repeat's own
inner `NodeSchema`, so a repeat that mixes atomic locals with IID trials is
handled by the same code that handles them at the top level.

### Parameter budget — the 1M cap is satisfied

`Partial_Pooling/report_group_budget.py` counts by constructing the real
modules, and self-checks by reassembling the flat winner to 463,364. Confirmed
against the built network:

| configuration | K=5 | K=20 | K=100 |
|---|---:|---:|---:|
| **h64, te256, d 2+2+0 (default)** | 760,837 | **761,797** | 766,917 |
| h64, one extra block anywhere | 909,381 | 910,341 | 915,461 |
| h64, d 3+3+0 | 1,057,925 | 1,058,885 | **over** |
| h96, d 2+2+0 | 1,288,581 | 1,290,021 | **over** |

* The default has 24% headroom and changes **no** per-block hyperparameter from
  the validated winner: two trunks where there was one.
* Exactly **one** block of headroom. `depth_readout=1` is legal; `3+3` is not.
* **The budget is nearly independent of K** — 0.8% from K=5 to K=100, and the
  difference is exactly the `K x hidden_size` slot table, asserted as a test. So
  a K sweep measures the representation, not the capacity.
* h96 needs te128, and `docs/followup_results.md` §3 measured te64 costing 45%
  on global NRMSE while barely moving validation loss. Time-embedding width is
  not a lever to trade for hidden width.
* If depth ever binds: adaLN is 66% of every block (98,688 of 148,544) and
  `analyze_time_embedding.py` measured its effective rank at **2**. A low-rank
  factorization is the documented escape hatch, and a separate change.

### The symmetry contract, as implemented

`slot="learned"` is the **default**. Without a slot term the across-subject
trunk is permutation-equivariant whether or not anyone asked, which is exactly
the symmetry `Group` declines to promise. Tested in both directions: under the
default, permuting subjects changes the scores; under the `slot="none"`
ablation it permutes them equivariantly. The slot table is initialized
`normal(0, 0.02)` rather than zero, so a *fresh* model already breaks the
symmetry.

`Group(count=1)` is deliberately **not** asserted equal to the flat trunk. The
topologies differ — the flat trunk is one transformer over
`[globals, locals, trials]`, K=1 is `within -> pool -> across -> broadcast` —
and the pooling bottleneck makes bit-identity unavailable regardless of
initialization. What is asserted is that K=1 is well formed. The no-silent-change
guarantee for the existing model comes from the frozen fixture instead.

## Three things the implementation surfaced

**adaLN-Zero makes a fresh trunk the identity.** `adaLN_modulation[-1]` is
zero-initialized, so `gate_msa = gate_mlp = 0` and every block is exactly the
identity at construction. A freshly built network ignores attention completely,
so every "this input must change the output" test passes vacuously and every
"this must not change" test passes for the wrong reason. `tests/test_group_transformer.py`
wakes the modulation first. This applies to the flat trunk too and is worth
knowing before writing any behavioural test against a fresh COMPASS model.

**`sample()` cannot infer a mask for a nested schema.** It builds
`[zeros(nodes - width), ones(width)]`, a contiguous trailing split. A `Group`
puts each repeat's latent locals *inside* its observed trials, so that mask
would condition the wrong nodes. `ScoreBasedInferenceModel.sample` now raises a
clear error for a grouped schema when no explicit `condition_mask` is given,
rather than building a plausible-looking wrong one. This is the library-level
form of the adapter work the plan flags for `infer_partial_pooling.py`.

**`Sampler` and `PFODE` disagree about what `data` holds.** `Sampler.sample`
fills `joint_data[condition_mask == 1]` from a flat `data`, so it wants the
observed values **alone** with a full-width mask; `log_prob` and
`newton_map_estimate` want the whole node vector. Pre-existing, easy to trip on,
now pinned by tests on both paths.

One real bug was found and fixed this way: `t` arrives with a leading dimension
of 1 from `MultiObsSampler` and the MAP routines, relying on broadcast, and the
first draft's `repeat_interleave` broke that. `_per_repeat` leaves a batch of 1
alone.

## Where the drop-in claim stands

`tests/test_group_transformer.py` exercises every consumer of the score on a
grouped model: `Sampler`, `PFODE.log_prob`, `MultiObsSampler` under
`gauss_jacobian` / `gauss_hierarchical` / `fnpe`, and
`newton_map_estimate(curvature="jacobian")` — which pushes forward-mode AD
through the pool, both trunks and the broadcast. `Sampler.py`, `PFODE.py` and
`MultiObsSampler.py` are unmodified.

## Step 3 — the benchmark wiring (done)

An arm is now one command. Training:

```bash
python Partial_Pooling/train_models.py \
    --preset small --device cuda \
    --arch tokenized --tokenization group --train-subjects 20 \
    --train-size 100000 --validation-size 5000 \
    --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4 \
    --mask-menu six_mixed --mask-bernoulli-fraction 0.25
```

`--train-subjects` is the first architecture flag that is also a **data** axis,
so the arm generates its own split (`partial_pooling-train-100000-subj20`) sized
by `--train-size`: 100,000 groups for the globals-matched arm, 100,000/K for the
subject-budget-matched one.

### What a grouped arm borrows, and why

It reuses the **preset split's normalization constants and test datasets
verbatim** (`artifact_names.base_training_tag`, `config.base_data_signature`).
Refitting normalizers on the same distribution would move every normalized input
by sampling noise, and regenerating test data would break the byte-identity that
makes a new arm comparable to the published ones. `data_pipeline` refuses to
refit for a grouped split and errors if the preset's constants are missing.

### Nothing that existed moved

Verified, not assumed, and pinned by `Partial_Pooling/tests/test_group_contracts.py`:

* `get_config("full").model_signature` is still `aacc35ff55f5604e`;
* `partial_pooling-train-400000`, `sde_joint-train-400000` and the published
  `sde_joint-tok-trial-h64d2x8m4-mask6b0p25-sub100000-train-400000` are
  byte-identical;
* four new config fields are popped at their defaults in both `data_dict` and
  `model_dict`, so no existing signature changes;
* **every published recovery run's `inference_signature` is unchanged** —
  checked by computing 120 signatures under the current and pre-change code on
  identical arguments, 0 differences. `grouping_seed` enters the payload only
  when it is actually set.

### The inference adapter

`Partial_Pooling/group_layout.py` holds the pure functions, tested without
running inference (`test_group_layout.py`, 40 tests). It maps 100 subjects to
`100/K` composed factors and back, and the property everything depends on --
un-grouping restores the test file's own subject order for *any* grouping,
contiguous or seeded -- is asserted for every K.

`--grouping-seed` re-assigns subjects to groups, and is part of the run
signature, so the partition-sensitivity study's re-runs land in separate
directories instead of overwriting one another.

A K=5 end-to-end run confirms the whole path: sampling composed 4 groups of 5
subjects; the MAP stage ascended 20 subjects' locals with a conditioned-globals
clamp error of exactly **0.0**; the artifact came back with `globals (7,)`,
`locals (20, 3)`, 7 global rows and 60 local rows -- the same shapes a flat arm
produces, through the same downstream code, which never learns that grouping
happened.

**One deliberate gap.** `--map-estimator joint_score` raises `NotImplementedError`
for a grouped arm: it reuses the sampler's per-observation covariance (shaped
`(subjects, 10, 10)`) and a flat `[theta, x]` row layout, neither of which
survives grouping. The paper-matched protocol runs
`--map-estimator kde_global_then_local_ascent`, which is adapted. Refusing beats
producing a silently wrong MAP.

### Two more things the wiring surfaced

* **`config.arch` and the network's `arch` are different things.** A grouped arm
  keeps `arch="tokenized"` (the benchmark's legacy-vs-schema switch, and every
  tokenized-only validation still applies) while the network is `arch="grouped"`.
  `config.network_arch` is the one to hand to `ScoreBasedInferenceModel`.
* **A stale `partial_pooling-train-2048-normalization.pt`** carries no
  `data_signature`, so the `smoke` preset cannot be used to exercise a grouped
  arm end to end. Same class of staleness as the 100,000-row split already noted
  in the benchmark gotchas. `small` is current and is what the tests use.

Also fixed in passing: `artifact_names.test_data_path` is a path builder, not a
test, and pytest was trying to collect it. One `__test__ = False` removes a
pre-existing collection error.

## Tier 3.4 — the density and the composition, against closed forms (done)

`tests/test_group_analytic.py`, 13 tests, CPU, under a minute. A mock score
network returns the **exact** score of the diffused posterior for a hierarchical
linear-Gaussian toy -- partial pooling in miniature -- so any discrepancy is the
PF-ODE and composition machinery rather than training:

    g        ~ N(mu0, sigma0^2)      one group-level parameter
    l_j | g  ~ N(g, tau^2)           one parameter per subject
    x_j | l_j ~ N(l_j, sx^2)         one observation per subject

laid out the way `Group` lays it out -- `[g, (l_0, x_0), (l_1, x_1), ...]` -- so
the latent coordinates are deliberately **non-contiguous**. That is the property
that breaks the flat `[theta, x]` assumptions elsewhere, and testing against it
is the point.

**The density is intact.** PF-ODE `log_prob` matches the analytic
`log N(theta; m, Sigma)` to within 0.05 at K=1 and K=3; exact and Hutchinson
divergence agree. On a *real* `GroupedTransformer`, the log-density is invariant
under permuting a subject's trials and moves when trials change subject -- which
ties the schema's one claimed symmetry to the PF-ODE's own arithmetic.

**Composition over groups is exact.** The same eight subjects, partitioned three
ways, against the closed-form `p(g | all x)` (mean -0.0851, sd 0.2881):

| arrangement | correction | posterior mean | posterior sd |
|---|---|---:|---:|
| 8 x K=1 | `gauss_hierarchical` | -0.0826 | 0.2950 |
| 8 x K=1 | `gauss_jacobian` | -0.0900 | 0.2740 |
| 4 x K=2 | `gauss_hierarchical` | -0.1052 | 0.2859 |
| 4 x K=2 | `gauss_jacobian` | -0.1049 | 0.2853 |
| 2 x K=4 | `gauss_hierarchical` | -0.1010 | 0.2913 |
| 2 x K=4 | `gauss_jacobian` | -0.0810 | 0.2748 |

Maximum error 0.020 on the mean and 0.015 on the sd, and **the three
arrangements agree with each other** -- asserted directly, since the exact
posterior is a one-dimensional conjugate update that cannot see the partition.
Tolerances are set at about 3x the measured error: loose enough for a different
BLAS, tight enough to catch a regression. This is the miniature of the
partition-sensitivity study, run before the full-scale one.

One note for whoever writes the next such test: the first version of this mock
did a `linalg.solve` per row per diffusion step -- about a million Python-loop
solves -- and the *test harness*, not COMPASS, was the bottleneck; it was killed
by its own timeout. The covariance depends only on K and the diffusion time is
shared across a call's rows, so one batched solve per step does it. 27 seconds
instead of over 40 minutes.

## Steps 4-5 — the splits and the arms (training done)

Six splits, 1.6 GB, generated in ~100 min. Measured cost **41 MB and ~2.7 min
per 100,000 subjects** on the 3-CPU cap; K=20 globals-matched (2,000,000
subjects) took 64 min. All six verified loadable with node counts matching their
models exactly, and the normalized inputs check out: sd 0.816 rather than 1.0,
which is exactly `sqrt(60/90)` -- 30 of the 90 observation channels are the
degenerate `censored` flag -- confirming the borrowed constants are applied
per subject as intended.

### Training cost: the two-level trunk pays for itself

| arm | tokens | s/batch | s/epoch | vs flat |
|---|---:|---:|---:|---:|
| flat `mask6mix` | 40 | 0.013 | 10.2 | 1.0x |
| K=5 | 172 | 0.024 | 18.6 | 1.8x |
| K=10 | 337 | 0.029 | 22.9 | 2.3x |
| K=20 | 667 | 0.047 | 36.4 | 3.6x |

**K=20 costs 3.6x the flat arm while carrying 16.7x the tokens**, which is the
whole point of pooling each subject before the subjects interact: the cost is
`O(K * 33^2) + O((K+7)^2)`, not `O((7 + 33K)^2)`. Peak memory 1.16 GB.

### Trained arms

| arm | params | tokens | epochs | best val | at epoch |
|---|---:|---:|---:|---:|---:|
| flat `mask6mix` | 463,364 | 40 | 90 | 113.92 | 70 |
| K=5 globals | 760,837 | 172 | 98 | 456.89 | 48 |
| K=10 globals | 761,157 | 337 | 98 | 915.23 | 48 |
| K=20 globals | 761,797 | 667 | 98 | 1783.06 | 48 |

All three under the 1,000,000 cap; all three ran 98 epochs and took their best at
epoch 48. Total wall clock ~3 hours for the lane.

**Patience 50, not the preset's 20 -- and the reason is measured.** The first
K=5 attempt at patience 20 stopped at epoch 40 with its best at epoch 20, while
the flat control ran 90 epochs with its best at 70. Both curves have the same
shape (a steep three-epoch fall then a grind improving 8.3% and 7.7%
respectively) and the same ~2.8% epoch-to-epoch validation noise, so patience 20
was firing on noise. Re-running at patience 50 moved K=5's best from 473.06 to
**456.89** and its best epoch from 20 to 48 -- so the first run was cut short,
not converged. The plan requires optimizer updates held comparable across arms;
98 epochs against the control's 90 satisfies that, 40 would not have. The
patience-20 logs are kept in `ladder_group/logs_patience20/`.

**Nothing is being concluded from per-node validation loss** (0.955-0.977 for the
grouped arms against 1.139 for the flat control). It is not the metric, it is not
a fair comparison -- different latent fractions under the mask menu and 1.64x the
parameters -- and this benchmark has already established that validation loss is
nearly blind to what the compositional sampler needs: `docs/followup_results.md`
section 3 measured the te64 arm at 0.05% worse validation loss and **45% worse
global NRMSE**. Recovery numbers or nothing.

### Scheduling constraint on this host

Evaluation cannot run concurrently with training here. Only one of the ten GPUs
is ever free, and `autocvd` blocks rather than intruding on the others, so the
lanes are sequenced: `train_lane.sh`, then `run_all_evals.sh` chained behind it.
The screen sat waiting for a GPU after training released one and another user
claimed it within a minute.

## First screen results — K=5, both Gaussian rules (30 datasets)

Scored with `score_bayesflow_metrics.py` and the new `score_shrinkage.py`
against the published `mask6mix` rows on exactly the same datasets.

### The mechanism criterion, which is the primary one

| parameter | mask6mix hier | gK5 hier | mask6mix jac | gK5 jac |
|---|---:|---:|---:|---:|
| **log_sigma_nu slope** | 0.339 | **0.677** | 0.331 | **0.740** |
| log_sigma_nu r | 0.653 | **0.871** | 0.676 | **0.847** |
| log_sigma_log_alpha slope | 0.532 | **0.351** | 0.541 | **0.365** |
| log_sigma_log_alpha r | 0.698 | **0.447** | 0.693 | **0.391** |
| beta_raw r | 0.915 | **0.694** | 0.915 | **0.690** |
| mean r over 7 globals | 0.799 | 0.809 | 0.803 | 0.793 |

**The hypothesis is confirmed for the parameter it was aimed at, and only that
one.** `log_sigma_nu` -- the quantity whose 0.339 slope motivated this entire
experiment -- roughly doubles, and its correlation rises from 0.653 to 0.871
against the paper's 0.918. That is a large, targeted effect.

**It does not carry the headline, because it comes with a trade.**
`log_sigma_log_alpha` moves the other way by a comparable amount (slope 0.53 ->
0.35, r 0.69 -> 0.42) and `beta_raw`'s correlation drops from 0.915 to 0.69, so
mean r is flat: 0.799 -> 0.809 under `gauss_hierarchical` and 0.803 -> 0.793
under `gauss_jacobian`.

The trade **replicates across two structurally different composition rules**
(Schur-reduced and pilot-free Jacobian), with the same sign and similar
magnitude in every one of the seven parameters. `docs/followup_results.md`
established these rules are genuinely different machinery, so this is a property
of the representation, not of a sampler or of 30-dataset noise -- and the slope
shifts (0.34-0.41 and 0.17-0.18) are an order of magnitude above the noise floor
that document warns about for global columns.

### Everything else improves

| metric | mask6mix hier | gK5 hier | mask6mix jac | gK5 jac |
|---|---:|---:|---:|---:|
| g NRMSE (prior) | 0.5145 | **0.4492** | 0.5587 | **0.4699** |
| g ECE | 0.2225 | **0.1746** | 0.2239 | **0.1755** |
| g cov95 | 0.6762 | **0.7095** | 0.6381 | **0.7190** |
| l NRMSE (prior) | 0.4255 | **0.3912** | 0.4255 | **0.3902** |
| l ECE | 0.0196 | 0.0221 | 0.0177 | 0.0248 |
| l cov95 | 0.9488 | 0.9473 | 0.9480 | 0.9502 |
| l w95 | 1.584 | **1.473** | 1.585 | **1.474** |

Global NRMSE improves 13-16%, global calibration error 21-22%, global coverage
moves 0.03-0.08 closer to nominal, local NRMSE 8%, and local intervals are 7%
sharper. The one regression is local calibration error, which rises past the
plan's `<= 0.020` constraint to 0.022-0.025.

### Against the plan's pre-declared criteria

| criterion | verdict |
|---|---|
| 1. `log_sigma_nu` slope >= 0.55 | **MET** (0.677 / 0.740) |
| 1. `log_sigma_log_alpha` does not fall from 0.532 | **NOT MET** (0.351 / 0.365) |
| 2. mean r > 0.870 | **NOT MET** (0.809 / 0.793) |
| 3. global cov95 >= 0.676 | **MET** |
| 3. global ECE <= 0.223 | **MET** |
| 3. local w95 <= 1.584 | **MET** |
| 3. local ECE <= 0.020 | **NOT MET** (0.022 / 0.025) |
| 4. partition stability | not yet run |

So the arm passes the mechanism criterion on its target parameter, fails it on
the hold, and does not reach the headline. Recorded as measured; the K=10 and
K=20 arms and the F-NPSE rule are still running.

### F-NPSE: the trade disappears, and this is the best result on record

The third rule tells a different story, and it is the important one.

| parameter | mask6mix F-NPSE | gK5 F-NPSE | d slope | d r |
|---|---:|---:|---:|---:|
| mu_nu | 0.496 | 0.870 | +0.374 | +0.029 |
| mu_log_alpha | 0.286 | 0.587 | +0.301 | -0.019 |
| mu_log_t0 | 0.942 | 1.112 | +0.170 | +0.003 |
| **log_sigma_nu** | 0.322 | **0.645** | **+0.323** | **+0.213** |
| log_sigma_log_alpha | 0.222 | **0.334** | **+0.112** | +0.006 |
| log_sigma_log_t0 | 0.178 | **0.804** | **+0.626** | **+0.392** |
| beta_raw | 0.831 | 0.881 | +0.050 | **-0.206** |
| **mean** | 0.468 | **0.747** | **+0.279** | **+0.059** |

**Every one of the seven slopes improves**, and mean r goes **0.790 -> 0.849**
against the paper's 0.870. That is the closest anything in this repository has
come; the previous best on record is `mask6mix` + `gauss_jacobian` at 0.803.

| metric | mask6mix F-NPSE | gK5 F-NPSE |
|---|---:|---:|
| g NRMSE (prior) | **0.3738** | 0.3792 |
| g ECE | 0.2909 | **0.1590** |
| g cov95 | 0.5000 | **0.7476** |
| l NRMSE (prior) | 0.4507 | **0.4105** |
| l ECE | 0.0332 | 0.0354 |
| l w95 | 1.709 | **1.569** |

Global calibration error nearly halves and global coverage moves 0.25 closer to
nominal, at unchanged global NRMSE. Local intervals sharpen 8% with local error
down 9%. The regressions are local calibration error (0.033 -> 0.035) and
`beta_raw`'s correlation.

**A reporting error worth recording, because it nearly caused a wrong call.**
The slope references in `score_shrinkage.py` come from
`docs/followup_results.md`, which tabulates `mask6mix` under
**`gauss_hierarchical` only**. Scored against those, gK5's F-NPSE
`log_sigma_log_alpha` slope of 0.334 reads as a regression from 0.532 -- when in
fact `mask6mix`'s own F-NPSE slope is 0.222 and gK5 *improves* it. A fixed
baseline from one sampler is simply the wrong comparison for another. The column
is now labelled `hier-ref` and the script prints a same-method delta table,
which is the comparison that means anything. The Gaussian-rule trade reported
above survives this correction -- it was measured same-method -- but the F-NPSE
picture was inverted by it.

### So what the K=5 arm actually shows

Sampler-dependent, and strongly so:

* under both **Gaussian** rules -- a large targeted win on `log_sigma_nu` bought
  with a real loss on `log_sigma_log_alpha` and `beta_raw`; mean r flat;
* under **F-NPSE** -- no trade at all: every slope up, mean slope +0.279, mean r
  +0.059 to **0.849**, calibration substantially better.

`docs/followup_results.md` saw a milder version of the same thing with the mask
menu ("under F-NPSE the menu wins every column outright"). Whatever the grouped
representation supplies, the Langevin/F-NPSE composition exploits it and the
Gaussian corrections partly fight it. That is a hypothesis about the samplers,
not an established mechanism, and it is worth stating as such.

### K=10 beats the paper's metric — and fails the mechanism criterion

| arm | method | mean r | mean slope | log_s_nu slope | log_s_nu r | g NRMSE | g cov95 | l w95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mask6mix | hier | 0.799 | 0.799 | 0.339 | 0.653 | 0.5145 | 0.676 | 1.584 |
| gK5 | hier | 0.809 | 0.828 | **0.677** | 0.871 | 0.4492 | 0.710 | 1.473 |
| **gK10** | hier | **0.922** | 0.741 | 0.367 | 0.844 | **0.2515** | 0.724 | 1.265 |
| mask6mix | jac | 0.803 | 0.821 | 0.331 | 0.676 | 0.5587 | 0.638 | 1.585 |
| gK5 | jac | 0.793 | 0.844 | **0.740** | 0.847 | 0.4699 | 0.719 | 1.474 |
| **gK10** | jac | **0.924** | 0.739 | 0.375 | 0.850 | **0.2507** | 0.724 | 1.273 |
| mask6mix | fnpse | 0.790 | 0.468 | 0.322 | 0.685 | 0.3738 | 0.500 | 1.709 |
| gK5 | fnpse | **0.849** | 0.747 | 0.645 | 0.897 | 0.3792 | 0.748 | 1.569 |

**K=10 reaches mean r 0.922-0.924 against the paper's 0.870**, reproducibly under
both Gaussian rules, with global NRMSE halved (0.51 -> 0.25). On the paper's own
published metric that is a win.

**It is not the win the hypothesis predicted, and the plan's own rule says to
say so.** `log_sigma_nu`'s slope moves only 0.339 -> 0.367, nowhere near the
0.55 target, and `log_sigma_log_alpha`'s falls from 0.532 to 0.385. The
pre-declared rule is explicit: if r improves while the slope does not, the
mechanism claim is wrong whatever the headline says.

What actually moved is visible in `sd(err)/sd`, which collapses across every
parameter (mu_nu 0.463 -> 0.253, log_sigma_log_t0 1.279 -> 0.367). Since
`r = slope * sd(truth)/sd(median)`, **r rose because the estimation noise fell,
not because the shrinkage lifted.** K=10's posterior medians rank datasets very
well while still spanning only ~37% of the truth's range: well-ordered but badly
compressed. That is a real and large gain -- it is why NRMSE halved -- and it is
a different gain from the one this experiment set out to produce.

Two further caveats on the headline:

* **The mean is carried by the wrong parameters.** gK10 beats the paper on the
  three means and `beta_raw` (0.964-0.989 against the paper's implied ~0.917
  average for those four) and on `log_sigma_log_t0` (0.935 vs 0.699). On the two
  scales this work targeted it remains *below* the paper: 0.844 vs 0.918 and
  0.777 vs 0.804.
* **Bias got worse.** Three parameters carry statistically significant offsets
  (|shift|/se 2.8, 2.8, 1.9-2.1) where `mask6mix` had one.

### K=5 and K=10 differ in kind, not degree

This is the most surprising thing in the screen and the main open question for
K=20:

* **K=5** lifts the *slopes* hard (log_sigma_nu 0.34 -> 0.68-0.74) and r only a
  little (0.79 -> 0.79-0.85);
* **K=10** barely touches the slopes (0.34 -> 0.37) and lifts *r* hard
  (0.79 -> 0.92), by cutting estimator variance.

Both patterns replicate across composition rules, so neither is noise. Verified
not to be a mix-up: the run configs name `...gK5...` and `...gK10...`
checkpoints and the logs record "composing 20 groups of 5 subjects" and
"composing 10 groups of 10 subjects" respectively. Whatever grouping does, it
does *two different things* at these two scales, and a monotone reading of K
would be wrong.

### The complete 30-dataset screen (9 runs)

Mean r over the 7 global parameters, against the paper's **0.870**:

| arm | gauss_hierarchical | gauss_jacobian | F-NPSE |
|---|---:|---:|---:|
| mask6mix (flat control) | 0.799 | 0.803 | 0.790 |
| gK5 | 0.809 | 0.793 | 0.849 |
| gK10 | 0.922 | 0.924 | 0.924 |
| **gK20** | **0.933** | **0.935** | **0.936** |

`log_sigma_nu` slope, the mechanism criterion (target 0.55, was 0.339):

| arm | hier | jac | F-NPSE |
|---|---:|---:|---:|
| mask6mix | 0.339 | 0.331 | 0.322 |
| gK5 | **0.677** | **0.740** | **0.645** |
| gK10 | 0.367 | 0.375 | 0.380 |
| gK20 | 0.437 | 0.442 | 0.421 |

Accuracy and calibration at gK20 against the flat control:

| metric | mask6mix (hier/jac/fnpse) | gK20 (hier/jac/fnpse) |
|---|---|---|
| g NRMSE | 0.515 / 0.559 / 0.374 | **0.258 / 0.249 / 0.257** |
| g ECE | 0.223 / 0.224 / 0.291 | **0.141 / 0.139 / 0.143** |
| g cov95 | 0.676 / 0.638 / 0.500 | **0.791 / 0.800 / 0.791** |
| l NRMSE | 0.426 / 0.426 / 0.451 | **0.296 / 0.301 / 0.338** |
| l w95 | 1.584 / 1.585 / 1.709 | **1.182 / 1.191 / 1.335** |
| l ECE | 0.020 / 0.018 / 0.033 | 0.024 / 0.025 / 0.035 |

### Verdict against the pre-declared criteria

| criterion | gK20 | |
|---|---|---|
| 1. mechanism: `log_sigma_nu` slope >= 0.55 | 0.42-0.44 | **NOT MET** |
| 1. mechanism: `log_sigma_log_alpha` holds at 0.532 | 0.41-0.44 | **NOT MET** |
| 2. headline: mean r > 0.870 | **0.933-0.936** | **MET** |
| 3. global cov95 >= 0.676 | 0.79-0.80 | MET |
| 3. global ECE <= 0.223 | 0.139-0.143 | MET |
| 3. local w95 <= 1.584 | 1.18-1.34 | MET |
| 3. local ECE <= 0.020 | 0.024-0.035 | **NOT MET** |
| 4. partition stability | not run | pending |

**The headline is beaten and the hypothesis is not confirmed.** Both statements
are true and the plan required saying so: "If r improves while the slope does
not, the mechanism claim is wrong and must be reported as such, whatever the
headline number says."

What the screen establishes:

* **Grouping is a large, real improvement** -- global NRMSE more than halved,
  global calibration error down ~37%, global coverage from 0.64-0.68 to 0.79-0.80,
  local intervals 25% sharper -- and it is **robust to the composition rule**:
  the three rules agree to within 0.003 on mean r at both K=10 and K=20, where
  the flat control's rules spread over 0.013 and gK5's over 0.056.
* **It does not work the way the hypothesis said.** The shrinkage slope never
  reaches 0.55 at any K. `sd(err)/sd` collapses instead (mu_nu 0.463 -> 0.212,
  log_sigma_log_t0 1.279 -> 0.296), so `r = slope * sd(truth)/sd(median)` rose
  because estimator *variance* fell. The medians rank datasets far better while
  still spanning only ~44% of the truth's range.
* **The mean is carried by the means, not the scales.** gK20 does now beat the
  paper on `log_sigma_log_alpha` (0.816-0.834 vs 0.804) and `log_sigma_log_t0`
  (0.956-0.963 vs 0.699), but `log_sigma_nu` -- the parameter that motivated the
  whole experiment -- stays at 0.849-0.861 against the paper's **0.918**.
* **Bias is worse.** gK20 carries four parameters with |shift|/se above 2, where
  `mask6mix` had one. Lower variance, higher bias.
* **K is not monotone in the slope** (0.68 at K=5, 0.37 at K=10, 0.44 at K=20)
  while it is monotone in r (0.81 -> 0.92 -> 0.93). K=5 and K=20 lift the slope;
  K=10 does not. No explanation is offered for the dip, and none should be
  invented.
* **Local calibration error is the one consistent regression**, failing the
  <= 0.020 constraint in all nine runs.

### Partition sensitivity (final, 9 groupings)

The exact posterior cannot depend on which subjects share a composed group, so
any dependence is an artifact this representation introduces. Measured on gK20
under `gauss_hierarchical`, contiguous grouping plus random re-assignments:

The contiguous grouping plus eight random re-assignments of the 100 test
subjects into five groups of twenty, gK20 under `gauss_hierarchical`:

| parameter | across-grouping spread | MC floor | excess | ~68% CI | slope spread |
|---|---:|---:|---:|---|---:|
| mu_nu | 0.145 | 0.126 | 1.17 | [0.88, 1.46] | 0.014 |
| mu_log_alpha | 0.111 | 0.129 | 0.85 | [0.64, 1.07] | 0.020 |
| mu_log_t0 | 0.043 | 0.123 | 0.36 | [0.27, 0.45] | 0.004 |
| **log_sigma_nu** | 0.209 | 0.114 | **1.74** | [1.30, 2.17] | 0.013 |
| log_sigma_log_alpha | 0.148 | 0.117 | 1.23 | [0.93, 1.54] | 0.016 |
| **log_sigma_log_t0** | 0.208 | 0.132 | **1.56** | [1.17, 1.95] | 0.032 |
| beta_raw | 0.064 | 0.126 | 0.52 | [0.39, 0.65] | 0.010 |

**Established.** Partition dependence on the group scales is real:
`log_sigma_nu`'s interval [1.30, 2.17] excludes 1.0, so its per-dataset
posterior medians genuinely move more under re-grouping than resampling the same
posterior explains. The three means and `beta_raw` sit at or below the floor.

**Not established.** Whether it exceeds the 1.5 threshold. The interval straddles
it. Nine groupings give the across-run sd a ~25% relative standard error, so
1.74 and 1.50 are not distinguishable, and the threshold is in any case one this
work chose rather than one the plan supplied. Quoting a bare FAIL would overstate
what nine runs resolve.

**Unaffected either way.** The aggregate quantities the claim rests on:

* mean r across groupings **0.9300-0.9347**, spread **0.0047**, against a 0.063
  gap to the paper;
* the `log_sigma_nu` slope spread **0.0127** against the 0.05 tolerance, passing
  with a 4x margin.

So: **the headline is robust to grouping; individual scale-parameter estimates
are measurably, if modestly, not.** Both halves belong in any write-up.

An interim read at three groupings put the worst excess at 1.36 and was reported
as passing. That was premature -- three runs estimate a spread far too poorly to
act on -- and the number rose to 1.74 as seeds accumulated. The lesson is the
same one the tolerance itself taught: **do not read a spread estimate before it
has enough replicates to be a spread estimate.**

### The plan's median tolerance was mis-specified, and the correction matters

The plan asked for "across-run sd of each global median under **10%** of the
within-run posterior sd". That criterion is **below the Monte-Carlo noise floor
and cannot be met by anything**. Each run draws 100 posterior samples, and the
median of 100 draws from a roughly normal posterior already has
sd = `sigma * sqrt(pi/200)` = **12.5%** of the posterior sd from resampling
alone. An arm with literally zero partition dependence would fail it.

The first run of `score_partition.py` duly printed `FAIL` at 0.18, which meant
nothing at all. The criterion is now stated as **excess over a bootstrapped
Monte-Carlo floor** (each run's own draws resampled 200 times), tolerance 1.5x.
The bootstrap floor comes out at 0.115-0.132 against the 0.125 the asymptotic
formula predicts, which is what makes it trustworthy.

The lesson generalizes past this study: **a stability tolerance on a
sampling-based estimate has to be stated relative to that estimator's own noise,
not to zero.** The slope criterion (< 0.05) needed no such correction because a
regression slope over 30 datasets is far better determined than a single
posterior median.

### The 100-dataset paper-matched confirmation — and a correction

Both rules, 100 test datasets, the protocol `docs/followup_results.md` calls
paper-matched:

| | flat control (100ds) | **gK20 hier** | **gK20 F-NPSE** | paper |
|---|---:|---:|---:|---:|
| **mean r** | 0.732 | **0.934** | **0.936** | **0.870** |
| log_sigma_nu slope | 0.389 | **0.617** | **0.606** | — |
| log_sigma_nu r | 0.681 | 0.891 | 0.891 | 0.918 |
| log_sigma_log_alpha slope | 0.431 | 0.486 | 0.448 | — |
| log_sigma_log_alpha r | 0.521 | 0.802 | 0.811 | 0.804 |
| log_sigma_log_t0 r | 0.425 | 0.958 | 0.961 | 0.699 |
| g NRMSE | 0.4508 | **0.2461** | 0.2466 | — |
| g ECE | 0.2697 | **0.1036** | 0.1173 | — |
| g cov95 | 0.6100 | **0.8343** | 0.8400 | — |
| l NRMSE | 0.4546 | **0.2977** | 0.3381 | — |
| l ECE | 0.0406 | **0.0174** | 0.0303 | — |
| l w95 | 1.753 | **1.152** | 1.306 | — |

**The 30-dataset screen understated the mechanism, and this progress note said so
wrongly.** The same gK20 run, scored on its own first 30, 50 and 100 datasets:

| datasets | log_sigma_nu slope | r |
|---:|---:|---:|
| 30 | 0.437 | 0.854 |
| 50 | 0.545 | 0.876 |
| **100** | **0.617** | **0.891** |

Nothing about the model changed; only the number of points in the regression.
The earlier conclusion here -- "r rose because estimator variance fell, not
because the shrinkage lifted" -- was drawn from a slope estimated on 30 datasets
and is **retracted**. Both happened: the slope rose from the control's 0.389 to
0.617, *and* `sd(err)/sd` fell from 0.741 to 0.496.

The error is instructive. `docs/followup_results.md` warns that 30-dataset
*global* numbers are too noisy to read at the hundredths, and that warning was
duly applied to `r` -- but not to the median-on-truth slope, which is estimated
from the very same 30 points and is if anything noisier. **A caution about
sample size applies to every statistic computed from that sample, not just the
one it was written next to.**

### Final verdict against the pre-declared criteria (100 datasets)

| criterion | gK20 hier | gK20 F-NPSE | |
|---|---|---|---|
| 1. `log_sigma_nu` slope >= 0.55 | 0.617 | 0.606 | **MET** |
| 1. `log_sigma_log_alpha` holds at 0.532 | 0.486 | 0.448 | **not literally met** |
| 2. mean r > 0.870 | **0.934** | **0.936** | **MET** |
| 3. global cov95 >= 0.676 | 0.834 | 0.840 | MET |
| 3. global ECE <= 0.223 | 0.104 | 0.117 | MET |
| 3. local w95 <= 1.584 | 1.152 | 1.306 | MET |
| 3. local ECE <= 0.020 | 0.017 | 0.030 | MET / **not met** |
| 4. partition stability | r spread 0.0047 | — | aggregate stable |

The `log_sigma_log_alpha` hold is the one criterion that fails, and it fails
against a number measured on a *different arm at 30 datasets*. The same-protocol
100-dataset control gives 0.431 and gK20 gives 0.486, so on a like-for-like
comparison it improves. Settling it properly needs `mask6mix` at 100 datasets,
which has not been run.

### What can honestly be claimed

* **On the paper's own published metric, gK20 beats it**: mean r 0.934-0.936
  against 0.870, confirmed on 100 datasets under two structurally different
  composition rules, at **761,797 parameters against the reference's ~718,000**
  -- essentially the same model size.
* **It beats the paper on two of the three group scales** -- `log_sigma_log_alpha`
  0.802-0.811 vs 0.804, `log_sigma_log_t0` 0.958-0.961 vs 0.699 -- and remains
  slightly below on **`log_sigma_nu`, 0.891 vs 0.918**, the parameter that
  motivated the work.
* **The mechanism hypothesis is supported.** Giving one factor several subjects
  lifts the shrinkage slope on the targeted parameter from 0.389 to 0.617 at
  matched protocol, and cuts estimator variance as well.
* **Caveats that belong with any quotation**: individual scale-parameter
  posterior medians are measurably grouping-dependent (excess 1.74 over the
  resampling floor, CI [1.30, 2.17]) even though the aggregates are not; F-NPSE
  misses the local-calibration constraint where `gauss_hierarchical` meets it;
  and the paper's 0.870 is hand-transcribed from a published figure and cannot
  be recomputed here.

### A benchmark bug found by the seed check: validation noise is shared across runs

Every grouped arm -- K=5, K=10, K=20, and a K=10 repeat at a different weight
initialisation -- took its best validation loss at **exactly epoch 48** and ran
exactly 98 epochs. Four independent runs agreeing to the epoch is not a
coincidence, and it is not a property of the models:

| arm | validation loss, epochs 44-54 |
|---|---|
| K=5 | 488 489 509 508 **457** 471 499 486 484 500 497 |
| K=10 | 962 973 992 998 **915** 942 981 974 958 972 975 |
| K=20 | 1931 1908 1947 1982 **1783** 1848 1946 1888 1934 1922 1906 |
| K=10, init seed 20250719 | 969 986 984 1007 **896** 941 982 961 961 967 975 |

Epoch-wise validation curves correlate **r = +0.85 to +0.88 between different
architectures** (K=5 against K=20 -- 172 tokens against 667) and between
different initialisations of the same architecture. Epoch 48 ranks **1 of 98** in
all four.

**Cause.** `Trainer._run_epoch` draws the diffusion time `t` and the noise from
the global torch RNG on the validation pass, with no per-epoch reseed. Every arm
consumes the same number of draws per epoch (same rows, same batch size), so the
validation `(t, noise)` sequence is in lockstep across runs and roughly 75% of
the epoch-to-epoch variance in validation loss is a **common random draw rather
than model quality**.

This is the same class of bug `docs/followup_results.md` already documents and
fixed -- "Validation masks are now also drawn once from a fixed seed rather than
redrawn every epoch, so early stopping and best-checkpoint selection score the
model on the same task each time" -- but the fix covered the *masks* only. The
times and the noise were left on the shared stream.

**What it does and does not invalidate.**

* Early stopping and best-checkpoint selection pick **the luckiest validation
  draw, not the best model**. Reported best-validation numbers are minima over 98
  correlated draws and are optimistically biased.
* It affects every arm identically -- the published flat controls included -- so
  *between-arm* comparisons remain fair, which is why the recovery results above
  stand. Those are computed on the test set with fresh sampling and never touch
  this stream.
* It does partly undercut the seed-variance check: two runs sharing a validation
  noise sequence agree on the best epoch by construction, so "both peaked at 48"
  is not evidence of seed stability. The best-loss *values* (896.49 against
  915.23) still are.

**Fix**, not applied here because it changes checkpoint selection for every
future arm and would break comparability with everything already trained:
reseed the validation RNG at the top of each validation pass from a fixed seed,
exactly as `_fixed_validation_masks` already does for the masks. That is a
decision for whoever owns the benchmark, not a side effect of this experiment.

### Screen cost, measured

`gauss_hierarchical` 81 min per arm, `gauss_jacobian` **289 min** -- the 3.6x
ratio `docs/ladder_progress.md` records for the flat model, reproduced here. The
nine-run screen is therefore ~20 hours, not the ~13 first projected.

## Next

Step 5 of the plan: the K in {5, 10, 20} splits are generating (globals-matched
at 100,000 groups and subject-budget-matched at 100,000/K), then train the arms,
screen at 30 datasets, run the partition study on the best K, and promote to the
100-dataset paper-matched run.

Measured generation cost: **41 MB and ~2.7 min per 100,000 subjects** on the
3-CPU cap, so the whole sweep is ~1.5 GB and ~100 minutes. Worth watching: the
filesystem is at 93%.

**K=15 is not available.** `train_subjects` must divide the 100 test subjects so
a dataset regroups evenly into composed factors, and 100/15 is not an integer.
Supporting it would mean dropping to 90 subjects -- breaking the paper-matched
protocol and the comparison to the published rows -- or a ragged final group,
which the fixed `(K, hidden_size)` slot table cannot take. K=20 is used instead;
25 and 50 also divide 100.
