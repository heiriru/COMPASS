# Inverse Kinematics

> **Sampler and training-recipe change.** Churn now lives in `compass.churn` and
> is the sampler default (`churn_eta=2.0`, corrector-free); `time_sampling` and
> the training condition-mask law are chosen through the COMPASS API. The
> defaults used by the commands below are no longer the ones the tables in this
> file were produced under. See [`docs/unified_method.md`](../docs/unified_method.md)
> for what changed, what is pinned to reproduce the old numbers, and which
> checkpoints are now stale.


Simulation-only reproduction of the inverse-kinematics benchmark from the
BayesFlow "Diffusion Models" example:

* Source: <https://bayesflow.org/v2.0.13/_examples/Diffusion_Models.html>
* Original benchmark: Ardizzone et al., <https://arxiv.org/pdf/2101.10763.pdf>

**Scope: everything the notebook does before it builds a network.** No model is
constructed, trained, or sampled from here. The forward model, the prior, the
draws and the pre-training figures — nothing else.

## The model

A robot arm on a vertical rail. Four parameters
`(height_arm, angle_1, angle_2, angle_3)`: the first slides the shoulder along
the rail, the other three are joint angles applied cumulatively. Segment lengths
are `0.5, 0.5, 1.0`, and the prior is a zero-mean spherical Gaussian with scales
`[0.25, 0.5, 0.5, 0.5]` — the height a quarter as wide as the angles.

```
x1 = l1 sin(a1) + l2 sin(a1+a2) + l3 sin(a1+a2+a3) + height
x2 = l1 cos(a1) + l2 cos(a1+a2) + l3 cos(a1+a2+a3)
```

Four parameters map to two observables, so the inverse problem is
non-identifiable by construction — a continuum of configurations reaches any
attainable end position. That is the entire point of the benchmark and what
`01_example_simulations.png` shows.

## Running

```bash
python inverse_kinematics/run_simulations.py                       # notebook defaults
python inverse_kinematics/run_simulations.py --simulations 100000 --seed 7
python -m pytest inverse_kinematics/tests -q
```

Takes about 20 seconds at the defaults, almost all of it in the rejection
sampling for the illustration. CPU only.

## Outputs

Written to `artifacts/`:

| File | Contents |
|---|---|
| `figures/01_example_simulations.png` | Six individual prior draws, then 120 configurations reaching one end position |
| `figures/02_prior_parameters.png` | The notebook's four-panel prior histogram |
| `figures/03_observation_space.png` | Where the end effector lands under the prior — **not in the notebook** |
| `simulations-10000.npz` | `parameters` `(n, 4)` and `observables` `(n, 2)` |
| `manifest.json` | Seed, prior scales, segment lengths, summary moments, rejection-sampling acceptance rate, figure list |

## Deliberate differences from the notebook

**No BayesFlow, Keras or JAX dependency.** The prior and forward model are a
handful of trigonometric operations, written directly against NumPy. None of
those packages is installed in this repository's environment, and taking on an
inference framework to draw from a 4-dimensional Gaussian would not be a fair
trade. `bf.make_simulator([prior, observation_model])` therefore has no
counterpart here; `simulator.simulate` does the same job.

**Explicit named generators.** The notebook calls the global `np.random.normal`,
which cannot be reproduced across processes. Every draw here comes from a
`numpy.random.Generator` derived from a root seed by an explicit *name*
(`generator("training")`, `generator("rejection")`, …), so the streams are
independent and individually reproducible, and adding a new draw site later
cannot shift an existing one. This follows `Partial_Pooling`'s convention.

**The arm illustration is reimplemented, not copied.** The notebook's opening
image is a static PNG whose source is not on the published page, and the helper
it calls elsewhere — `misc.diffusion.kinematics_helper.InverseKinematicsModel` —
ships with the documentation build rather than with BayesFlow, so it could not
be imported. `plots.plot_example_simulations` redraws it from the arm geometry
following the helper's described behaviour: three segments through four joints,
the target end position marked, one exemplar solid over a faint ensemble.

Two choices inside that figure are mine, and both are visible in the output:

* The prior draws get **one small axis each** rather than a single shared axis.
  Overlaid, twelve arms are unreadable — a configuration that folds back on
  itself cannot be told from two that happen to cross.
* The ensemble reaching a common end position comes from **plain rejection
  sampling against the prior**, which is exactly `p(parameters | observables)`
  up to the acceptance radius. No trained model and no approximate inference is
  involved, which matters: the non-identifiability is a property of the forward
  model, and demonstrating it with an estimator would beg the question. The cost
  is visible — at tolerance 0.01 the acceptance rate is about `3e-5`, so 120
  configurations take 3.6M draws.

`03_observation_space.png` is an addition, not a reproduction. The two
reproduced figures show the arm and the parameters but never the observation
space the inference problem is posed in, and it costs nothing once the draws
exist.

## Layout

```
inverse_kinematics/
├── simulator.py         prior, forward model, joint chain, rejection sampling
├── plots.py             the figures
├── run_simulations.py   entry point
├── runtime.py           3-thread CPU cap, applied before numeric imports
├── tests/               9 contract tests
└── artifacts/           figures, draws, manifest
```

Per the repository `CLAUDE.md` rule, `run_simulations.py` caps itself at three
threads before importing NumPy or matplotlib. There is no GPU path and no
`autocvd` call, because nothing here touches a GPU.

---

# Part 2 — Compositional inference with a shared shoulder angle

`angle_1` is promoted to a **global** coordinate shared by every observation;
each observation keeps its own `(height_arm, angle_2, angle_3)`. Node layout is
`[angle_1 | height_arm, angle_2, angle_3 | x1, x2]`, `hierarchy=[0]`, 6 nodes.
Note this is *not* the simulator's parameter order — `hierarchical.py` converts,
and a test pins the round trip.

```bash
python inverse_kinematics/train_model.py --config h16d2 --arch legacy
python inverse_kinematics/run_inference.py --arch legacy
python inverse_kinematics/plots_posterior.py
python inverse_kinematics/observation_sweep.py
```

## The referee

The forward map is deterministic, so there is no closed-form posterior. The grader
is `hierarchical.reference_global`: an **ABC posterior** for the shared angle,
`p(a | x) ∝ prior(a) · Πⱼ E_l[N(xⱼ; f(a,l), h²I)]`, Monte Carlo over the local
prior with common random numbers across the grid. No network is involved, so it
can referee the composition rules rather than restate one of them.

**It passed simulation-based calibration** (`sbc_reference.py`, 200 replicates):
mean rank 0.526, KS 0.0864 against a 95% critical value of 0.0960. A pass with
modest margin — enough to trust it for the effects reported here, not for much
finer ones.

## Results (legacy h16d2, 8 observations)

| rule | W1/σ | width ratio | bias/σ | endpoint err | s |
|---|---:|---:|---:|---:|---:|
| Langevin + F-NPSE | 0.481 | 0.981 | −0.48 | 0.094 | 18 |
| DPM2 + gauss_hierarchical + churn 2 | 0.520 | 1.014 | −0.52 | 0.096 | 46 |
| DPM2 + gauss_jacobian + churn 2 | 0.562 | 1.042 | −0.56 | 0.094 | 25 |

Composition preserves the network's fidelity: endpoint error under all three
rules matches the **0.097** of the single-observation posterior with no
composition at all. Widths are right. W1 is *entirely* bias — `mean_err ≈ −W1` in
every row.

`observation_sweep.py` locates that bias. At `n = 1` the arrow system collapses
to the raw network score, so composition is provably a no-op, and the bias there
is ≈ 0 (−0.08…+0.01 σ). It then grows with the number of observations combined:
−0.07 σ at n=2, −0.6 σ at n=4, −0.5 σ at n=8, **−1.2 σ at n=16**, with all three
rules within 0.04 σ of each other at every count. That is the signature of
*coherent* single-observation error, which composition preserves while averaging
independent error away by √n — not of any one rule's composition step.

## The capacity ladder does NOT transfer from partial pooling

Identical data and recipe; only the trunk differs. Endpoint error is how far a
posterior draw's arm tip lands from its own observed point (prior scale 1.38):

| model | params | val loss | endpoint error |
|---|---:|---:|---:|
| tokenized h16d2 | 78,833 | 3.58 | 0.676 |
| tokenized h32d3 | 223,425 | 3.48 | 0.760 |
| **legacy h16d2** | **367,136** | **2.43** | **0.097** |

Tripling the tokenized budget changes nothing, which rules out capacity and
leaves the trunk's inductive bias. The partial-pooling reduction worked because
99.3% of that model replicated one function of diffusion time across **90
exchangeable trial scalars**. Here there are 6 *heterogeneous* nodes and no
exchangeability to trade for, so per-node modulation buys real capacity — and the
fixed 256-dim time embedding floors any tokenized model near 40K parameters
anyway. **Use the legacy trunk here**; shrink `hidden_size`/`depth` within it if
a smaller model is wanted.

## Against the reference implementation

`run_bayesflow.py` runs the notebook's own method — `bf.networks.DiffusionModel`
in a `BasicWorkflow` — in a **separate venv** (`COMPASS/.bayesflow`), on the
identical forward model imported from `simulator.py`.

| | BayesFlow | COMPASS legacy |
|---|---:|---:|
| parameters | 509,460 | 367,136 |
| training | 1.3 min | 42 min |
| endpoint error | **0.0144** | 0.097 |

They are 6.7x better. The fair caveat: they fit a dedicated amortized posterior
`x → θ`, while COMPASS fits the joint `p(θ, x)` under random conditioning masks —
strictly harder, and what buys arbitrary conditioning and composition. Some gap
is by design; 6.7x is more than that excuses.

## Making the posterior bimodal

The arm has exactly one discrete symmetry, and it decides the design. With
`S(a) = Σ lₖ sin(cumₖ)` and `C(a) = Σ lₖ cos(cumₖ)`:

    (a1, a2, a3, h)  ->  (-a1, -a2, -a3, 2*x1 - h)

is an **exact involution of the likelihood** (verified to 4e-16): `S` flips, `C`
is untouched, so the end position is identical. Angle priors are symmetric and
invariant; the **height prior is the only symmetry breaker**, costing
`2*x1*(h - x1)/σ_h²`.

**Partition (settled by the symmetry, not by a sweep):** the global coordinate
must be an **angle** and `height_arm` must be **local**. The flip is shared
across rows, but the compensating height shift differs per row, so each row needs
its own height to absorb it. With `height_arm` global, row *j* could mirror only
if `h = 2*x1ⱼ - h` for every *j* at once.

**Trigger — select data, do not widen priors.** Keep the stock priors and use
observations whose tip sits near the rail height, `|x1ⱼ| < 0.15`; the penalty is
proportional to `x1ⱼ` and vanishes with it. Over 10 independent datasets per
count:

| n | fraction bimodal | median dip ratio | median &#124;Σ x1ⱼ&#124; |
|---:|---:|---:|---:|
| 2 | 0.40 | 0.887 | 0.133 |
| 4 | 0.50 | 0.491 | 0.131 |
| 8 | **0.90** | **0.077** | 0.144 |

Bimodality *strengthens* with observations. The last column shows why: the signed
penalty terms cancel rather than accumulate, so `|Σ x1ⱼ|` stays flat while extra
observations sharpen both modes.

```bash
python inverse_kinematics/bimodality_study.py --select-x1 0.15
```

### Negative result: widening the height prior does not work

It frees the mirror and destroys the information in `x1` simultaneously — at
`σ_h = 1` the posterior has already relaxed to the prior (sd 0.47 vs 0.50). It
also breaks the referee: the ABC kernel's effective sample size falls from
3,396/80,000 at `σ_h = 0.25` to **17/80,000 at `σ_h = 16`**, where the apparent
"two clean modes" (dip ratio 0.077) are noise. The tell was a posterior width
that stopped contracting with `n`. Retained as `--height-stds` so the negative
result stays reproducible; this is the same failure shape as the kernel-curvature
candidate in `NOTES_hierarchical_locals.md` — an estimator informative only where
it has become unreliable.

**Read single-dataset sweeps here with suspicion.** The penalty is a *sum of
signed terms*, so whether both branches survive is a property of the realized
dataset. A one-dataset-per-`n` sweep produced dip ratios of 0.642, 0.007, 0.999,
0.001 across n = 2, 4, 8, 16 — no trend at all. Only replicates gave the monotone
table above.

---

# Part 3 — Can compositional score modelling represent a bimodal shared posterior?

Setup (`run_bimodal.py`): `angle_1` global, `(height_arm, angle_2, angle_3)`
local, **stock priors**, 8 observations selected with `|x1ⱼ| < 0.15`, true shared
angle 0.6. **No retraining** — the bimodality is a property of the evaluation
dataset, not the training distribution, so the existing checkpoint is what should
be tested.

The reference is a hard target: modes at **−0.53 / +0.62** (on ±truth) with dip
ratio **0.002**.

## Composed shared posteriors (`14_bimodal_shared_posterior.png`)

| rule | modes | dip | frac>0 | W1/σ |
|---|---|---:|---:|---:|
| Langevin + F-NPSE | −0.72, −0.11 | 0.998 | 0.249 | 0.751 |
| gauss_hierarchical + churn | −0.01 | 1.000 | 0.511 | 0.902 |
| gauss_jacobian + churn | +0.00, +0.67 | 0.656 | 0.546 | 0.917 |

All three put a single blob **in the valley** where the reference has almost no
mass. `gauss_hierarchical` is worst-behaved structurally — one central mode, dip
1.000 — which is exactly what a constant per-observation covariance does to a
two-branch target. `gauss_jacobian` keeps partial structure (dip 0.656, second
lobe at +0.67 near the true +0.62), its state-dependent curvature earning its
cost. F-NPSE picks a side.

## But the blocker is the network, not the composition

The per-observation score has to carry two modes before any rule can preserve
them. Measured at the notebook's own `obs = [0, 1.5]` — the point where the
reflection is exactly free:

| sampler | dip | frac>0 | sd of angle_1 | endpoint err |
|---|---:|---:|---:|---:|
| dpm2 + 10 correctors | 1.000 | **0.240** | 0.467 | — |
| dpm2 + 2 correctors | 1.000 | 0.385 | — | — |
| dpm2 predictor-only | — | 0.503 | 0.345 | ≈ prior |
| dpm2 + churn 2 (n = 1) | 1.000 | 0.478 | 0.370 | 0.252 |
| **BayesFlow DiffusionModel** | **bimodal** | 0.553 | **0.821** | **0.014** |

Two separate effects, and they must not be conflated:

1. **Langevin correctors break the symmetry.** `frac>0` falls 0.503 → 0.385 →
   0.240 as correctors go 0 → 2 → 10. They relax the ensemble onto whichever
   branch it is nearest, which is precisely the failure mode a corrector has on a
   multimodal target. Corrector-free sampling restores `frac>0 = 0.503`, i.e. the
   exact symmetry the reflection demands.
2. **Even with the symmetry intact, the posterior is unimodal.** Every
   corrector-free configuration gives a *symmetric but single central* mode with
   `sd ≈ 0.37`, against BayesFlow's `sd = 0.821` covering both lobes. The network
   averages the two branches instead of representing them.

So the answer is: **the composition machinery is not what blocks bimodality — the
learned score is.** No composition rule can recover a mode the per-observation
score does not have, and this network's does not have it. That is consistent with
the same network being 6.7x behind BayesFlow on endpoint reconstruction (0.097 vs
0.0144): both are symptoms of a less accurate single-observation posterior.

Note also that predictor-only *without* churn returns approximately the prior
(`13_compass_posterior_single.png`) — the known initialisation failure that churn
exists to fix. Churn is currently wired into `MultiObsSampler._dpm_sampler` only,
not the single-observation `Sampler`, which is why the `n = 1` route is used above
to get a churned single-observation posterior at all.

**What would move this:** a better single-observation score, not a different
composition rule. The `gauss_jacobian` row is the encouraging part — it is the
only rule that retains any second mode, which matches its status as a strict
generalisation of the pilot form rather than a competing approximation.
