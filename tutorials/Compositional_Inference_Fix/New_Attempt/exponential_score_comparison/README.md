# Composing a score network on `g → ℓⱼ → xⱼ`

A hierarchy that is generative, non-Gaussian, and **exactly solvable at every
noise level** — including the diffused joint score, which is what makes it a
usable test bench for composition rules rather than one more plausible-looking
figure.

```
g        ~ Normal(μ_g, σ_g²)                    μ_g = 0, σ_g = 1
ℓ_j | g  = g + ε_j,      ε_j ~ Exponential(λ)   λ (rate) = 1
x_j | ℓ_j = ℓ_j + η_j,   η_j ~ Exponential(λ)
```

`x_j − g = ε_j + η_j` is a sum of two i.i.d. exponentials, so the local
parameter integrates out and the tall posterior is closed form:

```
p(g | x_1..N) ∝ exp[−(g − μ_g)² / (2σ_g²)] · exp(N λ g) · ∏_j (x_j − g) · 1[g ≤ min_j x_j]

d/dg log p(g | x) = −(g − μ_g)/σ_g² + N λ − Σ_j 1/(x_j − g)
```

Two properties make it hard in the way real hierarchical problems are hard: the
support has a **hard wall** at `min_j x_j` where the score diverges, and
`exp(N λ g)` presses the mass right up against it. At `N = 30` the posterior is
roughly `1/N` wide and strongly skewed. Nothing about it is Gaussian, which is
precisely the assumption every pilot-covariance composition rule makes.

The locals are exact too, and flat: `p(ℓ_j | g, x_j) = Uniform(g, x_j)` — the
exponentials cancel. So conditioned on `g` the local posterior has no unique
argmax, and its λ-smoothed mode (what annealed score ascent converges to) is the
midpoint `(g + x_j)/2`, which is also its mean.

## Why the *joint* score is degenerate — and why that is the point

```
p(g, ℓ_1..N | x) ∝ exp[−(g−μ_g)²/(2σ_g²)] · exp(N λ g) · ∏_j 1[g ≤ ℓ_j ≤ x_j]
```

is **flat in every ℓ_j**, and its `g`-score is the affine `−(g−μ_g)/σ_g² + Nλ`
with **no** `−Σ_j 1/(x_j − g)` term. All the structure lives in the indicator,
i.e. in the boundary, not in the gradient. Comparing composed scores at `t = 0`
would therefore be vacuous here: every asymptotically correct rule returns the
same affine function, and Fisher's identity does not rescue it because the
support of `ℓ_j` depends on `g`.

At a finite noise level the boundary is smoothed *into* the gradient, and the
comparison becomes sharp. Under the VESDE kernel `N(·, λ_t²)` (α_t ≡ 1) the local
integrals still factor given `g`:

```
p_λ(g_t, ℓ_t | x) ∝ ∫ dg exp[−(g−μ_g)²/(2σ_g²) + N λ g] · N(g_t; g, λ_t²)
                        · ∏_j [ Φ((x_j − ℓ_jt)/λ_t) − Φ((g − ℓ_jt)/λ_t) ]
```

— a **one-dimensional quadrature**. So the exact diffused joint score is
available at every noise level the sampler visits, and composition rules can be
graded directly, before any sampler error enters. That is what
`hierarchy.diffused_score` computes, and it is the backbone of this directory.

## Checking it

`validate_references.py` checks every closed form against an independent
numerical route: finite differences of a brute-force quadrature (~1e-9 relative),
the large-λ Gaussian limit, Monte-Carlo moments, and importance sampling against
the `Gamma(2, rate)` marginal likelihood.

Those all grade formulas against other formulas, so two further checks grade the
**simulator**, which none of them touch:

**The hierarchy is real.** A simulator that drew a fresh `g` for every
observation would pass every closed-form check above, because each `(g, ℓ_j, x_j)`
triple would still have the right marginal laws. What separates the hypotheses is
the *within-dataset* spread, by a clean factor of two:

| | shared `g` | fresh `g` per observation | measured |
|---|---|---|---|
| `Var_j(ℓ_j)` within a dataset | 1 | 2 | **0.986** |
| `Var_j(x_j)` within a dataset | 2 | 3 | **1.988** |
| `Var(mean_j ℓ_j)` between datasets, N=40 | 1.025 | 0.050 | **1.008** |

The last row is the sharpest: with `g` shared, the dataset mean keeps the *whole*
prior variance no matter how many observations it averages over — a 20×
separation, and it lands on the shared-`g` value.

**Simulation-based calibration.** Over 600 replicate datasets, the exact
posterior CDF evaluated at the true `g` is Uniform(0, 1): mean rank 0.4997
(KS 0.022) at N=4 and 0.4917 (KS 0.032) at N=30. This is the end-to-end check
that the posterior is the posterior *of this simulator*, and it is precisely
sensitive to the shared-`g` structure — without sharing, the tall posterior would
contract like `1/√N` around the wrong point and the ranks would pile into the
tails at N=30 while still looking passable at N=4.

**`simulate` (training draws) deliberately draws a fresh `g` per row**, and that
is not an inconsistency: the network is trained on single `(θ, x)` pairs from the
prior predictive, so each row is its own one-observation problem. Sharing `g` is a
property of the *tall dataset* the composition rules are evaluated on, not of the
training set. The checks assert both halves separately.

## The three composition rules

One network, one set of observations, one seed, one sample and step count.

| key | rule | what it costs |
|---|---|---|
| `gauss_hierarchical` | DPM-2 + `correction="gauss_hierarchical"` | a **pilot** covariance per observation (4096 draws each — a whole extra reverse pass) |
| `gauss_jacobian` | DPM-2 + `correction="gauss_jacobian"`, MAP also via `newton_map_estimate(curvature="jacobian")` | **pilot-free**; curvature from the network's own Jacobian |
| `langevin_fnpe` | annealed Langevin + `correction="fnpe"` | cheapest; targets *bridging* densities |

**Read the F-NPSE row with its caveat.** `fnpe` composes the score of Geffner et
al.'s bridging densities, which are deliberately *not* the diffusion marginals of
the tall posterior; they agree only as `λ → 0`. A large mid-λ error for F-NPSE is
the algorithm doing what it says, not a defect. The rungs that grade all three on
equal terms are the small-λ ones.

## What is measured, in three layers

**1. The composed score itself.** At states drawn from the exact diffused joint
`p_λ(g_t, ℓ_t | x)` — i.e. where the sampler actually is at that noise level —
against the exact quadrature target, over a λ ladder spanning the schedule. No
sampler, no Monte-Carlo error in the reference. The figure also carries a
**network floor**: the same network's single-observation score error on the same
ladder. That line is a **reference, not a floor**: composing `n` observations
averages their *independent* network errors down by ~`√n` while leaving
*coherent* (bias) error untouched, so a rule can legitimately land below it — and
`gauss_jacobian` does at small λ. Below the line means the network's error was
largely independent across observations and the rule averaged it away; far above
it means the rule is contributing error of its own.

**2. The sampled posterior.** W1, mean error and width ratio of the shared
marginal against exact `p(g | x)`, the same for the 30 local marginals, and the
*implied* marginal score `d/dg log KDE(draws)` against the closed form above,
over the central 90% of the posterior (at the wall the exact score diverges and
no kernel estimate can follow it — scoring there would grade the KDE).

**3. The MAP.** One procedure for all three rules, so it compares them rather
than the estimators: marginalize the draws onto `g`, take a Gaussian-KDE maximum
(parabolic-refined) for `ĝ`, clamp `g = ĝ`, then run annealed score ascent on each
local under `p(ℓ_j | ĝ, x_j)`. With `g` conditioned the locals are independent, so
stage 3 contains no composition at all and every compositional error collapses
onto the single scalar `ĝ`. The Jacobian arm additionally reports
`newton_map_estimate`, its native pilot-free alternative.

Local error is reported against **two** references, and they must be read
together: against the exact conditional mode `argmax_ℓ p(ℓ | ĝ, x_j)`, which
grades the ascent; and against the exact local posterior mean `E[ℓ_j | x_1..N]`,
which grades the whole pipeline. Their difference is the price of conditioning on
a point estimate instead of integrating over `g` — the only quantity in the table
that is a property of the *method* rather than of the sampler or the network.

## Sizing the network

<!-- RESULTS:CAPACITY -->

## Results

<!-- RESULTS:METHODS -->

## Oracle and trajectory diagnostics

`oracle_diagnostics.py` keeps the headline artifacts immutable and adds the
controls needed to separate initialization, numerical integration, learned-score,
and finite-noise composition error. It runs seed 0 by default and caches every
convergence cell in `artifacts/oracle_diagnostics/`.

```bash
source .COMPASS/bin/activate
python oracle_diagnostics.py --device cuda --seed 0 --run-analytic-sampling
python oracle_diagnostics.py --replot
```

The diagnostic artifacts are:

| File | Question answered |
|---|---|
| `00_oracle_endpoint.png` | Does exact-score DPM/Langevin work, and what remains with analytical row scores plus approximate composition? |
| `01_analytic_composition.png` | Rule error, signed bias, and dependence on distance from the hard wall with network error removed |
| `02_intermediate_density.png` | When each learned trajectory leaves its matching target: exact diffusion for DPM, F-NPSE bridge for Langevin |
| `03_convergence.png` | One-seed timestep, MCMC-step, terminal-corrector, and endpoint-epsilon sweeps for learned and oracle scores |
| `04_curl.png` | Relative Jacobian antisymmetry; the exact score and exact F-NPSE bridge calibrate numerical zero |
| `05_learned_jacobian.png` | Learned single-row score error versus learned score-Jacobian/Hessian error |

The exact-score DPM controls deliberately report two initializations. `exact
start` draws from the exact diffused joint at the schedule maximum and isolates
the integrator. `production start` uses the sampler reference Gaussian and
therefore measures initialization mismatch plus integration. Those must not be
conflated: a deterministic probability-flow map cannot forget an incorrect
initial law merely because its vector field is exact.

The numerical endpoint `eps` is also a bias--stability trade-off. Stopping at a
positive noise level avoids the singular hard-wall score and improves numerical
stability, but leaves a smoothed target. Reducing `eps` removes that smoothing
only if the learned/composed score remains accurate in the newly exposed
low-noise region.

Current seed-0 controls show three separable effects. The 5,000-draw exact
shared-marginal oracle is converged by 50 DPM2 steps (W1 0.031 posterior sigma),
whereas the same exact vector field starting from the production Gaussian holds
at W1 0.146 through 400 steps. The full-joint oracle magnifies that mismatch
because all 30 local coordinates also start from the wrong high-noise law:
exact joint start gives W1 0.106 sigma with 64 draws, production start 0.982.
With network error removed but the finite-noise rule retained, analytical
Jacobian-GAUSS still gives W1 0.614 sigma; analytical pilot-GAUSS gives 2.21.
Thus timestep error is not the headline bias: high-noise initialization and the
finite-noise composition approximation are both material. The learned
Jacobian predictor-only sweep is stable but wrong (W1 2.254, 2.244, 2.241,
2.240 sigma at 50, 100, 200, 400 steps). Production correctors improve that
to 0.563 sigma at 50 steps, but adding proportionally more non-conservative
corrector updates worsens it to 1.007 by 400 steps.

Reducing `eps` confirms the stability trade-off rather than curing it. The
exact marginal oracle changes only from W1 0.031 to 0.027 sigma when `eps` falls
from 1e-3 to 1e-4, while learned Jacobian-GAUSS worsens from 0.650 to 1.835.
The smaller endpoint exposes a region where its learned/composed field is less
reliable.

## Removing the correctors

`gauss_jacobian` with an exact row score is 45× worse without its two Langevin
correctors (W1 2.598 vs 0.057 posterior sigma). Two scripts take that apart, and
`NOTES_corrector_free.md` is the running log of what has been tried and what is
therefore ruled out — **read it before proposing a fix**.

`initialization_experiments.py` shows ~85% of the gap is the initial law: the
stock initializer starts every latent at zero, while the exact diffused joint at
`lambda_max = 3.89` sits at 1.39 (shared) and 1.5–4.2 (locals). An exact diffused
start takes 2.598 → 0.387.

`stochastic_predictor_experiments.py` closes the rest without any corrector step.
A Langevin corrector works by relaxing the ensemble onto the composed density
where that density is accurate; the reverse-time SDE has the same mechanism in
its *predictor*. Following Karras et al. (2022, Alg. 2), each level takes a short
step **up** in noise and then the deterministic DPM2 step down —
`sigma_hat = sigma_i + eta (sigma_i - sigma_{i+1})`, with `eta = 0` the stock
probability-flow ODE and `eta = 1` the exact reverse SDE. It costs **no extra
score evaluations**.

| eta | shared W1/sigma | width | local W1/sigma | calls |
|---|---|---|---|---|
| 0 (stock predictor-only) | 2.598 | 1.53 | 0.948 | 594 |
| 1 (exact reverse SDE) | 0.314 | 1.15 | 0.131 | 594 |
| 2 | 0.049 | 1.03 | 0.034 | 594 |
| **4** | **0.034** | **0.99** | **0.033** | **594** |
| 8 | 0.037 | 1.00 | 0.027 | 594 |
| *DPM2 + 2 correctors* | *0.057* | *1.06* | *0.028* | *1236* |

`eta = 4` beats the two-corrector arm on the shared marginal at 48% of its score
calls, and adding the best initializer on top changes nothing (0.035) — once the
predictor is stochastic, the initial law stops mattering. The composed field is
unchanged; only where along the noise axis it is evaluated changed.

With the **learned** score (`h16d2`) churn is worth more still: 0.036 (`eta = 2`,
`gauss_jacobian`) and 0.019 (`eta = 4`, `gauss_hierarchical`) against 0.265 and
0.270 for the same rules with two correctors, and 0.107 for Langevin + F-NPSE at
the same wall clock. The optimal `eta` drops from 4 to 2 when the score is
learned, and the width ratio (0.98 -> 0.95 -> 0.94 across `eta` = 2, 4, 8) is the
diagnostic that says so without a reference posterior.

## Was it the skew and the hard wall? No — it was the initial law

`symmetric_experiments.py` reruns the whole thing on two drop-in twins of
`hierarchy.py` that change only the noise law, matched on variance:
`hierarchy_gauss` (Normal — symmetric, no wall, **and** Gaussian, so the
composition rules are exact by construction) and `hierarchy_laplace` (Laplace —
the symmetrized exponential: symmetric, no wall, still non-Gaussian).

| noise law | J, ODE | J, churn 4 | H, ODE | H, churn 4 |
|---|---|---|---|---|
| Gaussian | 1.652 | **0.027** | 1.657 | **0.022** |
| Laplace | 1.412 | **0.058** | 1.552 | **0.161** |
| Exponential | 2.598 | **0.034** | 4.700 | **0.051** |

Predictor-only fails by 1.4–1.7 sigma on problems with no wall and no skew, so
the failure is the initial law everywhere. On both symmetric twins its width
ratio is **1.00** — a pure location shift — against 1.53 on the exponential: the
wall is what turns an initialization error into a *shape* error. And the two
composition rules are indistinguishable only where the problem is Gaussian and
the Gaussian composition is therefore exact; add non-Gaussianity and
`gauss_hierarchical` costs a factor of 2–3, consistent with its ~0.92 relative
Jacobian antisymmetry in `artifacts/oracle_diagnostics/exact_curl.csv`.

```bash
python validate_symmetric.py                   # grade both twins first
python symmetric_row_scores.py                 # and their exact row scores
python initialization_experiments.py --device cuda
python stochastic_predictor_experiments.py --device cuda
python stochastic_predictor_experiments.py --device cuda --score learned
python symmetric_experiments.py --device cuda
python symmetric_experiments.py --summary-only # redraw 07 from the CSVs alone
```

No script here edits `compass`, `compare.py`, `analytic_compare.py` or
`hierarchy.py`: each swaps one `MultiObsSampler` method — or, for the twins, the
`hierarchy` name inside `compare` — for the duration of a run and restores it
afterwards.

## Running

```bash
source .COMPASS/bin/activate
cd tutorials/Compositional_Inference_Fix/New_Attempt/exponential_score_comparison

python validate_references.py                  # check every closed form first

CUDA_VISIBLE_DEVICES=0 python capacity.py --config h8d1 --train-samples 200000
python plot_capacity.py --config h8d1

python compare.py                              # all three rules
python compare.py --stage score                # composed-score layer only
python compare.py --replot                     # redraw, no resampling
```

`capacity.py` runs one (config, simulations) cell per process so the sweep can be
spread over GPUs; rows are appended as they finish and existing rows are skipped,
so a killed sweep resumes. `compare.py`'s metrics CSV is merged by method name
rather than overwritten, so running one rule does not drop the other two.

Every script caps itself to 3 threads before importing torch, per the
repository's `CLAUDE.md` rule.

### The training recipe is a *step* budget, not an epoch count

An epoch on 10K simulations is twenty batches and an epoch on 200K is four
hundred. Holding epochs fixed across the data ladder would hand the large budgets
twenty times the optimization and then report the difference as a data effect.
`train.schedule` rescales `max_epochs` and `early_stopping_patience` by
`200000 / train_samples`, so every rung gets ≈156K gradient steps and only the
data varies.

## Artifacts

| File | Contents |
|---|---|
| `artifacts/capacity.csv` | One row per (architecture, simulation budget) |
| `artifacts/00_capacity.png` | The sizing decision: fidelity and score error vs capacity and vs data |
| `artifacts/methods.csv` | One row per composition rule, every metric in the figures |
| `artifacts/01_<method>.png` | Four-panel layout per rule (observations, shared posterior, locals, recovery) |
| `artifacts/02_method_comparison.png` | The head-to-head: composed score, posterior, marginal score, MAP budget |
| `artifacts/<method>.npz` | Draws, KDE curve, `ĝ`, local MAPs, implied and exact scores |
| `artifacts/network_floor.json` | The network's own single-observation score error per λ |
| `artifacts/models/<config>_n<samples>/` | Checkpoints |
| `artifacts/initialization_experiments.csv`, `05_initialization_experiments.png` | One row per initial-law variant, predictor-only |
| `artifacts/stochastic_predictor_experiments.csv`, `06_stochastic_predictor.png` | One row per churn strength `eta`, corrector-free |
| `artifacts/stochastic_predictor_<variant>/` | Four-panel layout and draws per `eta` |
| `artifacts/stochastic_predictor_experiments_learned.csv`, `06_stochastic_predictor_learned.png` | The same sweep with the trained network, plus a Langevin+F-NPSE timing reference |
| `artifacts/symmetric_experiments.csv`, `07_symmetric_summary.png` | Three noise laws x two rules x churn on/off |
| `artifacts/symmetric_<problem>_<arm>/` | Four-panel layout and draws per symmetric arm |
