# COMPASS as a model-comparison tool — validation plan / TODO

Started 2026-10-13. The goal is to answer one question with evidence:
**can a difference in COMPASS's model-comparison output be attributed to the models,
rather than to the network, the composition rule, the PF-ODE solver or the criterion?**

Every item below has the same structure:

- **Problem**: what is wrong or unverified, with the code location.
- **Implementation**: what to build (library code in `src/compass/`, experiments,
  tests and reports in `validation/<item>/`).
- **Experiments**: written so that each has a *known answer* or a *falsifiable
  prediction* stated *before* running it.
- **Critical thinking**: the ways the experiment could pass while the method is still
  wrong, and the controls that rule those out.
- **Status**.

General rules (apply to every item):

1. **Hypothesis first.** Write the predicted number or the predicted direction into the
   item's README before running. A result that only "looks reasonable" is not a pass.
2. **Separate the error sources.** Every new check is run at three levels, in this order:
   (a) closed form / exact likelihood, (b) the COMPASS pipeline with an **exact-score
   oracle** (`compass.experimental.oracles`), (c) the pipeline with a **trained**
   network. A failure at (a) is a criterion bug, at (b) a solver/rule bug, and at (c) a
   network problem. Never debug (c) before (a) and (b) pass.
3. **Noise floor before differences.** Every reported difference is compared with the
   Monte-Carlo or seed spread of the same quantity. A tolerance of "zero" is never the
   right one (see memory: stability tolerances vs the noise floor).
4. **Rates, not anecdotes.** Model-selection claims are rates over ≥ 200 replicated
   datasets with a binomial confidence interval, never a single dataset.
5. **Vary the seed of the test data**, not only the seed of the method. The IK
   calibration "deficit" was a measurement artifact of one test seed.
6. **Thread-pinned numerics** (3-thread cap, `Partial_Pooling/runtime.py`) and
   `autocvd` before any GPU use.
7. **Report what failed.** Negative results go into the item's README with the same
   detail as positive ones.
8. **Plot every test result**, unless a small table or a simple diagram shows it more
   efficiently (e.g. three numbers).
   - Figures are produced by a `plots.py` in the item's folder, from the *saved*
     results (no recomputation), and saved as PNG + PDF in `results/figures/`.
   - Use the shared style in `common/plotting.py`. Fixed colours: new method blue,
     legacy/baseline orange, exact reference aqua.
   - Show the measured value **with its uncertainty** (CI bars, seed bands) next to the
     **prediction** (line), so a pass or fail is visible at a glance.
   - Prefer per-dataset scatter against the exact value (y = x reference) over
     aggregate bars: it shows *which* datasets fail.
   - Look at every rendered figure before committing it (label collisions, hidden
     points, log scales where the data span decades).
   - Each item's README embeds its figures.

Legend: `[ ]` open · `[~]` in progress · `[x]` done and validated.

---

## A. Information criteria

### [x] A1 — parameter counting and sample size in AIC/AICc/BIC

**Problem.** `ModelTransfuser.compare` computes the criterion *per observation* with
the full latent dimension `k` and sums over observations (`ModelTransfuser.py:465`).

- Penalty = `N · 2k`. In multi-observation (hierarchical) inference `k` includes the
  global parameters, so a global is penalised N times instead of once.
- AICc uses `n = N` although every per-row term is fitted to one row, and falls back to
  plain AIC silently when `n − k − 1 ≤ 0`.
- BIC becomes `N · k · ln N`.

**Implementation.** `src/compass/information_criteria.py`:

- Explicit counting `K = k_global + N · k_local`.
- An explicit data count `n_data = N · n_per_observation`. The default
  `n_per_observation` is the number of observed scalars per row.
- BIC penalises each parameter block by the log of the data that inform it:
  `k_global · ln(n_data) + N · k_local · ln(n_per_observation)`.
- Per-row contributions that sum exactly to the total, for `obs_probs` and the
  cumulative plot.
- `compare(..., ic_convention="counted"|"legacy")`. The legacy option is kept only so
  that old numbers can be reproduced.

**Experiments.**

- *E-A1.1 Hand-computed values.* Unit tests on every formula: AIC, AICc and BIC for
  flat and hierarchical models, plus the per-row decomposition summing to the total.
- *E-A1.2 Overfit rate of one superfluous global* (the key test).
  - Two nested hierarchical models whose rows are `x_i = (l_i + ν ε, g + ν ε)`:
    M1 has a global offset `g`, M0 fixes `g = 0`. Data are drawn with `g = 0`.
  - With exact MLEs the likelihood-ratio statistic is exactly χ²₁, for every N.
  - **Prediction:** AIC picks M1 with probability `P(χ²₁ > 2) = 0.1573`, independent
    of N. BIC picks it with probability `P(χ²₁ > ln n_data)`, which falls with N.
  - **Legacy prediction:** almost never picks M1 (the penalty grows like N).
  - Check: the measured rate over R = 2000 datasets lies inside the 99% binomial
    interval of the prediction, for N ∈ {4, 16, 64, 256}.
- *E-A1.3 Power.* Same models with `g = δ ≠ 0`. The selection rate of M1 must match
  `P(noncentral χ²₁(λ = N δ²/ν²) > 2)`. This catches a criterion that gets the null
  right only by being insensitive.
- *E-A1.4 Superfluous local parameter, flat model* (each row has its own θ).
  - **Prediction:** the overfit rate is `P(χ²_N > 2N)`, which goes to 0 with N.
    The legacy counting gives the same AIC here, so this is a control that the new code
    did not break the non-hierarchical case.
- *E-A1.5 Through the real pipeline.* The same nested pair through
  `ModelTransfuser.compare` with exact-score oracle networks and the PF-ODE likelihood.
  The rate must match E-A1.2 within its interval. This is not possible before A2,
  because AIC theory needs the MLE.

**Plots.** Done: `a1_information_criteria/results/figures/fig_selection_rates.png`. Rate vs N against the χ² prediction for null, power and flat-local panels.

**Critical thinking.**

- The χ² prediction holds only if the plugged-in likelihood is the *maximum*. With the
  MAP the statistic is shrunk and the rate drops. A low rate can therefore mean "good
  penalty" *or* "MAP bias". Run E-A1.2 with exact MLEs first (pure criterion test), and
  only then through the pipeline.
- The model must be regular. If `g` sat on a boundary (e.g. a variance at 0), the null
  distribution would be ½χ²₀ + ½χ²₁ and the prediction changes. The test uses an
  unbounded location parameter on purpose.
- What does "n" mean for BIC in a hierarchical model? It is genuinely ambiguous (number
  of rows vs number of scalars). The convention is written down, and the test checks
  *consistency* (rate → 0) rather than the exact constant.
- **Conditional vs marginal AIC.** Counting locals as free parameters is the
  *conditional* AIC. It overcounts degrees of freedom when the locals are shrunk
  (Vaida & Blanchard 2005). The marginal AIC (locals integrated out, `K = k_global`) is
  the other defensible choice. COMPASS can compute its likelihood through the identity
  `p(x_i | g) = p(l, x_i | g) / p(l | x_i, g)` (both sides are PF-ODE densities). That
  is item A4.

**Status / results (2026-10-13).** Details are in `a1_information_criteria/README.md`.

- Closed-form selection rates match χ² predictions: 93/96 and 96/96 cells inside 99%
  intervals on two data seeds.
- The legacy counting had zero power for a real global effect from N = 16 on.
- Found and fixed a per-model AICc fallback that always picked the larger model in the
  flat case.
- AICc is very conservative with many locals, so prefer `aic_plain` for hierarchies.
- BIC is inconsistent for locals when ln(n_per_observation) ≤ 1 (warning added).
- Pipeline-level E-A1.5 passed together with E-A2.3: counted AIC through `compare()`
  gives a 0.155 overfit rate (prediction 0.157).

### [~] A2 — plug in the maximum-likelihood estimate, not the MAP

**Problem.**

- AIC and BIC are defined at the maximum of the likelihood, but `compare` plugs in the
  posterior MAP (`map_method="score"`). The MAP is pulled toward the prior, so
  `log p(x | θ_MAP) ≤ max_θ log p(x | θ)`. The fit term is biased in favour of models
  whose priors happen to be broad.
- In multi-observation mode, `map_method="score"` also fixes the global parameters at
  the posterior *mean*, not at any mode.

**Implementation.**

- `PFODE.log_prob(..., differentiable=True)`: the same Heun integrator, with
  `create_graph` through the exact divergence, so that `∂ log p(x | θ) / ∂θ` is the
  gradient of the very number the criterion uses.
- `compass.likelihood_maximization.maximize_likelihood`: L-BFGS-B ascent of the PF-ODE
  log-likelihood over the parameters, with box constraints (it needs only
  `model.log_prob(..., differentiable=True)`).
  - Works in posterior-standardised coordinates.
  - Shared globals are optimised once for all rows, and locals per row.
  - Multi-start from the MAP and several posterior draws.
  - The best value seen is kept, so the result never decreases below the initial log p.
  - Optimisation runs on a coarse grid; the final evaluation uses `log_prob_timesteps`.
- `compare(..., point_estimate="mle"|"map")`. The MLE is restricted to a box (default:
  the posterior sample range widened by `mle_box_width` posterior sds, or a
  user-supplied `theta_bounds`). Outside that box the network has never seen the
  parameters, and a "maximum" there is a network artifact.

**Experiments.**

- *E-A2.1 Exactness on a closed form.*
  - Linear-Gaussian likelihood `x = B θ + ν ε`, B full column rank, exact-score oracle.
  - The closed-form MLE is `(BᵀB)⁻¹Bᵀx` and the maximum log-likelihood is analytic.
  - Prediction: MLE error ≪ posterior sd; log-likelihood gap < 1e-2 nats per row.
- *E-A2.2 Why it matters: MAP bias vs prior strength.*
  - Sweep the prior sd `s ∈ {0.3, 1, 3, 10}·ν`.
  - Prediction: the MAP log-likelihood deficit is `≈ ½ ‖x − B θ_MAP‖² / ν²`, which grows
    as `s` shrinks. The MLE deficit stays at the solver floor for every `s`.
  - Plot the AIC error of the MAP plug-in against `s`.
- *E-A2.3 Selection rate with the real pipeline.* Rerun E-A1.2 through `compare` with
  oracle networks, using both `point_estimate="map"` and `"mle"`.
  - Prediction: MLE ≈ 0.157 (inside the binomial interval).
  - MAP < 0.157 under an informative prior on `g` (the χ² statistic is shrunk).
- *E-A2.4 Cost.* Wall time and score calls of the MLE relative to the existing MAP.
  The number of Adam steps and the coarse grid are part of the method; report the
  log-likelihood they leave on the table against a long run.

**Plots.** Done, in `a2_maximum_likelihood/results/figures/`:

- `fig_plugin_error.png`: AIC error vs prior scale, MAP vs MLE;
- `fig_pipeline_lr.png`: pipeline LR vs exact LR per dataset;
- `fig_nonconcave_shortfall.png`: shortfall below the global maximum per start strategy.
- Still to add: PF-ODE value at the MLE vs true likelihood vs optimiser steps on a trained network (the "too good MLE" check).

**Critical thinking.**

- **Boundary.** If the MLE runs into the box boundary, the network is being queried
  outside its training support. Count and report how often that happens; do not
  silently clip.
- **Non-concave likelihoods** (multimodal, e.g. IK mirror symmetry). Adam finds a local
  maximum, so multi-start is required. Check that the reported optimum does not depend
  on the start, and report the spread across starts.
- **A "too good" MLE on a trained network.** Maximising a learned density rewards
  wherever the network over-estimates density (adversarial-example logic). Compare the
  maximised PF-ODE value with the true likelihood at that θ: a gap that grows with the
  number of optimiser steps means the optimiser is exploiting network error. Validate
  on trained networks in B6/C9 before trusting `point_estimate="mle"` there.
- **Hierarchical models with free locals.** The conditional MLE sets each local to its
  per-row MLE. For a model with as many locals as scalars per row, the conditional
  likelihood can grow without bound as the noise goes to 0. That is a known
  pathology of conditional AIC, not a bug, and is another reason for A4.

**Status / results (2026-10-13).** Details are in `a2_maximum_likelihood/README.md`.

- E-A2.1 passed: PF-ODE MLE location error ~1e-3 posterior sd, log-likelihood deficit
  ≤ 2e-6 nats/row.
- E-A2.2: the MAP plug-in AIC error on 32 rows is 0.4 / 4.6 / 22.6 / 49.7 for s/ν =
  10 / 3 / 1 / 0.3. The default box binds at s/ν = 0.3 (3/32 rows, reported).
- E-A2.3 (real `compare()`, oracle networks, 400 null + 200 effect datasets):
  - the MLE plug-in makes the exact decision on 99.5% of datasets;
  - its rates match prediction: 0.155 vs 0.157 (null), 0.735 vs 0.721 (effect);
  - the MAP plug-in disagrees with the exact decision on 4–10% of datasets, and its LR
    is shrunk by a median 0.76 (predicted 0.75).
- The PF-ODE LR error is small but grows with the effect: −0.012 (null), −0.064
  (effect). M0's large-residual rows are over-estimated more, which is a B7 (tails)
  signal.
- Non-concave likelihoods (`nonconcave.py`, two unequal maxima):
  - a single start in the wrong basin never finds the global maximum;
  - 4 posterior draws find it on 97% of rows;
  - screening 32 draws and optimising the best 3 finds it on 99.75%, with a maximum
    shortfall of 0.005 nats. This is now the `compare()` default (`mle_screen=32`).
- "Too-good MLE" on trained networks (run within B6): the network value at the MLE
  over-states the exact likelihood by +0.22 / +0.08 / +0.03 nats/row (Gaussian, 100
  iterations) at 2¹² / 2¹⁴ / 2¹⁶ simulations, growing with iterations. For small
  budgets, cap the MLE iterations.
- Still open: E-A2.4, the cost split between MAP and MLE.

### [ ] A3 — uncertainty of model probabilities

**Problem.** A softmax over a sum of N per-row criteria is almost always 0 or 1.
Nothing reports how sure that verdict is.

**Implementation.**

- Pointwise per-row contributions (already returned by A1).
- The ΔIC between each pair of models with the standard error
  `sqrt(N · Var_i(ΔIC_i))`.
- Bootstrap over rows for the model-probability distribution.
- Pseudo-BMA+ weights (Yao et al. 2018) as the "stable" alternative to Akaike weights.

**Experiments.**

- *E-A3.1 Coverage of the ΔIC interval.* On the E-A1.3 nested pair with a known
  expected ΔIC (`E[ΔAIC] = N δ²/ν² + 1 − 2`), the 90% interval must cover the expected
  value about 90% of the time over 1000 datasets.
- *E-A3.2 Behaviour at the decision boundary.* At the δ where AIC is indifferent, the
  model probabilities should be spread out, not close to 0/1.
- *E-A3.3 Calibration of the weights.* Bin the reported weights over many datasets
  drawn from a mixture of the two models, and compare with the empirical frequency of
  the generating model. Report the expected calibration error with a bootstrap interval.

**Plots.**

- Coverage curve: nominal vs empirical coverage of the ΔIC interval, with binomial bands.
- Reliability diagram of the Akaike / pseudo-BMA weights: binned reported weight vs empirical frequency, with bootstrap bands.
- Bootstrap distribution of the model weight on one example dataset near the decision boundary.

**Critical thinking.**

- The SE formula assumes independent rows. In the hierarchical case, rows share the
  globals, so the pointwise terms are correlated through ĝ. Check the SE against the
  replicate-to-replicate spread, not only against itself.
- Akaike weights are not posterior model probabilities. Whatever is plotted must not be
  labelled `P(M | x)` unless it is an evidence (A4). Fix the plot labels.

### [ ] A4 — fully Bayesian criteria: evidence, marginal AIC, PSIS-LOO/WAIC

**Problem.** Information criteria are asymptotic proxies. COMPASS can do better,
because the PF-ODE gives normalised densities for any mask.

**Implementation.** Each estimator needs an identity that holds exactly for any θ*, so
its spread over θ* doubles as a diagnostic of network coherence.

- **Single-row evidence:** `log p(x) = log p(θ*, x) − log p(θ* | x)` at any θ*. Both
  terms are PF-ODE densities (all nodes latent, and θ latent given x). This is the
  candidate's formula.
- **Hierarchical evidence:** `log p(x_{1:N}) = Σ_i log p(x_i | g*) + log p(g*) −
  log p(g* | x_{1:N})`.
  - `p(x_i | g*)` comes from the per-row identity with the locals integrated out.
  - `log p(g* | x_{1:N})` needs a density of the *composed* posterior. Candidates:
    - a Gaussian (or mixture) summary of the composed draws;
    - a PF-ODE on the composed score (exact for the composed field; the field may not
      be conservative, see B8);
    - bridge sampling between a fitted proposal and the composed draws.
- **Marginal AIC:** `−2 Σ_i log p(x_i | ĝ) + 2 k_global`.
- **PSIS-LOO / WAIC:** `log p(x_i | θ_s)` over S posterior draws per row. This costs
  S PF-ODE solves per row, so it needs the learned divergence head
  (`test_instantaneous_divergence.py`) or Hutchinson with a few probes.

**Experiments.**

- *E-A4.1 Closed-form evidence.* Linear-Gaussian models with exact `log p(x)`:
  - single-row identity with the oracle at 20 θ* values per x;
  - spread over θ* must be at the PF-ODE solver floor;
  - error against the closed form below 0.05 nats.
- *E-A4.2 Coherence with trained nets.* The same identity with a trained network. The
  spread over θ* measures how inconsistent the network's conditionals are.
  - Prediction: it is larger in the posterior tails.
  - Report it as a function of `|θ* − θ_MAP| / sd`.
  - Choose θ* = MAP if the spread is smallest there.
- *E-A4.3 Bayes-factor accuracy and selection calibration.* Nested and non-nested pairs
  with closed-form evidence: report the log-BF error and its seed spread, then
  calibrate posterior model probabilities as in E-A3.3. These, unlike Akaike weights,
  *should* be calibrated if the evidence is right.
- *E-A4.4 PSIS-LOO.* Compare with exact LOO (closed form for linear-Gaussian), and
  report Pareto-k̂ diagnostics.

**Plots.**

- Identity check: log p(x) from the evidence identity vs θ*-distance from the MAP (in posterior sds), oracle vs trained network.
- Log Bayes factor from COMPASS vs the closed form, per dataset (y = x).
- Reliability diagram of the posterior model probabilities.
- PSIS-LOO: Pareto-k̂ histogram; pointwise LOO vs exact LOO scatter.

**Critical thinking.**

- Evidence is **prior-sensitive** in a way the IC is not. A broad, vague prior is
  penalised. Make sure every candidate's training prior is the prior you *mean*, and
  report the Bayes factors' sensitivity to a prior rescale.
- The identity's θ*-spread can be small while both terms are wrong in the same
  direction (a shared trunk error). Coherence is necessary, not sufficient; the
  closed-form check in E-A4.1/E-A4.3 is still required.

---

## B. PF-ODE likelihood

### [x] B5 — the low-λ end and the endpoints of the PF-ODE integral

**Problem.**

- *Corrected 2026-10-13.* The original statement claimed the PF-ODE integrates through
  untrained noise levels. It does not: `log_sigma` training draws λ down to
  λ(eps = 1e-3) = 0.0317, and `log_prob` starts at the same eps.
- What remains open:
  - (i) the low-λ band, where the learned score shrinks toward the conditional mean
    (memory: score calibration), dominates the density;
  - (ii) the σ(eps) smoothing biases models with narrow likelihoods;
  - (iii) eps below the trained floor extrapolates.
- **Found while building B5/B6: the t = 1 endpoint.**
  - The default VESDE (`sigma = 25`) only reaches λ_max = 9.8.
  - The isotropic endpoint density N(μ, λ_max² I) leaves a model-dependent bias
    (+0.007 nats Gaussian, +0.039 bimodal, with exact scores) that more steps do not
    remove.
  - Fixed by a fitted-Gaussian endpoint, with covariance from the score Jacobian (now
    `endpoint="gaussian"`): +0.0002 / +0.0005 at 400 steps.
  - Whether it is also better with *trained* networks is part of E-B6.1, because
    network Jacobians are unreliable.

**Experiments.**

- *E-B5.1 Error vs eps.* On linear-Gaussian, heavy-tailed (Student-t noise) and
  bimodal likelihoods with a closed form, measure the trained-network log p error for
  `eps ∈ {1e-3, 3e-3, 1e-2, 3e-2}`.
  - Decompose the error into the per-λ-band contribution of the divergence integral
    (accumulate `I` by band).
  - Prediction: the error is dominated by `λ < λ_min,train`.
- *E-B5.2 The three remedies,* compared against the closed form:
  1. train down to `eps` (time sampling floor);
  2. evaluate every model at the same `σ(eps)` and accept the smoothing as part of the
     model;
  3. Richardson-extrapolate `eps → 0` from 3 values.
- *E-B5.3 Smoothing vs data scale.* With nearly deterministic simulators (noise sd
  `≤ σ(eps)`), the `σ(eps)`-smoothed density differs from the model's by
  `O(σ(eps)² / ν²)`. Sweep ν/σ(eps) and confirm that scaling.

**Plots.**

- log p error vs eps for each likelihood family, log-log.
- Stacked contribution of each λ-band to the divergence-integral error.
- Error vs ν/σ(eps), with the predicted O(σ(eps)²/ν²) line.
- ΔlogL error between a model pair vs eps: does it cancel?

**Critical thinking.**

- A smaller absolute error after a remedy is not enough. What matters for comparison
  is whether the error is the **same across candidate models** (it cancels in ΔIC) or
  model-dependent (it does not). Report the error of ΔlogL across a model pair, not
  only per model.
- The smoothing is *on the observations x*. A model with a narrow likelihood loses
  more density to smoothing than one with a broad likelihood. That is a systematic
  bias toward broad-noise models, so check it explicitly.

### [x] B6 — log-likelihood accuracy against ground truth

**Experiments.**

- *E-B6.1 Error budget.* For trained networks on four likelihood families
  (linear-Gaussian, Student-t, bimodal mixture, IK forward kinematics with analytic
  noise), measure log p error against:
  - solver timesteps {25, 50, 100, 200};
  - divergence method: exact, Hutchinson (1, 4, 32 probes), learned divergence head;
  - the t=1 prior approximation (alter σ_max).
  
  Report the error per factor with the other factors at their finest setting.
- *E-B6.2 Error vs training budget and network size.* Plot log p RMSE against
  simulations ∈ {2¹², 2¹⁴, 2¹⁶}. This tells you what budget a given ΔIC resolution
  needs.
- *E-B6.3 ΔlogL error, not logL error.* For a model pair, the quantity that drives the
  decision is the per-row difference. Measure its error and its seed spread; compare
  with the true ΔlogL between the models.

**Plots.**

- Error-budget bars per factor: solver steps, divergence method, prior approximation.
- log p RMSE vs simulation budget, log-log, with seed bands.
- Per-row ΔlogL scatter, network vs exact.

**Critical thinking.**

- Hutchinson's estimator is unbiased for the divergence, but `log p` is not affected
  linearly by its noise once the noise is fed through a later `exp`/softmax. Report the
  bias in Akaike weights, not only in log p.
- Do not use the training simulator's own noise draws as test data. Test x must be
  fresh draws, including some from the tails.

**Status / results (2026-10-14).** Details are in `b5_b6_pfode_accuracy/README.md`
(B5 and B6 together, 30 trained networks).

- Per-row log p RMSE at 2¹⁶ simulations: 0.11 (Gaussian), 0.17 (Student-t), 0.19
  (bimodal).
  - It falls roughly as N^−0.45 and stays 4–9× above the exact-score floor.
  - It is biased low by 0.03–0.05.
- **B5 hypothesis falsified.** The low-λ band contributes nothing. The error comes from
  the mid and high bands (λ > 0.2, largest at λ > 5), so the likelihood needs more
  training weight at *high* noise (→ C10/C11).
- **Fitted-Gaussian endpoint:** exact with oracle scores, but worse with trained
  networks in every case. The default is back on `endpoint="isotropic"`.
- **Cheaper likelihood at no cost:** 25–50 steps and 1 Hutchinson probe are as good as
  100 steps with the exact divergence for trained networks.
- **Model differences do not cancel** (difference error ≈ per-model error).
- **The wrong model's network over-estimates out-of-range data** by +0.8 to +2.2
  nats/row, independent of budget. The Gaussian network gives −10 to −50 where the
  truth is −480. This is a bias toward the misspecified model (→ B7 OOD flag).
- 16-row decisions agree with exact 95–100% at 2¹⁶ and 90–93% at 2¹².
- The tail rows (least likely 5%) have 3–4× the bulk error.

### [ ] B7 — tail and out-of-distribution observations

**Experiments.**

- *E-B7.1 Error vs typicality.* Bin the test rows by the true `log p(x | θ)` quantile
  and plot log p error per bin.
  - Prediction: the error grows in the tails, and the criterion is dominated by exactly
    those rows (the largest −2 log p terms).
  - Measure each bin's share of the total ΔIC.
- *E-B7.2 OOD x.* Observations outside the training x-support, e.g. a misspecified
  noise level.
  - Check whether the PF-ODE density goes wrong in a consistent direction (always too
    high or always too low).
  - Design an OOD flag from the joint PF-ODE density `log p(x)`, the A4 identity's
    θ*-spread, or a posterior predictive check.

**Plots.**

- log p error vs typicality quantile of the row (binned, with CI).
- Cumulative share of total ΔIC vs rows sorted by |contribution|.
- OOD flag ROC: in- vs out-of-distribution rows.

**Critical thinking.**

- A comparison that is decided by three tail rows is a statement about the network's
  tails, not about the models. Always report the leave-the-top-k-out sensitivity of the
  verdict.

### [ ] B8 — consistency of the PF-ODE density with the sampler

**Problem.** Network Jacobians are about 30% antisymmetric (memory). The learned score
is not a gradient field, so the PF-ODE flow density and the SDE sampler's distribution
can disagree.

**Experiments.**

- *E-B8.1 Normalisation.* In one or two observed dimensions, integrate `p(x | θ)` on a
  grid (or by importance sampling for d ≥ 3). The integral must equal 1 within the
  quadrature error. Repeat for 20 θ.
- *E-B8.2 Density–sample agreement.* Draw 10⁵ likelihood samples with the sampler and
  compare their histogram, or a held-out log-score, with the PF-ODE density
  (KS on the probability integral transform `F_PFODE(x_sample)` in 1-D).
- *E-B8.3 Curl vs error.* Measure the antisymmetric part of the network Jacobian along
  the PF-ODE path, and correlate it with the density error of E-B6.

**Plots.**

- Integral of p(x | θ) per θ, against 1, with quadrature error bars.
- PIT histogram of sampler draws under the PF-ODE CDF (should be flat).
- Jacobian antisymmetry vs density error scatter.

**Critical thinking.**

- The PF-ODE density is a valid normalised density of the *flow* even when the score
  is not conservative. "Normalised" (E-B8.1) can therefore pass while "equals what the
  sampler produces" (E-B8.2) fails. Both are needed.

---

## C. Network

### [x] C9 — unequal network error across candidate models

**Problem.** Each candidate model has its own network. A better-trained wrong model
can beat a worse-trained right one.

**Experiments.**

- *E-C9.1 Seed spread vs model gap.*
  - Train each candidate with 5 seeds (training data *and* initialisation).
  - Compute the per-row and total ΔlogL and ΔIC for every seed pair.
  - The comparison is only resolved where the true gap exceeds about three times the
    seed spread. Report the smallest resolvable gap.
- *E-C9.2 Checkpoint selection.* Validation noise is shared across runs (r ≈ 0.87), so
  early stopping picks the luckiest draw. Compare final epoch, best validation epoch,
  and an EMA of the weights on ΔIC error.
- *E-C9.3 Swap test.* Train both models on the *same* simulator (identical models,
  different seeds, different names).
  - Prediction: model probability 50/50 within the seed spread.
  - Anything systematic is pure network bias. This is the cheapest and most telling
    control of the whole plan.

**Plots.**

- Per-seed ΔIC with seed spread vs the true gap, one point per model pair.
- Swap test: histogram of the model probability for identical models (should centre on 0.5).
- ΔIC error by checkpoint rule.

**Critical thinking.**

- Seed spread must include the training data draw, not only the initialisation.
  Otherwise the spread is underestimated.

**Status / results (2026-10-14).** Details are in `c9_network_seeds/README.md`.

- **Swap test (identical models, different seeds), own-family data, 2¹⁶:**
  - 0–2% spurious confident verdicts at N = 16, but 4–50% at N = 64.
  - At 2¹² simulations: 21–29% at N = 16.
  - On misspecified data: 18–47% at N = 16.
- **Full `compare()` pipeline swap:** no confident verdict in 16 datasets, but a
  systematic per-seed offset of ≈ 0.011 nats/row (P = 0.454, 2.3 SE). It grows like N.
- **Resolvable Gaussian-vs-Student-t gap at N = 16** (3 × seed sd): 1.4 nats at 2¹⁶,
  4.2 nats at 2¹².
- **Best-validation checkpoint 30–70% worse than final epoch:** fixed the
  Trainer/`train_models` save path accordingly.

### [ ] C10 — amortisation gap in the likelihood direction

**Experiments.**

- *E-C10.1 Error over the prior.* Map the likelihood-direction score error and the log
  p error over a grid of θ covering the prior, including the prior edges.
  - Prediction: the error is worst at the prior edges, which is where the MLE of
    a misspecified model goes.
- *E-C10.2 Mask menu.* The mask menu decides how much gradient goes to the likelihood
  direction ("uniform six spends 87% on generating x"). Compare menus on log p
  accuracy, not only on posterior accuracy.

**Plots.**

- Heatmap of the log p error over the θ grid (prior support marked).
- log p error by mask menu.

**Status (2026-10-14).** Partly addressed by `likelihood_improvement/`.

- The **inference mask menu** (1/3 posterior, 1/3 likelihood, 1/3 Bernoulli) plus
  **EDM high-noise loss weighting** cut the 2¹⁶ log p error by 32–37% on
  non-Gaussian likelihoods and remove the low bias, with an equal or better posterior.
- Both are opt-in API options now.
- E-C10.1 (error over the prior support) is still open.

### [ ] C11 — matched simulation budgets

**Experiments.**

- *E-C11.1 Budget asymmetry.* Model pair with true ΔIC known. Give one model 4× the
  simulations, and measure the shift in ΔIC and in model probability.
- *E-C11.2 Budget rule.* Decide whether "equal simulations" or "equal error on a held-out
  log p" is the fair rule. The second is measurable only with a closed form; propose a
  proxy (e.g. held-out denoising loss in the lowest λ band) and check it correlates with
  the true log p error.

**Plots.**

- Shift in ΔIC and in model probability vs budget ratio between the two models.
- Proxy (held-out low-λ loss) vs true log p error scatter.

---

## D. Composition / hierarchy

### [ ] D12 — ModeTree into the library, with learned scores

**Problem.** ModeTree fixes the multimodal global-posterior failure at n ≥ 128, but only
with oracle scores and outside `src/compass`.

**Experiments.**

- *E-D12.1* GM, IK reach-only, and folded-|g| with *trained* networks, at
  n ∈ {8, 32, 128}. Grade with:
  - branch masses against the exact reference;
  - W1 within each branch;
  - endpoint residual for IK (W1 is too noisy there, see memory).
- *E-D12.2 Network mirror bias.* A 1% branch bias per row becomes a factor of e^{0.01n}
  in branch mass. Predict the trained-network branch mass from the measured per-row
  bias and check that the prediction holds.

**Plots.**

- Branch mass vs n: exact, oracle and trained network.
- Predicted vs measured branch mass from the per-row mirror bias (y = x).

### [ ] D13 — two moons is score-limited

- Fix the network first (TSM on globals). Measure per-group g-score error at low λ
  before and after.
- Composition rules are only re-tested once the summed error is below the signal.

**Plots.**

- Per-group g-score error vs λ, before and after the network fix.

### [ ] D14 — composed-posterior error on the comparison benchmarks

- Errors in the composed posterior feed the MAP/MLE initialisation (A2), the evidence
  (A4) and the PSIS-LOO draws. On the A-item benchmarks, measure composed-posterior
  error against MCMC references at N ∈ {8, 32, 128}.
- Propagate that error to ΔIC: run the criterion with exact posterior draws vs composed
  draws.

**Plots.**

- Composed-posterior W1 / mean error vs N against the MCMC reference.
- ΔIC with composed vs exact posterior draws, per dataset (y = x).

### [ ] D15 — multi-level composition

- Derived (memory, 2026-10-12) but not implemented. Needed only if a candidate model
  has more than two levels.
- Validate on a three-level linear-Gaussian tree with the exact oracle before trained
  networks.

**Plots.**

- Three-level oracle: composed vs exact marginal per level (mean/sd scatter).

---

## E. End-to-end validation

### [ ] E16 — known-answer suite

Each pair gets an exact reference, so log-BF and ΔIC errors can be computed:

- polynomial order (nested, linear-Gaussian, closed-form evidence);
- number of mixture components (non-nested, bridge-sampling or numerical evidence);
- pooled vs unpooled vs partially pooled hierarchy;
- a nonlinear pair (e.g. Lotka–Volterra vs a damped oscillator) with an MCMC + bridge
  sampling reference.

**Plots.**

- One panel per pair: log BF and ΔIC error vs N, with seed bands.

### [ ] E17 — calibration of model selection (model-level SBC)

- Draw a model from the model prior, then data from that model.
  - Reliability diagram of reported model probabilities.
  - Confusion matrix, and how both change with N.
- Prediction for evidence-based probabilities: calibrated. For Akaike weights:
  over-confident (they are not posterior probabilities).

**Plots.**

- Reliability diagrams (evidence vs Akaike weights).
- Confusion matrices vs N.

### [ ] E18 — misspecification

- Data from neither model: a noise law, a missing component, an outlier fraction.
- Measure how extreme the model probabilities become, and whether the OOD flag of E-B7.2
  fires before the verdict becomes confidently wrong.

**Plots.**

- Model probability vs misspecification strength.
- OOD-flag rate on the same axis.

### [ ] E19 — baselines

- BayesFlow evidential / hierarchical model comparison (Radev et al. 2021;
  Elsemüller et al. 2023).
- NPE + bridge sampling.
- Exact references where possible.
- Same simulation budgets (C11). The BayesFlow vs COMPASS budget lesson from
  "budget-matched 0.928" applies here as well.

**Plots.**

- Accuracy vs simulation budget per method (log x).
- Calibration curves overlaid.

### [ ] E20 — reporting standard

- Fixed thread counts.
- Seeds for data, training and inference all stated.
- Every number with its noise floor.
- Every claimed improvement checked under a second test-data seed.

**Plots.**

- None. This is a checklist.

---

## Findings log (side discoveries while working through the plan)

- **2026-10-13 `compare()` was broken by default on real models.**
  - It passed `corrector_steps=5` to the sampler, and since churn became the default
    (`churn_eta=2`) the sampler refuses that combination.
  - Mocked tests did not notice because they bypass the sampler.
  - Fixed: `corrector_steps=None`, so the sampler's own default applies.
- **2026-10-13 `compare()` takes one `multi_obs_inference` / `hierarchy` for all models.**
  - Comparing a hierarchical model with a non-hierarchical one therefore needs two
    calls, combined with `information_criteria.compare_criteria` (as
    `a2_maximum_likelihood/pipeline_selection.py` does).
  - TODO: per-model `hierarchy`.
- **2026-10-13 tests ahead of source.** 12 tests in `tests/` fail on HEAD and on the
  working tree alike (`test_instantaneous_divergence.py`, 5 PF-ODE sampler tests in
  `test_pfode_vpsde_analytic.py`). They call features that do not exist in `src/`:
  - `equation=` on `sample`;
  - `divergence_head`;
  - `divergence="learned"`.

  Restore those features or retire the tests. The learned divergence head is needed
  by A4 (PSIS-LOO).
- **2026-10-13 `ModelTransfuser.py` runs `warnings.filterwarnings('ignore')` at import.**
  This silences every warning process-wide, including the library's own. Remove it.
- **2026-10-14 the fitted-Gaussian PF-ODE endpoint was briefly the default** (B5, exact
  with oracle scores) and is reverted: worse with every trained network, and it
  explodes where the network Jacobian is flat.
- **2026-10-14 RNG hygiene.** `torch.distributions` samplers (`Chi2`, `Gamma`, …) ignore
  a `generator` and use the global RNG. A seeded test set built with them is not
  reproducible across processes. It invalidated the first B6.3 comparison. Use
  generator-based constructions only.
- **2026-10-14 GPU selection race.** Parallel workers calling `autocvd` at the same moment
  pick the same GPU and run out of memory. Stagger worker starts.
- **2026-10-14 `ModelTransfuser.train_models` could not work on a single GPU/CPU.**
  - It reloads `{path}/{name}.pt`, but the Trainer only wrote `{name}_checkpoint.pt`
    (best validation epoch). The final model was saved only in the multi-GPU branch.
  - Fixed: the final-epoch model is now also saved as `{name}.pt`, after
    `optimizer.eval()`. Schedule-free AdamW's averaged weights are only active in eval
    mode; without validation data the model used to end on raw training weights.
  - `train_models` also never passes `time_sampling`, so it trains with the "mixture"
    default (memory: `log_sigma` is 3× better). Open.
- **2026-10-14 the best-validation checkpoint is the worse model** (C9.2): 30–70% larger
  log p error than the final epoch in every family and budget. The validation DSM loss
  is too noisy to select on.
- **2026-10-14 a default `compare()` call costs about 8–10 min** for two models × 16
  rows on a shared GPU, almost all of it the PF-ODE MLE (4 starts × 50 L-BFGS
  iterations with double backprop). E-A2.4: MLE ≫ MAP. 20 iterations / 2 starts suffice
  at 2¹⁶ (B6).
- **2026-10-15 maximum-likelihood fine-tuning through the PF-ODE breaks the shared
  network's posterior** (samples 7–10% too wide) while not improving per-row log p.
  Not usable with one network.
  - The score-based endpoints let the optimiser beat Gibbs' inequality (held-out NLL
    3.42 < exact 4.04).
  - `endpoint="fixed"` (normalised base, stored moments) is the correct choice whenever
    a PF-ODE likelihood is optimised. See `likelihood_improvement/README.md`.
- **2026-10-18 the networks were under-trained by early stopping.** Patience 40
  (COMPASS default: 20) halts at ~50–120 epochs. Patience 100 gives −15–33% log p error
  and a better posterior. 2¹⁸ simulations give −31–65%.
  - Depth/width results are confounded by early stopping.
  - At 2¹⁸ the isotropic-endpoint solver floor (λ_max = 9.8) becomes a visible share of
    the error (bimodal bias +0.03).
  - Details in `likelihood_improvement/README.md`.
