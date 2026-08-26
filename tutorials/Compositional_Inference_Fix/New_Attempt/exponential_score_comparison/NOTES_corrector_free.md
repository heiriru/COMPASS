# Getting a corrector-free DPM2 onto the exact shared posterior

Running log of what was tried, what it measured, and what is therefore ruled
out. **Read the "Ruled out" table before proposing anything** — several obvious
ideas are already dead with numbers attached.

The goal: reproduce `artifacts/01_gauss_jacobian_analytic_correctors2.png`
(W1 = **0.057** posterior sigma on the shared marginal, width ratio 1.06,
local W1 0.028) using DPM2 with **zero corrector steps**, while keeping the
compositional score modelling valid — i.e. still composing the exact
single-observation score with `correction="gauss_jacobian"`, no oracle knowledge
of the tall posterior anywhere in the sampler.

Reference arms, all with the exact single-observation score (`methods_analytic.csv`):

| arm | shared W1/σ | width | local W1/σ | score calls |
|---|---|---|---|---|
| `gauss_jacobian_analytic_correctors2` (**target**) | 0.057 | 1.06 | 0.028 | 1236 |
| `gauss_jacobian_analytic` (10 correctors) | 0.059 | 1.05 | 0.028 | 3804 |
| `gauss_jacobian_analytic_predictor_only` | **2.598** | 1.53 | 0.948 | 594 |

---

## The diagnosis the existing artifacts already support

Two separable error terms, not one.

**1. The initial law (~85% of the 2.6 σ).** `MultiObsSampler._initial_sample`
puts every latent at **zero** plus `lambda_max` noise. The exact diffused joint
at `lambda_max = 3.89` is centred on `E[g|x] = 1.39` and, for the locals, on
`(E[g|x] + x_j)/2 ∈ [1.5, 4.2]`. A probability-flow ODE is a deterministic
transport: it returns the pushforward of whatever it starts from and has no
mechanism to forget it. `initialization_experiments.csv` measures this directly —
an exact diffused start (A0) takes 2.598 → 0.387.

**2. The finite-noise composition rule (~0.4 σ, the residual).**
`03_rule_versus_sampler.png` panel 1 grades the composed field alone against the
exact quadrature: `gauss_jacobian` sits at 0.10 relative RMS at λ = 0.05, peaks
at **0.65 near λ = 0.5**, and returns to 0.13 at λ = 2. The field is good at both
ends of the schedule and wrong in the middle, and the ODE transports that
mid-schedule error straight to t = 0.

**Why a Langevin corrector fixes both at once.** It relaxes the ensemble onto the
*composed* density at the current noise level. Because the composed field is
nearly exact for small λ, the late correctors re-equilibrate onto very nearly the
right density — erasing the initial-law error and the accumulated mid-λ transport
error in one move. The mechanism that matters is **stochastic relaxation while
the field is accurate**, not "corrector steps" as a label.

---

## Ruled out (do not retry)

| Idea | Evidence | Verdict |
|---|---|---|
| More timesteps | 2.598 @ 100 vs 2.606 @ 50 steps; oracle sweep 2.254 / 2.244 / 2.241 / 2.240 σ at 50/100/200/400 | Integrator is converged. Dead. |
| Higher DPM order (3) | Same argument — a converged ODE does not become a different ODE | Dead without a new experiment. |
| Smaller endpoint `eps` | learned Jacobian-GAUSS worsens 0.650 → 1.835 going 1e-3 → 1e-4 (README) | Exposes the region where the composed field is least reliable. Dead. |
| Fix the initializer alone | Best *principled* start A0 (exact diffused joint) = 0.387; VP SDE (terminal law genuinely forgets the data) = 0.511; wide schedule σ=25 = 0.330; diffused-prior start = 1.315 | Necessary, ~7× short of the target. Not sufficient. |
| A1b "single-observation posterior means" start | 0.094 — closest predictor-only result before this work | **Fortuitous.** A0 has the *correct* initial law and scores 4× worse (0.387), so A1b's number is error cancellation between a wrong start and a wrong field, not a fix. Width ratio 0.896 (too narrow) shows the residue. Do not build on it. |
| More/denser λ grid in the mid-λ band | Same converged-ODE argument as timesteps | Dead. |

---

## Tried here: turn the predictor into the reverse SDE (`stochastic_predictor_experiments.py`)

**Idea.** If the mechanism is stochastic relaxation, get it from the *predictor*
rather than from extra corrector steps. Karras et al. (2022, Algorithm 2): at
each level take a short step **up** in noise, then run the deterministic DPM2 step
from the raised level down to the next one. Noise-up + a longer ODE-down *is* the
reverse-time SDE. It is a predictor, not a corrector, and it costs **no extra
score evaluations** — 594 calls, the predictor-only budget, versus 1,236 for the
two-corrector arm.

**Parameterization.** In noise-scale coordinates the VE probability-flow ODE is
`dx = -σ s(x,σ) dσ` and the reverse SDE is
`dx = -2σ s dσ + sqrt(2σ|dσ|) dw`. Injecting variance `σ̂² - σ² ≈ 2σδ` and then
integrating the ODE from `σ + δ` down to `σ_next` reproduces the SDE exactly when
`δ = σ - σ_next`. So

    σ̂ = σ_i + eta · (σ_i - σ_{i+1})

with **eta = 0** the untouched probability-flow ODE (today's predictor-only arm),
**eta = 1** the exact reverse SDE, and **eta > 1** the generalized family of
Karras eq. (6) — a stronger Langevin term that leaves the marginals invariant for
an exact score and only trades discretization error for mixing.

**Why the sweep goes up to eta = 8.** The schedule is geometric with ratio
r = 0.9526 over 100 steps, so eta = 1 injects `2σ²(1-r) = 0.095 σ²` of variance
per level, while the two-corrector arm's Langevin steps inject
`2 · 2 · snr · σ² = 0.8 σ²`. Matching the corrector arm's *noise budget* at zero
extra score cost needs eta ≈ 8; Karras' stability cap `gamma ≤ sqrt(2) - 1`
allows eta ≤ 8.7 on this schedule.

**Isolation.** Nothing in `compass`, `compare.py` or `analytic_compare.py` is
touched. `MultiObsSampler._dpm_sampler` is swapped for the churned variant for
the duration of one run and restored afterwards, exactly as
`initialization_experiments.py` swaps `_initial_sample`. The replacement
*refuses* a nonzero corrector request rather than ignoring it, so no arm can
silently stop being corrector-free. Artifacts land in
`artifacts/stochastic_predictor_*/` and `artifacts/06_stochastic_predictor.png`.

### Result — the target is met at eta ≥ 2, and beaten at eta = 4

Seed 0, 30 observations, 3,000 draws, 100 steps, exact single-observation score,
`correction="gauss_jacobian"`, **no corrector steps of any kind**
(`artifacts/stochastic_predictor_experiments.csv`):

| arm | eta | shared W1/σ | mean err/σ | width | local W1/σ | marginal-score RMSE | calls | s |
|---|---|---|---|---|---|---|---|---|
| `P0_baseline` | 0 | 2.598 | +2.598 | 1.535 | 0.948 | 1.880 | 594 | 207 |
| `C_eta1` | 1 | 0.314 | +0.309 | 1.148 | 0.131 | 0.346 | 594 | 206 |
| `C_eta2` | 2 | 0.049 | +0.020 | 1.025 | 0.034 | 0.269 | 594 | 206 |
| **`C_eta4`** | **4** | **0.034** | **+0.016** | **0.995** | **0.033** | **0.198** | **594** | **206** |
| `C_eta8` | 8 | 0.037 | +0.011 | 0.996 | 0.027 | 0.228 | 594 | 206 |
| `C_eta4_A1b` | 4 | 0.035 | +0.024 | 0.991 | 0.037 | 0.192 | 594 | 205 |
| *target* `..._correctors2` | — | *0.057* | *+0.044* | *1.056* | *0.028* | *0.185* | *1236* | *430* |

`P0_baseline` reproduces `gauss_jacobian_analytic_predictor_only` to five digits
(2.5982 vs 2.598241), so the harness is measuring the same thing as
`analytic_compare.py` and the improvement is not a change of yardstick.

**`C_eta4` beats the two-corrector target on the shared marginal** — W1 0.034 vs
0.057, mean error 0.016 σ vs 0.044 σ, width ratio 0.995 vs 1.056 — matches it on
the locals (0.033 vs 0.028) and the KDE-implied marginal score (0.198 vs 0.185),
at **48% of the score calls** (594 vs 1,236) and **48% of the wall clock**
(206 s vs 430 s). `artifacts/stochastic_predictor_C_eta4/01_C_eta4.png` overlays
the exact density as well as `01_gauss_jacobian_analytic_correctors2.png` does.

Three things the sweep settles:

- **It is a monotone, saturating knob, not a tuned coincidence.** 2.598 → 0.314 →
  0.049 → 0.034 → 0.037 across eta = 0, 1, 2, 4, 8. The curve flattens at the
  Monte-Carlo/quadrature floor between eta = 2 and 8, so no eta was fitted to the
  answer — anything in [2, 8] lands on the target.
- **The exact reverse SDE (eta = 1) is not enough**, and the reason is
  quantitative: this schedule injects only `0.095 σ²` per level at eta = 1
  against the two-corrector arm's `0.8 σ²`. The predicted eta ≈ 8 needed to match
  that budget brackets the observed plateau, which is what makes the noise-budget
  argument above load-bearing rather than decorative.
- **Churn subsumes the initialization fix.** `C_eta4_A1b` (churn + the best
  initializer from `initialization_experiments.py`) is 0.035 against `C_eta4`'s
  0.034 — indistinguishable. So once the predictor is stochastic the initial law
  stops mattering, and the A1b start can be dropped. That is the falsifiable half
  of the claim, and it held.

**Compositional validity is untouched.** The composed field is still
`correction="gauss_jacobian"` over the exact single-observation score, evaluated
by the same `_get_score` path; only *where along the noise axis the sampler
evaluates it* changed. No pilot, no tall-posterior reference, and no oracle enters
the sampler — the `hierarchy` closed forms are used solely to grade the output.
Shared-coordinate synchronization across observation rows is preserved because the
injected noise comes from `_shared_noise`, and `compare.draw` asserts it (max
row-to-row deviation < 1e-5) on every arm.

**Recommendation.** eta = 4, i.e. `sigma_hat = sigma_i + 4 (sigma_i - sigma_{i+1})`
before each DPM2 step, with correctors off. Cheapest and most accurate arm
measured on this problem. Caveat before promoting it into `compass`: the plateau
width in eta was established on *one* schedule (geometric, 100 steps, VESDE
sigma = 8) and *one* seed. eta is defined relative to the schedule's own step
length, so it should transfer across step counts by construction, but that has
not been measured here.


---

## Not tried — and, after the result above, not needed for this target

Kept so nobody spends a day rediscovering that they were unnecessary.

1. **Restrict churn to a λ band** (`--churn-sigma-min`/`--churn-sigma-max`, wired
   but unused, both at their permissive defaults in every arm above). The
   motivation was that the field is worst at λ ≈ 0.5, so churning only *below* it
   would avoid re-injecting samples into the inaccurate band — Karras'
   `S_tmin`/`S_tmax`. Unnecessary: full-schedule churn already lands at the
   Monte-Carlo floor. Worth revisiting only if a harder problem shows churn
   *hurting* at mid-λ.
2. **`S_noise` above 1** (`--s-noise`, wired, left at 1.0). Karras' empirical fix
   for over-contraction. The width ratio at eta = 4 is 0.995, so there is nothing
   for it to fix here.
3. ~~Churn + a correct initializer~~ — **tested and closed**: `C_eta4_A1b` is
   indistinguishable from `C_eta4`. Churn subsumes the initializer.
4. **A better composition rule at mid-λ.** The 0.65 relative RMS peak at λ ≈ 0.5
   remains, and is now the binding floor: it is what stops the eta sweep
   improving past ~0.034. Fixing it is a research change to
   `_compositional_score` (beyond the Gaussian / Tweedie-second-order
   approximation), not a sampler knob. Not required for this target — but it is
   the honest answer to "what is left".

## Symmetric-noise twins: was it the skew and the wall? (No.)

`symmetric_experiments.py`, on two drop-in replacements for `hierarchy.py` that
change only the noise law, both matched to `Var(eps) = Var(eta) = 1`:

- **`hierarchy_gauss`** — Normal noise. Symmetric, no wall, **and Gaussian**, so
  the Gaussian/Tweedie composition every rule rests on is *exact at every noise
  level*. Whatever error remains is the sampler, the pilot's Monte-Carlo noise,
  or the KDE.
- **`hierarchy_laplace`** — Laplace noise: the *symmetrized exponential*.
  Symmetric and no wall, but still non-Gaussian, so the rules stay approximate.
  This is the twin that separates skew from non-Gaussianity.

Exact single-observation score throughout, no corrector steps anywhere, W1 /
posterior sigma (Monte-Carlo floor at n = 3,000 is 0.024):

| noise law | J, ODE | **J, churn 4** | H, ODE | **H, churn 4** |
|---|---|---|---|---|
| Gaussian — shared | 1.652 | **0.027** | 1.657 | **0.022** |
| Gaussian — local | 0.577 | **0.023** | 0.580 | **0.024** |
| Laplace — shared | 1.412 | **0.058** | 1.552 | **0.161** |
| Laplace — local | 0.504 | **0.031** | 0.552 | **0.056** |
| Exponential — shared | 2.598 | **0.034** | 4.700 | **0.051** |
| Exponential — local | 0.948 | **0.033** | 1.136 | **0.097** |

(`J` = `gauss_jacobian`, `H` = `gauss_hierarchical`; figure
`artifacts/07_symmetric_summary.png`, table `artifacts/symmetric_experiments.csv`,
four-panel layouts in `artifacts/symmetric_<problem>_<arm>/`.)

**1. The predictor-only failure is not the skew or the wall.** It is 1.4-1.7
sigma on both symmetric twins, which have neither. It is the initial law, on
every noise law tested.

**2. The symmetric twins show that error in a purer form than the exponential
one does.** Predictor-only width ratio is **1.00** on both symmetric problems --
a pure location shift with the shape intact -- against **1.53** on the
exponential. On the Gaussian twin this is exact rather than empirical: the rules
are exact there, so the ODE's field is exact, the flow map is affine, and a wrong
initial *mean* can only translate the answer. The hard wall is what converts an
initialization error into a *shape* error.

**3. Churn repairs all six cases**, 1.4-4.7 sigma down to 0.022-0.161.

**4. The two rules are indistinguishable only when the problem is Gaussian.** On
the Gaussian twin both sit on the sampling floor and within noise of each other
(0.022-0.027 shared, 0.023-0.024 local) -- as they must, since the Gaussian
composition is exact by construction there. Add non-Gaussianity and the gap opens
at once: Laplace 0.161 vs 0.058 shared, 0.056 vs 0.031 local. **This is the
control `exact_curl.csv` needed**: `gauss_hierarchical`'s ~0.92 relative Jacobian
antisymmetry costs nothing on a Gaussian problem and a factor of 2-3 as soon as
the problem is not Gaussian.

Wall clocks in `symmetric_experiments.csv` (2 s to 1076 s) are dominated by
row-score cost -- the Gaussian row score is closed-form linear algebra, the
Laplace one a 769-node quadrature under `jvp` -- and say nothing about method
cost. Use the learned-score timings below for that.

### Provenance of the two twins

Neither is trusted on assertion. `validate_symmetric.py` grades every closed form
against a route the module does not use: the diffused score against a
**brute-force 2-D quadrature** of `int int p(g,l|x) N(g_t;g,lam^2) N(l_t;l,lam^2)`
(1e-12 Gaussian, 2e-5 Laplace, over lam = 0.05 to 3), the local references
against Monte-Carlo moments, the simulator against its own noise moments and the
within-dataset spread that distinguishes a shared `g` from a fresh one, and the
whole posterior by SBC. `symmetric_row_scores.py` grades the row scores against
the N = 1 reference (~1e-9, at the floor set by `VESDE` storing `sigma` in
float32) and `jvp` against central differences (~1e-8).

One SBC note worth keeping: Laplace SBC was **borderline at 400 replicates**
(KS 0.0700 against a 0.0680 critical value) and resolved at 1600 (KS 0.0156,
mean rank 0.4992). It shrank rather than persisted, which is what separates
Monte-Carlo noise from a bias in the reference -- but the 400-replicate run on
its own would have been a false alarm, and a smaller one would have been a
missed one.

## The learned score, and cost

`--score learned --config h16d2` (200K simulations), same GPU and session, with a
same-session Langevin+F-NPSE arm so the wall clocks are comparable
(`artifacts/stochastic_predictor_experiments_learned.csv`):

| method | eta | wall | calls | shared W1/σ | width | local W1/σ |
|---|---|---|---|---|---|---|
| `gauss_jacobian` predictor-only | 0 | 93 s | 594 | 2.321 | 1.58 | 0.921 |
| `gauss_jacobian` churn | 2 | 115 s | 594 | **0.036** | 0.98 | 0.043 |
| `gauss_jacobian` churn | 4 | 103 s | 594 | 0.045 | 0.95 | 0.043 |
| `gauss_jacobian` churn | 8 | 89 s | 594 | 0.062 | 0.94 | 0.041 |
| `gauss_hierarchical` predictor-only | 0 | 123 s | 198 | 3.826 | 2.17 | 1.013 |
| `gauss_hierarchical` churn | 4 | 106 s | 198 | **0.019** | 1.01 | 0.112 |
| Langevin + F-NPSE | — | 105 s | 1000 | 0.107 | 1.10 | 0.042 |

against the published learned corrector arms (`methods.csv`):
`gauss_jacobian_correctors2` 0.265 (53 s), `gauss_hierarchical_correctors2`
0.270 (45 s), `gauss_jacobian` with 10 correctors 0.663 (186 s).

**Churn beats every learned corrector arm by 7-14x** and F-NPSE by 2.4-5.6x at
equal wall clock. It also lands where the *exact* score lands (0.036 vs 0.034),
which is not a contradiction of the README's "network is worth 5-11x" note:
composing 30 observations averages *independent* network error down by ~sqrt(30),
and once a sampler relaxes onto the field instead of transporting an initial law
through it, that averaging is what survives.

**The optimal eta moves down with a learned score, and the width ratio says so.**
Exact score: width 0.995 -> 0.996 across eta = 4 -> 8. Learned: 0.98 -> 0.95 ->
0.94 across eta = 2 -> 4 -> 8, with W1 degrading monotonically. Stronger Langevin
driving pushes the ensemble harder onto the *learned* field's invariant density,
so a slightly over-confident network gives a progressively too-narrow posterior.
Use **eta = 2 with a learned score, eta = 4 with an exact one** -- and read the
width ratio to tell which side you are on, which needs no reference posterior.

### Cost: only same-session wall clocks mean anything

**Cross-session timings on this host are worthless.** Identical work
(`gauss_jacobian` predictor-only, 594 calls) took 26.8 s in the original
`compare.py` run, 93.0 s in one of this script's runs, and `C_eta2` took 115.0 s
on one GPU and 39.8 s on another -- a 2.9x spread driven by GPU load, not by
anything algorithmic. Every cost claim here therefore comes from **one process,
one device, one session**, and the corrector arms were re-run inside this harness
rather than quoted from `methods.csv`. They reproduce their published accuracy
exactly (0.2653 and 0.2697), which is what licenses comparing them at all.

Learned score, all five arms in a single session
(`R_*` = corrector references, no churn):

| arm | wall | calls | shared W1/σ | width | local W1/σ |
|---|---|---|---|---|---|
| **`gauss_jacobian` + churn eta 2** | **39.8 s** | 594 | **0.036** | 0.98 | 0.043 |
| `gauss_hierarchical` + churn eta 4 | 47.1 s | 198 | **0.019** | 1.01 | 0.112 |
| Langevin + F-NPSE | 48.9 s | 1000 | 0.107 | 1.10 | 0.042 |
| `gauss_hierarchical` + 2 correctors | 58.8 s | 412 | 0.270 | 1.01 | 0.079 |
| `gauss_jacobian` + 2 correctors | 85.4 s | 1236 | 0.265 | 0.99 | 0.035 |

So against its own composition rule, churn is **2.1x faster and 7.4x more
accurate** than the two-corrector arm, and against F-NPSE **1.2x faster and 3.0x
more accurate**. The mechanism is simply that churn buys its Langevin relaxation
with zero extra score evaluations, while correctors buy it with 642 more.

Note the call counts do not order the wall clocks: `gauss_hierarchical` bills 198
calls but pays a 4096-draw pilot per observation on top, and `gauss_jacobian`
bills 3 counted calls per evaluation because of the `jvp`. Use seconds, not
calls, and only within a session.

**`gauss_hierarchical` + churn: best shared, worst local.** 0.019 shared (the best
number in the table) but 0.112 local, ~2.7x worse than everything else. The same
split appears with the exact score (0.051 shared / 0.097 local). Its composed
field carries ~0.92 relative Jacobian antisymmetry against `gauss_jacobian`'s
~0.09, and the curl evidently does not corrupt the shared marginal but does
corrupt the locals. Do not read 0.019 as "the best method".

## What this does *not* claim

The exact single-observation score is used throughout, so none of these numbers
include network error. The learned-score arm (`--score learned --config h16d2`)
has not been run with churn; `03_rule_versus_sampler.png` panel 4 suggests the
network is worth 5–11× *once a sampler relaxes onto the field*, so the learned
churned arm should be expected to land well above 0.034 — that is a separate
measurement, not a contradiction.
