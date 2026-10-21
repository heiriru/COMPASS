# Oracle bridge experiment on the Gaussian-mixture benchmark

**No network was trained or loaded.** Every single-observation score, every score
Jacobian, the prior score and Hessian, and the tall posterior together with its
diffusion are the exact closed forms of
`Gaussian_mixtures_compositional_score_modelling`. Code:
`ModeTree/experiments/gm_oracle.py` (exact objects),
`gm_bridges.py` (rules and samplers), `gm_metrics.py` (grading),
`exp6_gm_oracle_bridges.py` (stages), `report_gm_oracle.py` (tables),
`figures_gm_oracle.py`. Tests: `ModeTree/tests/test_gm_oracle.py` (10 passing)
pin the exactness of every object the experiment rests on.

---

## The one-sentence answer

> **With perfect single-observation information the remaining multimodal GM
> problem is, in this order: a sampler problem first, then a bridge-definition
> problem — and it is not a Linhart approximation problem at small `n`.**

Unpacked:

1. **The sampler was the largest single error in the whole comparison, and it
   was an error in the *control*.** The deterministic DPM2 predictor, handed the
   *exact* diffused tall score, does not recover the exact posterior: it
   plateaus at W1/sd = 0.219 / 0.139 / 0.097 at n = 8 / 32 / 128, against a
   Monte-Carlo floor of 0.019 / 0.023 / 0.028. More steps, a lower terminal
   `eps` and a final denoising step change nothing. The cause is the *start*:
   the probability-flow ODE is a deterministic map and cannot repair the
   difference between `N(0, lambda_max^2 I)` and the true
   `p_{lambda_max}(theta | x)`; under a bimodal target that difference *is*
   branch mass, and no later step can move it. Three independent fixes confirm
   the diagnosis -- an exact start, a much larger `lambda_max`, and Karras churn
   -- and the last is COMPASS's own shipped default.
2. **Once the sampler is fixed, the constructions separate cleanly by `n`.** At
   n = 8 and n = 32 oracle JAC with a stochastic predictor is essentially exact
   (W1/sd = 0.0100 and 0.0555 against floors of 0.019 and 0.023) while the
   F-NPSE bridge, run to demonstrated Langevin convergence, settles around
   0.19-0.21 — an order of magnitude worse, and worse than what the *same*
   Langevin sampler achieves against the true score (0.014-0.050).
3. **At n = 128 with a genuinely bimodal target every construction fails** while
   the true score with the corrected sampler reaches the floor. That is
   Outcome 4 of the decision table, localised to large `n`. (In the *standard*
   setting n = 128 is only nominally bimodal — see below — and the arms do
   better there for that reason, not because the rules work.)

---

## 0. What the existing implementation actually does

Read before anything was written; three findings matter.

**COMPASS's `fnpe` is not Geffner's literal rule, and it is right not to be.**
`MultiObsSampler._compose_scores` computes
`(1-n)(1-t) * prior_score + sum_j s_j` with the **diffused** prior score. The
paper's bridge uses the *undiffused* prior, `p(theta)^{(1-n)a(t)}`. The
difference is not cosmetic here. The literal bridge's quadratic coefficient at
large `|g|` is `(n-1)a(t)/tau_g^2 - n/(V_1 + lambda^2)`, which for this model
(`tau_g = 0.4`, `V_1 = 0.140`) is **positive — i.e. the bridge is not
normalisable — over most of the schedule**: measured improper at
t = 0.05, 0.1, 0.25, 0.5, 0.75 for n = 8, and the arm run against it produces
finite numbers of order 1e150. COMPASS's docstring already claims this; it is
now measured. **Every F-NPSE result below therefore uses the diffused-prior
bridge, which has the same `t -> 0` limit and is proper everywhere.**

**COMPASS's `damping` is faithful to Arruda.** `_damping_factor` with
`damping_at_data = 1` and the exponential schedule is exactly
`d(t) = d_1^t`, matching BayesFlow's `compositional_bridge`
(`bayesflow/networks/inference/diffusion/diffusion_model.py:831`), whose
default `damping_at_noise = n^-0.5` is the paper's. One deviation, documented
and implemented the BayesFlow way here: BayesFlow multiplies the *whole*
compositional score by `d(t)`, COMPASS multiplies only the shared block.

**COMPASS's adaptive solver is the Arruda/BayesFlow solver.**
`_adaptive_sampler` is the Jolicoeur-Martineau (2021) adaptive stochastic Heun
that `two_step_adaptive_step` implements, with the same `e_rel = 0.1`,
`safety = 0.9` and `exponent = 0.9`. One deviation: COMPASS's default
`adaptive_abs_tol = 0.002576` is ten times tighter than BayesFlow's `0.02576`.
The BayesFlow value is used here and the tolerance ladder is run explicitly.

**`gauss_hierarchical` uses a constant pilot precision** (`_effective_global_factors`
caches on `(1/var_t, n)`), Schur-reduced onto the shared block. The oracle
version therefore uses each row posterior's *exact total covariance*; the
`gauss_jacobian` version uses the *exact row Hessian* through Tweedie's
identity, with COMPASS's own spectral clamp and information projection applied
unchanged. No moment-matched prior curvature enters the JAC arm: its prior
precision comes from the exact diffused **mixture** prior Hessian.

## The bridge coordinate

COMPASS's own diffusion time already runs on `[eps, 1]` with 1 at maximum
noise, so Geffner's `a = (T - t)/T` with `T = 1` transfers as `a(t) = 1 - t`
with `tau = t`. That is the mapping used throughout. Note it is *not* uniform
in `log lambda`: `lambda(t)` is exponential in `t`, so `tau = 0.5` is
`lambda = 1.70`, a quarter of `lambda_max = 6.43`. The bridge pictures are
therefore indexed by a `tau` that *is* uniform in `log lambda`, and each panel
states the `lambda` it is at, so the two conventions are never confused.

## Validation of the endpoints

`tests/test_gm_oracle.py` pins, to 1e-10 or better:

* the clean factorisation `p(g, l_1..l_n | X) = p(g)^{1-n} prod_i p(g, l_i|x_i)`
  up to a constant (spread 5.7e-14 over random states);
* `q_0^F(theta) = p(theta | x_{1:n})` for both the clean-prior and the
  diffused-prior F-NPSE bridge, and the same endpoint for the damped, GAUSS and
  JAC rules;
* the exact diffused tall score, row scores, row Hessians and prior Hessian
  against autograd.

## A property of the benchmark that has to be stated first

The branch separation of this model is
`2 mu_0 / (tau_g^2 sqrt(1/tau_g^2 + n/v))` within-branch sds, which **falls like
`1/sqrt(n)`**:

| n | separation (within-branch sds) | exact dip ratio | exact modes |
|---:|---:|---:|---:|
| 8 | 8.56 | 0.0004 | 2 |
| 32 | 5.32 | 0.099 | 2 |
| 128 | 2.52 | **0.9987** | 2 |

At n = 128 the standard setting's posterior is **bimodal in name only** — a
0.13% dimple — and its *diffused* marginal at the terminal noise level is
outright unimodal. A multimodal comparison at n = 128 in that setting measures
nothing. The whole programme is therefore also run at `mu_0 = 7.5`, which
changes nothing else and restores a separation of 8.58 sds and a dip ratio of
0.00035 at n = 128 — the same difficulty the standard setting has at n = 8.
Results are reported for both and labelled.

---

## 2. The essential sampler control — and its failure

`TRUE-SCORE-DPM2`, the deterministic second-order DPM predictor of
`MultiObsSampler._dpm_solver_2_step` fed the exact diffused tall score:

| n | 20 steps | 50 | 100 | 200 | 500 | MC floor |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.2203 | 0.2192 | 0.2191 | 0.2191 | 0.2190 | 0.019 |
| 32 | 0.1406 | 0.1388 | 0.1386 | 0.1385 | 0.1385 | 0.023 |
| 128 | 0.1001 | 0.0972 | 0.0968 | 0.0968 | 0.0967 | 0.028 |
| 128, `mu_0=7.5` | 0.1625 | 0.1616 | 0.1615 | 0.1614 | 0.1614 | 0.019 |

(W1/sd; the corresponding branch-mass TV is 0.082 / 0.055 / 0.026 / 0.063 and the
width ratio 1.07 / 1.03 / 1.02 / 1.04.)

Perfectly converged in step count, and 5-11x above the floor. Neither a lower
terminal time (`eps` = 1e-4, 1e-5) nor a final Tweedie denoising step moves it.

### What it is

Not the solver. The **start**. Three independent interventions each recover the
floor, and they agree on the mechanism:

| n = 8 | W1/sd | TV | width |
|---|---:|---:|---:|
| `N(0, lambda_max^2)` start (as shipped) | 0.2190 | 0.082 | 1.070 |
| exact `p_{lambda_max}` start | **0.0469** | 0.018 | 1.022 |
| `lambda_max` x 4 | 0.1126 | 0.043 | 1.051 |
| `lambda_max` x 16 | 0.0853 | 0.034 | 1.042 |
| `lambda_max` x 16 **and** exact start | 0.0456 | 0.018 | 1.016 |

| n = 128, `mu_0 = 7.5` | W1/sd | TV | width |
|---|---:|---:|---:|
| `N(0, lambda_max^2)` start | 0.1614 | 0.063 | 1.040 |
| exact `p_{lambda_max}` start | **0.0297** | 0.014 | 1.018 |
| `lambda_max` x 4 | 0.0513 | 0.021 | 1.014 |
| `lambda_max` x 16 | 0.0231 | 0.005 | 0.998 |
| `lambda_max` x 16 and exact start | 0.0151 | 0.001 | 0.999 |

`lambda_max = lambda(1) = 6.43` is simply not large enough for `N(0, lambda^2 I)`
to be `p_lambda(theta | x)`: the posterior mean is offset by ~0.2-0.6 per
coordinate and in `n+1 = 129` dimensions that offset is not negligible. A
deterministic ODE carries the resulting error all the way to `t = 0`.

### The fix, which is COMPASS's own default

Karras churn re-injects noise before each step and makes the trajectory
stochastic, so the map is no longer deterministic and the start's error is
forgotten. Sweeping churn on the exact score:

| churn | n=8 | n=128 | n=128, `mu_0=7.5` |
|---:|---:|---:|---:|
| 0.0 | 0.2190 | 0.0967 | 0.1614 |
| 0.1 | 0.0957 | 0.0559 | 0.0902 |
| 0.2 | 0.0124 | 0.0396 | 0.0303 |
| **0.5** | **0.0117** | **0.0443** (0.0279 at 200 steps) | **0.0153** |
| 1.0 | 0.0278 | 0.0617 | 0.0186 |
| 2.0 | 0.0260 | 0.0828 | 0.0614 |
| MC floor | 0.019 +- 0.006 | 0.028 +- 0.006 | 0.019 +- 0.006 |

(500 steps unless noted.) **Any churn between 0.2 and 1.0 brings the control to
within one to two Monte-Carlo floor widths at every `n` and in both settings,
against 5-11 floor widths at churn = 0.** At n = 8 and at n = 128 with
`mu_0 = 7.5` it is at the floor outright; at n = 128 in the standard setting the
best settings land at 0.028-0.044 against a floor of 0.028 +- 0.006, i.e. at or
just above it. `churn = 0.5` is used throughout as the corrected setting because
it is the only value that is at or near the floor in all three columns. Plain
reverse-SDE Euler-Maruyama at 1000 steps also works at small `n` (0.040 at
n = 8, 0.017 at n = 32) but is worse at n = 128 (0.069).

So the control passes, and **every compositional DPM2 arm below is reported at
`churn = 0.5`**; the `churn = 0` numbers are kept in `TABLE.md` because they are
what a reader would get from the deterministic settings, and because the
*ordering* of the methods changes between the two — which is the point.

### The ordering really does change

| n = 8, DPM2, 500 steps | churn = 0 | churn = 0.5 |
|---|---:|---:|
| ORACLE-GAUSS | 0.1930 | 0.2819 |
| ORACLE-JAC | 0.3891 | **0.0100** |

With the deterministic sampler GAUSS looks twice as good as JAC. With the
corrected sampler JAC is 28x better than GAUSS and at the Monte-Carlo floor.
Any comparison of composition rules run on the deterministic predictor is
measuring the predictor.

---

## 3-4. Method A — F-NPSE, and its Langevin convergence

The literal paper bridge is improper here (section 0), so `ORACLE-FNPSE` is the
diffused-prior bridge sampled with annealed Langevin, step
`delta = snr lambda^2 / n` on the shared coordinate (COMPASS's
`_hierarchy_scale`), update `y <- y + delta S + sqrt(2 delta) eps`.

**The control that makes the ladder interpretable.** The same annealed Langevin
run against the *exact* diffused tall score (step not shrunk by `1/n`, since the
true posterior does not contract):

| n = 8 | L=10 | L=20 | L=50 | L=100 | L=200 | L=500 |
|---|---:|---:|---:|---:|---:|---:|
| true score | 0.0248 | 0.0296 | 0.0407 | 0.0142 | 0.0375 | 0.0262 |
| F-NPSE bridge | 0.2252 | 0.1155 | **0.0781** | 0.0972 | 0.2136 | 0.2052 |

| n = 32 | L=10 | L=20 | L=50 | L=100 | L=200 | L=500 |
|---|---:|---:|---:|---:|---:|---:|
| true score | 0.0744 | 0.0583 | 0.0198 | 0.0501 | 0.0286 | - |
| F-NPSE bridge | 0.8743 | 0.7525 | 0.4449 | 0.1725 | **0.0751** | 0.1893 |

| n = 128 | L=10 | L=50 | L=200 |
|---|---:|---:|---:|
| true score | 0.0685 | **0.0243** | 0.0346 |
| F-NPSE bridge | 0.8504 | 0.6456 | 0.3118 |
| F-NPSE width ratio | 0.244 | 0.489 | 0.789 |

(W1/sd, T = 100 levels, snr = 0.05; floors 0.019, 0.023 and 0.028. The n = 128
ladder was truncated to three rungs for cost.)

**The Langevin machinery is sound**: against the true score it sits at the
Monte-Carlo floor at every effort level, at both `n`, and does not drift. So
anything the F-NPSE arm gets wrong belongs to its bridge.

**The F-NPSE metric does not improve monotonically with Langevin effort.** It
falls to a minimum at intermediate effort (L = 50 at n = 8, L = 200 at n = 32)
and then *rises again and settles* near 0.19-0.21 at both `n`. That is the
plateau the experiment asked for, and it is not at zero: the extra Langevin
steps are doing their job, equilibrating the chain to the bridge at each level,
and **the bridge's own equilibrium has the wrong branch mass.** The favourable
intermediate numbers (0.078, 0.075) are under-converged chains that happen to
lag at a better answer — reporting one of those as "ORACLE-FNPSE" would be
reporting a chain that has not finished.

Sensitivity confirms this is not a step-size artefact: at n = 8, T = 100,
L = 100, snr = 0.01 gives 0.1804 and snr = 0.2 gives 0.2339, bracketing the
snr = 0.05 value of 0.0972; the level count moves it the same way
(T = 50: 0.0754, T = 200: 0.2159 — *more* levels is worse).

**n = 128 is a different situation and must not be read as a plateau.** There
the F-NPSE arm is still falling at the top of the explored budget (0.850 ->
0.646 -> 0.312) with a width ratio climbing 0.244 -> 0.489 -> 0.789, i.e. the
chain is *under-relaxed*, not converged; a larger step (snr = 0.2) reaches
0.127. So the honest n = 128 statement in the standard setting is "0.13-0.31 and
still improving at the budget explored", not a converged number. That is
consistent with the rest of the picture rather than in tension with it: at
n = 128 the standard target is effectively unimodal, so there is no branch mass
for the bridge to get wrong, and with enough Langevin effort it can approach the
right answer. The wrong-mass plateau appears precisely where the target is
genuinely bimodal.

**Converged `ORACLE-FNPSE` is therefore W1/sd ~ 0.19-0.21 at n = 8 and n = 32,
about 10x the Monte-Carlo floor and about 10x what the same sampler reaches on
the true score** — and at n = 128 the arm was not run to convergence.

**The Langevin control holds at every `n`**: against the true score it reaches
0.0234-0.0685 at n = 128 as well (floor 0.028), so nothing in the F-NPSE result
is the Langevin machinery.

### Is the large-`n` F-NPSE failure the bridge, or a step size that cannot move?

The width ratio of the n = 128 F-NPSE samples is 0.03-0.31, which says the chain
has barely moved rather than moved somewhere wrong. Geffner's prescription
shrinks the shared-coordinate step by `1/n` because the *bridge* contracts by
`1/n` there. That is right at the noise end, where the bridge's shared width is
`lambda/sqrt(n)`, and wrong at the data end, where its width is the posterior's
own `sqrt(V)` and does not depend on `n` at all. At n = 128 and `lambda_min` the
prescribed step is `snr lambda^2 / n ~ 4e-7` against a target width of 0.09, so
the final levels need ~1e4 steps each merely to relax.

Repeating the arm with the shared step **unscaled** separates the two:

| F-NPSE, W1/sd (branch TV) | step = `1/n` | step unscaled |
|---|---|---|
| n = 8, L=200, snr=0.05 | 0.214 (0.088) | 0.221 (0.094) |
| n = 8, L=500, snr=0.2 | 0.254 (0.113) | 0.218 (0.097) |
| n = 32, L=200, snr=0.05 | 0.075 (0.022) | 0.379 (0.140) |
| n = 32, L=500, snr=0.2 | 0.243 (0.106) | **diverges** (7.5e6) |
| n = 128, L=500, snr=0.2 | 0.147 (0.048) | **diverges** (1.2e7) |
| n = 128 `mu_0=7.5`, L=500, snr=0.2 | 0.585 (0.228) | **diverges** (4.3e6) |

Three things follow.

1. **At n = 8 the scaling is irrelevant.** Both settings reach the same
   ~0.19-0.25 with the same wrong branch mass, so the plateau there is the
   bridge's and nothing else.
2. **The `1/n` scaling is not gratuitous.** Remove it and the chain blows up at
   n >= 32, because the bridge really is `n` times sharper at the noise end.
3. **A single scalar cannot serve both ends**, so at n = 128 the arm is
   *simultaneously* under-relaxed (width ratio 0.44 even at `snr = 0.2`,
   `L = 500`) and wrong-massed (TV 0.228), and neither setting of the one free
   parameter fixes either. Pushing the step up improves W1 from 0.897 to 0.585
   while making the branch mass *worse* (TV 0.171 -> 0.228).

So the large-`n` F-NPSE failure is **not** a tuning problem that a better
constant would solve, and it is not cleanly separable into "bridge" and
"sampler" either: within Geffner's own one-parameter prescription there is no
setting that converges. Fixing it would need a *level-dependent* step schedule
that interpolates between `1/n` at the noise end and `1` at the data end, which
nobody has specified. That is a concrete, well-posed piece of follow-up work and
it is the only place in this experiment where the F-NPSE arm deserves the
benefit of the doubt.

(The standard setting at n = 128 behaves differently again — `snr = 0.2` reaches
0.083 with width ratio 0.962 — because that target is effectively unimodal, so
there is no branch mass to get wrong and the chain only has to relax.)

---

## 5-7. Method B — the Arruda damped bridge

Implemented as `d(t) [ (1-n)(1-t) s_prior,lambda + sum_i s_i ]` with
`d(t) = d_1^t`, multiplying the whole composed score (BayesFlow's convention),
and sampled both with the Jolicoeur-Martineau adaptive stochastic Heun at
BayesFlow's defaults and with the corrected DPM2.

**Damping does not help when the scores are perfect.** At n = 8, every solver
ranks `d_1` monotonically and `d_1 = 1` — i.e. no damping — wins:

| n = 8 | d1=1 | d1=0.5 | d1=0.354 (`n^-1/2`) | d1=0.2 | d1=0.1 |
|---|---:|---:|---:|---:|---:|
| DPM2, churn=0.5, 500 steps | **0.1652** | 0.2077 | 0.2312 | 0.2819 | 0.3448 |
| branch-mass TV | 0.0315 | 0.0660 | 0.0805 | 0.1110 | 0.1445 |
| adaptive Heun, rtol=1e-3 | **0.6250** | 0.6772 | 0.7146 | 0.7926 | 0.8210 |

`d_1 = 1` is exactly the diffused-prior F-NPSE rule, and the two arms agree to
the digit (0.1652 both), which is a useful internal check.

At n = 32 and n = 128 the picture is mixed rather than reversed. Under
Euler-Maruyama damping buys a little and then loses:

| Euler, 1000 steps | d1=1 | d1=0.5 | d1=0.2 | d1=0.1 | d1=`n^-1/2` |
|---|---:|---:|---:|---:|---:|
| n = 32, W1/sd | 0.6913 | 0.6418 | **0.6257** | 0.6391 | 0.6275 |
| n = 32, branch TV | 0.108 | 0.141 | 0.189 | 0.213 | 0.194 |
| n = 128, W1/sd | 0.4002 | **0.3564** | 0.5745 | 0.6869 | 0.7001 |
| n = 128, branch TV | 0.027 | 0.056 | 0.132 | 0.147 | 0.149 |

The best damped setting is 3-10% better in W1 than the undamped one and
2-5x worse in branch mass, which is the trade the bridge pictures predict: a
smaller composed field means a shallower barrier and a wider answer, not a
better-placed one. Under DPM2 with churn the damped field is *unstable* at
n >= 32 — most `d_1` settings diverge outright — so the only usable sampler for
this arm at large `n` is the plain reverse SDE. At n = 128 with a genuinely
bimodal target (`mu_0 = 7.5`) every setting is 1.4-5.4 or diverges.

**So Arruda's damping is a stabiliser for learned scores, not a better bridge.**
With exact scores it trades width for branch mass and mostly loses: it makes
the composed field smaller, which shrinks the barrier its own bridge builds
(section 13: valley depth 7.12 -> 5.71 at n = 8, 33.0 -> 26.5 at n = 128), but
it does not move the barrier or fix the mass.

**Tolerance convergence goes the wrong way.** Tightening `rtol` from 1e-1 to
1e-3 makes every damped arm *worse* (n = 8, `d_1 = 1`: 0.461 -> 0.475 -> 0.625;
NFE 1762 -> 10164 -> 20000). The same phenomenon as the Langevin ladder: a more
faithful integration converges more faithfully to the damped bridge's own,
wrong, answer. The converged `ORACLE-ARRUDA` is the `rtol = 1e-3` number, not
the flattering `rtol = 1e-1` one.

---

## 8-9. Method C — Linhart GAUSS and JAC with oracle quantities

C1 `gauss` weights each row by `Sigma_i^-1 + lambda^-2 I` built from the row
posterior's **exact total covariance**, Schur-reduced onto the shared block;
its prior precision is the exact second moment of the mixture prior. C2 `jac`
builds the same object from the **exact row Hessian** through
`Sigma_{t,i} = lambda^2 (I + lambda^2 J_i)`, using COMPASS's own spectral clamp
and information projection, and takes its prior precision from the exact
diffused *mixture* prior Hessian. No moment-matched curvature, no numerical
Jacobian.

**The DPM2 step ladder is converged** (churn = 0, W1/sd): at n = 8 GAUSS runs
0.1919 / 0.1939 / 0.1932 / 0.1930 / 0.1930 and JAC 0.3880 / 0.3889 / 0.3891 /
0.3891 / 0.3891 over 20 / 50 / 100 / 200 / 500 steps; at n = 32 and n = 128 the
same flatness holds by 100 steps. So neither rule's error is the integrator's.

**Under the corrected sampler** (DPM2, churn = 0.5, 500 steps):

| | n = 8 | n = 32 | n = 128 | n = 128, `mu_0 = 7.5` |
|---|---:|---:|---:|---:|
| ORACLE-JAC | **0.0100** | **0.0555** | 0.2853 | 0.5640 |
| branch-mass TV | 0.0025 | 0.0121 | 0.0618 | 0.2215 |
| width ratio | 0.993 | 1.057 | 1.318 | 1.293 |
| ORACLE-GAUSS | 0.2819 | 0.8163 | 3.3704 | 6.9528 |
| GAUSS width ratio | 0.857 | 1.385 | 3.968 | 7.693 |
| MC floor | 0.019 | 0.023 | 0.028 | 0.019 |

JAC's own error grows steadily with `n` — 0.010, 0.056, 0.285, and 0.564 once
the n = 128 target is genuinely bimodal — roughly a factor of five per factor of
four in `n`, with the width ratio drifting from 0.99 to 1.32. It is exact at
small `n` and not at large `n`; it is simply the *least* wrong construction at
every `n`.

**On the full metric set, not just W1**, oracle JAC at n = 8 is
indistinguishable from 2000 exact draws of the target:

| n = 8 | W1/sd | W2/sd | branch TV | +mass | valley | dip | width | mean err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact draws | 0.0135 | 0.0419 | 0.0020 | 0.748 | 0.0125 | 0.176 | 0.997 | 0.009 |
| ORACLE-JAC | 0.0100 | 0.0404 | 0.0025 | 0.752 | 0.0145 | 0.179 | 0.993 | 0.008 |
| ORACLE-GAUSS | 0.2819 | 0.5539 | 0.1085 | **0.859** | 0.0075 | 0.213 | **0.857** | 0.282 |
| exact value | 0 | 0 | 0 | 0.750 | 0.0142 | - | 1 | 0 |

Every column agrees, including the two that a mean-and-covariance summary
cannot see — branch mass and valley mass. GAUSS misses on exactly those: it
puts 0.859 of the mass on the dominant branch against a true 0.750 and
under-covers the valley by half.

**Oracle JAC is essentially exact at n = 8 and n = 32** — 0.0100 is below the
Monte-Carlo floor of a 2000-draw sample, 0.0555 is about twice it — and it is
the only construction in the comparison that gets there. **Oracle GAUSS fails,
and fails worse as `n` grows**, with the failure visible as a width ratio that
runs away to 4-8x: a Gaussian summary of a *bimodal* row posterior has a total
variance dominated by the branch separation, not the within-branch width, so it
under-weights every row by a factor that compounds with `n`. That is a property
of the rule, not of any approximation to it: these are the rows' exact second
moments.

This also settles the earlier appearance that GAUSS beats JAC. It does, under
the deterministic predictor, and only there.

---

## 10. Method D — the naive additive score, as a negative control

`(1-n) s_prior,lambda + sum_i s_i,lambda` fed straight to ordinary reverse
diffusion, with no Geffner or Linhart correction added.

| | n = 8 | n = 32 | n = 128 |
|---|---:|---:|---:|
| DPM2, churn = 0, 500 steps | 0.9259 | 0.9220 | 0.5405 |
| width ratio | **0.310** | **0.234** | 0.481 |
| DPM2, churn = 0.5, 500 steps | 0.1434 | 0.2304 | 0.5244 |
| width ratio | 1.061 | 1.163 | 1.759 |

The deterministic arm collapses the posterior to a third or a quarter of its
width and gets monotonically worse with more steps (n = 32: 0.817 at 100 steps,
0.877 at 200, 0.922 at 500) — the over-subtracted prior term sharpens the field,
and a better-integrated wrong field sharpens it further. The finite-time
non-commutation is directly visible in the field error (section 14): the naive
field is **114x** the RMS of the true score at n = 128, `lambda = 0.45`.

Interestingly the naive rule is *not* the worst arm once the sampler is
corrected (0.143 at n = 8 beats GAUSS's 0.282 and F-NPSE's converged 0.21).
That is not a recommendation: its error has no `n`-uniform sign and it is the
arm whose field error is largest by far.

---

## 11. The fifth diagnostic — the same bridge, the wrong sampler

`GEFFNER-DPM2` takes the exact F-NPSE bridge score and feeds it to ordinary
reverse diffusion:

| | n = 8 | n = 32 | n = 128, `mu_0 = 7.5` |
|---|---|---|---|
| Langevin (its own sampler), converged | ~0.21 | ~0.19 | 0.36 (best explored) |
| DPM2, churn = 0.5, 500 steps | 0.1652 | **diverges** (1.5e43) | **diverges** (NaN) |
| Euler-Maruyama, 1000 steps | 0.4619 | 0.6913 | 5.3177 |

**This is the cleanest demonstration in the experiment that a bridge score is
not a diffused score.** The same field that annealed Langevin traverses stably
at every `n` blows up under the reverse diffusion for `n >= 32`, because
`(1-n)(1-t) s_prior` is deliberately not the drift of the reverse SDE
corresponding to the original forward process. The F-NPSE score is correct for
its artificial bridge and wrong for the ordinary reverse diffusion, exactly as
the distinction in section 14 requires.

---

## 13. The intermediate bridges — where the transport actually goes wrong

Each rule's bridge is cut along the slice that pins every local at its exact
clean conditional mode `l_i(g) = a g + b x_i` (branch-independent for this
model), and the potential along that slice is obtained by integrating the
directional derivative `S_g + a sum_i S_{l_i}`. Figure: `bridges.png`,
`bridges_mu7.5.png`.

**Read the coloured curves against the black `true` curve, not against the
dashed one.** The dashed line is the exact diffused *marginal* `p_lambda(g|x)`
and the coloured lines are *profiles* along the slice; profiling out the locals
makes a curve narrower than the marginal, which is why the two differ at high
noise even for the exact score. `true` is the same profile construction applied
to the exact diffused tall score, so it is the like-for-like reference, and it
is what the depth and location comparisons below use. The dashed curve is kept
only to show where the target's mass actually is.

**When the two modes first emerge.** The exact target is unimodal at high noise
and bifurcates late. Every rule bifurcates at the wrong time and in the wrong
place:

| n = 8 | `tau`=1.0 | 0.75 | 0.5 | 0.25 | 0.1 | 0.0 |
|---|---:|---:|---:|---:|---:|---:|
| exact marginal, modes | 1 | 1 | 2 | 2 | 2 | 2 |
| true score, slice | 1 | 1 | 2 | 2 | 2 | 2 |
| `jac` | 1 | **2** | **3** | 2 | 2 | 2 |
| `fnpe` / `arruda` | 1 | 1 | 2 | 2 | 2 | 2 |
| `gauss` | 1 | 1 | 2 | 2 | 2 | 2 |
| `naive` | 1 | 1 | **3** | 2 | 2 | 2 |

* **`jac` bifurcates a whole level too early** and in the wrong place: at
  `tau = 0.75` it already has two modes at `+-1.84` where the target is still
  unimodal and its eventual modes are at `-1.14` and `+1.21`. The same happens
  at n = 32 (`+-1.85`) and n = 128 (`+-1.78`). At `tau = 0.5` it adds a
  *spurious third, central* mode which carries 8% of the mass at n = 8, 32% at
  n = 32 and **98.7%** at n = 128.
* **`fnpe`, `arruda` and `gauss` build a barrier that is far too deep.** At
  n = 8, `tau = 0.5`, the true slice has a valley 2.76 log-units deep and the
  F-NPSE bridge has **7.12**; `gauss` has 3.70. At n = 32 and n = 128 the true
  slice is unimodal or barely split at that level while those three bridges
  show three modes with essentially all mass on one, and a barrier
  **33.0 log-units** deep at n = 128.
* **`naive` does the opposite at low noise** — it *flattens*: the barrier is
  systematically shallower than the truth (6.14 vs 6.46 at n = 8 `tau`=0.25;
  0.34 vs 1.16 at n = 32) and its modes are pulled inward (`-1.09/+1.16` vs
  `-1.14/+1.21`), which is the width collapse seen in the samples.
* **Damping does not delay bifurcation.** `arruda-d1=0.2` has the *identical*
  mode locations to `fnpe` at every level and every `n`; the only thing it
  changes is the barrier depth (7.12 -> 5.71 at n = 8, 33.0 -> 26.5 at n = 128).
  It softens the wall without moving it.
* **`gauss` puts a barrier where the target has none.** At n = 128 it is the
  only arm still showing two modes at `tau = 0.1` and `tau = 0.0` (valley depth
  0.01 and 0.00, masses 0.27/0.73) where the exact marginal is unimodal.

**Does probability have to cross a low-density barrier? Yes, and that is the
mechanism.** Every rule's bridge forms its branch split earlier and deeper than
the true diffusion does. Once a 7-to-33 log-unit barrier exists, neither
Langevin nor reverse diffusion moves mass across it, so whatever split the
bridge fixed at that level is the split that survives to `t = 0`. That is why
running *more* Langevin steps makes F-NPSE worse: it equilibrates harder to a
landscape whose mass was already wrongly assigned.

---

## 14. Score-field error, and what it does and does not mean

RMS error on the shared coordinate relative to the RMS of the exact score, at
states drawn from the true diffused tall posterior (full tables per `n` in
`TABLE.md`):

| n = 128 | `tau`=1.0 | 0.75 | 0.5 | 0.25 | 0.1 | 0.0 |
|---|---:|---:|---:|---:|---:|---:|
| `naive` | 12.0 | 70.7 | **114.4** | 10.2 | 1.3 | 0.40 |
| `gauss` | 0.014 | 0.21 | 1.52 | 0.78 | 0.26 | 0.13 |
| `jac` | 0.086 | 0.59 | 2.17 | 7.69 | 0.84 | 0.19 |
| *`fnpe` (bridge)* | *121.7* | *63.5* | *80.8* | *7.9* | *1.03* | *0.32* |
| *`arruda` (bridge)* | *23.5* | *26.9* | *64.6* | *7.7* | *1.02* | *0.32* |

For `gauss` and `jac` this **is** the quantity the rule is trying to make small,
and it says the composed field misses the true diffused score by 1.5-7.7x its
own size in the middle of the schedule even with exact inputs. For `fnpe` and
`arruda` it is **not an error** — those are the scores of deliberately
different intermediate distributions, and the numbers are shown in italics only
to make the size of that intended difference visible. Nothing in this report
calls `S_t^F - S_t^true` an F-NPSE score error.

The relevant endpoint question for F-NPSE and Arruda is instead: *does their
sampler traverse their own bridge and land on the exact clean posterior?* The
answer is that it traverses the bridge correctly (the Langevin control is at the
floor, the adaptive solver converges) and lands about 0.19-0.21 away, because
the bridge's own `t = 0` limit is reached through a landscape that assigned the
branch mass wrongly at `tau ~ 0.5`.

---

## 15. Decision

The outcome is **not uniform in `n`**, and saying so is the result.

**At n = 8 and n = 32: Outcome 2.** Oracle JAC with a stochastic predictor is
essentially exact (0.0100 and 0.0555 against floors of 0.019 and 0.023; branch
mass TV 0.0025 and 0.0121; width ratio 0.99 and 1.06). F-NPSE remains
inaccurate at ~0.19-0.21 after demonstrable Langevin convergence, with its own
sampler shown sound by the true-score control, and the damped bridge is no
better and usually worse. **The true-diffusion correction is the valuable
thing, and the artificial compositional bridge is not sound for multimodality
even with perfect information.**

**At n = 128 with a genuinely bimodal target (`mu_0 = 7.5`): Outcome 4.**
Everything fails — JAC 0.564, naive 0.576, GAUSS 6.95, Arruda 1.44-5.4 or
divergent — while the true score with the corrected sampler reaches the
Monte-Carlo floor (0.0153). F-NPSE reaches 0.585 at its best stable setting and
is **not converged there**: no setting of its one free step parameter both
converges and stays stable (previous section), so its n = 128 number is a lower
bound on its error rather than a measurement of its bridge. In the standard
setting at n = 128 the same arms score better (JAC 0.285, naive 0.524,
GAUSS 3.37, F-NPSE 0.083), but that target has a dip ratio of 0.9987, so those
numbers measure how well a rule handles a nearly unimodal posterior, not
multimodal transport. The bridge plots say why: at
`tau = 0.5` every rule has already split the target into a landscape with a
12-to-33 log-unit barrier and put essentially all the mass on one side, and no
sampler recovers from that.

**And the sampler comes before both.** Before any of the above could be read,
the deterministic predictor had to be replaced, because with the exact score it
was itself 5-11x above the floor and it *reordered the methods*.

### What this implies for the project

1. **Do not read any composition-rule comparison that was run on the
   deterministic DPM2 predictor**, including earlier ones in this project. Rerun
   with churn (COMPASS's own default) or with a reverse-SDE sampler.
2. **Prioritise JAC, not F-NPSE, at small and moderate `n`.** With exact inputs
   JAC is the only rule that reaches the floor, and the F-NPSE bridge's residual
   error is a property of the bridge that more sampling effort makes worse, not
   better.
3. **The `n = 128` failure is not a score problem.** With exact scores and a
   validated sampler every construction still fails. For the DPM2-sampled rules
   (JAC, GAUSS, naive, Arruda) the binding constraint is that each builds its
   branch split at the wrong noise level and behind a barrier orders of
   magnitude too deep — a question about *when* a rule is allowed to commit to
   a mode split. For F-NPSE the constraint is different and more tractable: its
   prescribed step schedule cannot converge at large `n` at all, and a
   level-dependent schedule interpolating `1/n` -> `1` is an unexplored and
   well-posed fix.
4. **Arruda damping is a stabiliser for learned scores, not a better bridge.**
   With exact scores it never helped at n = 8, helped marginally and at a large
   cost in branch mass at n = 32, and helped nowhere at n = 128.
5. **GAUSS's failure mode compounds with `n`.** Its width ratio runs 0.86 ->
   1.39 -> 3.97 -> 7.69 as `n` grows, because the second moment of a bimodal row
   posterior is dominated by the branch separation. No amount of better moment
   estimation fixes that; it needs curvature, which is what JAC uses.

---

## 16. The required table

Generated into `TABLE.md` (standard setting) and `TABLE_mu7.5.md` (the wide
setting at n = 128), one block per `n`, with every column the specification
asks for — W1/sd, W2/sd, mode recall, branch-mass TV, positive-branch mass,
valley mass, dip ratio, mode location error, width ratio, posterior mean error,
wall clock, row-score evaluations and NFE — plus, underneath each block, the
full convergence ladders (sampler steps, churn, start and schedule, Langevin
effort, damping and tolerance), the score-field error table, and the F-NPSE
bridge properness table.

## 17. The deliverable sentence

> With perfect single-observation information, the remaining multimodal GM
> problem is **a sampler problem first** — the deterministic predictor cannot
> recover the exact posterior even from the exact score, and fixing it changes
> which composition rule wins — and, once that is fixed, **a bridge-definition
> problem**: at n = 8 and n = 32 the Linhart/JAC construction with exact
> curvature is essentially exact while the F-NPSE and damped bridges converge to
> the wrong branch mass, and at n = 128 with a genuinely bimodal target every
> construction fails because each one commits to its branch split at the wrong
> noise level and behind a barrier that is one to two orders of magnitude too
> deep. **It is not a Linhart approximation problem at small `n`, and it is not
> a network problem anywhere in this experiment.**
