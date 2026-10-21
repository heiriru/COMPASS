# Bridge topology, bifurcation-aligned JAC, and an adaptive F-NPSE step

*Oracle experiment on the Gaussian-mixture benchmark. No network was trained
or loaded. Every single-observation score, every score Jacobian, the prior
score and Hessian, and the tall posterior and its diffusion are the exact
closed forms of `Gaussian_mixtures_compositional_score_modelling`. The
datasets, the noise schedule, the sampler settings and the metric set are
those of `exp6_gm_oracle_bridges`, reused unchanged.*

---

## The question, answered in one sentence

**No, and partly.** Oracle JAC's small-`n` behaviour cannot be extended to
strongly multimodal `n = 128` by correcting only its bifurcation geometry — a
scalar `alpha(tau, n)` between naive composition and JAC cannot reproduce the
true bridge's mode structure at all in the middle of the bridge (at
`mu_0 = 7.5`, `n = 128` there is **no** `alpha` in `[0, 1]` that matches the
true mode count at **17 of 41** noise levels), and every schedule chosen from
bridge geometry makes the endpoint **worse** at every `n` tested (`n = 8`:
0.0100 → 0.0711; `n = 32`: 0.0555 → 0.4291; `n = 128` standard: 0.2853 →
1.3745; `n = 128`, `mu_0 = 7.5`: 0.5640 → 3.5271). F-NPSE's large-`n`
*sampling* problem, by contrast, **is** solvable with a noise-level-dependent
Langevin step — at standard `n = 128` a sigmoid schedule reaches width ratio
0.970 at `L = 500` while mixing an order of magnitude more than the fixed
`1/n` step, and at `n = 32` the geometry-aware step halves the endpoint error
(0.2430 → 0.1226) — but that buys nothing where the bridge is the problem: at
`mu_0 = 7.5`, `n = 128` the best adaptive schedule and the fixed `1/n` step
agree to three digits (W1/sd 0.5801 vs 0.5853, branch TV 0.229 vs 0.228), and
both converge, as effort increases, to a distribution that is too narrow and
has the wrong branch mass.

**Decision: Outcome E.** A scalar correction is insufficient. The next COMPASS
method should be an explicitly branch-aware / multimodal representation, not a
calibrated JAC — with the qualification that the level-dependent sampler
schedule is a real and cheap win in the regime where mixing, rather than the
bridge, is the binding constraint.

---

## 1. What is measured, and why not endpoint W1

The previous experiment graded bridges by endpoint `W1`. That says *that* a
rule is wrong, not *where*. This experiment therefore measures, at every noise
level, the four quantities that decide whether a reverse chain can end up with
the right branch mass at all:

* `tau_split` — the first level, moving from the noise end towards the data
  end, at which one central basin becomes two stable modes separated by a
  valley;
* `m_pm(tau)` — where those modes sit;
* `Delta E_barrier(tau) = min_k [log q(m_k) - log q(valley)]` — how deep the
  wall between the branches is, i.e. how hard it is for a sampler to
  re-balance the branches once it has split;
* `w_+(tau)` — the mass the bridge puts to the right of *its own* valley.

### The one-dimensional reduction, and its own error

Every field lives on the tall state `theta = (g, l_1..l_n)`. The reduction to
a curve in `g` is the *profiled slice*: each local is pinned at its exact clean
conditional mode `l_i(g) = a g + b x_i`, so

    slope(g) = S_g(theta(g)) + a sum_i S_{l_i}(theta(g))

is a genuine directional derivative along a one-dimensional cut and its
integral is that cut's log density. Roots are bracketed on an 8001-point grid
and refined with Brent; `+ -> -` is a mode, `- -> +` a valley. A stationary
point whose basin carries less than `1e-3` of the curve is treated as a
wrinkle, not a branch (`gm_topology.MASS_FLOOR`) — at `1e-6` a numerically
tiny shoulder counts as a mode and every barrier statistic downstream becomes
meaningless.

The slice is a cut, not a marginal, so the two are different curves by
construction. What matters is that they agree on *topology*, and they do. For
the `true` bridge at `n = 8` (full table in `TABLE.md`):

| tau | slice modes | marginal modes | slice `w_+` | marginal `w_+` | slice barrier | marginal barrier |
|---|---|---|---|---|---|---|
| 1.00 | 1 | 1 | 0.4957 | 0.4993 | 0.000 | 0.000 |
| 0.75 | 1 | 1 | 0.3918 | 0.4656 | 0.000 | 0.000 |
| 0.50 | 2 | 2 | 0.7503 | 0.7528 | 2.759 | 1.262 |
| 0.25 | 2 | 2 | 0.7500 | 0.7500 | 6.600 | 6.461 |
| 0.10 | 2 | 2 | 0.7500 | 0.7500 | 7.592 | 7.583 |
| 0.00 | 2 | 2 | 0.7500 | 0.7500 | 7.804 | 7.803 |

Mode count and branch mass agree everywhere. Barrier depth agrees to 2% once
the branches are resolved and differs by a factor of 2 at the single level
where the valley has just opened (`tau = 0.5`), which is the level where a
depth of a couple of log units is most sensitive to the cut. **All barrier
comparisons below are therefore between arms on the same slice**, never
between a slice and a marginal.

### The bridge coordinate

`tau` here is uniform in `log lambda`, 0 at `lambda_min = 0.0317` and 1 at
`lambda_max = 6.431` — the coordinate the samplers' own schedules are uniform
in (`_schedule` is a `logspace` in lambda). It is **not** the coordinate the
bridges use internally: `ComposedField.tau` is COMPASS diffusion time, where
`a(t) = 1 - t` is literally Geffner's `a = (T - t)/T`. The two disagree
strongly because `lambda(t)` is exponential in `t`. No bridge definition was
changed; the log-lambda coordinate is used only to *index* levels and to
parameterise `alpha`, which is a sampler-facing object.

### The `mu_0 = 7.5` setting is mandatory, and here is the number

At the standard benchmark the branch separation is
`2 mu_0 / (tau_g^2 sqrt(1/tau_g^2 + n/v))` within-branch sds, which falls like
`1/sqrt(n)`. The measurement: **at standard `n = 128` the true bridge has one
mode at every one of the 41 levels, including `lambda_min`** — `tau_split` is
undefined because the posterior is not bimodal in any operational sense. At
`mu_0 = 7.5` the same `n = 128` posterior has `tau_split = 0.725` and a
terminal barrier of 7.09 log units. Every multimodal claim below is made at
`mu_0 = 7.5`; the standard `n = 128` case is retained as a secondary control
and is labelled as such.

---

## 2. The true topology (section 3)

| setting | n | `tau_split` | within-branch sd | terminal barrier | terminal `w_+` |
|---|---|---|---|---|---|
| standard | 8 | 0.700 | 0.2739 | 7.804 | 0.7500 |
| standard | 32 | 0.575 | 0.1701 | 2.200 | 0.7507 |
| standard | 128 | *none* | 0.0915 | 0.000 | 0.3828 (unimodal) |
| mu_0 = 7.5 | 128 | 0.725 | 0.0915 | 7.090 | 0.7500 |

The true bridge bifurcates **later** (at lower noise) as `n` grows in the
standard setting, and the terminal barrier collapses from 7.8 to 2.2 to
nothing. At `mu_0 = 7.5` the bifurcation is early (0.725) and the barrier is
deep — the genuinely hard case.

---

## 3. JAC's topology error (section 4)

`Delta tau_split = tau_split^arm - tau_split^true`; positive means the arm
splits **too early**.

| arm | n=8 | n=32 | n=128 (standard, true: no split) | n=128, mu_0=7.5 |
|---|---|---|---|---|
| `jac` | **+0.100** | **+0.200** | splits at 0.050 (spurious) | **-0.500** |
| `gauss` | -0.025 | +0.075 | splits at 0.425 (spurious) | -0.600 |
| `fnpe` | 0.000 | +0.100 | no split | -0.375 |
| `naive` | -0.050 | -0.250 | no split | -0.550 |
| `arruda` d1=0.2 | 0.000 | +0.100 | no split | -0.375 |

Three findings, all of which contradict a simple "JAC splits too early"
picture:

1. **The sign of JAC's error flips with the regime.** JAC splits too early at
   small `n` (+0.10, +0.20) and far too *late* at the genuinely multimodal
   `n = 128` (-0.50). A correction calibrated to delay bifurcation at `n = 8`
   pushes the `mu_0 = 7.5` case further in the wrong direction.
2. **At standard `n = 128` both precision-weighted rules invent a bifurcation
   that the truth does not have** — `jac` at `tau = 0.05`, `gauss` at
   `tau = 0.425`. The additive rules (`fnpe`, `naive`, `arruda`) correctly
   stay unimodal here. This is a clean statement of the Linhart
   construction's failure mode at large `n` that the previous experiment could
   only see as an endpoint number.
3. **The failure at `mu_0 = 7.5` is not a timing error but a wrong
   *sequence*.** At `tau = 0.75` JAC has two modes where the truth has one,
   at positions **10.2 within-branch sds** from the truth and behind a barrier
   of **14.2** log units where the truth has none; it then *merges* back to a
   single mode at `tau = 0.5` and `0.25` where the truth is bimodal; and only
   re-splits at `tau = 0.225`. GAUSS does the same thing more violently
   (66.7 sds, barrier 76.1). F-NPSE at `tau = 0.5` has **no stationary point
   at all** on the slice.

`snapshots_mu7.5.png` shows the mechanism directly, on a log-density axis
because on a linear one the JAC bridge at intermediate noise is a vertical
line: at `tau = 0.5` and `tau = 0.25` the JAC/TOPO-JAC bridge is a razor-thin
spike many times narrower than the true bridge, while F-NPSE is a
correspondingly narrow spike displaced to `g ~ +3.1`. The over-concentration
of the composed field at intermediate noise, not the timing of a split, is
what the pictures actually show.

---

## 4. Constant alpha cannot do it (sections 5-6)

`S^(alpha) = S^naive + alpha [S^JAC - S^naive]`, with `alpha = 0` recovering
naive and `alpha = 1` recovering JAC to machine precision (tested; both
endpoints reproduce to 1e-12, and the field is verified affine in `alpha`).
Because the blend is affine and the slice is a linear functional of the field,
the whole `alpha` family is spanned by two slice evaluations per level, so
every `alpha` on the grid is exact rather than interpolated.

Sampled with the validated churned DPM2 (`churn = 0.5`, 200 steps):

| n | setting | `alpha` minimising the topology loss | its loss | W1 at `alpha`=0 | W1 at `alpha`=1 |
|---|---|---|---|---|---|
| 8 | standard | 0.7 | 1.318 | 0.2465 | 0.0734 |
| 32 | standard | 0.0 | 2.814 | 0.1698 | 0.1452 |
| 128 | standard | 0.2 | 1.274 | 0.4446 | 0.5638 |
| 128 | mu_0=7.5 | 0.9 | **6.919** | 0.5988 | 0.7647 |

The answer to "can one constant `alpha` preserve JAC's `n = 8/32` behaviour
while materially improving `n = 128`?" is no, and the `mu_0 = 7.5` row says
why: the topology loss **never falls below 6.9** for any `alpha`, because the
mode-count penalty is being paid at most levels no matter what `alpha` is. The
constant-`alpha` sweep at `mu_0 = 7.5` also never improves the endpoint — W1
runs 0.599 (`alpha`=0) up through 4.669 (`alpha`=0.7) and back to 0.765
(`alpha`=1), with branch TV pinned at 0.16-0.23 and `tau_split` stuck at 0.2
(0.15 at `alpha = 0`) against a true 0.725 for *every* `alpha`.

---

## 5. Topology-matched alpha, and why it fails (sections 7-8)

`alpha*(tau)` is chosen at each of 41 levels by a hierarchical objective — mode
count first (penalty 10 per extra or missing mode, which dominates everything
else), then mode positions in within-branch sds, then barrier depth in log
units, then branch mass. **The endpoint plays no part in the choice**;
selecting `alpha` on final W1 would make the experiment circular.

The decisive diagnostic is how often the objective is *saturated* — how many
levels have no `alpha` in `[0, 1]` that reproduces the true mode count:

| setting | n | saturated levels | best smooth fit | fit RMSE |
|---|---|---|---|---|
| standard | 8 | 5 / 41 | piecewise, 5 knots | 0.101 |
| standard | 32 | 1 / 41 | piecewise, 5 knots | 0.180 |
| standard | 128 | 0 / 41 | piecewise, 5 knots | 0.267 |
| mu_0 = 7.5 | 128 | **17 / 41** | piecewise, 5 knots | 0.281 |

At `mu_0 = 7.5`, over `tau` in roughly `[0.125, 0.5]` the loss sits at
10.5-11.1 for every `alpha` — the mode count cannot be matched, and `alpha*`
consequently jumps between 0 and 1 from level to level because the objective is
flat to within 0.2 while saturated. The "schedule" recovered there is noise,
and the fits say so: an RMSE of 0.28 on a quantity confined to `[0, 1]` is not
a fit.

Sampled (`ORACLE-TOPO-JAC`, churned DPM2, 500 steps, matched against
`ORACLE-JAC` under identical settings):

| setting | n | ORACLE-JAC | ORACLE-TOPO-JAC | ratio |
|---|---|---|---|---|
| standard | 8 | 0.0100 | 0.0711 | 7.1x worse |
| standard | 32 | 0.0555 | 0.4291 | 7.7x worse |
| standard | 128 | 0.2853 | 1.3745 | 4.8x worse |
| mu_0 = 7.5 | 128 | 0.5640 | 3.5271 | 6.3x worse |

Every schedule tried — the raw per-level `alpha*`, a constant, 3- and 5-knot
monotone splines, a sigmoid in `tau`, a quadratic in `log lambda` — lands in
the same band; the spread across schedules at `mu_0 = 7.5`, `n = 128` is
3.01-5.08 against JAC's 0.564. The one place the correction is not simply
destructive is `n = 8` at 200 steps, where TOPO-JAC scores 0.021-0.035 against
`alpha = 1`'s 0.073 at the same budget; at 500 steps the ordering reverses.
Both numbers are within a few multiples of the 2000-draw Monte-Carlo floor
(0.019 at `n = 8`), so the honest statement is that at `n = 8` the correction
is **neither a clear gain nor a clear loss**, and at every larger `n` it is a
clear loss.

**This is the negative case of section 8's own decision rule: little
improvement, so a scalar JAC-strength correction is insufficient and a
branch-aware representation is probably necessary.**

---

## 6. How simple can alpha be, and what does it track (section 9)

Fitted across all three `n` in the standard setting:

| form | RMSE against `alpha*` |
|---|---|
| `alpha(n)` only | 0.2888 |
| `alpha(tau)` only | 0.3042 |
| separable `a(tau) b(n)` | 0.2514 |
| `sigma(a0 + a1 log n + a2 log lam + a3 log n log lam)` | 0.2712 |

None of these is a fit. They span 0.251-0.304 RMSE on a quantity confined to
`[0, 1]`, and the ordering is uninformative: the most flexible form
(`separable`, which has a free scale per `n`) is best by 0.02, which is far
less than the gap to anything usable. Sampled, they reproduce the TOPO-JAC
result (`n = 8`: 0.058-0.079; `n = 32`: 0.337-0.386; `n = 128`: 1.303-1.600).

Correlation of `alpha*` with quantities JAC already computes — this is the test
of whether an oracle-free proxy exists:

| n | row variance | curvature magnitude | prior/data precision ratio | precision condition number | log lambda |
|---|---|---|---|---|---|
| 8 | -0.384 | **0.813** | 0.147 | 0.005 | **-0.778** |
| 32 | -0.286 | 0.504 | -0.059 | -0.045 | -0.586 |
| 128 | -0.109 | 0.155 | 0.381 | 0.210 | 0.024 |
| 128, mu_0=7.5 | -0.484 | 0.181 | 0.165 | 0.122 | -0.291 |

`alpha*` tracks curvature magnitude and `log lambda` well at `n = 8` and the
correlation decays to nothing exactly where a correction would be needed. There
is no oracle-free proxy at large `n`.

---

## 7. The F-NPSE step schedule: mixing is fixable, the bridge is not (sections 10-13)

The shared-coordinate step is `delta = snr lambda^2 [(1-w(tau))/n + w(tau)]`,
so `w = 0` reproduces Geffner's `1/n` and `w = 1` removes the scaling. The
initial ensemble's shared width follows the schedule's own `w(1)` rather than a
fixed `1/n`, so a schedule is never compared against the unscaled arm while
starting from a different ensemble. Stability and mixing are reported
separately throughout, as required: a stable chain that never moved is not a
success, and neither is a fast chain that occasionally diverges.

### Screening (T = 100, L = 100, snr = 0.2)

At **standard `n = 128`**: the unscaled step, `power p in {0.5, 1}` and every
sigmoid with `tau_0 >= 0.5` and `k <= 10` **diverge** (max |state| reaching
`1e7`). What survives splits into two groups — schedules that stay near `1/n`
until late (`power p >= 2`, sigmoid `tau_0 = 0.25`) which mix hard
(level-to-level mean movement 0.49-0.61 against 0.053 for fixed `1/n`) but
land on a worse branch mass, and two that get both: `sigmoid k=20, tau_0=0.5`
(width ratio **1.000**, TV 0.056, movement 0.742) and the geometry-aware
variance step (width 1.012-1.028, TV 0.023-0.049). Fixed `1/n` sits at width
**0.923** with movement 0.053 — stable, under-dispersed, barely moving, which
is exactly the diagnosis the previous experiment made.

### The convergence ladder, standard `n = 128`

| schedule | L=50 | L=100 | L=200 | L=500 |
|---|---|---|---|---|
| fixed `1/n` | 0.313 / w 0.806 | 0.127 / 0.923 | 0.083 / 0.962 | 0.147 / 0.957 |
| `sigmoid k=20, tau0=0.5` | 0.366 / 1.172 | 0.281 / 1.000 | 0.223 / 0.946 | **0.158 / 0.970** |
| `variance, snr_v=0.05` | 0.081 / 1.070 | 0.162 / 1.028 | 0.181 / 0.976 | 0.155 / 0.971 |

All three converge to width ~0.96-0.97 and W1 ~0.15 by `L = 500`. Here the
schedule is not needed: the target is effectively unimodal and fixed `1/n`
reaches the same place.

### The real test: `n = 128`, `mu_0 = 7.5`

| schedule | L=50 | L=100 | L=200 | L=500 |
|---|---|---|---|---|
| fixed `1/n` | 1.045 / w 1.174 / TV 0.148 | 0.735 / 0.812 / 0.179 | 0.632 / 0.602 / 0.204 | 0.585 / 0.442 / **0.227** |
| `sigmoid k=5, tau0=0.25` | 0.833 / 0.912 / 0.171 | 0.605 / 0.792 / 0.168 | 0.570 / 0.600 / 0.201 | 0.550 / 0.505 / 0.216 |
| `sigmoid k=20, tau0=0.25` | 0.784 / 1.006 / 0.150 | 0.647 / 0.733 / 0.184 | 0.584 / 0.575 / 0.206 | 0.580 / 0.438 / 0.229 |

Every stable schedule behaves the same way, and the direction is the point:
**as Langevin effort increases the chain gets narrower (width 1.01 → 0.44) and
its branch mass gets worse (TV 0.150 → 0.229)** while W1 creeps down. That is
not an under-relaxed chain — it is a chain converging to its own stationary
distribution, and that distribution is too narrow and puts `w_+ = 0.978`
against a true 0.750. The F-NPSE bridge at `mu_0 = 7.5`, `n = 128` has
essentially collapsed onto one branch.

The geometry-aware variance step **diverges** in this setting, and the reason
is informative: it reads the ensemble's own `Var(g)`, which at `mu_0 = 7.5` is
dominated by the branch separation rather than by the bridge's local width, so
it clips to the unscaled step — and the unscaled step is unstable at `n = 128`.
The `1/n` contraction is about the stiffness of the summed likelihood score,
not about the marginal width, and a rule that estimates the latter does not
recover it.

---

## 8. Final cross-comparison (section 14)

Churned DPM2 at 500 steps for the field-based arms; annealed Langevin at
`T = 100, L = 500, snr = 0.2` for the F-NPSE arms. The adaptive F-NPSE
schedule is the one whose ladder width ratio came closest to 1 — selected on
mixing, never on W1.

### Standard setting

| arm | n=8 | n=32 | n=128 |
|---|---|---|---|
| TRUE-SCORE | 0.0117 | 0.0189 | 0.0443 |
| ORACLE-JAC | 0.0100 | 0.0555 | 0.2853 |
| ORACLE-GAUSS | 0.2819 | 0.8163 | 3.3704 |
| ORACLE-TOPO-JAC | 0.0711 | 0.4291 | 1.3745 |
| FNPE-FIXED | 0.2541 | 0.2432 | 0.1466 |
| FNPE-ADAPTIVE | 0.2355 | 0.1226 | 0.1577 |

### `mu_0 = 7.5`, `n = 128` — the mandatory multimodal test

| arm | W1/sd | branch TV | width ratio | `w_+` (true 0.750) |
|---|---|---|---|---|
| TRUE-SCORE | **0.0153** | **0.0030** | 1.010 | 0.753 |
| ORACLE-JAC | 0.5640 | 0.2215 | 1.293 | 0.528 |
| ORACLE-GAUSS | 6.9528 | 0.2330 | 7.693 | 0.517 |
| ORACLE-TOPO-JAC | 3.5271 | 0.1975 | 4.324 | 0.552 |
| FNPE-FIXED | 0.5853 | 0.2275 | 0.442 | 0.978 |
| FNPE-ADAPTIVE | 0.5801 | 0.2285 | 0.438 | 0.979 |

The true-score control sits at 0.0153 with TV 0.0030 and width 1.010, so the
sampler is not the limitation here — everything in this table above the control
is bridge error. The two families fail in opposite directions: JAC-type rules
put too *little* mass on the positive branch (0.52-0.55 against 0.750) and are
too wide; F-NPSE puts essentially *all* of it there (0.978) and is less than
half as wide as it should be.

`ORACLE-JAC`, `ORACLE-GAUSS` and the F-NPSE fixed-step arm reproduce
`exp6`'s numbers to four decimals at every `n` and in both settings, which is
the cross-experiment consistency check on the reused machinery.

---

## 9. The seven questions (section 16)

1. **Is JAC's large-`n` multimodal failure mainly incorrect bifurcation
   timing/barrier geometry?** No. Timing is wrong (`Delta tau_split = -0.50`),
   but the deeper fault is a wrong *sequence* — a spurious early split at
   10.2 within-branch sds behind a 14.2-log-unit barrier, a merge back to
   unimodal across the levels where the truth is bimodal, and a late re-split
   — together with gross over-concentration of the composed field at
   intermediate noise.
2. **Can a scalar or low-dimensional `alpha(tau, n)` fix it?** No. At
   `mu_0 = 7.5`, `n = 128` no `alpha` in `[0, 1]` matches the true mode count
   at 17 of 41 levels, and every schedule fitted to the topology objective
   makes the endpoint 4.8-7.7x worse than plain JAC.
3. **How simple can `alpha` be?** The question does not arise: none of the
   simple forms fits (RMSE 0.251-0.304 on a `[0, 1]` quantity), and `alpha*`'s
   correlation with JAC-computable geometry (0.81 with curvature magnitude at
   `n = 8`) decays to 0.16-0.18 exactly where a correction is needed.
4. **Does the correction preserve JAC's `n = 8/32` accuracy?** At `n = 8`,
   roughly — 0.0100 → 0.0711 at 500 steps, 0.073 → 0.027 at 200 steps, both
   within a few Monte-Carlo floor widths. At `n = 32` it does not: 0.0555 →
   0.4291, a factor of 7.7.
5. **Can a level-dependent F-NPSE step avoid both high-noise divergence and
   low-noise under-relaxation?** Yes, in the near-unimodal regime. At standard
   `n = 128`, `sigmoid k=20, tau_0=0.5` is stable at every budget, reaches
   width ratio 1.000 at `L = 100` and 0.970 at `L = 500`, and moves the
   ensemble mean 0.742-0.983 per level against 0.053-0.084 for fixed `1/n`,
   while the unscaled step and every schedule that relaxes `1/n` too early
   diverges outright. The clearest endpoint gain is at `n = 32`, where the
   geometry-aware step halves F-NPSE's error (0.2430 → 0.1226) at equal
   budget.
6. **After fixing its sampler schedule, does F-NPSE still converge to the
   wrong branch mass?** Yes, and more clearly than before. At `mu_0 = 7.5`,
   `n = 128` every stable schedule converges to TV 0.216-0.229 and
   `w_+ = 0.978` against a true 0.750, with the error *growing* monotonically
   in Langevin effort.
7. **At `n = 128`, `mu_0 = 7.5`, which dominates: bridge topology,
   sampling/mixing, or both?** Bridge topology, decisively. The true-score
   control under the same sampler reaches 0.0153 / TV 0.0030 / width 1.010, so
   there is no sampler budget left to blame; and the adaptive and fixed F-NPSE
   schedules agree to three digits, so the schedule is not the binding
   constraint either.

---

## 10. Decision and what it implies (section 17)

**Outcome E.** Neither route works at genuine multimodality: topology-corrected
JAC is 6.3x worse than plain JAC there, and adaptive F-NPSE is
indistinguishable from fixed F-NPSE. A scalar interpolation between two
unimodal-in-spirit composition rules cannot represent a bridge whose two
branches must be tracked separately — which is what the 17 saturated levels
mean concretely.

The next COMPASS method should therefore be **an explicitly branch-aware /
multimodal representation**, not a calibrated JAC and not a faster bridge
sampler. Two results point at what it needs:

* the failure is a *representation* failure, not a *calibration* one: at the
  levels that matter, both `alpha = 0` and `alpha = 1` have the wrong number
  of modes, so no point on the segment between them can be right;
* the two rule families err in opposite directions on branch mass (JAC 0.53,
  F-NPSE 0.98, truth 0.75), which is the signature of two constructions each
  committing to a single branch structure rather than carrying both.

### What this does *not* license

* It does not retire the level-dependent Langevin step. That is a real,
  cheap improvement in the regime where mixing binds (standard `n = 128`:
  width 0.923 → 0.970, an order of magnitude more movement, no divergence),
  and it costs one scalar schedule. It should be kept whatever the bridge
  becomes.
* It does not indict the sampler anywhere in this experiment. The true-score
  control is at or near the Monte-Carlo floor at every `n` and in both
  settings (0.0117 / 0.0189 / 0.0443 standard; 0.0153 at `mu_0 = 7.5`).
* It says nothing about learned scores. Nothing here was trained; these are
  the *ceilings* that exact single-observation information buys.

### Caveats a reader should hold

* `alpha` was restricted to `[0, 1]`. Extrapolation beyond JAC (`alpha > 1`)
  or against it (`alpha < 0`) was not tested; the saturated-level result makes
  it unlikely to help, since the mode count is wrong at both endpoints, but it
  is not ruled out by anything measured here.
* The topology objective's weights (`10` per mode, `1` per within-branch sd of
  mode position, `0.25` per log unit of barrier, `4` per unit of branch mass)
  were fixed in advance and not tuned. Where the objective is saturated the
  weights are irrelevant; where it is not, a different weighting would move
  `alpha*` somewhat.
* The `variance` step is ensemble-adaptive (the step at each level is read
  from the ensemble entering it, frozen for that level's inner steps). It is
  reported as a diagnostic, not as an exact sampler.
* One dataset seed per `n`. Conclusions rest on effect sizes of 5-10x and on
  topology counts, not on differences near the noise floor.
* Everything is one-dimensional in the shared coordinate. The locals are
  profiled out, and a multimodal structure that lived only in the joint would
  not be seen.
