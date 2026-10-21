# IK: exact MCMC reference and recovery plots

Three tasks:

1. an MCMC reference posterior for the IK problem, checked for convergence;
2. recovery plots for n = 1 (all four parameters local) over many simulated
   datasets, then for the global/local hierarchy, to see whether the network can
   do the inference at all;
3. all of the above again with **height_arm also global**: two globals
   (angle_1, height), two locals (angle_2, angle_3).

Results and numbers: see [Results](#results). Everything below is reproducible
from `COMPASS/` with the commands in [Reproduce](#reproduce).

## The reference: exact, noise-free, no ABC kernel

The forward model is deterministic, so `p(x | theta)` is a point mass and there
is no likelihood to hand an MCMC sampler over all four parameters. The earlier
referee (`hierarchical.reference_global`) smoothed it with a Gaussian ABC kernel,
which biased it at useful bandwidths (on the rail-height data h = 0.05 pulled the
n = 128 modes from ±0.54 to ±0.40).

Two parameters can be integrated out **exactly** instead (`ik_exact.py`). Given
`angle_1` and `height`, the end position fixes joint 2 and leaves a two-link arm,
so `cos(a3) = (|r|² − L2² − L3²)/(2 L2 L3)`: two elbow solutions, with
`|det ∂x/∂(a2,a3)| = L2 L3 |sin a3|`:

    p(x | a1, h) = Σ_solutions N(a2; 0, .5) N(a3; 0, .5) / (L2 L3 |sin a3|)

It is zero outside the workspace annulus and has integrable 1/sqrt peaks on its
edges. For the angle-only layout the height is also integrated out,
`p(x | a1) = ∫ N(h; 0, .25) p(x | a1, h) dh`. The allowed heights are two
intervals known in closed form, and the substitution `u = lo + (hi−lo)(1−cos t)/2`
removes both edge singularities, so Gauss–Legendre converges spectrally
(24 nodes already agree with 400 to 6 digits).

Validated three ways:

| check | result |
|---|---|
| every solution reproduces x through `simulator.forward` | max residual 4e-15; the true (a2, a3) is always among them |
| `∫ p(x | a1, h) dx = 1` (uniform MC over the annulus, 10M points) | 1.0005 ± 0.0036, 1.0009 ± 0.0031, 0.9980 ± 0.0037 |
| against a small-bandwidth kernel estimate from 4M simulations | agrees within MC error (e.g. 0.0800 vs 0.0815 ± 0.004; p(x∣a1) 0.648 vs 0.633 ± 0.022) |
| 2π wraps of a2, a3 | change log p by < 1.3e-8 on prior draws (so the MCMC sums only the principal branch) |

## The samplers (`ik_mcmc.py`)

Both are batched over datasets and chains. Step sizes adapt during warmup only.
Every move is Metropolis–Hastings with a symmetric or explicitly corrected
proposal, so the retained draws come from a fixed reversible kernel.

**angle_1 global (and n = 1)** — *collapsed* sampler: Metropolis on `a1` against
its exact marginal likelihood, heights integrated out by the quadrature, then
every `h_j` drawn exactly from `p(h_j | a1, x_j)` (inverse CDF on 256 cells of
the regularised variable) and `(a2_j, a3_j)` from the discrete elbow law. Moves:
adapted random walk, the arm's exact mirror `a1 → −a1`, and an independence
proposal from the prior.

*Why collapsed:* the obvious full-state sampler over `(a1, h_1..h_n)` (kept as
`sample_angle`) fails to mix at n ≥ 8. With the heights fixed, moving `a1` by δ
forces every `a2_j` to move by −δ, so a1's conditional is far narrower than
its marginal: R̂ 1.2 at n = 8, ≈ 4 at n = 32. The diagnostics caught it
(and it agrees with the collapsed sampler where it does mix, n ≤ 2).

**angle_1 + height global** — the state is only 2-D, but the data pin *joint 2*
(vertical `v = h + L1 sin a1`, horizontal `L1 cos a1`). In `(a1, h)` that is a
thin curved ridge, and near a1 = 0 it is two-lobed in ±a1 (cos is even). A sampler
in `(a1, h)` gave R̂ up to 1.7 at n = 32. The chain therefore runs in `(a1, v)`
(unit Jacobian): walks in each coordinate, a covariance-adapted joint walk, the
involution `a1 → −a1` at fixed `v` (joint 2 does not move), and after warmup an
independence proposal from a multivariate t fitted to warmup.

**Repair pass:** after the main run, datasets with R̂ > 1.01 are rerun with
4× and then 16× longer chains (same number of retained draws).

## Convergence evidence

* rank-normalised split R̂ (folded and bulk, Vehtari et al. 2021) and bulk ESS
  for **every** continuous coordinate (angle_1 and every height), per dataset:
  `artifacts/<setting>/reference_diagnostics.json`;
* against an MCMC-free estimate (`check_reference.py`): the quadrature grid
  of p(a1 | x) for the angle layout; importance sampling (1M draws, defensive
  prior component) for the angle+height layout, whose 1/sqrt edges defeat a 2-D
  grid. KS is compared with its 95% floor `1.36 sqrt(1/ESS + 1/ESS_check)`;
* an independent sampler (the full-state one) where it mixes;
* SBC of the reference itself, since every truth came from the simulator.

## Results

Settings: 500 single-observation datasets; 100 tall datasets per n ∈ {2, 8, 32}
per layout, globals and locals drawn from the prior, no selection. Network draws:
1,000 per dataset. Reference: 2,000 (angle layout) or 4,000 (angle+height) draws
per dataset. Composition: churned DPM2 (eta 2), 100 steps, eps 1e-3, as in
`Arruda/compare_compass_ik.py`. Networks: `legacy_h16d2_n200000_six_weighted_log_sigma`
(angle_1 global) and the new `..._globalheight` (angle_1 + height global; same rows,
same recipe, six-conditional roles 2 + 2; 52 min; best validation loss at epoch
299/300, so like the first one it was still improving at the step cap).

"W1/sd" = median over datasets of W1(network, reference) / reference sd, and the
**floor** is the same statistic between the two halves of the reference's own chains
(≈ 0.05 everywhere). "sd ratio" = median network sd / reference sd.

### 1. The reference is converged and correct

| setting | datasets | max R̂ | min bulk ESS | KS within 95% floor (MCMC-free check) |
|---|---:|---:|---:|---|
| n = 1 | 500 | 1.007 | 1,220 | 18/20 (max 1.22× floor); independent sampler agrees to 0.065 sd |
| angle, n = 2 / 8 / 32 | 100 each | 1.007 / 1.009 / 1.010 | 1,338 / 804 / 657 | 18/20, 19/20, 18/20 |
| angle+height, n = 2 / 8 / 32 | 100 each | 1.013 / 1.010 / 1.009 | 644 / 832 / 611 | 38/40, 37/40, 40/40 |

Across all 200 comparisons 12 exceed the 95% floor (6%, against 5% expected);
the worst is 1.36× the floor (KS 0.043). The reference also passes its own SBC on
every global (KS p 0.15–0.91, `metrics.json`). Its recovery
plots are the **ceiling** for any method: the problem is non-identifiable, so even
the exact posterior recovers angle_3 at n = 1 with R² 0.21.

Locals: SBC on the *pooled* locals of a dataset is invalid (they share the
global's posterior error, so their ranks are correlated), and at first gave the
reference itself p ≈ 0 at n = 32. `metrics.json` therefore tests one observation
per dataset. Coverage uses all of them.

### 2. n = 1: the network can do the inference

`figures/single_recovery.png`, `figures/single_vs_reference.png`

| parameter | reference R² / 90% cov | network (angle) W1/sd, sd ratio | network (angle+height) W1/sd, sd ratio |
|---|---|---|---|
| angle_1 | 0.58 / 0.90 | 0.10, 0.99 | 0.11, 1.01 |
| height | 0.54 / 0.91 | 0.11, 0.99 | 0.14, 1.02 |
| angle_2 | 0.33 / 0.89 | 0.09, 1.00 | 0.10, 1.03 |
| angle_3 | 0.21 / 0.90 | 0.09, 1.00 | 0.09, 0.99 |

Both networks sit at about twice the MCMC floor (0.05), with the right widths and
calibrated SBC (every p ≥ 0.075). 3 of 500 datasets (0.6%) have a posterior mean off by more than 1 reference-sd
in some parameter; they are the outliers in the posterior-mean scatter.

### 3. Hierarchy: fine at n = 2, degrading with n

`figures/hier_angle_*.png`, `figures/hier_angle_height_*.png`

Global parameters (gauss_hierarchical; gauss_jacobian_FD is within noise of it
everywhere):

| layout | n | exact sd | network sd | sd ratio | W1/sd | RMSE exact → network |
|---|---:|---:|---:|---:|---:|---|
| angle | 2 | 0.228 | 0.246 | 1.07 | 0.23 | 0.247 → 0.259 |
| angle | 8 | 0.103 | 0.129 | 1.26 | 0.51 | 0.095 → 0.118 |
| angle | 32 | 0.041 | 0.068 | 1.79 | 1.12 | 0.041 → 0.066 |
| angle+height: angle_1 | 2 | 0.222 | 0.252 | 1.21 | 0.47 | 0.233 → 0.274 |
| angle+height: angle_1 | 8 | 0.077 | 0.148 | 2.44 | 1.79 | 0.114 → 0.178 |
| angle+height: angle_1 | 32 | 0.015 | 0.077 | 18.8 | 17.0 | 0.044 → 0.094 |
| angle+height: height | 32 | 0.007 | 0.036 | 24.3 | 19.5 | 0.022 → 0.049 |

* **Recovery against truth hides this.** At n = 32 the network's recovery plot
  looks almost perfect (R² 0.97–0.98, 90% coverage 0.84–0.98), because its extra
  width roughly absorbs its extra error. Against the exact posterior it is
  1.8× too wide (angle) and ~20× too wide (angle+height), with posterior means
  off by a median of 0.7 and 9 reference-sd respectively.
* **Errors are unbiased on average but grow with n** (`hier_*_vs_reference.png`):
  the mean-error histograms stay centred at 0 while their spread widens. The
  per-observation error does not average out as fast as the posterior contracts.
* **Not the sampler's leftover noise.** eps 1e-3 leaves noise of sd 0.032 on
  every draw, which is larger than the angle+height n = 32 posterior. With
  eps = 1e-5 (sd ≈ 0.003) the sd ratio moves only 1.79 → 1.69 (angle) and
  18.8 → 17.5 (angle+height).
* **Why angle+height is so much harder.** With the height shared, every
  observation's workspace annulus constrains the *same* joint-2 position, so the
  exact posterior contracts much faster than 1/√n (sd 0.22 → 0.015 from n = 2 to 32,
  15×, against √16 = 4× for a regular model) and becomes non-Gaussian near the
  constraints. A Gaussian composition of per-observation scores cannot follow
  hard support boundaries.
* Locals are mildly affected in the angle layout (W1/sd 0.12–0.19 for the
  angles, 0.17–0.31 for height; sd ratio 1.02–1.15). In the angle+height layout they inherit the globals' error
  (W1/sd 0.5–0.9 at n = 8–32), because there the locals are almost determined by
  the globals.

**Answer to "can the network do the inference at all":** the single-observation
posterior, yes, close to exact. The composed hierarchical posterior works at small
n and in a regular (angle-only) hierarchy, but it becomes 2× too wide by n = 32.
With height global it fails against the exact posterior by n = 8, while still
*looking* calibrated in a truth-based recovery plot.

## Why (almost) never two modes, and why a mode is not always the truth

`bimodality.py` → `artifacts/bimodality/` (`constructed_posteriors.png`, `mirror_penalty.png`,
`bimodality.json`). Everything uses the exact posterior, not a network and not an ABC kernel.

**The only exact symmetry is expensive on natural data.** `(a1, a2, a3, h) → (−a1, −a2, −a3, 2x1 − h)`
leaves every observation's likelihood unchanged. Only the height prior distinguishes a point
from its mirror twin, by exactly `Σ_j −2 x1_j (x1_j − h_j)/σ_h²` (σ_h = 0.25, so a factor of 32).
Unless the tip sits at rail height (`x1 ≈ 0` or `x1 ≈ h`) this is large:

| | per observation (prior predictive) | exact posterior, n = 1 / 2 / 8 / 32 (median) |
|---|---|---|
| log p(mirror) − log p(point) | median −22 nats; only 6% of observations favour the mirror | −27 / −60 / −227 / −910 nats |
| bimodal exact posteriors (angle_1 global) | | 3% / 1% / 0% / 0% |

The mirror branch's weight is roughly e^(−22 n). A second mode needs *every* observation to
have its tip at rail height, so it has to be constructed (the `selected` / rail-height sets).
On natural data the reference is right to be unimodal, and a method that found two modes
there would be wrong.

**With height global there is a different, cheap near-symmetry.** Joint 2 sits at
`(h + L1 sin a1, L1 cos a1)`. Flipping `a1 → −a1` while holding joint 2 fixed leaves the
arm's reach unchanged and costs only the a2 priors (`a2 → a2 + 2a1`), which is small when
|a1| is small. So in the angle+height layout, two lobes at ±a1 are *common* on natural data
whenever the true angle_1 is near 0: 49/100 datasets put more than 10% of their mass on each
sign at n = 2, 41 at n = 8 and 19 at n = 32. The composed network gets that sign split roughly
right (|ΔP(a1>0)| median 0.03–0.13) but is 1.4–2.5× too wide there.

**On constructed data the exact posterior need not sit at the truth, and that is correct.**
Both constructed sets are generated at a1 = 0.6 and then *filtered* (rail: `|x1 − h| < 0.05`,
i.e. the tip is at the shoulder's height; x1sel: `|x1| < 0.15`):

| rail-height, n | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| exact, standard model: peaks | −0.69, +0.23 | −0.69 | −0.65 | −0.56 | +0.57 | −0.53, +0.56 |
| exact, standard model: P(a1 > 0) | 0.18 | **0.01** | **0.00** | **0.01** | 0.99 | 0.91 |
| exact, selection-aware: peaks | ±0.61 | ±0.60 | ±0.60 | ±0.60 | ±0.61 | ±0.60 |
| exact, selection-aware: P(a1 > 0) | 0.51 | 0.46 | 0.51 | 0.48 | 0.58 | 0.56 |

* On filtered data the tip is at rail height, so the mirror term is a sum of *tiny terms with
  random signs*. It performs a random walk in n, and the standard-model posterior commits to
  whichever branch that walk currently favours. On this set that is the *wrong* branch from
  n = 8 to 32 (P(a1 > 0) = 0.00–0.01, a peak at −0.65), and then it flips (0.99 at n = 64). |a1|
  is recovered (E|a1| = 0.60); its sign is not identified by these data, and the exact
  posterior reports whatever the realised data happen to favour. That is not a sampler failure.
* The standard model is also *mis-specified* for filtered data: every row was kept with a
  probability `P(selected | a1)` that depends on a1. Dividing each row's likelihood by it (the
  selection-aware posterior, computed exactly) gives modes at **±0.60 at every n** with
  P(a1 > 0) ≈ 0.5: the truth and its mirror, and both are correct.
* The old ABC reference (h = 0.05) put the rail-height peaks at ±0.40 for n = 64–128. That was the
  kernel acting as observation noise, not the data (`Arruda/artifacts/ik/rail_height/reference_a0.6_tol0.05_h*.npz`: shrinking h moves them outward). The exact
  standard posterior peaks at −0.53/+0.56 and the selection-aware one at ±0.60.
* COMPASS gauss_hierarchical puts one broad bump near 0, between the two branches at every n
  (orange in the figure). Its per-observation scores do not carry the two mirror branches
  sharply enough for composition to keep them apart.

**Methods against the MCMC posterior on the rail-height set** (`plot_constructed_methods.py` →
`artifacts/bimodality/constructed_methods_vs_mcmc.png`; MCMC R̂ ≤ 1.002, KS vs the exact grid ≤ 0.012):

| n | MCMC P(a1>0) | Arruda (BayesFlow, d = 1) W1/sd, P+ | gauss_hierarchical | gauss_jacobian_FD | Langevin F-NPSE |
|---|---|---|---|---|---|
| 1 | 0.43 | 0.04, 0.43 | 0.14, 0.40 | 0.14, 0.40 | 0.16, 0.39 |
| 8 | 0.01 | 0.41, 0.02 | 4.05, 0.24 | 3.90, 0.24 | 3.73, 0.21 |
| 32 | 0.01 | 1.51, 0.00 | 5.57, 0.46 | 5.36, 0.40 | 5.43, 0.43 |
| 64 | 0.99 | 12.7, 0.00 | 4.99, 0.69 | 5.18, 0.56 | 4.71, 0.72 |
| 128 | 0.91 | 4.02, 0.00 | 1.60, 0.67 | 1.58, 0.55 | 1.47, 0.77 |

Arruda follows the MCMC from n = 1 to 32 only because the MCMC happens to sit on the negative
branch there. Arruda is *always* negative (the network's mirror bias summed n times), so when
the exact posterior flips to +0.57 at n = 64 it stays at −0.75, outside the support. The three
COMPASS rules are indistinguishable from each other: one broad bump near 0, between the branches,
from n = 4 on.

The **near-rail-tip set** (`|x1| < 0.15`, second row of the figure) settles the Arruda mechanism.
There the MCMC posterior is genuinely two-moded at ±0.6 for every n ≥ 8, mostly on the *positive*
side (P(a1>0) 0.65–0.81 at n = 8–32, 0.61 at n = 128). Arruda still ends on the negative branch
(P+ 0.19 at n = 8, 0.00 from n = 32), W1 1.0–1.6 sd. The COMPASS rules give one broad lump
leaning positive (P+ 0.62–1.00), 0.6–1.1 sd away, never two sharp modes.

**The n = 1 MCMC was checked against plain rejection sampling** from the simulator (prior draws kept
when the tip lands within 0.005 of the observation; 5×10⁸ draws, 43k accepted; no shared code):
mean −0.079 ± 0.002 vs −0.083 ± 0.004, sd 0.370 vs 0.369, P(a1>0) 0.435 vs 0.434, W1 0.013 sd,
KS p = 0.41. The broad n = 1 posterior is correct: one observation cannot pin angle_1, because
the other three joints compensate.

**What would fix the methods** (target = the MCMC posterior; on these filtered sets not even the
exact posterior can tell +0.6 from −0.6):
* *Arruda*: symmetrise the network score under the exact mirror, `½[s(θ, x) − s(−θ, Mx)]` with
  `M(x1, x2) = (−x1, x2)` (valid because p(θ|x) = p(−θ|Mx) and every prior is symmetric), or
  train on mirror-augmented rows. This removes the per-pair log-odds bias of ≈ −0.4 that
  composition adds up.
* *COMPASS rules*: the same symmetrisation, plus a composition that is not a single Gaussian per
  observation (the mixture / mode-mixture compositions), since the per-observation posteriors
  here have two branches and hard reach edges.
* *Filtered data*: put the filter into the model (divide by P(kept | angle_1)); otherwise even
  exact inference lands on a random branch.

### Getting a bimodal target without filtering (`bimodality_options.py`)

Share of exact posteriors with modes at both ±truth (natural, unfiltered data drawn from the
same prior; 50 datasets, or 20 for σ_h = 5 and 10; `artifacts/bimodality/bimodality_options.*`):

| height prior σ_h | a1 = 0.25: n = 8 / 32 / 128 | a1 = 0.6: n = 8 / 32 / 128 |
|---|---|---|
| 0.25 (stock) | 0 / 0 / 0 | 0 / 0 / 0 |
| 1 | 0 / 0.02 / 0 | 0 / 0 / 0 |
| 2 | 0.10 / 0.38 / 0.02 | 0.34 / 0 / 0 |
| 5 | 0.30 / 0.90 / 0.50 | 0.50 / 0.40 / 0 |
| **10** | **0.50 / 0.95 / 1.00** | 0.75 / 1.00 / 0.45 |
| filtered \|x1\| < 0.15, σ_h 0.25 | 1.00 / 1.00 / 0.40 | |

Why σ_h has to be *that* wide: the mirror gap per observation is `−2(hS + S²)/σ_h²`
(S = the arm's vertical reach). Its mean, ≈ −1.2/σ_h², adds up linearly in n; its random
part, ≈ 1.6/σ_h, only like √n. At σ_h = 2 and n = 128 that is −38 ± 9 nats, so one branch
wins; at σ_h = 10 it is −1.5 ± 1.8 nats, so both survive. The posterior stays informative
(x2 carries information independent of the height), and the modes sit at ±truth, with no
selection effect.

**Recommended target:** σ_h = 10, true angle_1 = 0.25, natural data. That needs a network trained
under that prior. With heights spanning ±30, the VESDE σ_max = 8 in `train_model.py` is too small
for x1, so the observables need standardising or σ_max raising.

**Summary.** No method (and no reference) finds two modes on natural data because the height
prior charges ~22 nats per observation to the mirror branch. Two modes appear only when every
row is filtered to the rail height, and then the sign is genuinely unidentified. The standard
posterior's branch choice is a coin flip of the realised data, not an error. The exception is
the angle+height model, where a cheap ±a1 flip at fixed joint 2 creates two lobes whenever
|angle_1| is small.

### The bimodal target, run (`wide_height.py`, `run_wide_height.sh`)

sigma_h = 10, true angle_1 = 0.25, 10 natural datasets of 128 rows. Both networks retrained under
that prior (COMPASS: `..._six_weighted_log_sigma_sdh10`, VESDE sigma 60, 59 min; Arruda: paper
setting, 1000 epochs, `Arruda/artifacts/ik/wide_sdh10`). Every module reads the height scale from
`IK_SD_H` (unset = the stock 0.25, nothing else changes).

**Reference.** Grid of 40,001 points (a 1,401-point grid is too coarse: at n = 128 each mode is a
few 1e-3 wide and the KS check against MCMC failed on all 10 sets for that reason alone). Collapsed
MCMC: max R-hat 1.009 over 50 (dataset, n) cells, KS within floor in 48/50 (two at 0.032/0.029 and
0.025/0.023). The target is not two Gaussian bumps: x1 carries almost nothing (h is nearly flat), so
angle_1 is constrained only by reach in x2 and the posterior piles up at the **edge of its
feasible support**, a hard cutoff that shrinks to |angle_1| ~ 0.25 -- spikes at +-0.25 by n = 32.
The cutoff is exact reach: with h free, the arm reaches x2 only if |x2 - L1 cos a1| <= L2 + L3, so
|a1| <= arccos((max_j x2_j - 1.5) / 0.5); it matches the grey edge on every set (d0: 0.58 at n = 1,
0.35 at n = 8). Inside it, density rises toward the edge (the fully stretched arm is a Jacobian
singularity), and the product over n turns that into the spikes. n = 1-2 is a plateau, not
bimodal; two modes appear from n = 8. Their weights are set by the weak x1 terms and vary by set:
P(a1 > 0) at n = 128 is 0.41-0.93, so the minor mode holds >= 25% in 7/10 sets but only 7% on d0.

Median over the 10 sets (W1 to the exact posterior / its sd; error in P(angle_1 > 0); share of sets
with peaks at both +-0.25):

| method | W1/sd n=1 / 8 / 32 / 128 | P+ err n=128 | both modes n=32 / 128 |
|---|---|---|---|
| MCMC | 0.014 / 0.016 / 0.014 / 0.018 | 0.007 | 1.0 / 1.0 |
| Arruda (BayesFlow) | 0.09 / 0.38 / 0.90 / **3.0** | 0.65 | 0.1 / 0 |
| gauss_hierarchical | 0.09 / 0.33 / 0.40 / 0.76 | 0.24 | 0.1 / 0 |
| gauss_jacobian_FD | 0.09 / 0.37 / 0.38 / 0.71 | 0.23 | 0 / 0 |
| Langevin F-NPSE | 0.13 / 0.40 / 0.41 / 0.81 | 0.37 | 0.1 / 0.1 |

No method finds the two modes. The COMPASS rules give one smooth bump centred at 0 (where the exact
posterior has no mass by n = 32), shrinking with n but never splitting; Arruda follows the same bump
to n = 32 and then collapses to a single wrong point (about -0.65 on dataset 0, outside the support).
At n = 1 all four are close (W1/sd ~0.1) but too smooth at the hard support edges at +-0.6.
Figures (MCMC row on top; `SHOW=<d>` picks the set): `artifacts/wide_height/wide_height_methods_d4.png`
(balanced modes) and `_d0.png` (lopsided); numbers: `wide_height_summary.json`,
`mcmc_checks.json`.

## Reproduce

From `COMPASS/` (the network scripts pick a GPU through `autocvd`):

```bash
P=.COMPASS/bin/python; R=inverse_kinematics/recovery
$P $R/train_height_global.py                                   # ~52 min GPU
$P $R/run_reference.py --kind single --count 500
for n in 2 8 32; do
  $P $R/run_reference.py --kind hier --layout angle --n $n --count 100
  $P $R/run_reference.py --kind hier --layout angle --n $n --count 100 --repair 2
  $P $R/run_reference.py --kind hier --layout angle_height --n $n --count 100 --warmup 4000 --draws 500 --thin 20
  $P $R/run_reference.py --kind hier --layout angle_height --n $n --count 100 --warmup 4000 --draws 500 --thin 20 --repair 2
done
$P $R/check_reference.py
$P $R/run_network.py --kind single --layout angle --count 500
$P $R/run_network.py --kind single --layout angle_height --count 500
for l in angle angle_height; do for n in 2 8 32; do $P $R/run_network.py --kind hier --layout $l --n $n --count 100; done; done
for l in angle angle_height; do $P $R/run_network.py --kind hier --layout $l --n 32 --count 100 --methods gauss_hierarchical --eps 1e-5; done
$P $R/plot_recovery.py
$P $R/bimodality.py
```

The n = 64 / 128 runs checkpoint as they go (`--resumable`, per-dataset network files):
`stop_runs.sh` stops everything safely and `resume_runs.sh` picks up where it stopped.

## Files

| file | role |
|---|---|
| `ik_exact.py` | closed-form likelihood, quadrature, exact conditional draws, grid posteriors |
| `ik_mcmc.py` | the samplers and R̂ / ESS |
| `layouts.py`, `datasets.py` | the two partitions; seeded test sets |
| `run_reference.py`, `check_reference.py` | reference + convergence evidence |
| `train_height_global.py`, `run_network.py` | new network; network posteriors |
| `plot_recovery.py` | figures and `artifacts/metrics.json` |
| `artifacts/<setting>/` | `data.npz`, `reference.npz`, `network_*.npz`, diagnostics |
