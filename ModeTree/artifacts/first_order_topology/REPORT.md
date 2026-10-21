# Does the learned first-order score still contain the multimodal topology?

**Answer: it depends on the checkpoint, and the two we have give opposite
answers.** The IK network retains the correct two-mode topology at every `n`
tested; the Gaussian-mixture network has lost it entirely. Nothing below uses a
score Jacobian, a Hessian, a Tweedie covariance, Newton, or any ModeTree
curvature code.

## Method

The composition is COMPASS's own `ComposedField(rule="uncorrected")`, verified
by reading `modetree/diffusion.py`:

    S_g = (1-n) * s_prior + sum_i s_{i,g}   ==   s_prior + sum_i [s_{i,g} - s_prior]

That branch calls `row_jacobians` nowhere and applies no cross-factor; its local
block is the raw per-row local score. Only forward evaluations are used. Roots
come from sign changes refined by Brent -- never Newton.

The state is hierarchical, so for each candidate `g` the locals are **profiled**
by Adam on their own row score (first-order only), and the global score is read
at `(g, l*(g))`.

**What integrating that curve gives, exactly.** At `l*(g)` the local score
vanishes, so

    d/dg log p(g, l*(g)|X) = d_g log p + (d_l log p) . dl*/dg = d_g log p

with the second term exactly zero. The composed global score read at the
profiled locals is therefore the exact total derivative of the **profile** log
posterior. Integrating it reconstructs the profile posterior, *not* the
marginal: the Occam factor from integrating the locals out is absent. Mode
locations agree between profile and marginal when the local curvature is roughly
constant in `g`; mode **masses** generally do not. Every density below is the
profile one.

## Control: the machinery is correct

IK has no analytic score, so the Gaussian-mixture benchmark -- which has both an
exact diffused score and a closed-form tall posterior -- validates the pipeline.

| n | t=1e-2 | t=1e-3 | t=1e-4 |
|---|---|---|---|
| 8  | 0.0405 | 0.0041 | 0.0004 |
| 16 | 0.1026 | 0.0104 | 0.0010 |
| 32 | 0.2269 | 0.0234 | 0.0024 |

W1/sd of the reconstructed profile posterior against the exact posterior, oracle
score. Monotone `lambda^2` convergence at every `n`, ending three orders of
magnitude below anything the learned scores achieve. Mode locations recover to
0.000-0.009. **Mode recall 1.00 in all 9 oracle cells, 0 false modes.**

## Question A -- does the learned IK score contain two `+ -> -` roots? YES

Roots merged at separation 0.05 (see "root stability" below):

| n | t | basins | locations | reference | mode error | recall | false |
|---|---|---|---|---|---|---|---|
| 8  | 1e-2 | 2 | -0.490, +0.675 | -0.410, +0.458 | 0.217 | 1.00 | 0 |
| 8  | 1e-3 | 2 | -0.563, +0.577 | | 0.153 | 1.00 | 0 |
| 8  | 1e-4 | 2 | -0.497, +0.539 | | 0.087 | 1.00 | 0 |
| 16 | 1e-2 | 2 | -0.514, +0.699 | -0.490, +0.475 | 0.224 | 1.00 | 0 |
| 16 | 1e-3 | 3 | -0.588, +0.125, +0.593 | | 0.118 | 1.00 | **1** |
| 16 | 1e-4 | 4 | -0.508, -0.022, +0.090, +0.548 | | 0.073 | 1.00 | **2** |
| 32 | 1e-2 | 2 | -0.477, +0.670 | -0.422, +0.480 | 0.190 | 1.00 | 0 |
| 32 | 1e-3 | 2 | -0.555, +0.540 | | 0.133 | 1.00 | 0 |
| 32 | 1e-4 | 2 | -0.483, +0.499 | | 0.060 | 1.00 | 0 |

**Mode recall is 1.00 in all nine cells**, and the mode-location error falls
monotonically as terminal noise drops, at every `n`. These are the same datasets
on which `gauss_jacobian` and F-NPSE return a unimodal blob (dip 0.93-1.00
against a reference dip of 0.000-0.068) and ModeTree found no branches at all.

This is **Case B**: topology retained, modes displaced outward by a systematic
bias that shrinks with `lambda`.

## Question B -- does integrating it reproduce the posterior? Partly

Locations yes, masses no. At `n=32, t=1e-4` the two basins are at -0.483/+0.499
against a reference -0.422/+0.480, but the reconstructed profile puts nearly all
mass on one mode, and the valley is far too deep (dip 0.000 vs reference 0.000
at n=32, 0.002 vs 0.068 at n=8). W1/sd is 0.87-1.35 for IK against an oracle
floor of 0.0004.

Part of the mass error is structural -- profile posteriors are sharper than
marginals -- but the oracle control bounds that contribution: with an exact
score the same procedure reproduces the reference dip to 0.011 vs 0.011. The
residual is genuine learned-score bias.

## Question C -- how does it change with n?

Topology is retained at n=8, 16 and 32 alike. Mode-location error at the best
terminal time improves slightly with n (0.087 -> 0.073 -> 0.060), and W1/sd
improves (1.17 -> 1.31 -> 0.87). There is no sign of collapse up the ladder for
this network.

## Question D -- how much faster?

Same datasets, same 3-CPU cap:

| n | first-order scan | gauss_jacobian | ModeTree-JAC |
|---|---|---|---|
| 8  | ~17 s | 64.7 s | 361.7 s (no branches) |
| 16 | ~29 s | 121.3 s | 470.6 s |
| 32 | ~56 s | 240.2 s | 176.7 s (no branches) |

**4.3x faster than `gauss_jacobian`**, 8-21x faster than ModeTree-JAC, and the
only one of the three that recovers the topology. This is with a deliberately
wasteful 1201-point grid; targeted multi-start search would use a small fraction
of those evaluations.

## The Gaussian-mixture network: Case C, and a specific cause

Same pipeline, same day, different checkpoint:

| n | exact posterior modes | learned basins (t=1e-3) | prior modes |
|---|---|---|---|
| 8  | +-1.13 / 1.20 | -1.963, -0.008, +2.144 | +-2.5 |
| 16 | +-0.72 / 0.79 | -1.915, -0.012, +2.098 | +-2.5 |
| 32 | +-0.42 / 0.49 | -1.887, -0.014, +2.087 | +-2.5 |

**Mode recall 0.00 in 7 of 9 cells.** At `t=1e-2` the field collapses to a single
mode at the origin (W1/sd 9.5, 14.5, 24.2). At lower `t` two basins appear, but
pinned near the **prior** modes: the true posterior contracts by 2.5x across the
ladder while the learned basins move by 0.06. The GM network's composed global
score is very nearly data-independent.

That retrospectively explains every GM result in this project -- the samplers'
mass at +-2.2, the error growing with `n`, and ModeTree finding no branches
(there was one basin to find, and it was the prior's).

## Root stability (Case D), honestly

Root **counts** are not stable. They change with grid resolution (601 / 1201 /
2401), with profiling iterations (300 / 900), and with local initialization
(cold vs warm-started across `t`). The cause is that **the local profiling
problem is itself multimodal**: `l*(g)` is not unique, so different
initializations give different fine structure in `S(g)`.

Root **clusters** are stable. Every configuration puts one basin near -0.5 and
one near +0.55, and the reconstructed density is stable to 4 decimals in W1/sd
(1.3119-1.3129 across all n=16 t=1e-4 configurations). Two kinds of extra root
appear:

* **micro-roots** inside a basin, spanning ~0.005-0.02 -- removed by merging at
  a tolerance two orders of magnitude below the true mode separation (~1.0);
* **stable false modes** near the origin at n=16 (e.g. +0.125, persisting across
  300 and 900 iterations) -- 0.46 from the nearest true mode, so *not* removable
  by merging.

So first-order mode search on this network needs a merge step and will still
occasionally report a false basin.

## What this means for the architecture decision

* **IK branch -- the modes are in the first-order field.** Curvature/JAC is the
  bottleneck, exactly as the decision tree's left branch describes. A direct
  denoising-covariance head plus first-order multi-start search is the right
  next step, and it should be both more accurate and ~4x cheaper than the
  Jacobian route.
* **GM branch -- the modes are not in the first-order field.** No covariance
  head can repair that checkpoint; its score training is what needs fixing.
* **Mode masses need the marginal, not the profile.** Profiling establishes
  topology but cannot give correct mode weights in a hierarchical model. This is
  where a moment/covariance head earns its keep for a second reason: it supplies
  the local integration in closed form instead of by optimization.
* **Before generalizing:** two checkpoints, two answers. Run this diagnostic on
  any network before assuming which branch applies to it. It costs ~1 minute per
  cell.

## Limitations

* The reconstructed density is the **profile** posterior throughout; Variant B
  (marginalizing the locals) was not implemented, and no conclusion about mode
  *masses* should be drawn without it.
* `t=1e-5` was not evaluated; `t=1e-4` is already near the low end of the
  network's trained noise range and the learned curves degrade below `t=1e-3`.
* The IK reference is ABC with its own noise floor; the reference score is a
  smoothing spline on `log p` masked to where the density exceeds 1e-3 of its
  peak, and its tails should not be read as structure.
* Single dataset per `(n, regime)`; no replicate seeds.
* The 10-D partial-pooling experiment (task sections 14-21) is not included here.
