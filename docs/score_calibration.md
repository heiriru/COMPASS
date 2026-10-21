# Is the learned score miscalibrated at small noise? Yes — and it cannot be corrected post hoc

Measured 2026-09-10 with
`tutorials/Compositional_Inference_Fix/New_Attempt/exponential_score_comparison/score_calibration.py`.
Artifacts: `.../artifacts/score_calibration/{calibration.csv,01_score_calibration.png}`.

## Why this measurement

The composed posteriors in `inverse_kinematics` are systematically too narrow
(widths 0.72–0.85 of reference, 0.68 at n=16) and a Fisher-identity transplant
of the marginal score collapsed to width 0.156. Both are consistent with the
network's score having **the right shape but the wrong scale** — an error that
is invisible at n=1 and amplified by composition, which sums `n` row scores
against an `(n-1)`-fold prior subtraction.

In `inverse_kinematics` that could only be probed through a KDE, so the
"slope 0.80" number there was confounded with the density estimate. The
exponential hierarchy admits the **exact** diffused score at every noise level
(`hierarchy.diffused_score`), so here the learned field is regressed directly
on the truth with no density estimate anywhere.

For each `λ` and each observation, states are drawn from the exact diffused
single-observation posterior `p_λ(g_t, l_t | x_j)` — the states the sampler
actually visits — and `learned = slope · exact + intercept` is fitted by OLS.
Never pooled across `λ`: the score magnitude runs as `1/λ`, so a pooled fit
would measure the ladder rather than the network. 8 observations, 4096 states,
seed 0.

## Result 1 — the shrinkage is real, and it is a capacity effect

OLS slope on the shared coordinate (1.000 = calibrated):

| λ | h128d6 | h32d3 | **h16d2** | h8d1 | h4d1 |
|---|---|---|---|---|---|
| 0.01 | 0.958 | 0.890 | **0.837** | 0.854 | 0.561 |
| 0.02 | 0.984 | 0.924 | **0.871** | 0.877 | 0.570 |
| 0.05 | 0.986 | 0.958 | **0.902** | 0.873 | 0.565 |
| 0.10 | 0.970 | 0.969 | **0.942** | 0.868 | 0.563 |
| 0.20 | 1.002 | 0.980 | **0.987** | 0.947 | 0.669 |
| ≥0.5 | 1.00–1.01 | 1.00–1.01 | **0.99–1.02** | 0.98–1.04 | 0.98–1.07 |

Three things this settles:

1. **The slope deficit exists and is not a KDE artifact.** It is measured here
   against an exact reference.
2. **It lives entirely at small `λ`.** Every model on the ladder is calibrated
   to within 2% for `λ ≥ 0.5`. This is a terminal-noise problem.
3. **It is monotone in capacity.** `inverse_kinematics` runs **h16d2**, which
   sits at 0.90 / 0.84 — the same neighbourhood as the 0.80 measured through
   the KDE there. The ceiling model h128d6 is at 0.986.

The local coordinate is consistently worse than the shared one at small `λ`
(h16d2: 0.895 at λ=0.05, 0.784 at λ=0.01).

Correlation collapses at the bottom for **every** model — h128d6 is at
r = 0.675 at λ=0.01 — which reproduces the `r = 0.579 at σ = 0` seen in
`inverse_kinematics`. Part of that is this model's hard wall at `min_j x_j`,
where the exact score diverges; but capacity clearly moves it (h4d1: 0.313).

## Result 2 — post-hoc affine recalibration is refuted

The proposed cheap fix was `s → (s − intercept) / slope`, fitted on held-out
simulations. Relative RMSE before → after, shared coordinate:

| λ | h128d6 | h16d2 | h8d1 | h4d1 |
|---|---|---|---|---|
| 0.01 | 1.047 → **1.092** | 1.190 → **1.408** | 2.133 → **2.471** | 1.794 → **3.035** |
| 0.05 | 0.094 → 0.093 | 0.264 → **0.271** | 0.733 → **0.820** | 0.805 → **1.177** |
| 0.10 | 0.058 → 0.051 | 0.141 → 0.134 | 0.357 → **0.382** | 0.601 → **0.730** |
| 0.50 | 0.022 → 0.020 | 0.038 → 0.033 | 0.097 → 0.095 | 0.226 → 0.228 |

**It makes the error worse at exactly the noise levels where the slope deficit
is largest**, and helps only marginally where the network was already
calibrated — i.e. where there is nothing to gain.

The reason is that slope and noise are not separable here. A slope of 0.90 with
`r = 0.965` is what a network does when it cannot resolve its target: it
**regresses to the conditional mean**. Restoring the variance by dividing by
0.90 restores the scale *and* amplifies the uncorrelated part, which at small
`λ` is the dominant term. The shrinkage is not a gain error sitting on top of a
good estimate; it *is* the estimate being unresolved.

## What this means for the plan

- The hypothesis motivating step 1 is **confirmed as a diagnosis** and
  **refuted as a cheap fix**. There is no correction factor to apply.
- The lever is capacity and small-`λ` training, not post-processing. The
  ladder shows the deficit closing monotonically (0.565 → 0.902 → 0.986 at
  λ=0.05 across h4d1 → h16d2 → h128d6), so it is reachable — `inverse_kinematics`
  is simply running a model four rungs below the calibrated one.
- Anything that composes in score space inherits this, because composition
  evaluates the score at the noise levels where it is least resolved.
  Methods that avoid the score at small `λ` — the density product — do not.

## A caveat that needs its own run

`h16d2_n200000_log_sigma` is **uniformly worse** than `h16d2_n200000` for
λ ≥ 0.05 (rel RMSE 0.691 vs 0.264 at λ=0.05), while `h8d1_n200000_log_sigma` is
clearly **better** than `h8d1_n200000` (0.320 vs 0.733). One checkpoint each, so
this is more likely early-stopping luck than a capacity-dependent property of
`log_sigma` — the DSM validation loss is known to plateau in noise here, and
patience was raised to 150 for exactly that reason. It should not be read as
evidence against `log_sigma` until both are retrained on several seeds.

## Result 3 — the causal link to composed width is **refuted**

The point of Results 1 and 2 was a prediction: if the small-`lambda` slope
deficit is what makes composed posteriors too narrow, the composed width should
recover as capacity rises. It does not. `compare.py --stage sample`, 30
observations, 2000 draws, 100 steps, seed 0, GPU 6, artifacts in
`composed_ladder.csv`:

| config | slope @ λ=0.05 | rule | composed W1/σ | mean err/σ | **width ratio** |
|---|---|---|---|---|---|
| h4d1 | 0.565 | jacobian | 0.692 | 0.692 | **1.066** |
| h8d1 | 0.873 | jacobian | 0.449 | 0.446 | **0.992** |
| h16d2 | 0.902 | jacobian | 0.034 | 0.003 | **0.999** |
| h32d3 | 0.958 | jacobian | 0.076 | 0.059 | **1.013** |
| h4d1 | 0.565 | hierarchical | 0.676 | 0.676 | **0.999** |
| h8d1 | 0.873 | hierarchical | 0.479 | 0.479 | **0.995** |
| h16d2 | 0.902 | hierarchical | 0.064 | 0.004 | **1.089** |
| h32d3 | 0.958 | hierarchical | 0.081 | 0.078 | **1.112** |
| h128d6 | 0.986 | hierarchical | 0.173 | 0.128 | **1.225** |

(The h128d6 Jacobian arm OOMs at 10.5 GiB inside `_row_score_jacobian`'s `jvp`,
and the footprint does not fall with `--num-samples`, so it is a fixed cost of
that rule at this width. The hierarchical arm covers the rung instead.)

**No width deficit appears anywhere.** Even h4d1, with a 44% slope deficit,
composes to width ratio 1.066 — slightly *too wide*. Under `gauss_hierarchical`
the widths are ≥ 1 throughout and get **worse with capacity**, the opposite of
the prediction.

Two things follow.

1. **The `inverse_kinematics` width deficit (0.72–0.85, 0.68 at n=16) is not
   caused by score miscalibration.** This ladder rules out the competing
   explanation, leaving the one already established there: the Gaussian summary
   collapses two unequally-weighted branches into one narrow blob, and
   compounding that `n` times is what shrinks the width. It is a *rule* defect
   on multimodal posteriors, not a *network* defect.
2. **What capacity does buy is the mean, and it has a threshold.** Below it the
   composed error is a pure location error — `W1/σ` equals `mean_err/σ` to three
   decimals at h4d1 and h8d1, the √n bias amplification. Crossing h8d1 → h16d2
   drops composed `W1/σ` 13-fold (0.449 → 0.034). Above the threshold more
   capacity does **not** help and under `gauss_hierarchical` actively hurts
   (h128d6 is the worst rung at 0.173).

So "train a bigger score" is not the fix either. `inverse_kinematics` runs
h16d2, which on this problem sits *at* the threshold rather than below it.

## Revised conclusion

Step 1 set out to test whether score calibration is the binding constraint on
composed accuracy. Measured end to end, it is not:

- the shrinkage is real and capacity-graded (Result 1),
- it cannot be corrected post hoc, because it is regression to the mean rather
  than a gain error (Result 2),
- and it does not propagate into composed width at all (Result 3).

The binding constraint on a **unimodal** hierarchy is the per-observation mean,
which capacity fixes up to a threshold and no further. The binding constraint on
a **multimodal** one is the Gaussian summary itself, which no amount of network
quality repairs. That is an argument for the dispatcher (step 2) and against any
further single-rule tuning — the two failure modes have different cures.

## Not done

Re-running the ladder at other seeds and observation counts; measuring the same
calibration on the `inverse_kinematics` checkpoint against an exact reference
(there isn't one); confirming the predicted composed-width recovery by grading a
composed posterior from a higher-capacity checkpoint.
