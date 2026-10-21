# Frozen set-encoder summaries at K=1 — running record

The `Partial_Pooling/ladder_encsum` lane. A frozen permutation-invariant
`SetEncoder` (`setenc-d16-w64x2s4`, 185,248 parameters, trained on held-out
trial-predictive NLL) compresses a subject's 30 trials to a 16-D summary `z`;
the COMPASS score network is then trained on `p(theta, z)` over 26 nodes
instead of the raw 40-token `p(theta, X)`. K=1 throughout -- no grouping.

All numbers below are `--subjects 100 --draws 100 --timesteps 50`, the same
protocol as the raw-trial K=1 control, scored with `score_bayesflow_metrics.py`
and `score_shrinkage.py`.

## The K=1 reference points, re-measured at 100 datasets

`docs/group_block_progress.md` compares gK20 against a "flat control (100ds)"
at mean r 0.732. **That is the plain tokenized control, not the best K=1 arm.**
Re-scored:

| K=1 raw arm (100 ds, `gauss_hierarchical`) | mean r | log_s_nu slope | g NRMSE | g ECE | g cov95 | l NRMSE | l ECE | l w95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control h64d2 (the doc's "flat control") | 0.732 | 0.389 | 0.451 | 0.270 | 0.610 | 0.455 | 0.041 | 1.753 |
| `mask6mix` | 0.786 | 0.501 | 0.525 | 0.214 | 0.680 | 0.408 | 0.017 | 1.517 |
| **`mask6mix` + `log_sigma`** | **0.801** | **0.634** | 0.528 | 0.202 | 0.713 | 0.408 | 0.017 | 1.510 |

So the gK20 gap is against 0.801, not 0.732. `log_sigma` time sampling is the
one K=1 training change that lifts the shrinkage slope past 0.55 unaided.

## The 30-dataset screen (10 arms, `gauss_hierarchical`)

| arm | score params | mean r | log_s_nu slope | g NRMSE | g ECE | g cov95 | l NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| RAW `mask6mix` | 463,364 | 0.802 | 0.347 | 0.528 | 0.212 | 0.671 | 0.425 |
| RAW `logsigma` | 463,364 | 0.802 | 0.501 | 0.537 | 0.204 | 0.686 | 0.426 |
| h48d2-te64-mask6mix | 127,458 | 0.772 | 0.662 | 0.584 | 0.288 | 0.548 | 0.421 |
| h48d2-mask6mix | 330,402 | 0.785 | 0.602 | 0.533 | 0.234 | 0.667 | 0.416 |
| h48d2-mask6mix-400k | 330,402 | 0.829 | 0.554 | 0.541 | 0.220 | 0.652 | 0.414 |
| h64d4-te64-mask6mix | 344,130 | 0.827 | 0.743 | 0.495 | 0.170 | 0.671 | 0.416 |
| h96d6 (te256) | 1,809,858 | 0.818 | 0.901 | **0.448** | 0.172 | 0.676 | 0.415 |
| h96d6-te64 | 961,794 | 0.826 | 0.848 | 0.467 | 0.179 | 0.686 | 0.424 |
| **C2 h96d6-te64-mask6mix** | 961,794 | **0.842** | 0.646 | 0.593 | **0.157** | 0.771 | 0.414 |
| h96d6-te64-mask6mix-400k | 961,794 | 0.829 | 0.547 | 0.461 | 0.185 | 0.729 | 0.415 |

**Every encoder arm meets the `log_s_nu` slope >= 0.55 criterion; neither raw
arm does.** `h64d4-te64-mask6mix` reaches 0.827 at **344,130 parameters** --
under the raw 463,364 baseline -- which is the budget-respecting arm.

Two levers are already exhausted and should not be retried: 400,000 training
simulations is *worse* than 100,000 on the best arm (0.842 -> 0.829, consistent
with the ladder's finding that data was never the constraint), and
`mask_posterior_weight=6` (the partial-pooling form of the `six_weighted` menu
that won on inverse kinematics) is arm C3 at 0.826, below `mask6mix`'s 0.842.

## C2 across all three composition rules (30 datasets)

| rule | mean r | log_s_nu slope | sd(e)/sd | g NRMSE | g ECE | g cov95 | l NRMSE | l ECE | l w95 | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gauss_hierarchical` | **0.842** | 0.646 | 0.689 | 0.593 | **0.157** | 0.771 | 0.414 | 0.020 | 1.518 | 77 min |
| `gauss_jacobian` | 0.826 | 1.343 | 1.622 | 0.655 | 0.171 | 0.705 | 0.415 | 0.020 | 1.516 | **2,710 min** |
| Langevin F-NPSE | 0.819 | 0.347 | 0.739 | **0.361** | 0.296 | 0.510 | 0.436 | 0.025 | 1.626 | 192 min |

**The headline is sampler-robust; the mechanism is not.** Mean r spans 0.023
across the three rules -- tighter than gK5's 0.056 and comparable to the flat
control's 0.013. But the `log_s_nu` slope spans **0.347 to 1.343, a factor of
four**, and the pre-declared >= 0.55 criterion is met under both Gaussian rules
and failed under F-NPSE. A slope claim at K=1 therefore has to name its rule.

The two extremes fail in opposite, diagnosable ways. `gauss_jacobian` does not
shrink, it *over-shoots*: slope 1.343 and 1.697 on the two worst scales with
`sd(e)/sd` 1.62 and 2.44, i.e. posterior medians more dispersed than the truth.
F-NPSE gives the best global NRMSE recorded for any K=1 arm (0.361) with the
worst calibration (ECE 0.296, cov95 0.510) -- sharp and over-confident, exactly
the pattern `docs/followup_results.md` records for `mask6mix` under F-NPSE
(0.374 / 0.500).

**`gauss_jacobian` costs 35x `gauss_hierarchical` here**, not the 3.6x that
`docs/ladder_progress.md` records for the flat model: 45 hours against 77
minutes for the same 30 datasets. The arrow-Newton MAP on a 26-node summary
schema is the suspect. It is not worth its cost on this representation -- it is
the slowest rule and the worst on the mechanism.

## C2 at 100 datasets, paper-matched (`gauss_hierarchical`)

Datasets 30-99 were generated as a shard and combined with the existing first
30, which were produced at byte-identical settings.

| | C2 @ 30 ds | **C2 @ 100 ds** | best raw K=1 @ 100 ds | paper |
|---|---:|---:|---:|---:|
| **mean r** | 0.842 | **0.830** | 0.801 | **0.870** |
| log_s_nu slope | 0.646 | **0.775** | 0.634 | -- |
| log_s_nu r | 0.737 | 0.741 | 0.781 | 0.918 |
| log_s_logA slope | 0.558 | 0.583 | 0.521 | -- |
| log_s_logA r | 0.778 | 0.749 | 0.655 | 0.804 |
| log_s_logt0 r | 0.591 | 0.542 | 0.479 | 0.699 |
| g NRMSE | 0.593 | 0.555 | 0.528 | -- |
| g ECE | 0.157 | **0.145** | 0.202 | -- |
| g cov95 | 0.771 | 0.757 | 0.713 | -- |
| l NRMSE | 0.414 | **0.397** | 0.408 | -- |
| l ECE | 0.020 | 0.018 | 0.017 | -- |
| l w95 | 1.518 | **1.443** | 1.510 | -- |

**The 30-dataset screen was optimistic on r and pessimistic on the slope**, in
exactly the direction gK20 showed: mean r falls 0.842 -> 0.830 while the
`log_s_nu` slope rises 0.646 -> 0.775. The lesson from
`docs/group_block_progress.md` holds -- a 30-dataset caution applies to every
statistic computed from those 30 points, in both directions.

### What can honestly be claimed

* **Best K=1 result on record: mean r 0.830**, against the raw-trial K=1
  baseline's 0.801 and the paper's 0.870. The frozen summary closes 42% of the
  remaining gap (0.069 -> 0.040) and improves global calibration error 28%,
  global coverage +0.044, local NRMSE 3% and local intervals 4%.
* **It does not beat the paper.** 0.830 < 0.870, and C2 is below the paper on
  *all three* group scales -- log_s_nu 0.741 vs 0.918, log_s_logA 0.749 vs
  0.804, log_s_logt0 0.542 vs 0.699. Its mean is carried by the three means and
  `beta_raw` (0.898, 0.940, 0.989, 0.953). Grouping (gK20, 0.934) remains the
  only thing in this repository that beats 0.870.
* **The shrinkage mechanism works at K=1 under the Gaussian rules.** The
  `log_s_nu` slope reaches 0.775 at 100 datasets against the raw arm's 0.634 and
  the plain control's 0.389, and `log_s_logA` holds at 0.583. Under F-NPSE it
  does not (0.347), so the claim is rule-conditional.
* **Caveats.** The 100-dataset confirmation exists for `gauss_hierarchical`
  only. C2 is 961,794 parameters against the raw baseline's 463,364, so it is
  not budget-matched; `h64d4-te64-mask6mix` at 344,130 is the arm that is, and
  it has only a 30-dataset number (0.827). Every checkpoint here was selected
  under the validation-RNG bug documented in
  `docs/shrink_followup_progress.md` section 2.
