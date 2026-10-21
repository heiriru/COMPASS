# Follow-up experiments: mask menus, churn, time embedding, output heads

All arms share the Stage-4 winner's backbone (trial tokens, `h64/d2/x8/m4`,
463,364 parameters), the same 100,000 simulations, the same normalization
constants and the same test datasets. Recovery is the paper-matched protocol
(100 subjects x 100 draws x 50 DPM steps, `gauss_hierarchical`, churn eta=2)
truncated to the **first 30 test datasets** for every arm, including the frozen
100-dataset references, so every row is scored on datasets 0-29.

Caveat that applies to every global number below: a global metric has one sample
per dataset, so 30 datasets makes global NRMSE, ECE and `r` noticeably noisier
than the paper's 100. Local metrics get 30 x 100 subjects = 3,000 units and are
solid. Differences smaller than a few hundredths in the global columns should
not be read as real.

## 1. Training on the six conditionals

`compass.ConditionalMenu` draws the training condition mask from an explicit
menu instead of a per-token Bernoulli. `SIX_CONDITIONALS` is

    p(g,l|x)  p(g|l,x)  p(l|g,x)  p(x|g,l)  p(x,l|g)  p(x,g|l)

which is every non-degenerate conditional over a (global, local, observation)
partition -- the two excluded are the joint prior (conditions nothing) and the
fully conditioned mask (no latent coordinate, zero loss).

These are the masks inference actually evaluates:
`infer_partial_pooling.py:439` and `:949` use `[0]*10 + [1]*90` = p(g,l|x), and
`:830` uses `[1]*7 + [0]*3 + [1]*90` = p(l|g,x), the hierarchical score-ascent
mask. Under Bernoulli(0.33) the first has probability
`0.67**10 * 0.33**30 = 6.5e-17` and is never drawn.

**Uniform weighting over the six is wrong.** The three conditionals with the
observations latent carry 5-7x the loss of the three with them observed, so a
uniform menu spends roughly four fifths of its gradient on generating `x`. The
fix is `mask_posterior_weight` (upweight the observation-conditioned half) or
`mask_menu="six_mixed"` (75% menu, 25% Bernoulli).

Per-conditional validation loss, all arms scored by the same code on the same
rows, times and noise (`score_conditional_losses.py`; lower is better):

| arm | params | p(g,l\|x) | p(g\|l,x) | p(l\|g,x) | p(x\|g,l) | p(x,l\|g) | p(x,g\|l) |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 5,957,984 | 95,098 | 74,673 | 17,876 | 882,567 | 926,191 | 1,043,146 |
| control | 463,364 | 95,263 | 74,062 | 15,007 | 551,652 | 741,440 | 698,660 |
| mask6 (uniform) | 463,364 | 97,375 | 74,487 | 16,099 | 356,985 | 454,680 | 433,475 |
| mask6-pw4 | 463,364 | 94,885 | 73,758 | 14,640 | 358,302 | 455,343 | 434,534 |
| mask6-pw8 | 463,364 | 94,443 | 73,514 | 14,334 | 358,607 | 455,851 | 434,676 |
| **mask6mix** | 463,364 | **94,056** | **73,379** | **13,987** | **355,811** | **453,172** | **432,190** |
| te64 | 192,836 | 95,769 | 74,266 | 15,562 | 473,101 | 636,237 | 613,611 |
| pph | 463,949 | 95,397 | 74,072 | 15,477 | 421,687 | 569,445 | 497,566 |

`mask6mix` wins every column, including -6.8% on p(l|g,x), the score-ascent
conditional.

## 2. Churn: eta=8 is not reachable

`MultiObsSampler._make_sampling_schedule` builds the schedule with
`torch.logspace(sigma_max -> sigma_min)`, so sigma is **geometric** and the
relative step `(sigma_i - sigma_i+1)/sigma_i` is identical at every step. The
churn is `gamma = min(eta * rel_step, GAMMA_CAP)` with
`GAMMA_CAP = sqrt(2) - 1 ~ 0.414`. A constant `rel_step` makes that clamp
all-or-nothing: above one threshold eta, every step pins to the same gamma and
the sampler stops depending on eta.

Verified on one dataset, same seed, `dpm2_gauss_hierarchical`:

| pair | posteriors identical |
|---|---|
| eta=0 vs eta=3 | no |
| eta=3 vs eta=4 | no |
| eta=4 vs eta=5 | **yes, bit-identical** |
| eta=4 vs eta=8 | **yes, bit-identical** |
| eta=5 vs eta=8 | **yes, bit-identical** |

With `sigma_max = 9.845` and 50 steps the threshold is
`eta_crit = 0.414 / (1 - r)`, about 3.2-4.0 here. So `--churn-eta 8` is the
maximum churn the sampler can express, and the production runs labelled eta=4
and eta=8 are the same sampler -- which is why they scored identically to four
decimals across nine statistics.

Effect of maximum churn against eta=2 (30 datasets):

| method | eta | g NRMSE | g ECE | g cov95 | l ECE |
|---|---:|---:|---:|---:|---:|
| `gauss_hierarchical` | 2 | 0.436 | 0.300 | 0.595 | 0.041 |
| `gauss_hierarchical` | max | 0.451 | 0.308 | 0.552 | 0.047 |
| `gauss_jacobian` | 2 | 0.438 | 0.298 | 0.567 | 0.042 |
| `gauss_jacobian` | max | 0.453 | 0.319 | 0.510 | 0.046 |

More churn does not fix global calibration; it makes it slightly worse on both
methods. Global coverage is 0.51-0.60 against a nominal 0.95, so the failure is
over-confident global posteriors, and injecting more noise per step does not
reach it. A genuine high-churn sweep needs `GAMMA_CAP` raised or made
schedule-relative.

## 3. Time embedding 256 -> 64

`time_embedding_size` sets the input width of every adaLN modulation, not just
the clock, so it compounds:

| time_emb | total | t_embedder | block adaLN | final |
|---:|---:|---:|---:|---:|
| 256 | 463,364 | 131,584 | 197,376 | 33,156 |
| 64 | **192,836** | 33,088 | 49,920 | 8,580 |

**58% of the model**, and the trained 256-wide embedding does not use its width:
sweeping t and taking singular values of the modulation vectors gives effective
rank **2** at 99% of energy and 4 at 99.9%, identically in `t_embedder`, both
blocks and the final layer (`analyze_time_embedding.py`).

On the training objective this is nearly free: best validation loss 139.360 at
192,836 parameters against 139.289 at 463,364, a 0.05% difference for 58% fewer
parameters.

**But recovery does degrade**, and this is the interesting part: global NRMSE
0.632 vs 0.436, local NRMSE 0.506 vs 0.475, local ECE 0.064 vs 0.041, and wider
local intervals (w95 2.018 vs 1.838). The score-matching loss is almost blind to
what the compositional sampler needs -- a 0.05% loss difference maps to a 45%
global NRMSE difference. Do not size this network by validation loss alone.

## 4. Per-position output heads: hypothesis disconfirmed

`per_position_head` gives each of the 10 theta scalars its own readout instead
of one shared direction per block (IID blocks stay shared, so trials remain
exchangeable). Cost: 585 parameters.

It slightly improves the training objective (best val 138.567 vs 139.289) and it
does **not** fix local over-coverage: local coverage goes 0.965 -> 0.974, i.e.
further from nominal, and local ECE 0.041 -> 0.055. The shared output head is
not what causes the over-coverage.

What does fix it is the mask menu. `mask6mix` reaches local coverage 0.949
against a nominal 0.95 and local ECE 0.020, half the control's 0.041 and a third
of legacy's 0.054. **Local over-coverage is a training-distribution problem, not
an architecture problem.**

## Recovery, all arms

`gauss_hierarchical`, churn eta=2, 30 datasets x 100 subjects x 100 draws:

| arm | params | g NRMSE | g ECE | g cov95 | l NRMSE | l ECE | l cov95 | l w95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 5,957,984 | 0.765 | 0.174 | 0.714 | 0.438 | 0.054 | 0.922 | 1.557 |
| control | 463,364 | **0.436** | 0.300 | 0.595 | 0.475 | 0.041 | 0.965 | 1.838 |
| mask6 | 463,364 | 0.600 | 0.216 | 0.667 | 0.483 | 0.031 | 0.945 | 1.794 |
| mask6-pw4 | 463,364 | 0.634 | **0.199** | 0.671 | 0.435 | **0.019** | 0.947 | 1.600 |
| mask6-pw8 | 463,364 | 0.541 | 0.226 | 0.643 | 0.430 | 0.023 | 0.947 | 1.592 |
| **mask6mix** | 463,364 | 0.515 | 0.223 | **0.676** | **0.426** | 0.020 | **0.949** | **1.584** |
| te64 | **192,836** | 0.632 | 0.220 | 0.619 | 0.506 | 0.064 | 0.977 | 2.018 |
| pph | 463,949 | 0.581 | 0.225 | 0.614 | 0.493 | 0.055 | 0.974 | 1.939 |

Median correlation `r`, the paper's own global metric:

| arm | log s_nu | log s_logA | log s_logt0 | mean r (7 globals) |
|---|---:|---:|---:|---:|
| **paper** | 0.918 | 0.804 | 0.699 | **0.870** |
| legacy 5.96M | 0.563 | 0.558 | 0.768 | 0.707 |
| control 463k | 0.601 | 0.452 | 0.450 | 0.726 |
| mask6-pw8 | 0.722 | 0.581 | 0.552 | 0.783 |
| **mask6mix** | 0.653 | **0.698** | 0.575 | **0.799** |

`mask6mix` closes about half the remaining gap to the paper at the same 463,364
parameters and zero extra cost, concentrated on the group-scale parameters that
were weakest.

## mask6mix across all three inference methods

The mask menu is a training change, so it should be checked against every
sampler the benchmark deploys, not just the one it was selected on.

| method | arm | g NRMSE | g ECE | g cov95 | l NRMSE | l ECE | l cov95 | l w95 | mean r |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Langevin + F-NPSE | control | 0.414 | 0.307 | 0.443 | 0.513 | 0.056 | 0.977 | 2.018 | 0.766 |
| Langevin + F-NPSE | **mask6mix** | **0.374** | **0.291** | **0.500** | **0.451** | **0.033** | **0.961** | **1.709** | **0.790** |
| DPM2 + `gauss_hierarchical` | control | **0.436** | 0.300 | 0.595 | 0.475 | 0.041 | 0.965 | 1.838 | 0.726 |
| DPM2 + `gauss_hierarchical` | **mask6mix** | 0.515 | **0.223** | **0.676** | **0.426** | **0.020** | **0.949** | **1.584** | **0.799** |
| DPM2 + `gauss_jacobian` | control | **0.438** | 0.298 | 0.567 | 0.476 | 0.042 | 0.966 | 1.839 | 0.713 |
| DPM2 + `gauss_jacobian` | **mask6mix** | 0.559 | **0.224** | **0.638** | **0.426** | **0.018** | **0.948** | **1.585** | **0.803** |

Under **F-NPSE the menu wins every column outright** -- it is strictly better on
accuracy, calibration, sharpness and correlation at the same 463,364 parameters.
Under the two Gaussian-composition rules it wins every local column, every
calibration column and the paper's correlation metric, and loses only global
NRMSE (0.436 -> 0.515 and 0.438 -> 0.559).

Two things follow. `mask6mix` + F-NPSE gives the best global NRMSE recorded here
for any arm on any method (0.374), and `mask6mix` improves the paper's own global
metric on all three methods, with the largest gain on `gauss_jacobian`
(0.713 -> 0.803 against the paper's 0.870).

The global NRMSE regression under Gaussian composition is the one place the menu
costs something. It is consistent with what the coverage columns say: the menu
widens the global posteriors (cov95 0.595 -> 0.676), which is a large calibration
gain and a small accuracy loss, since NRMSE is computed over draws and charges
for posterior width.

## Two bugs found along the way

**`Trainer.loss_fn` returned a batch mean against a batch sum.**
`sigma_t` is `(B,1,1)` and `torch.sum(..., dim=(1,2))` is `(B,)`, so their
product broadcast to `(B,1,B)` -- an outer product pairing every example's
sigma^2 with every other example's residual. `mask_loss_weights` could not work,
and because `_run_epoch` divides by the batch size, the short final batch of an
epoch (8 rows of a 5,000-row validation split) carried sixteen times the weight
of a full one. Validation swung +-40% between epochs and three arms early-stopped
on a lucky 8-row draw, all at the identical epoch. Restored to the
batch-size-invariant expression every existing checkpoint was trained under, with
`mask_loss_weights` implemented correctly on top
(`tests/test_loss_batch_invariance.py`).

**`runtime.py:configure_cpu_limit` always chose CPUs 0, 1, 2.** It took
`available[:3]`, which honours the three-CPU cap but pins every concurrent script
to the same three cores. Measured: CPUs 0/1/2 at 100% while the other 69 sat
~90% idle, run queue 51 on a 72-core machine, and recovery datasets slowed from
~3 to ~50 minutes each. Now offset by PID; the cap is unchanged
(`tests/test_cpu_affinity_spread.py`).

Validation masks are now also drawn once from a fixed seed rather than redrawn
every epoch, so early stopping and best-checkpoint selection score the model on
the same task each time (`tests/test_validation_mask_stability.py`).

## Where the remaining bias actually is

Global NRMSE decomposes as `bias^2 + posterior variance` (it is computed over
draws), while `mean r` is a correlation of the posterior *median* and is blind to
width. Measured on `mask6mix`, 30 datasets:

| arm | method | NRMSE | bias | width |
|---|---|---:|---:|---:|
| control | `gauss_hierarchical` | 0.4359 | 0.3335 | 0.2234 |
| mask6mix | `gauss_hierarchical` | 0.5145 | **0.3283** | **0.3010** |
| control | F-NPSE | 0.4140 | 0.3712 | 0.1659 |
| mask6mix | F-NPSE | 0.3738 | **0.3343** | 0.1634 |

So the menu's apparent NRMSE regression under DPM2 is entirely width: bias fell
slightly, and the control was badly under-covered (cov95 0.595 against a nominal
0.95) so the extra width is a calibration gain that NRMSE bills as a loss.

Regressing the posterior median on the truth separates offset from shrinkage.
`mask6mix`, `gauss_hierarchical`, 30 datasets:

| parameter | shift (sd) | \|shift\|/se | slope | sd(err)/sd |
|---|---:|---:|---:|---:|
| mu_nu | 0.074 | 0.9 | 0.782 | 0.471 |
| mu_log_alpha | 0.012 | 0.1 | 1.057 | 0.573 |
| mu_log_t0 | 0.002 | 0.0 | 1.040 | 0.191 |
| **log_sigma_nu** | **0.362** | **2.5** | **0.339** | 0.782 |
| **log_sigma_log_alpha** | -0.160 | 1.2 | **0.532** | 0.731 |
| log_sigma_log_t0 | -0.239 | 1.0 | 0.897 | 1.301 |
| beta_raw | 0.040 | 0.5 | 0.948 | 0.429 |

The group **means** track essentially perfectly. The group **scales** are shrunk
toward a constant: a slope of 0.339 mechanically caps that parameter's `r` near
0.65, which is the 0.653 measured. This is the whole of the remaining gap.

### What the shrinkage is not

Slope of the posterior median on the truth, same network (`mask6mix`):

| rule | s_nu | s_logA | s_logt0 |
|---|---:|---:|---:|
| `gauss_hierarchical` | 0.339 | 0.532 | 0.897 |
| `gauss_jacobian` (pilot-free) | 0.331 | 0.541 | 1.069 |
| Langevin F-NPSE | 0.322 | 0.222 | 0.178 |
| `gauss_hierarchical`, 400 steps | 0.333 | 0.541 | 0.957 |

Three structurally different composition rules and an 8x finer solver all give
`s_nu ~ 0.33`. Ruled out as causes:

* **Composition rule** -- identical across Schur, Jacobian and F-NPSE.
* **Discretisation** -- 50 -> 200 -> 400 steps moves bias 0.328 -> 0.343 -> 0.349
  and `mean r` 0.799 -> 0.801 -> 0.802, i.e. nothing (slightly worse).
* **Churn** -- saturates at eta ~ 3.5 and is worse than eta=2 (see above).
* **Composition prior** -- `_normalized_global_prior` uses `GLOBAL_PRIOR_MEAN/STD`,
  which is exactly what `sample_hierarchical_prior` draws from. Not misspecified.
* **Damped composition**, including Arruda et al.'s own estimator -- see below.
* **Capacity and simulation budget** -- established saturated earlier.

What remains is the score network's own representation of the group scales.

### Damped composition does not transfer

`MultiObsSampler` has always carried `"damping"` (its docstring: "the
error-damped compositional estimator of Arruda et al. 2026", the same idea as the
reference implementation's `compositional_bridge_d0`), plus `gauss_damping`,
`hybrid` and `hybrid_damping`. None were exposed in `INFERENCE_METHODS`, and none
could run at all until the `terminal_corrector_counts` fix. Now exposed as
`langevin_damping`, `dpm2_gauss_damping`, `dpm2_hybrid`, `dpm2_hybrid_damping`.

Langevin pair, same network, first 16 datasets:

| arm | g NRMSE | bias | width | mean r | g ECE |
|---|---:|---:|---:|---:|---:|
| F-NPSE | **0.4016** | **0.3607** | 0.1682 | **0.771** | 0.2805 |
| damping (Arruda) | 0.5391 | 0.4296 | 0.2401 | 0.745 | **0.2709** |

Damping *raises* bias. `hybrid` and `hybrid_damping` produce global coverage of
0.81 and 1.000 against a nominal 0.95 -- posteriors so wide as to be
uninformative -- and cost 58x more per dataset than the reference, because they
need a per-dataset pilot precision estimate that `gauss_jacobian` is explicitly
built to avoid. Both hybrids were stopped early on those grounds.

### Why the scales are hard here, and what would fix them

The scales are identified from the *spread* of the subject-level parameters, but
the network models one subject at a time and the sampler composes 100
single-subject posteriors. Scale information can reach the globals only through
the global-local correlation inside each single-subject factor.

That ceiling is soft, not hard -- composing more subjects does help:

| subjects | s_nu | s_logA | s_logt0 | m_nu |
|---:|---:|---:|---:|---:|
| 20 | 0.256 | 0.214 | 0.217 | 0.554 |
| 100 | 0.289 | 0.337 | 0.439 | 0.750 |

but sublinearly, and 100 is already the paper's protocol. At fixed n=100 the
training distribution moves the same quantity much harder: `mask6mix` takes
s_logA from 0.337 to 0.532 and s_nu from 0.289 to 0.339.

The paper does not face this constraint in the same form: its global network is a
`SetTransformer` over the whole set of subjects, so it observes the spread
directly instead of reconstructing it from composed single-subject factors. That
is the ladder's Stage 5/6 (a learned summary), deliberately not built. It is the
highest-leverage remaining change for the group scales, and it can be added as an
*additional* conditioning route for the globals while keeping the joint
`p(theta, X)` model, so the raw likelihood is not given up.
