# Latent attention + Arruda/Schur composition — running record

Started 2026-09-30. Derivation: `docs/arruda_schur_derivation.md`.
Nothing in `src/compass/*.py` or in any benchmark directory was edited.

| piece | where |
|---|---|
| latent-attention trunk (`none` / `global` / `hierarchical` / `full`) | `src/compass/experimental/latent_attention.py` |
| `arruda_schur`, `arruda_schur_gauss` rules; estimators `direct` / `schur` / `laplace`; ancestral locals | `src/compass/experimental/marginal_schur.py` |
| exact linear-Gaussian oracle network | `src/compass/experimental/oracles.py` |
| tests (23) | `tests/test_experimental_latent_attention.py`, `tests/test_experimental_marginal_schur.py` |
| benchmark drivers (in-process patching only) | `experiments_arruda_schur/{gm,ik,pp,twins}.py` |
| every output | `experiments_arruda_schur/artifacts/`, logs in `experiments_arruda_schur/logs/` |

Rule names used below: `gauss_jacobian_FD` is the production joint-state rule.
`arruda_<est>` is the plain Arruda sum on the marginal state, and
`arruda_gauss_<est>` is the precision-weighted marginal rule.

## 1. Verified on exact oracles (tests)

- **Attention modes.** Each mode has exactly the zero pattern in the network
  Jacobian that the derivation's table predicts. `none` reproduces the
  production trunk to 1e-5, the state dict is unchanged, and latent IID trials
  never couple.
- **Schur marginal.** Exact for any carried `l` on a jointly Gaussian factor
  (1e-4).
- **Laplace with the log-det term.** Recovers the exact marginal of a scale
  parameter (`l ~ N(μ, e^{2s})`) to 2e-3 from an arbitrary starting `l`.
  Without the log-det term the `∂/∂log σ` score is low by exactly
  `v/(σ²+v)` per subject, matching the analytic value to 8 digits.
- **Composed posteriors, exact linear-Gaussian oracle** (H=2, J=12):
  - `arruda_schur_gauss` and production `gauss_jacobian_FD` hit the exact
    mean *and* width within 0.15 sd.
  - The plain Arruda **sum** gets the mean right but is **~0.83× too narrow**.
    It integrates the bridge `p_t(g)^{1−J} ∏ p_t(g|x_j)`, which is not a
    diffused posterior.
  - Ancestral locals match the exact local posteriors.
  - Mean-field factors (the production trunk's) are measurably wrong under the
    same rule.
- Regression: the log-det gradient must take its baseline curvature at the
  same local state as its shifts (`test_log_det_gradient_is_taken_at_one_local_state`;
  see §5).

## 2. Training: validation loss by attention mode

The data, the validation split and the recipe are the same within each benchmark.

| benchmark | none | global | hierarchical | full |
|---|---:|---:|---:|---:|
| partial pooling (C2 recipe) | ~44 (production) | 43.7 | 41.6 | **29.9** |
| IK angle+height (h32d3wide) | 1.28 | 1.26 | pending | pending |
| GM (h32d3wide) | not retrained (production checkpoint reused) | – | 2.65 | 2.63 |

Only `full`, where locals and globals see each other, moves the loss
materially, and only on partial pooling (7 globals, 3 locals).

## 3. Partial pooling, K = 1, encoder summary

Matched protocol: datasets 0–9, 100 subjects, 100 draws, 50 steps, and the
same std-RMSE scoring as the published C2 screen.

| network / rule | global std-RMSE | local std-RMSE |
|---|---:|---:|
| production `none` / `gauss_hierarchical` (C2, published files) | **0.709** | 0.351 |
| production `none` / `gauss_jacobian` (C2, published files) | 1.106 | 0.350 |
| `full` / `gauss_jacobian_FD` | 0.719 | **0.272** |
| **`full` / `gauss_hierarchical`** | **0.551** | 0.276 |

**Headline: `full` attention with the production `gauss_hierarchical` rule lowers
the global error 22% and the local error 21%**, on the same 10 datasets and the
same protocol, at 80–160 s per dataset (the Jacobian rule takes 350–550 s).
With 10 datasets this is not yet a significance test; 30 are needed to confirm.

- **The Jacobian rule needs a joint score.** On the mean-field trunk it
  over-shoots badly. With `full` attention it matches the best production
  global error and improves the locals by about 23%.
- **The Arruda marginal-state rules lose on this trained network** (dataset 0,
  10 draws, after the §5 fix):

  | rule | global | local |
  |---|---:|---:|
  | `gauss_jacobian_FD` | 0.51 | 0.19 |
  | `arruda_gauss_laplace` | 1.33 | 0.53 |
  | `arruda_schur` | 1.50 | 0.39 |
  | `arruda_direct` | 1.53 | 0.50 |
  | `arruda_laplace` | 2.31 | 1.00 |

  The working explanation: the marginal rules query the learned joint score at
  Newton modes or carried locals for every global sample, including at high
  noise, far from the joint posterior, and composing 100 subjects multiplies
  those errors. The joint-state rule keeps each l_j tied to g.
- `arruda_laplace_damped` crashes: the damped path skips the denoise clamp, so
  nothing bounds it. It was not pursued.

## 4. Gaussian mixture

**Exact scores (oracle), asymmetric regime** (exact branch mass 0.75),
W1/sd, 1 replicate:

| n | gauss_jacobian_FD | arruda_direct | arruda_schur | arruda_laplace |
|---:|---:|---:|---:|---:|
| 8 | 0.59 | 0.51 | 0.32 | **0.13** |
| 16 | 1.23 | 0.44 | 0.41 | **0.19** |
| 32 | 2.48 | 0.49 | 0.39 | **0.25** |
| 64 | 4.59 | 0.46 | pending | pending |

The production rule doubles its error with every doubling of n; the marginal
rules stay roughly flat.

**Exact scores, symmetric regime** (exact mass 0.50), n=8:

| rule | W1/sd |
|---|---:|
| `arruda_schur` | **0.086** |
| `arruda_direct` | 0.149 |
| `arruda_laplace` | 0.206 |
| `arruda_gauss_laplace` | 0.485 |
| `arruda_gauss_direct` | 0.488 |
| `gauss_jacobian_FD` | 0.515 |

Precision weighting adds nothing on this bimodal target.

**Trained `hierarchical` network:** every rule is poor by n = 32, and the
plain sum collapses onto the wrong branch (mass 0.00). The Laplace rows on
this network are double-counting controls, because its g-output is already
marginal.

## 5. Bug found and fixed (2026-09-30)

`_logdet_gradient` took its baseline log det H_ll at the local state *before*
the last Newton step and its shifted curvatures *after* it. Every column was
therefore (Δ log det along the Newton step) / (FD step): identical in all
coordinates and divergent as λ → 0 (≈ −60,000 per coordinate on partial
pooling).

It was invisible on every oracle, because there H_ll does not depend on l.
The GM oracle Laplace rows above ran on the old code, and on GM this has
little effect.

A trust region on the Newton step (3·√(s_l² + λ²) per coordinate) was added
in the same pass. It did not change the partial-pooling result.

## 6. Inverse kinematics, angle+height, n = 8, 30 datasets

W1/sd against the exact MCMC reference, with the sd ratio in brackets.

| network | rule | angle_1 | height |
|---|---|---:|---:|
| `none` (tokenized) | `gauss_hierarchical` | 1.89 (2.12×) | 2.88 (3.79×) |
| `none` | `gauss_jacobian_FD` | 1.93 (2.21×) | 2.71 (3.94×) |
| `none` | `arruda_direct` | 2.18 (1.69×) | 2.00 (2.74×) |

For comparison, the legacy network in `recovery/README.md` with
`gauss_hierarchical` scores 1.79 (2.44×) on angle_1 (100 datasets). The
`full`, `global` and `hierarchical` networks are queued.

### 6b. IK, all four attention modes (final; `artifacts/figures/10_ik_attention_modes.png`)

Angle+height layout, n = 8, the same 30 datasets throughout. W1/sd against the
exact MCMC reference, with the sd ratio in brackets.

| network / rule | angle_1 | height |
|---|---:|---:|
| `none` / `gauss_hierarchical` | 1.89 (2.12×) | 2.88 (3.79×) |
| `none` / `gauss_jacobian_FD` | 1.93 (2.21×) | 2.71 (3.94×) |
| `global` / `gauss_jacobian_FD` | 1.64 (3.23×) | 2.73 (4.25×) |
| **`global` / `arruda_direct`** | **1.06 (1.26×)** | **1.17 (1.56×)** |
| `hierarchical` / `gauss_jacobian_FD` | 1.27 (2.11×) | 2.01 (3.35×) |
| **`hierarchical` / `arruda_direct`** | **1.08 (1.38×)** | **1.09 (1.56×)** |
| `hierarchical` / `arruda_gauss_direct` | 1.06 (1.87×) | 1.63 (3.12×) |
| `full` / `gauss_jacobian_FD` | 1.21 (2.08×) | 2.28 (3.55×) |
| `full` / `gauss_hierarchical` | 1.79 (2.25×) | 1.85 (2.77×) |
| `full` / `arruda_laplace` | 2.16 (2.65×) | 5.11 (5.77×) |
| `full` / `arruda_gauss_laplace` | 1.90 (2.74×) | 4.46 (5.11×) |

At n = 32 the `none` network is 13–35 W1/sd off under every rule. The other
modes were not run at n = 32.

### 4b. GM, production `none` network (2 replicates)

Every rule loses the positive branch (mass ≤ 0.03 at n ≥ 32), and W1/sd is
about 10 at n = 128. On this mean-field network `arruda_laplace` ≡
`arruda_direct` and `arruda_gauss_laplace` ≡ `arruda_gauss_direct` to 3
digits. That is what the derivation predicts: with no latent–latent attention,
the Schur gain and the log-det term are identically zero, so the match is a
consistency check.

The posterior figures (`artifacts/figures/9_gm_posteriors_{oracle,none,hierarchical}.png`)
make the split clear. With exact scores the marginal rules track the exact
bimodal posterior at every n from 8 to 64, while `gauss_jacobian_FD`'s branches
drift outward. With trained networks every rule collapses into the valley
(`none`) or onto the wrong branch (`hierarchical`). GM is limited by the
networks, not by the rules.

### Noise twins (exact scores), final

W1/σ; the Laplace rows were re-run with the fixed log-det code.

| rule | Gaussian | Laplace | exponential |
|---|---:|---:|---:|
| production `gauss_jacobian(_FD)` | 0.077 | 0.097 | **0.243** |
| `arruda_direct` | **0.062** | **0.063** | 0.560 |
| `arruda_gauss_direct` | 0.087 | 0.110 | 0.393 |
| `arruda_schur` | 0.066 | 0.239 | 1.77 |
| `arruda_laplace` | 0.066 | 0.97 | 1.18 |
| `arruda_gauss_laplace` | 0.068 | 1.23 | 1.28 |

The exponential twin's local conditional is Uniform(g, x_j), flat with a
g-dependent support, so it falls outside the class where Schur and Laplace
apply. The derivation says as much, and production wins there.

### 3b. Partial pooling, 30-dataset confirmation

| arm (datasets 0–29) | global std-RMSE | local std-RMSE | mean r |
|---|---:|---:|---:|
| production `none` / `gauss_hierarchical` | 0.718 | 0.339 | 0.843 |
| production `none` / `gauss_jacobian` | 1.133 | 0.338 | 0.825 |
| **`full` / `gauss_hierarchical`** | **0.554** | **0.250** | **0.850** |

Datasets 0–9, every arm (`artifacts/figures/11_partial_pooling_final.png`):

| network / rule | global std-RMSE | local std-RMSE | mean r |
|---|---:|---:|---:|
| `global` / `gauss_jacobian_FD` | 0.565 | 0.348 | 0.845 |
| `global` / `arruda_direct` | 1.399 | 0.339 | 0.792 |
| `hierarchical` / `gauss_jacobian_FD` | 0.672 | 0.278 | 0.856 |
| `hierarchical` / `arruda_direct` | 1.101 | 0.304 | 0.855 |

On partial pooling the plain Arruda sum keeps a high mean r but has a large
std-RMSE, which indicates a consistent offset or scale error rather than
scatter. The derivation's ~0.83× width bias of the bridge, summed over 100
subjects, is the first suspect. It has not been verified.

## 7. Conclusions (2026-10-01)

1. **Latent attention helps, and the mode matters per benchmark.**
   - On partial pooling, `full` with `gauss_hierarchical` lowers the global
     std-RMSE by 23% and the local by 26% (30 datasets).
   - On IK, every non-`none` mode improves on production.
   - Letting the globals attend each other (`global`) is what fixes
     `gauss_jacobian(_FD)`: on partial pooling its global std-RMSE goes from
     1.11 to 0.57 and its mean r from 0.75 to 0.85. The rule's Jacobian
     machinery needs a joint score.
2. **The Arruda method works when the network supplies the marginal score by
   construction.** With `global`/`hierarchical` attention, the g-output is the
   learned joint marginal score of p_t(g | x_j), which is what Arruda's global
   network learns. `arruda_direct` then gives the best IK result of any arm
   (W1/sd 1.06–1.08 / 1.09–1.17 against production's 1.89 / 2.88, and widths
   1.3–1.6× instead of 2–4×). It also beats production on two of the three
   twins with exact scores, and on the GM oracle at every n.
3. **Reaching the marginal by a Schur/Laplace reduction of a `full` network's
   joint score is exact on oracles but fragile on learned networks.**
   - It needs second derivatives of the learned score.
   - It assumes l | g is Gaussian, which fails for IK and the exponential twin.
   - It lost to the joint-state rules on every trained benchmark.

   Preferred route: get the marginal from the architecture
   (`global`/`hierarchical`), not from the Jacobian.
4. **Open issues.**
   - The plain sum on partial pooling has an offset (high r, poor RMSE).
   - GM trained networks fail under every rule (network-limited).
   - Untested: `hierarchical` / `arruda_direct` beyond n = 8 on IK, and the
     30-dataset partial-pooling runs for the `global`/`hierarchical` arms.

## Matched rerun, C2 `none` vs C2 + `full` (2026-10-01)

Fresh inference for both networks under one protocol: 30 datasets, 100 subjects,
100 draws, 50 DPM2 steps, seed 7, `pp.py --tag rerun`. Scored by `score_rerun.py`
(std-RMSE averaged over datasets; Arruda mean r over the 7 globals).

| network | rule | global std-RMSE | local std-RMSE | mean r |
|---|---|---:|---:|---:|
| C2 published files | gauss_hierarchical | 0.718 | 0.339 | 0.843 |
| C2 `none` | gauss_hierarchical | 0.751 | 0.338 | 0.839 |
| C2 `none` | gauss_jacobian_FD (refresh 20 below lambda 0.3) | 1.088 | 0.338 | 0.820 |
| C2 `none` | gauss_jacobian_FD_r1 (refresh every step) | 1.093 | 0.338 | 0.820 |
| `full` | gauss_hierarchical | **0.546** | 0.250 | **0.850** |
| `full` | gauss_jacobian_FD | 0.709 | **0.247** | 0.840 |
| `full` | gauss_jacobian_FD_r1 | 0.732 | **0.247** | 0.838 |

- The dense refresh changes nothing; the lagged FD schedule is enough here.
- `full` attention: global RMSE -27 % (hier) / -35 % (FD); local -26 %; mean r +0.01 to +0.02.
- r is limited by globals 4 and 6 (about 0.6), which attention does not lift; global 4 drops from 0.71 to 0.64.

### Hierarchical attention (g->g, l->l, l->g; g never -> l) + gauss_jacobian_FD (2026-10-01)

Same protocol and seed as the matched rerun, 30 datasets.

| network | rule | global std-RMSE | local std-RMSE | mean r | per-global r |
|---|---|---:|---:|---:|---|
| C2 `none` | gauss_jacobian_FD (churn 2) | 1.088 | 0.338 | 0.820 | 0.90 0.95 0.98 0.62 0.75 0.59 0.95 |
| `hierarchical` | gauss_jacobian_FD (churn 2) | 0.628 | 0.254 | **0.869** | 0.95 0.93 0.99 0.51 0.81 0.92 0.98 |
| `hierarchical` | gauss_jacobian_FD churn 4 | 0.684 | 0.256 | 0.859 | 0.95 0.90 0.98 0.53 0.82 0.86 0.98 |
| `full` | gauss_hierarchical | 0.546 | 0.250 | 0.850 | 0.97 0.96 0.98 0.64 0.80 0.63 0.97 |

- Hierarchical + FD reaches the paper's 0.870 on these 30 datasets, mainly by lifting global 6 from 0.59 to 0.92; global 4 falls from 0.62 to 0.51.
- Churn 4 is slightly worse than the default churn 2 on every global metric.
- 30-dataset r moves by ~0.01 between datasets subsets (C2: 0.842 -> 0.830 from 30 to 100); confirm at 100 datasets before quoting 0.869 as a result.

### Observation sweep, `hierarchical` + gauss_jacobian_FD, churn 2 vs 4 (2026-10-02)

n = subjects composed per dataset (each a 30-trial set summary); the test set has 100, so the
sweep runs n = 10, 25, 50, 100 on the same 30 datasets and seed. Figures:
`experiments_arruda_schur/artifacts/figures/12_nsweep_recovery_*.png`, `13_nsweep_metrics.png`
(`plot_nsweep.py`).

| rule | n | global std-RMSE | local std-RMSE | mean r | 90 % coverage |
|---|---:|---:|---:|---:|---:|
| churn 2 | 10 | 0.627 | 0.278 | 0.785 | 0.857 |
| churn 2 | 25 | **0.510** | 0.253 | 0.861 | 0.867 |
| churn 2 | 50 | 0.565 | 0.258 | 0.856 | 0.871 |
| churn 2 | 100 | 0.628 | 0.254 | **0.869** | 0.833 |
| churn 4 | 10 | 0.625 | 0.276 | 0.786 | 0.848 |
| churn 4 | 25 | 0.542 | 0.252 | 0.849 | 0.867 |
| churn 4 | 50 | 0.636 | 0.262 | 0.853 | 0.838 |
| churn 4 | 100 | 0.684 | 0.256 | 0.859 | 0.800 |

- Global RMSE is lowest at n = 25 and *rises* with more subjects; coverage falls at n = 100.
- The cause is `log_sigma_nu`: its r falls 0.64 -> 0.51 as n grows. In the recovery panels a
  growing number of datasets with true log_sigma_nu in [-3, -1] collapse to a posterior median
  near -5.5 (sigma_nu -> 0, full pooling of nu) with intervals spanning the whole range; the
  others are recovered tightly. Collapsed datasets (median < -4.5, truth > -3.5): churn 2
  0, 0, 3, 5 of 30 at n = 10, 25, 50, 100; churn 4 0, 1, 4, 5. The count grows with n, so it is a composition
  failure, not lack of data.
- Every other global improves or holds with n (mu_log_alpha 0.70 -> 0.93, log_sigma_log_t0
  0.72 -> 0.92, log_sigma_log_alpha 0.68 -> 0.81).
- Churn 4 is never better than churn 2 beyond n = 10, and is worse at n >= 50.

### log_sigma_nu collapse: diagnosis (2026-10-02, from the saved posteriors, no new runs)

- The 5 datasets that collapse at n=100 (ids 2, 7, 10, 13, 20) are those where
  the true sigma_nu (sd of nu_j 0.07-0.39) is at or below one subject's own nu
  uncertainty from 30 trials (posterior sd 0.12-0.27). These are the datasets where sigma is only weakly identified.
- The collapse is a second mode, not a shift: 58-94 % of the draws sit at about -5.5 and
  the rest stay near the truth. Locals then shrink to mu (sd of nu_j medians 0.01-0.02).
- -5.5 is 4.5 prior sd below the prior N(-1, 1). Even a flat likelihood below
  about -2.5 would put the exact posterior median near -3, so -5.5 is an artefact, and
  the network never trained there.
- In `hierarchical` mode the global score does not see the locals, so this is
  not the Neal-funnel feedback. Composed global score = sum of n per-subject
  marginal scores - (n-1) prior: per-subject errors scale with n, the prior's
  restoring pull scales with 1. At log_sigma = -5.5 the pull is +4.5, so a per-subject
  bias of only -0.045 balances it at n=100, whereas n=10 would need -0.45.
  This is inferred, not yet measured. Test: evaluate the composed log_sigma_nu score along a line for
  dataset 2 at n=10 vs 100.
- At n=100 against the paper: log_sigma_nu r 0.51 vs 0.918, log_sigma_log_alpha 0.81 vs 0.804,
  log_sigma_log_t0 0.92 vs 0.699, the three means + beta_raw average 0.96 vs about 0.92 implied.
  Lifting log_sigma_nu alone to 0.74 would give a mean r of 0.902, and to 0.918 would give 0.927.

### Setup audit against upstream case_study4 (rev 363186e, 2026-10-02)

The DDM benchmark and the 0.870 come from Arruda et al.'s tutorial review
(bayesflow-org/diffusion-experiments case_study4). They are not from the ICLR compositional
paper, which has no DDM experiment. Upstream recovery figure copied to
`docs/figures/upstream_cs4_partial_pooling_global_recovery.png`. Its per-global r are mu_nu 0.874,
mu_log_alpha 0.891, mu_log_t0 0.939, log_sigma_nu 0.918, log_sigma_log_alpha 0.804,
log_sigma_log_t0 0.699, beta 0.967 (beta scale), mean 0.870.

- Hyperpriors: identical to ours (all 7 means/sds match `simulators/priors.py`).
- Training data: 1 subject x 30 trials per sim (upstream 32,768 sims x 500 epochs; ours 100k). Same.
- Test protocol: 100 datasets x 100 subjects x 30 trials, 100 draws. Same, but our recent arms used 30 datasets only.
- Simulator: upstream counts t0 inside max_time and outputs (choice, rt); ours adds a censor flag and log RT. Negligible.
- Network: upstream trains a global-only posterior p(g | x_j) (SetTransformer summary 16-d) and a separate
  local consistency model p(l | g, x_j), sampled ancestrally. Ours is a joint p(g, l | x).
- **Composition differs substantially.** Upstream uses the error-damped mini-batch estimator:
  - 3 random subjects per step (scaled by 100/3);
  - adaptive two-step solver, up to 1000 steps;
  - `compositional_bridge_d0=0.1`, i.e. the whole composed score is multiplied by
    d(t) = 0.1^(1-t). At t -> 0 it is x0.1, which roughly tempers the posterior to the power 0.1.
- Upstream log_sigma_nu has the same small-sigma floor: truth < -2 is estimated at about -1.3 to -1.6,
  worse than ours (-1.9). Its r comes from a wider truth range (-3.4..2.1 over 100 datasets vs our
  -3.9..0.2 over 30) and from the absence of collapses. That absence is plausibly due to the damping, but this is untested.
