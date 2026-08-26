# Validating `correction="gauss_jacobian"`

Evidence for the pilot-free hierarchical GAUSS rule derived in
[`../jacobian_curvature.md`](../jacobian_curvature.md). Two experiments, and
they answer two *different* questions — conflating them is the easy mistake.

Both use an **analytic score**, so the network is perfect by construction and
every deviation from the reference posterior is composition error and nothing
else. That is what makes these decisive about the composition rule rather than
about training.

## Experiment 1 — `gaussian`: the null test

Each observation's joint `p(g, l_j | x_j)` is exactly Gaussian. This is the one
case where the Tweedie construction

```
Σ_t,j = λ²(I + λ²∇s_j)   →   Λ_j = Σ_0,j⁻¹ + λ⁻²I
```

reduces to the pilot form **algebraically**. So `gauss_jacobian` *must* tie
`gauss_hierarchical` here — a difference would mean a bug. **A tie is the pass
condition, not a disappointing result.**

What this experiment does show is the cost side. `gauss_hierarchical` is handed
the **exact analytic covariance** — an oracle it never has in practice, standing
in for a full extra reverse-diffusion pass (`precision_est_samples` draws per
subject) plus five hyperparameters. `gauss_jacobian` is given nothing and
matches it. The `network_calls` panel is the real result.

## Experiment 2 — `nongaussian`: the discriminating test

Each joint is a **two-component Gaussian mixture** with unequal weights (0.7 /
0.3), different component shapes, and **opposite-sign g/l correlation**. No
single covariance matrix describes it, which is exactly the assumption every
pilot-covariance rule makes. This is the only place an accuracy difference can
appear.

The exact tall-data target for the shared coordinate is computable on a grid:

```
p(g | x_1..x_n)  ∝  p(g)^(1-n) ∏_j p(g | x_j)
```

with `p(g | x_j)` the analytic 1-D marginal of observation `j`'s mixture — so
the comparison is against ground truth, not against another sampler.

`gauss_hierarchical` is given the **exact moment-matched covariance** of each
mixture: the best constant Gaussian that exists, strictly better than any pilot
run could estimate. The comparison is deliberately generous to the baseline.

## Experiment 3 — `new_method.py`: global mode first, locals afterwards

A different way to *use* `gauss_jacobian`, not a different composition rule.
Experiments 1 and 2 ask how good the composed shared posterior is; this one asks
what happens if you commit to its mode and solve for the locals conditionally:

1. **Joint draws** — DPM-2 + `correction="gauss_jacobian"`, dense Langevin
   correctors on (`corrector_steps_interval=1`, `corrector_steps=10`,
   `final_corrector_steps=3`, `snr=0.2`), exactly the sampler settings behind
   `02f_dpm2_gauss_global_local.png`.
2. **Marginalize** — `p(g | x)` is the shared column of those draws. Projection
   is exact Monte Carlo marginalization; what the corrector sweeps contribute is
   the MCMC refinement of the draws *being* projected — each sweep is a Langevin
   kernel on the composed bridging density, so the marginal is refined rather
   than read off a pure ODE trajectory.
3. **Mode** — a Gaussian KDE (Scott bandwidth) over the shared draws, evaluated
   on a 4001-point grid and sharpened by a parabolic fit through the peak, so
   the reported `ĝ` is the maximizer of the density estimate rather than of the
   grid it was rendered on.
4. **Locals at fixed `g`** — clamping `g = ĝ` makes `p(l_j | ĝ, x_j)` depend on
   observation `j` alone. **No composition, no correction and no shared
   precision enter stage 4 at all**; it is `n` independent single-observation
   annealed score ascents, run as one batch with the shared coordinate
   *conditioned* instead of latent.

The structural point is that every compositional error mode is confined to
stages 1–3 and collapses onto **one scalar**. Get `ĝ` right and the locals cost
`n` single-observation problems and inherit no composition error at all. The
matching risk is just as sharp: the locals are conditioned on a point estimate,
so they carry no shared-parameter uncertainty, and any bias in `ĝ` propagates
*coherently* into all `n` of them — which is why the figures report the local
error against two references, the exact local posterior mean **and** the exact
conditional mode `argmax_l p(l | ĝ, x_j)`. The second isolates whether the
ascent solved the problem it was given; the first shows what conditioning on a
point estimate cost.

The pipeline runs on three problems:

| `--problem` | Score | References |
|---|---|---|
| `gaussian` | analytic, experiment 1's exactly-Gaussian joints | closed form throughout |
| `nongaussian` | analytic, experiment 2's mixtures | grid quadrature of the exact tall-data target |
| `trained` | the 20K-parameter checkpoint of `../local_std_1_obs_noise_0.02_extremely_small/` | that run's analytic Gaussian truth, reused unchanged from its `reference.npz` |

Each produces the four-panel layout of `02f_dpm2_gauss_global_local.png`
(observations, shared posterior, per-observation locals, local recovery) with a
second dashed vertical line in the shared-posterior panel for the KDE MAP.
`nongaussian` is specified as a set of posteriors and has no data-generating
truth, so its red line is the *exact posterior mode* rather than a true `g`, and
its "observations" panel plots each observation's own `E[g | x_j]` — the closest
thing that problem has to an observational-space coordinate.

## Running

```bash
source .COMPASS/bin/activate
cd tutorials/Compositional_Inference_Fix/New_Attempt/jacobian_validation

python experiments.py --experiment gaussian
python experiments.py --experiment nongaussian
python plot.py

python new_method.py                    # all three problems
python new_method.py --problem gaussian # one of them
python new_method.py --replot           # redraw figures, no resampling
```

`new_method.py` reserves a GPU via `autocvd` only when the `trained` problem is
in the run; the analytic problems stay on CPU. Its metrics CSV is merged by
problem name rather than overwritten, so running one problem does not drop the
other two.

Rows are appended to `artifacts/<experiment>_metrics.csv` **as they complete**,
and re-running skips rows already present — so a killed run keeps its partial
results and resumes. `--seeds`, `--observations`, `--num-samples` and
`--timesteps` are all configurable; the defaults are what the committed CSVs
were produced with.

Both scripts cap themselves to 3 threads before importing torch, per the
repository's `CLAUDE.md` rule.

## Artifacts

| File | Contents |
|---|---|
| `artifacts/gaussian_metrics.csv` | Null-test metrics, one row per (n, method, seed) |
| `artifacts/nongaussian_metrics.csv` | Discriminating-test metrics |
| `artifacts/01_gaussian_reduction.png` / `.pdf` | Null test: mean error, width ratio, network calls |
| `artifacts/02_nongaussian_advantage.png` / `.pdf` | Discriminating test: W1, mean error, width ratio |
| `artifacts/03_new_method_<problem>.png` | Experiment 3, one four-panel figure per problem |
| `artifacts/new_method_<problem>.npz` | Its draws, KDE curve, `ĝ`, local MAPs and exact conditional references |
| `artifacts/new_method_metrics.csv` | Experiment 3 metrics, one row per problem |

The CSVs are the table view for the figures: every plotted number is readable
there, which is what lets the third series carry its own direct label rather
than depending on colour contrast alone.

## Results (seed 0, 3000 draws, 150 steps)

Each experiment is run twice: with a **perfect** analytic score, and with an
imperfect one (`--score-noise 0.05`, a per-observation bias plus a linear warp
applied in denoiser space). The second row is the one that reflects real use.
The first row alone is misleading, because with a perfect score there is very
little for any correction to fix: the unweighted sum `(1-n) s_prior + sum_j s_j`
is *exactly* the score of the target as t -> 0, so `uncorrected` is
asymptotically correct and only suffers from the intermediate-t bridge.

### Non-Gaussian (`02_nongaussian_advantage.png`) -- the decisive one

| eps | n | method | W1 | mean err / sigma | width ratio |
|---|---|---|---|---|---|
| 0 | 4 | uncorrected | 0.106 | 0.124 | 0.510 |
| 0 | 4 | gauss_hierarchical | 0.090 | 0.378 | 1.375 |
| 0 | 4 | **gauss_jacobian** | **0.057** | 0.241 | **1.157** |
| 0 | 16 | uncorrected | **0.047** | **0.145** | 0.496 |
| 0 | 16 | gauss_hierarchical | 0.077 | 0.692 | 1.777 |
| 0 | 16 | **gauss_jacobian** | 0.055 | 0.472 | **1.358** |
| 0.05 | 4 | uncorrected | **0.070** | **0.039** | 0.812 |
| 0.05 | 4 | gauss_hierarchical | 0.120 | 0.505 | 1.709 |
| 0.05 | 4 | **gauss_jacobian** | 0.083 | 0.318 | 1.413 |
| 0.05 | 16 | uncorrected | 0.199 | 1.782 | 0.656 |
| 0.05 | 16 | gauss_hierarchical | 0.125 | 0.943 | 1.541 |
| 0.05 | 16 | **gauss_jacobian** | **0.103** | **0.921** | **1.121** |

Three things to read off this.

**`gauss_jacobian` beats `gauss_hierarchical` in every non-Gaussian cell** --
both noise levels, both observation counts, all three metrics -- against a
baseline holding the *exact* moment-matched covariance. Same algebra, same
sampler, same everything: only the source of `Lambda_j` differs. That is the
claim the method makes, and it holds.

**`uncorrected` is strong with a perfect score and collapses without one.** Its
mean error goes 0.039 (n=4) -> 1.782 (n=16) at eps=0.05: a 45x degradation from
adding observations. That is the n-amplification of a *coherent* score bias --
the error adds across observations while the posterior contracts like
1/sqrt(n), and an unweighted sum has nothing to attenuate it. This is precisely
what the precision-weighted corrections exist to prevent, and it is invisible
in the eps=0 row.

**The Gaussian-approximation barrier.** The exact target here is 31% (n=4) and
44% (n=16) *narrower* than the Gaussian approximation of the composition can
represent (see the diagnostic in the derivation note). So the best a Gaussian
composition rule could do is width ratio 1.31 / 1.44. `gauss_hierarchical`
lands outside that (1.375 / 1.777); `gauss_jacobian` lands inside it
(1.157 / 1.358), and at eps=0.05, n=16 reaches 1.121. Getting *past* the
Gaussian barrier is only possible because the weighting varies with the state.

### Gaussian (`01_gaussian_reduction.png`) -- the null test, and a real cost

With a perfect score the two tie, as they must (0.006/0.021/0.027 vs
0.018/0.021/0.031 mean error at n=2/8/32) -- the baseline holding an exact
analytic covariance that in practice costs a whole extra reverse-diffusion pass.

With an imperfect score **`gauss_jacobian` is slightly worse than the pilot**
(mean error 0.319 vs 0.196 at n=2, 1.765 vs 1.603 at n=8, 0.771 vs 0.603 at
n=32). This is the fragility risk stated up front, and it is real: when the
single-observation posterior genuinely *is* Gaussian, a constant covariance is
the correct answer, a pilot estimate averages network noise away over hundreds
of draws, and differentiating a noisy network instead adds variance for no bias
reduction. `jacobian_refresh > 1` is the knob to test here, since averaging
curvature over a window of steps is the same kind of smoothing a pilot performs
over samples.

### What this means in practice

Use `gauss_jacobian` when the single-observation posteriors are **non-Gaussian**
-- skewed, heavy-tailed, multimodal, or with state-dependent global/local
coupling. That is where its bias reduction outweighs its variance cost, and it
is also the regime where the whole GAUSS family currently plateaus (see the GMM
rows of `Gauss_test/artifacts/figure2/metrics.csv`). When the posteriors are
close to Gaussian, the pilot rules are competitive or slightly better on
accuracy, and the argument for `gauss_jacobian` reduces to cost: no pilot run,
no oracle, five fewer hyperparameters.

### Experiment 3 (`new_method.py`, seed 0, 30 observations, 3000 draws, 100 steps)

| problem | ĝ error vs exact mode | shared width ratio | local error vs exact mean | local error vs exact p(ℓ\|ĝ,x) mode |
|---|---|---|---|---|
| gaussian | **0.05 σ** | 1.013 | **0.02 σ** | 0.005 σ |
| nongaussian | **0.11 σ** | 1.143 | 0.14 σ | 0.005 σ |
| trained | 1.86 σ | 0.934 | 1.78 σ | 0.62 σ |

The two analytic problems say the pipeline works, and the last column says *why*
it works: **conditioned on ĝ, the ascent recovers the exact local mode to 0.005 σ
in both.** Stage 4 contributes essentially nothing to the error budget — as it
should, since with `g` clamped there is no composition left to get wrong. On the
Gaussian problem the locals land 0.02 σ from their exact posterior means, against
the 0.10 σ that `02f_dpm2_gauss_global_local.png` reaches on the analogous
learned problem with an oracle covariance bank.

The interesting number is the non-Gaussian problem's 0.14 σ. The ascent is exact
(0.005 σ), so that residual is entirely **the price of conditioning on a point
estimate rather than integrating over `g`** — visible in the recovery panel as a
systematic tilt on the observations whose mixtures are most skewed, where
`argmax_l p(l | ĝ, x_j)` and `E[l_j | x_1..n]` genuinely differ. This is the
method's defining trade, isolated: it is not sampler error and no amount of
sampling removes it.

The `trained` row is the honest failure case and should not be read as a verdict
on stage 4. The 20K-parameter checkpoint composed with `gauss_jacobian` puts the
shared posterior 1.86 σ high (ĝ = 1.032 against an exact posterior mode of 0.706),
and all 30 locals inherit that single shift *coherently* — the recovery panel is a
line parallel to the diagonal, offset by roughly the shared error. Against the
mode it was actually given, the ascent is still 0.62 σ, which at this run's
conditional σ ≈ 0.02 is 0.012 in absolute terms. So the pipeline faithfully solved
the problem it was handed; what failed upstream was the composed shared posterior
on a very small network at obs. noise 0.02. Note that
`02f_dpm2_gauss_global_local.png` in that same directory is *not* a like-for-like
baseline: it was produced with `Gauss_global_local` handed an oracle
single-observation covariance bank, not with a pilot-free rule.

The two structural properties this experiment measures, independent of any
problem: stage 4 costs `map_timesteps * map_iterations` = 600 network calls
against the sampler's 3804, and it needs no correction, no covariance and no
shared precision at all.

## Reading the figures honestly

- **Tie in experiment 1 = correct.** Read the third panel (network calls) for
  what was actually gained.
- **Experiment 2 is the accuracy claim.** If `gauss_jacobian` does not beat
  `gauss_hierarchical` there, the accuracy argument for it fails and the case
  reduces to cost and the removal of the pilot run.
- **Width ratio has a reference line at 1.0.** Over-wide and under-wide are both
  errors; the bar height alone does not say which.
- **In experiment 3, read the two local error columns together.** The error
  against the exact conditional mode grades the ascent; the error against the
  exact posterior mean grades the whole pipeline. Their difference is the cost
  of conditioning on a point ĝ, and it is the only quantity in the table that is
  a property of the *method* rather than of the sampler or the network.

## Not covered here

Robustness to a *noisy* score. Differentiation amplifies high-frequency error,
so `∇s` is rougher than `s` on a poorly trained network, whereas a pilot
covariance averages hundreds of draws and is smooth. `gauss_jacobian` could
plausibly lose to a well-estimated pilot in that regime. Adding a
score-perturbation arm (as in the ε sweep of `Gauss_test/`) is the natural next
experiment, and `jacobian_refresh` is the knob to test as a smoother, since
averaging curvature over a window is what the pilot does in the sample
dimension.
