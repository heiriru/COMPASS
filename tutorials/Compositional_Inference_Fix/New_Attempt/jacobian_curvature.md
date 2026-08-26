# Pilot-free hierarchical GAUSS: `correction="gauss_jacobian"`

Successor to `derivation.md`, which derived the arrow/Schur elimination behind
`correction="gauss_hierarchical"`. That elimination is kept unchanged here. What
changes is where the per-observation weighting matrix `Λ_j(t)` comes from.

## 1. What the pilot covariance actually costs

Every covariance-aware rule in `MultiObsSampler` weights observation `j`'s score
by the inverse of its **backward-kernel covariance**

```
Σ_{t,j} = Cov(θ₀ | θ_t, x_j)
```

and every one of them builds it as `Σ_{t,j} = (Σ_{0,j}⁻¹ + λ_t⁻² I)⁻¹` from a
single constant `Σ_{0,j}` estimated once at `t = 0` by
`estimate_posterior_moments`. Two separate problems follow.

**It is exact only for Gaussian single-observation posteriors.** The step from
`Σ_{0,j}` to `Σ_{t,j}` is the conjugate-Gaussian update; it presumes
`p(θ₀ | x_j) = N(·, Σ_{0,j})`. When that fails, the weights are wrong at every
`t`, and the error is multiplied by `n`. This is visible in the repo's own
Figure 2 sweep: on the GMM task with an *exact* score, both `gauss` and
`full_gaussian` plateau around 0.06–0.19 sliced Wasserstein while Langevin
reaches 0.006.

**It costs a second full reverse-diffusion pass.** `precision_est_samples`
draws per subject, before inference starts, plus five hyperparameters
(`precision_est_samples`, `precision_est_timesteps`, `precision_est_batch_size`,
`covariance_shrinkage`, `covariance_nugget`) with no principled setting.

## 2. Tweedie gives `Σ_{t,j}` in closed form

Work in the sampler's rescaled coordinates `y = x / α(t)`, where every supported
SDE is variance-exploding with variance `λ_t²`, so

```
p_t(y | x_j) = ∫ N(y; θ₀, λ_t² I) p(θ₀ | x_j) dθ₀.
```

Differentiating under the integral twice gives Tweedie's first- and
second-order identities, both **exact for any** `p(θ₀ | x_j)`:

```
s_j(y)   = ∇ log p_t(y | x_j) = ( E[θ₀ | y, x_j] − y ) / λ_t²
∇s_j(y)  = ∇² log p_t(y | x_j) = ( Cov(θ₀ | y, x_j) − λ_t² I ) / λ_t⁴
```

Rearranging the second one is the whole method:

```
Σ_{t,j}(y) = λ_t² ( I + λ_t² ∇s_j(y) ) = λ_t² ( I − λ_t² H_j(y) ),   H_j = −∇s_j
Λ_j(t, y)  = Σ_{t,j}(y)⁻¹
```

The weighting matrix GAUSS needs is a property of the score network the sampler
is *already evaluating*, read off its Jacobian at the current state. No pilot
run, no Gaussian assumption, no constant-in-`t` approximation.

### It strictly generalizes the pilot form

Suppose `p(θ₀ | x_j)` really is `N(μ_j, Σ_{0,j})`. Then
`p_t(· | x_j) = N(μ_j, Σ_{0,j} + λ_t² I)`, so
`∇s_j = −(Σ_{0,j} + λ_t² I)⁻¹` and

```
Σ_{t,j} = λ² ( I − λ² (Σ₀ + λ²I)⁻¹ ) = λ² Σ₀ (Σ₀ + λ² I)⁻¹
Λ_j     = Σ_{t,j}⁻¹ = Σ_{0,j}⁻¹ + λ_t⁻² I
```

— exactly the matrix `_effective_global_factors` builds from the pilot estimate.
So `gauss_jacobian` is not an alternative approximation competing with GAUSS; it
is GAUSS with its one remaining assumption removed. `tests/test_gauss_jacobian_composition.py`
checks both statements numerically: the precision itself
(`test_jacobian_precision_equals_pilot_precision_on_gaussian_score`) and the
full composition through `_compositional_score`
(`test_gauss_jacobian_matches_gauss_hierarchical_given_matching_covariance`).

### Does it actually help?

The reduction above means a Gaussian benchmark **cannot** show a difference —
`gauss_jacobian` ties the pilot rule there by construction, and a tie is the
pass condition. The accuracy question only has an answer where the
single-observation posterior is non-Gaussian.

[`jacobian_validation/`](jacobian_validation/) runs both. On per-observation
Gaussian mixtures with opposite-sign global/local correlation, against a
`gauss_hierarchical` holding the *exact moment-matched* covariance:

| n | method | W1 to exact | mean err / σ | width ratio |
|---|---|---|---|---|
| 4 | gauss_hierarchical | 0.090 | 0.378 | 1.375 |
| 4 | **gauss_jacobian** | **0.057** | **0.241** | **1.157** |
| 16 | gauss_hierarchical | 0.077 | 0.692 | 1.777 |
| 16 | **gauss_jacobian** | **0.055** | **0.472** | **1.358** |

The gap widens with `n`, as a systematic weighting error must once the composed
posterior contracts like `1/sqrt(n)`.

### Everything downstream is unchanged

`Λ_j(t)` is the only input the rest of the pipeline takes. The arrow
elimination of `derivation.md` — marginal global precision
`Λ_j = P_j[gg] − P_j[gl] P_j[ll]⁻¹ P_j[lg]`, composed solve, local
cross-correction `s_l,j + P_j[ll]⁻¹ P_j[lg] (s_g,j − s_composed)` — carries over
verbatim, as do `_project_information`, `_schur_global_precision` and
`_solve_composed_global`. The only code-level difference is that the precisions
now carry a sample axis, since they depend on the state.

## 3. Conditioning

Every `λ`-smoothed density satisfies `0 ≼ H_j ≼ λ⁻² I` exactly (the upper bound
is the pure-noise limit, the lower bound is log-concavity of the Gaussian
smoothing kernel dominating at small `λ`). The eigenvalues of `H_j` are clamped
into `[0, (1 − floor)/λ²]`, which bounds

```
λ² I  ≼  Σ_{t,j}  ≼  (λ²/floor) I        ⟺        λ⁻² I  ≼  Λ_j  ≼  (floor·λ²)⁻¹ I
```

Two consequences worth stating. `Λ_j ≽ λ⁻² I` means no observation's backward
kernel can ever be *wider* than the pure-noise kernel, which is what keeps the
composed precision well posed. And `Λ_j ≼ (floor·λ²)⁻¹ I` stops a locally
degenerate Jacobian from claiming a point mass. This is the same projection
`_map_curvature_blocks` already applies on the Newton path, for the same reason.

Where the network's Jacobian is genuinely indefinite the smoothed
single-observation posterior is genuinely non-log-concave there — multimodality,
typically at small `t`. The clamp then reports "no information beyond the noise
floor in that direction", which is the honest answer, and
`covariance_diagnostics` still tracks how often it fires via
`negative_information_fraction`.

## 4. Cost

Per refresh: `D = len(hierarchy) + len(local latents)` forward-mode JVPs, where
`D` is the size of the *latent block* — single digits — independent of the
observation count `n` and of `nodes_size`. Observation rows do not mix inside
the network, so one tangent that perturbs coordinate `k` in every row returns
column `k` of *every* observation's Jacobian in one pass (`_row_score_jacobian`).

`jacobian_refresh` amortizes further: curvature varies far more slowly in `t`
than the score does, so refreshing every 5–20 composed-score evaluations costs
a few percent. Against that, the pilot run it replaces is a complete extra
reverse-diffusion pass.

### What a refresh may and may not lag

`Λ_j(t)` must **never** be cached wholesale. It contains an exact `λ_t⁻² I`
term, and the composed precision subtracts `(n−1)` copies of
`Λ_prior(t) = Σ_prior⁻¹ + λ_t⁻² I` built at the *current* time. Reuse a `Λ_j`
from an earlier `t` and those two `λ⁻²` terms no longer cancel; since both
diverge as `λ` shrinks, the composed precision goes indefinite for `n > 1` and
the solve blows up — partway down a real schedule, not at any single time, which
is why a single-time unit test does not catch it.

What is cached instead is the **λ-free information**

```
Σ̂_{0,j}⁻¹ := Λ_j(t) − λ_t⁻² I
```

(PSD by construction, since `Λ_j ≽ λ_t⁻² I`), and `Λ_j(t) = Σ̂_{0,j}⁻¹ + λ_t⁻² I`
is rebuilt exactly at every evaluation. A lagged refresh then lags only the
network's opinion about `Σ_{0,j}` — precisely the quantity the pilot-covariance
rules hold fixed for the *entire* trajectory. So `jacobian_refresh = k` degrades
gracefully towards ordinary GAUSS with a pilot covariance re-derived `k` times
along the way, and on a genuinely Gaussian problem it is exact for every `k`
(`tests/test_gauss_jacobian_composition.py::test_jacobian_refresh_tracks_the_unlagged_composition`).

Note on backends: the fused scaled-dot-product-attention kernels have no
forward-mode AD rule, so `_row_score_jacobian` routes attention through the math
backend for the tangent passes only (`_forward_ad_attention`). Ordinary score
evaluation keeps the fused kernel. This also fixes
`newton_map_estimate(curvature="jacobian")`, which previously raised
`NotImplementedError` on backends that select a fused kernel.

## 5. Pairing with the Newton MAP

`newton_map_estimate(correction="gauss_jacobian", curvature="jacobian")` makes
the whole hierarchical estimate pilot-free: the composition weights and the
Newton curvature both come from the same network Jacobian, and nothing has to
be carried over from a posterior-sampling run. `jacobian_refresh` governs the
composition; `curvature_refresh` governs the Newton step; they are independent.

## 6. Certifying the result

`certify_composition` (on `MultiObsSampler`, exposed as
`ScoreBasedInferenceModel.certify_composition`) evaluates the **exact**
tall-data target at the returned samples. The factorization

```
p(g, l₁..l_n | x₁..x_n)  ∝  p(g)^(1−n) ∏_j p(g, l_j | x_j)
```

(each `l_j`'s own prior appears in exactly one factor, so only the shared prior
is over-counted) turns the target into single-observation conditional densities,
and `PFODE.log_prob` evaluates each without a KDE. So

```
log q(θ) = (1 − n) log p(g) + Σ_j log p(g, l_j | x_j) + const
```

is available at any point, for any `n`, at the cost of `n × num_samples` PF-ODE
evaluations — once, on the returned samples, not per sampler step.

**What it certifies.** The densities come from the same trained network that
produced the samples, so this isolates *composition* error and is blind to the
network's own error. It is the right instrument for choosing between
corrections, tuning `jacobian_refresh`, or deciding whether Langevin correctors
helped — and the wrong one for asking whether the network is trained well
enough. Before this, no measurement in COMPASS could distinguish those two.

The composed sampler's own density is intractable, so weights are
self-normalized against an explicit proposal fitted to the returned samples
(`proposal="gaussian"` for one moment-matched Gaussian over the whole latent
vector, `"factorized"` for a better-conditioned block-diagonal version at large
`n`, or your own `log_proposal`). The ESS measures how far the composed samples
sit from the exact target; `resample=True` returns draws corrected towards it.

## 7. Measuring the curl

`composition_curl` reports `||J − Jᵀ||_F / ||J||_F` for the Jacobian `J` of the
*composed* field on the latent block, by central differences.

A score is a gradient, so its Jacobian is symmetric. A composed score is
assembled from `n` per-observation scores weighted by `n` precision matrices,
and nothing in that construction forces conservativity; the denoised-prediction
clamps break it further. The reverse-diffusion predictor merely integrates the
field, but a **Langevin corrector is an MCMC kernel whose invariant distribution
is defined by the field being a gradient**. Where the antisymmetry is large,
correctors circulate samples instead of refining them.

This is the missing measurement behind the open question in `README.md`:
`gauss_hierarchical` with dense correctors scored 0.732σ on the global parameter
against 0.176σ for the same correction with correctors switched off. Circulation
under a non-conservative field is the mechanism that predicts exactly that
signature, and `composition_curl` measures it directly instead of leaving it to
be inferred from a regression in the final metric.

Finite differences rather than autodiff, deliberately: the composition contains
eigendecompositions and clamps whose derivatives are ill-conditioned or
undefined, while the field itself is perfectly well defined. Cost is `2P`
composed-score evaluations per probed time, with `P` the probed coordinate
count. Cross terms between the locals of *different* observations vanish
identically in both directions, so probing a subset of observation rows loses
nothing.
