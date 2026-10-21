# Arruda-style composition on the all-in-one transformer, via the Schur complement

Code: `src/compass/experimental/{latent_attention,marginal_schur,oracles}.py`.
Tests: `tests/test_experimental_{latent_attention,marginal_schur}.py`.
The production modules are untouched. Everything here is opt-in through
`enable_latent_attention(...)` and `install_marginal_schur(...)`.

## 0. Notation

Each subject `j = 1..J` has data `x_j`, local parameters `l_j ∈ R^L`, and shares
the global parameters `g ∈ R^H`:

    p(g, l_{1:J}, x_{1:J}) = p(g) ∏_j p(l_j | g) p(x_j | l_j, g).

The diffusion is COMPASS's VESDE. We work in the sampler's `y = x/α`
coordinates, where the noise is `N(0, λ_t² I)`. For any density `p`,
`p_t = p * N(0, λ_t² I)`. The network, conditioned on `x_j`, returns the
diffused score of whatever conditional its mask asks for. Under the mask
`p(g, l | x)` we write it as `S_j = (S_g, S_l)`, its Jacobian as
`J_j = ∂S_j/∂(g, l)`, and the curvature as `H = −½(J + Jᵀ)`, with blocks
`H_gg, H_gl, H_ll`.

## 1. Two exact identities, two composition targets

**Joint identity.** From the factorisation,

    p(g, l_{1:J} | x_{1:J}) ∝ p(g)^{1−J} ∏_j p(g, l_j | x_j),                (1)

because `p(g, l_j | x_j) ∝ p(g) p(l_j|g) p(x_j|l_j,g)`. The joint-state rules
(`gauss_hierarchical`, `gauss_jacobian(_FD)`) target (1). Their state is
`(g, l_1..l_J)`, and the g-component of the composed score at `t = 0` is
`(1−J)∇log p(g) + Σ_j S_g(g, l_j)`, the joint score evaluated at each
subject's own carried `l_j`.

**Marginal identity** (Arruda et al., Prop. A.1). Integrating (1) over every `l_j`:

    p(g | x_{1:J}) ∝ p(g)^{1−J} ∏_j p(g | x_j).                             (2)

Arruda et al. target (2) on a state that holds `g` only. Each factor needs the
*marginal* score `∇_g log p(g | x_j)`. The locals are then drawn ancestrally,
`l_j ~ p(l_j | g, x_j)`, which is (1) read as `p(g|x_{1:J}) ∏_j p(l_j|g,x_j)`.

**Diffused bridges.** At noise level `t` the two rules follow
`q_t ∝ p_t(g)^{1−J} ∏_j p_t(g, l_j | x_j)` and
`q_t ∝ p_t(g)^{1−J} ∏_j p_t(g | x_j)`. Neither is the diffusion of the
posterior, but both reduce to (1) and (2) at `t = 0`. The prior factor must be
the **exactly diffused** prior. With a clean prior evaluated at the noisy
point, the bridge is improper at intermediate `t` (our oracle runs), and it was
BayesFlow's largest error source.

## 2. What the transformer can represent: the attention mode decides

**Lemma (a restricted denoiser learns a marginal).** Denoising score matching
fits `f_g(g_t, ·)` to `∇_{g_t} log N(g_t; g_0, λ²)` in L2. If the architecture
makes `f_g` depend on `(g_t, x)` only, the minimiser is
`E[∇_{g_t} log p(g_t|g_0) | g_t, x] = ∇_{g_t} log p_t(g_t | x)`: the score of
the *marginal* over every input `f_g` cannot see. If `f_{g_k}` depends on
`(g_{k,t}, x)` only, the minimiser is `∂_k log p_t(g_{k,t} | x)`.

The production shared trunk attends from every token to *conditioned* tokens
only (`key_padding_mask = 1 − c`), and its head is token-wise. So under
`p(g,l|x)`, `S_{g_k}` depends on `(g_{k,t}, x)` alone. The trained factor is
`∏_k p_t(g_k | x_j)`, which is mean-field. The latent-attention modes change
what the lemma returns:

| mode | `g` attends | `l` attends | `S_g` under `p(g,l|x)` is | `p(g,l|x)` represented as |
|---|---|---|---|---|
| none (production) | – | – | `∂_k log p_t(g_k|x)`, per coordinate | `∏_k p(g_k|x) ∏_m p(l_m|x)` |
| global | g | – | `∇ log p_t(g|x)`, jointly marginal | `p(g|x) ∏_m p(l_m|x)` |
| hierarchical | g | g, l | `∇ log p_t(g|x)`, jointly marginal | `p(g|x) p(l|g,x)` (exact form) |
| full | g, l | g, l | `∇_g log p_t(g, l|x)`, the joint | `p(g,l|x)` (exact form) |

In every mode, items of one IID block never attend each other, so the
likelihood `p(x|g,l)` factorises over exchangeable trials exactly as before.
The tests check each row of this table by the zero pattern of the network
Jacobian.

**Why "none" composes to the wrong posterior.** Composing (2) with
`∏_k p(g_k | x_j)` in place of `p(g | x_j)` targets
`∏_k p(g_k)^{1−J} ∏_j p(g_k|x_j)`, which is not (2). For a
location/scale pair `l_j ~ N(μ, σ²)`, the per-subject σ-factor is
`p(σ) ∫ N(l_j; μ, σ²) p(μ) dμ`. It measures dispersion about the *prior* mean of
μ, not about the other subjects. On the saved partial-pooling posteriors the
log-σ error rises with `((l̄ − m₀)/s₀)²` (partial Spearman +0.37 / +0.25 /
+0.87), as this predicts. The oracle test
`test_mean_field_factors_compose_to_the_wrong_posterior` shows the same failure
with exact per-coordinate scores.

## 3. From the joint score to the marginal score

Needed only for the `full` mode. `global`/`hierarchical` return the marginal
directly (estimator `direct`).

**Fisher identity.** `∇_g log p_t(g | x) = E_{l ~ p_t(l | g, x)}[S_g(g, l)]`.
Differentiate `log ∫ p_t(g, l | x) dl` under the integral.

### 3.1 Schur estimator (Gaussian marginalisation)

Let `log p_t(g, l)` be locally quadratic, with Hessian `−H` and score `S` at
the point `(g, l)`. Integrating the quadratic over `l` gives the Schur
complement `H_gg − H_gl H_ll⁻¹ H_lg` as the marginal precision, and the
marginal score

    ŝ_schur = S_g − (∂S_g/∂l) (∂S_l/∂l)⁻¹ S_l = S_g + A H_ll⁻¹ S_l,       A = ∂S_g/∂l.   (3)

(3) is Newton's step to the conditional mode `l* = l + H_ll⁻¹ S_l`, with
`S_g` then carried along it to first order. **It is exact for every `l` when
`p_t(g, l | x)` is jointly Gaussian**, because `S` is then affine and the
marginal score is independent of `l`
(`test_marginal_is_exact_for_any_local_state_on_a_gaussian`).

### 3.2 Where Schur fails: `g`-dependent local precision

If `l | g` is Gaussian with a precision `C(g)` that depends on `g`, then
`S_g` is quadratic in `l`, and

    ∇_g log p_t(g | x) = S_g(g, l*) − ½ ∇_g log det H_ll(g, l*).            (4)

This is the Laplace formula, and it is exact in this class. At the mode,
`S_l = 0`, so (3) returns only the first term. The dropped term is the
`l`-volume, and **it has a definite sign for a scale parameter**. For
`l ~ N(μ, e^{2s})`, `x ~ N(l, v)`, we get `H_ll = e^{−2s} + 1/v`, and the
missing piece of `∂_s log p(μ, s | x)` is `+v/(σ² + v)` per subject. Summed
over `J` subjects it is a bias of `J·v/(σ²+v)` toward small σ, which does not
shrink relative to the evidence as `J` grows. Our log-σ gap is exactly the
kind of quantity this affects. `test_schur_without_log_det_is_biased_...`
checks it to 2e-3.

A "control variate" reading of (3), evaluated at a conditional *sample*
instead of the mode, does not rescue it. The gain `A H_ll⁻¹` varies with the
sample and correlates with `S_l`. In the example the result is biased by
`−2 e^{−2s}/H_ll`. The unbiased Monte Carlo version is plain `S_g` at a
conditional sample (Fisher), with no gain.

### 3.3 Laplace estimator (the default for `full`)

1. Newton to the conditional mode: `l ← l + H_ll⁻¹ S_l`. This uses a
   finite-difference Jacobian at the current `(g, l)` and one batched network
   call. The mode is warm-started from the previous solver step (`g` moves by
   `O(Δt)`), so one step per evaluation suffices after `cold_newton_steps` at
   the start.
2. `ŝ = S_g(l) + A (l* − l) − ½ ∇_g log det H_ll`, where `∇_g log det` comes
   from forward differences in each global coordinate (`H` extra local
   Jacobians, refreshed on the `jacobian_refresh` schedule).

This is exact when `l | g` is Gaussian: the joint-Gaussian case, and the
`g`-dependent-precision case (`test_laplace_log_det_term_recovers_...`).
Partial pooling is nearly in that class at low noise. At high noise every
diffused conditional is close to `N(·, λ²)` and hence Gaussian. `H_ll` is
clamped into `[1/((κ s_l)² + λ²), (1−floor)/λ²]`. The upper end is Tweedie's
exact bound. The lower end says a conditional is at most `κ = 10` times wider
than the diffused local prior, which prevents division by a degenerate
curvature.

## 4. Composition on the marginal state

With per-subject marginal scores `ŝ_j(g_t)`:

**Arruda sum** (`correction="arruda_schur"`):

    s(g_t) = d(t) [ (1 − J) w(t) ∇ log p_t(g_t) + Σ_j ŝ_j(g_t) ].           (5)

Here `w = 1` is the diffused-prior bridge of §1 and `w = 1−t` is Geffner's
bridge. Damping `d(t)` is Arruda's stabiliser. It is off by default, because
with exact scores damping never improved the answer in our oracle runs; it
only helps learned scores that would otherwise diverge.

**Precision-weighted** (`correction="arruda_schur_gauss"`). Linhart et al.'s
GAUSS composition applied to the marginal factors:

    Λ_j = [λ²(I − λ² H^marg_j)]⁻¹,  H^marg = H_gg − H_gl H_ll⁻¹ H_lg,
    s = Λ⁻¹ [ Σ_j Λ_j ŝ_j + (1−J) Λ_prior ∇log p_t ],  Λ = Σ_j Λ_j + (1−J) Λ_prior.   (6)

`Λ_j` is Tweedie's backward precision for the *marginal* factor. Its
curvature is the Schur complement, by §3.1. Each information term
`Λ_j − Λ_prior` is projected onto the PSD cone, the same guard as
`gauss_jacobian`, so `Λ ≥ Λ_prior` for all `t`. (6) targets the diffused
posterior itself rather than the bridge, and is exact for Gaussian factors.

**Locals.** The state's local coordinates follow their own score `S_l` as
auxiliary variables. After the composed pass, each subject's locals are redrawn
from `p(l | g, x_j)` (mask `g, x` conditioned) at every global draw. This is
Arruda's Eq. 12, and our network trains this conditional directly.

## 5. Relation to `gauss_jacobian(_FD)`

`gauss_jacobian_FD` is the joint-state counterpart. It composes `S_g(g, l_j)`
weighted by the Schur-reduced *joint* backward precisions, and it moves each
`l_j` with the arrow-elimination cross-correction. For jointly Gaussian factors,
its `g`-marginal equals (6). Both are exact in that case, which is the oracle
test's `gauss_jacobian_FD` row. Which one the network can support depends on
the attention mode:

- **none:** `S_g` does not depend on `l` and `∂S_l/∂g = 0`. The joint rule
  composes per-coordinate marginals, and its locals never see the composed
  `g`. Neither rule can repair the mean-field factor.
- **global:** the `g` factor is right. The locals are still decoupled from
  `g`, so the joint rule's locals are wrong, and the marginal rule plus
  ancestral locals is the consistent choice.
- **hierarchical / full:** the joint score is coherent, and both rules are
  valid. The marginal rule needs `direct` (hierarchical) or `laplace` (full).

## 6. Cost per composed-score evaluation

`D = H + L` latent coordinates per subject; rows = subjects × samples.

| rule | network row-evaluations |
|---|---|
| direct | 1 |
| schur | 1 + (D + 1) |
| laplace | newton_steps·(D + 1) + H·(L + 1)/refresh (the carried-state call is spent too) |
| gauss_jacobian_FD | 1 + (D + 1)/refresh |

## 7. What is not claimed

- The network is still an approximation. "Exact" above means exact given
  exact scores. The oracle tests test the rules, not a trained network.
- (4) is exact only when `l | g` is Gaussian. Where the local conditional is
  multimodal (inverse kinematics), `laplace` is an approximation, and the
  joint-state rule or `direct` on a `hierarchical` network is the principled
  alternative.
- Observation mini-batching (Arruda's Eq. 8) is not implemented for these
  rules. At `J = 100` it is not needed.
