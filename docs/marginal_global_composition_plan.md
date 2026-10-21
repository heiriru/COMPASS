# Marginal-global compositional score modelling in the all-in-one transformer

Plan, 2026-09-28. Source paper: Arruda et al., *Compositional amortized inference
for large-scale hierarchical Bayesian models*, ICLR 2026
(`Papers/Compositional_score_dumping.pdf`).

**Goal.** Infer the global parameters `g` by composing per-subject scores the
way Arruda et al. do, using the network we already have, which learns
`p(g,l|x)`, `p(l|g,x)` and the other four conditionals. The model-comparison
path must keep working: `ModelTransfuser.compare`, meaning posterior draws,
then the MAP, then the PF-ODE `log p(x | g, l)`, then AICc/BIC.

---

## 1. The two facts the plan is built on (measured today)

### 1.1 In the deployed trunk, the network's `p(g,l|x)` g-output is already a *marginal* score. It is also mean-field.

`TokenizedTransformer` (`trunk="shared"`, which every Partial_Pooling ladder arm
uses) sets `key_padding_mask = 1 - c`. **Only conditioned tokens are attention
keys**, and the output head is token-wise, so a latent token never sees another
latent token. I checked this on a built model with adaLN woken up, using
the probe that becomes a Phase 1 test. The latent-to-latent Jacobian is
**exactly 0.0** under both `p(g,l|x)` and `p(l|g,x)`. The diagonal and the
`d latent / d x` entries are non-zero.

What follows from that:

- Under mask `x`, the output for `g_k` is a function of `(g_k,t, x_j, t)` only.
  Among such functions, the minimiser of denoising score matching is
  `∂/∂g_k log p_t(g_k,t | x_j)`. That score is **marginal over `l`**, which is
  what Arruda's global network is trained to give, **and** marginal over the
  other `g` coordinates, which Arruda's network is not.
- The legacy `ConditionTransformer` is different: its flattened
  `(N·H, N)` readout couples every token. The move to the tokenized trunk
  therefore also changed *what the network can represent*, in addition to the
  parameter count.

The composed global posteriors we already have show this. On
`ladder_k1/mask6mix`, `gauss_hierarchical`, 100 datasets × 100 draws, the median
of the maximum |off-diagonal correlation| over 21 global pairs is **0.21**
(maximum 0.32). The noise floor for zero correlation at 100 draws is about 0.23.
The pilot covariances (256 draws) are equally diagonal (0.15 against a floor of
about 0.17).

### 1.2 Composing mean-field factors is biased in a way we can predict, and the bias is present in our results

The composition identity `p(g|x_1:J) ∝ p(g)^(1-J) ∏_j p(g|x_j)` needs the
**joint** per-subject factor. If you substitute `∏_k p(g_k|x_j)`, the
per-subject factor for a scale parameter `σ_k` becomes
`p(σ_k) ∫ N(l_jk; μ_k, σ_k²) p(μ_k) dμ_k`. That factor measures how far subject
j sits from the **prior mean of μ_k**, when it should measure how far it sits
from the other subjects. The prediction is that the composed `log σ_k` is
inflated by roughly `((l̄_k − m0_k)/s0_k)²`. The true posterior, with 100
subjects, does not depend on that quantity.

I checked this on the same 100 saved datasets, with no reference posterior. The
table gives the partial Spearman correlation of the log-sigma error with
`((l̄ − m0)/s0)²`, controlling for the true `log σ` and the true `μ`:

| parameter | partial ρ | p | OLS slope per unit z² |
|---|---:|---:|---:|
| `log_sigma_nu` | +0.373 | 1.3e-4 | +0.094 |
| `log_sigma_log_alpha` | +0.248 | 1.3e-2 | +0.148 |
| `log_sigma_log_t0` | +0.872 | 3.8e-32 | +0.184 |

The sign is the predicted one and the effect is large. This is one arm under one
rule; Phase 0 replicates it. It is also consistent with the grouped `gK20` arm
doing better (0.934 against 0.801 for K=1): a factor that holds 20 subjects sees
their dispersion jointly. I state this as consistency only, not as proof.

**This changes what should be taken from the paper.** Their damping, schedule
and mini-batch tricks all *stabilise the sum*. The part that makes their
estimator *correct* is that each summand is a **joint marginal global score**,
and locals are then sampled ancestrally. Our network already has the ancestral
half (`p(l|g,x)`). It has half of the marginal half (l is integrated out) and is
missing the joint half (the g coordinates are independent).

---

## 2. Arruda ingredient by ingredient against COMPASS today

| Ingredient | Arruda et al. | COMPASS now | Action |
|---|---|---|---|
| Per-group factor | joint marginal `s(η_t, Y_j)` from one MLP over all of η | coordinate-wise marginal (§1.1) | **fix in the architecture (§3.1)** |
| What gets composed | η only; locals by ancestral `θ_j ~ q(θ|η, Y_j)` | joint state `(g, l_1..l_J)` with arrow elimination | **new sampler mode (§3.2)** |
| Prior term | `(1−J)(1−t)∇log p(η_t)` | `_diffused_gaussian_prior_score` exists; the global prior is exactly Gaussian `N(M,S)`, so the diffused prior score is closed form | reuse. Must be time-aware and diffused (measured: BayesFlow's clean prior at the noisy point was a major error source) |
| Initial noise | `N(0, I)`, not `N(0, I/J)` (post-publication note) | VE initialisation at σ_max | check `_initial_sample`; never shrink by 1/J |
| Solver | adaptive second-order SDE (Jolicoeur-Martineau), abs 0.002576, rel 0.1 | `method="adaptive"` already exists with the same defaults | reuse |
| Damping `d(t)` | exponential, d0 ≈ 1, d1 ~ 1/√J; d1 < 1 helps calibration | `damping_at_data/at_noise`, five schedules, default 1/√n | reuse and tune |
| Mini-batch (Eq. 8, unbiased) | M ≈ 10% of J | implemented but **refused when locals exist** ("supported only when every latent coordinate is shared") | unlocked for free once the composed state is g only |
| Inference-time schedule shift | cosine shift s, less time at high noise | nothing for the VESDE | add a `sigma_start` / time-grid warp knob |
| Parameterisation | v-prediction, SNR conditioning, standardised data | ε-like output scaling, standardised | optional arm; see the residual-parameterisation memory |
| Hyperparameters | tuned amortised on simulated data, criterion RMSE + calibration error | none | Phase 3, on **validation** datasets and never the 100 test datasets |
| Regime | unimodal targets (shrinkage) | IK and GM are multimodal, and there Arruda lost to our rules; **partial pooling is unimodal** | partial pooling is the right benchmark. Say this scope out loud |

We already measured that damping never helps with exact scores: it stabilises
learned scores and does not make the bridge better. So damping and the shift are
Phase 3 knobs, not the hypothesis.

---

## 3. Design

### 3.1 Role-ordered latent attention (`latent_attention="hierarchical"`)

This is a new option on `TokenizedTransformer`. The default `"none"` is today's
behaviour and must stay bit-identical to the frozen tokenized fixture.

Rule: a token attends to **every conditioned token** and to **latent tokens of
its own role or of a role above it**, in the order `g ≻ l ≻ x`. One exception:
latent items of the same `IID` block do not attend each other.

| Conditional (mask) | g latents see | l latents see | x latents see |
|---|---|---|---|
| `p(g,l|x)` | g, x | g, l, x | – |
| `p(l|g,x)` | – | g, l, x | – |
| `p(g|l,x)` | g, l, x | – | – |
| `p(x|g,l)` | – | – | g, l (**not** other trials) |
| `p(x,l|g)` | – | g, l | g, l |
| `p(x,g|l)` | g, l | – | g, l |

Why the direction matters:

- **g must not see latent l.** If it could, its output would be the joint score
  `∇_g log p_t(g_t, l_t|x)` and no longer the marginal. With the rule above, the
  denoising-score-matching optimum for the g-output under `x` is
  `∇_g log p_t(g_t | x_j)`: joint over g and marginal over l. That is exactly
  Arruda's global score. The l-output becomes `∇_l log p_t(l_t | g_t, x)`, so the
  pair is still a coherent joint score, and single-subject `p(g,l|x)` sampling
  becomes a real joint for the first time.
- **Trials must not see other latent trials.** Trials are exchangeable and
  conditionally independent given θ (`ddm_sde.py`). Today `p(x|g,l)` factorises
  over trials by construction; the rule keeps that, so the PF-ODE likelihood
  used for model comparison keeps its structure.
- Role membership comes from `mask_roles` / `hierarchical_roles`, which the
  six-conditional menu already needs. It is implemented as a per-sample
  `attn_mask` of shape `(B·heads, L, L)`, which is negligible at 40 tokens.
  `GroupedTransformer` is out of scope for v1.

### 3.2 Two-stage ancestral sampler: `correction="marginal_global"`

This goes in `MultiObsSampler` next to the existing rules, so every current
correction keeps working.

1. **Stage G (composed, g only).** The state is `(num_samples, |g|)`. At each
   solver step:
   - evaluate the network with mask `x` for a mini-batch of M subjects. The l
     token values are irrelevant to the g-output by construction (a test
     asserts this);
   - compose `ŝ = d(t)·[(1−J)·s_prior,t(g) + (J/M)·Σ_{j∈batch} s_j(g_t)]`, with
     the exact diffused Gaussian prior;
   - integrate with the adaptive SDE solver (or DPM2 with churn, as a control)
     and the optional σ_start shift.
2. **Stage L (ancestral).** For each g draw, sample `l_j ~ p(l | g, x_j)` with
   mask `gx` through the ordinary single-observation `Sampler`, batched over
   subjects × draws. This is Arruda's Eq. 12, and our network already trains
   this conditional.

It returns the same `(g, l_1..l_J)` layout as today, so `compare`, the MAP
code and the metrics scripts do not change. The composed field is now a sum of
gradients, so `composition_curl` should be about 0. It is a cheap sanity check,
and it is also what makes the global MAP by score ascent legitimate.

The cost per step is M network rows, with no pilot, no JVPs and no arrow solve.
`gauss_jacobian` cost 35× `gauss_hierarchical` on the encoder representation,
and this mode avoids that entirely.

### 3.3 Model comparison is unchanged by construction

`compare(..., multi_obs_inference=True, correction="marginal_global")`:

- **Draws:** from §3.2.
- **Global MAP:** KDE or mean-shift on the Stage G draws (the existing
  `kde_global_then_local_ascent` global stage).
- **Local MAP:** the existing ascent with mask `gx`.
- **`log p(x | g_MAP, l_MAP)`:** the PF-ODE with mask `gl`, the same conditional
  and code path as today.
- **Parameter counts and AICc/BIC:** unchanged.

The only way model comparison could regress is if the new attention degrades
`p(x|g,l)`. Phase 4 tests exactly that.

---

## 4. Experiments, in order of hypothesis, with gates

### Phase 0: confirm the mechanism with no training (about 1 day)

- **0a. Analytic toy, exact scores, no network.** Set up
  `l_j ~ N(μ, σ²)`, `x_j | l_j ~ N(l_j, τ²)`, with μ and log σ on a 2-D grid.
  Compare three things:
  - the exact posterior;
  - the composition of exact **per-coordinate** marginals (what we do now);
  - the composition of exact **joint** marginals `p(μ, log σ | x_j)` (Arruda).

  Then push the joint-marginal field through our adaptive + damped sampler to
  get the **rule floor** that every trained arm is graded against. The
  prediction: the per-coordinate version is biased by about `(x̄ − m0)²`, and the
  joint version reaches the Monte Carlo floor.
- **0b. Replicate §1.2 on other arms.** Use `ladder_k1/logsigma`, `gK20`, the
  encoder-summary C2 arm, and a **legacy** (coupled readout) arm. The prediction
  is that the z² dependence is weaker for gK20 and for legacy. If legacy shows it
  just as strongly, the mechanism is something else, and we stop and re-plan.
- **0c.** Check that `tutorials/Hierarchical_Linear_Quadratic.py --quick` runs
  on the current tree. There are 18 known failing tests elsewhere.

**Gate:** 0a shows that the joint-marginal composition removes the bias, and 0b
replicates the effect on at least two tokenized arms.

### Phase 1: code, CPU-only tests

- **Architecture tests.**
  - The default path is bit-identical to `freeze_tokenized.py`.
  - With adaLN woken (a fresh model is the identity: known trap):
    - `∂g_out/∂l_in = 0` under every mask;
    - `∂g_k/∂g_m ≠ 0` under `x`;
    - `∂l/∂g ≠ 0` under `x`;
    - `∂trial_i/∂trial_k = 0` under `gl`.
  - Checkpoint round-trip; `architecture_payload` records the option.
- **Sampler tests.**
  - With `J = 1`, it reduces to the single-observation sampler.
  - The mini-batch estimator is unbiased (mean over batches equals the full sum).
  - The prior term is the diffused one.
  - On the Gaussian toy with an exact-score stub, it matches the analytic
    posterior to within the MC floor.

### Phase 2: one training run

Retrain the best K=1 recipe unchanged except for `latent_attention`: tokenized
trial, h64 d2 heads 8 mlp 4, `six_mixed` with Bernoulli fraction 0.25,
`log_sigma` time sampling, 100k simulations. Use the **same split,
normalisation constants and test datasets** (`data_signature` preserved), so the
numbers stay comparable with `followup_results.md`.

Train a **second seed of the control** too. The per-conditional validation
losses (`score_conditional_losses.py`, fixed validation masks) are then judged
against seed-to-seed noise and not against zero. Validation noise is shared
across architectures, so compare on identical masks.

### Phase 3: 2×2 factorial on 100 test datasets

The two factors are network {control, hierarchical} and sampler
{`gauss_hierarchical`, `marginal_global`}. The design separates "fix the factor"
from "change the sampler":

- control + `marginal_global` is plain mean-field Arruda and is predicted to be
  biased;
- hierarchical + `gauss_hierarchical` tests whether the old rule benefits from
  a coherent joint score.

**Metrics:** mean r (targets 0.801 now and 0.870 in the paper), the three
`log_sigma` slopes, NRMSE, ECE / cov95, the §1.2 z² diagnostic (predicted to
vanish), wall clock, and `score_network_calls`.

Tune `d1`, `sigma_start` and M on **separate simulated validation datasets**
(Arruda's criterion: RMSE + calibration error), freeze them, then run on test.

### Phase 4: model-comparison regression gate

- Build tokenized versions of the linear and quadratic models in
  `Hierarchical_Linear_Quadratic.py` (the tutorial currently uses the legacy
  `nodes_size` net), with and without hierarchical attention.
- **Pass:** AIC/BIC selection accuracy over (n, c_true) is not worse than the
  control beyond seed noise, and `log p(x|θ)` at fixed θ agrees between the two
  networks, and with the analytic value where one exists.
- Run an end-to-end `ModelTransfuser.compare(..., correction="marginal_global")`
  smoke test.

### Phase 5: scaling, where Arruda's stabilisers should matter

At J = 100, damping, the shift and mini-batching may do nothing. Generate new
test sets with J ∈ {1,000, 10,000} subjects under a new signature. Compare
stability, runtime and the metrics for `marginal_global` against
`gauss_hierarchical`, which cannot mini-batch, so its cost grows with J.

### Phase 6 (optional, only after Phase 4 passes): compositional evidence

Every piece of `log p(x_1:J) = Σ_j log p(x_j) + log ∫ p(g)^(1−J) ∏_j p(g|x_j) dg`
comes from the same network (PF-ODE under `x` and `gl`). That would be a true
Bayes-factor alternative to AICc at the MAP. It is an extension, not a
replacement.

---

## 5. Risks and falsifiers

- **Hierarchical attention hurts `p(x|g,l)`.** Mitigated by the no-trial-to-trial
  rule; Phase 4 is the gate.
- **The z² signal has another cause.** 0a (exact scores) and the legacy arm in
  0b separate these.
- **The factor becomes correct but the per-subject network bias grows with J.**
  In the IK work, a per-observation bias was summed n times. Partial pooling is
  unimodal, which is Arruda's favourable regime, and d1 < 1 partly damps this.
  Grade against the 0a rule floor.
- **The fix is right but the slope does not move,** because a single subject
  carries too little σ information for this capacity. Then `marginal_global`
  composes K-subject factors from the grouped arm (J/K factors) or encoder
  summaries, which combine rather than compete with this plan.
- **VE σ=25 against their VP cosine.** The shift knob is our port of their
  schedule result. Do not assume their s values transfer.
- **Wording.** No "exact likelihood". This method is compositional score
  modelling proper: composed scores inside the reverse SDE, unlike the tilt.

## 6. Files

**New or changed:**
- `src/compass/TokenizedTransformer.py`: attention option.
- `src/compass/MultiObsSampler.py`: `marginal_global` mode.
- `SBIm` / `Sampler` plumbing for the option.
- `Partial_Pooling/config.py`: `latent_attention` in the architecture tag.
- Tests under `tests/` and `Partial_Pooling/tests/`.
- A running record in `docs/marginal_global_progress.md`.

**Untouched:** every existing correction, the checkpoints, the test datasets and
the frozen fixtures.

**Decisions taken without asking** (switch any of them if you disagree):
- partial pooling is the benchmark, not IK or GM;
- v1 excludes `GroupedTransformer`;
- AICc/BIC stays the model-comparison criterion, with evidence as optional
  Phase 6;
- hyperparameters are tuned on validation datasets only.
