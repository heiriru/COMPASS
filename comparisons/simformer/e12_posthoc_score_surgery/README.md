# E12 — Post-hoc prior and likelihood modification

**Paper:** A1.3 and A1.4, Fig. A3.

## What they did

Because the model represents the joint, Bayes' rule decomposes the posterior
score into a prior term and a likelihood term:

    ∇ log p_t(θ_t | x_t) = ∇ log p_t(θ_t) + ∇ log p_t(x_t | θ_t)

and the second term can be recovered from the model itself as
`s_φ(θ_t, t | x_t) − s_φ(θ_t, t)`. A new configuration is then an affine
rescaling of the two pieces:

    α₁·(s_φ(θ_t,t) + β₁)  +  α₂·(s_φ(θ_t,t|x_t) − s_φ(θ_t,t) + β₂)

α tempers, β shifts. For Gaussians the mapping is exact:
`α = σ₀²/σ²`, `β = (μ − μ₀)/σ₀²`. They demonstrate on a 1-D toy
(θ ~ N(0,3²), x₁ ~ N(2 sin θ, 0.5²), x₂ ~ N(0.1 θ², 0.5|x₁|)) and are honest
about the limits — increasing the prior variance works less well than
decreasing it.

## Why this one is worth doing

It is cheap, it needs no new library feature (two score evaluations and an
affine combination), and it is directly relevant to COMPASS's own concerns.
COMPASS already reasons about prior scores explicitly in the compositional
samplers — `MultiObsSampler._diffused_gaussian_prior_score`,
`_prior_precision_at`, the `prior=`/`local_prior=` arguments — so "swap the
prior after training" is a question this codebase asks in a different dialect.

There is also a standing COMPASS result that bears on it directly:
`docs/score_calibration.md` found the learned score shrinks toward the
conditional mean below λ ≈ 0.2, that this is a capacity effect, and that **affine
recalibration makes it worse**. This experiment is the same shape of
intervention. Expect the paper's α/β surgery to inherit that failure at small
noise scales, and design the sweep to show *where* it breaks rather than
reporting a single number.

## To build

1. `toy.py` — the A1.4 toy, with its exact joint density (all three factors are
   Gaussian conditionals, so the ground truth is available by quadrature on a
   grid).
2. `run.py` — sweep α, β for prior and likelihood separately; compare against the
   exact modified posterior; plot as Fig. A3 does (exact vs approximate overlay).
3. Extend to one benchmark task from e01 with a shifted prior, which is the case
   anyone would actually use.

## Watch out for

- The identity `∇ log p_t(x_t|θ_t) ≈ s_φ(·|x_t) − s_φ(·)` needs both an
  unconditional and a conditional score from the same network. Under COMPASS's
  key-padding convention, the unconditional (all-latent) mask leaves the
  attention with **no keys**; it works only because of `add_bias_kv`. Check that
  the all-latent score is actually meaningful for the checkpoint being used
  before building anything on this difference — if the arm was not trained with
  the joint mask in its menu, it will not be.
