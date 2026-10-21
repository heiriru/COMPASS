# E10 — Hodgkin–Huxley with interval conditioning — **BLOCKED (small)**

**Paper:** §4.4, Fig. 7; guidance accuracy in Fig. A15/A16; algorithm in §3.4
and Alg. 1; model details in A2.2.

## What they did

Seven-parameter Hodgkin–Huxley simulator (Pospischil et al. 2008
parameterisation, 200 ms, 4 mA injected 50–150 ms), reduced to the Gonçalves et
al. (2020) summary statistics **plus** a metabolic-energy statistic computed from
sodium charge. Infer the posterior from the voltage summaries alone; then
generate the posterior-predictive energy distribution *from the model*; then
re-infer with the energy constrained to its lowest 10% quantile, using guided
diffusion, and show the constrained posterior predictives satisfy the constraint
while the voltage traces still match.

The guidance term is general:
`s(x̂,t | c) ≈ s(x̂,t) + ∇ log σ(−s(t)·c(x̂))` for constraints `c(x̂) ≤ 0`, with
`s(t) → ∞` as `t → 0`. An interval upper bound `u` is just `c(x̂) = x̂ − u`.

## Why this cannot run today

COMPASS's samplers have `cfg_alpha` (classifier-free guidance strength) but no
general constraint-guidance term. Nothing consumes a user-supplied `c(x̂)`.

## Route in — this is the cheapest of the four blocked experiments

The term is additive on the score inside the sampler loop: roughly a 30-line
change in `Sampler._get_score` / `MultiObsSampler._get_score` accepting a
`constraints=` callable and an `s(t)` schedule. It also unlocks e12's post-hoc
prior/likelihood surgery, which is the same mechanism.

Once that exists, the remaining work is a Hodgkin–Huxley simulator and the
summary statistics — substantial, but ordinary.

## Recommendation

Implement the guidance term in the library (it is generally useful and small),
then decide whether the HH simulator is worth the effort or whether the
guidance can be demonstrated on a task already in the repository.
