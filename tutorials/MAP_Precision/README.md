# Arrow-Newton hierarchical MAP — implementation and A/B results

`MultiObsSampler.newton_map_estimate` is a **separate** estimator alongside
`map_estimate`, added so the two can be compared directly. See `PLAN.md` for the
diagnosis and the remaining phases.

## What changed

`map_estimate` is byte-for-byte unchanged in behaviour. Its 240-line validation
and state-preparation block was moved verbatim into `_configure_map_estimate`,
which both estimators now call, so an A/B isolates the ascent step and nothing
else. Both seek the same fixed point `s(z, t) = 0` at each annealing level.

| | `map_estimate` | `newton_map_estimate` |
|---|---|---|
| step | `z + lambda**2 * s` | arrow-Newton solve |
| block coupling | Gauss-Seidel alternation | simultaneous, via Schur elimination |
| curvature | none (implicitly `lambda**-2 I`) | `H_j = -grad(s_j)` by JVP, or `(Sigma_0,j + lambda**2 I)^-1` |
| safeguards | `denoise_clamp` | PSD projection, trust region, merit backtracking |
| smoothing bias | uncorrected | Richardson extrapolation in `lambda**2` |
| default `denoise_clamp` | `5.0` | `None` |

The step solves the arrow system directly, which is the same block elimination
as the score composition in `../Compositional_Inference_Fix/New_Attempt/derivation.md`,
applied to the Newton system instead:

```
S      = sum_j M_j + (1 - n) H_prior,   M_j = H_j[gg] - H_j[gl] H_j[ll]^-1 H_j[lg]
d_g    = S^-1 (s_g - sum_j H_j[gl] H_j[ll]^-1 s_l,j)
d_l,j  = H_j[ll]^-1 (s_l,j - H_j[lg] d_g)
```

`curvature="jacobian"` (the default) obtains `H_j = -grad(s_j)` exactly by
forward-mode AD. Rows are independent inside the network, so one tangent per
latent coordinate returns that column of *every* row's Jacobian in a single
pass: the JVP count is the latent width (single digits), not the observation
count. It needs no pilot covariance, so it works for every `correction`.
`curvature="gaussian"` instead reuses an already-estimated
`posterior_covariance` and requires a correction in
`SCHUR_GAUSSIAN_CORRECTIONS`.

`_project_map_information` mirrors `_project_information`: writing
`H_j = H_prior + I_j` makes `S = H_prior + sum_j I_j` an identity, so projecting
each `I_j` onto the PSD cone makes `S` positive definite for every `n` by
construction. The Newton direction is therefore always an ascent direction, and
no eigenvalue floor is load-bearing for correctness.

## A/B results

`compare_map_methods.py`, 140 runs per method: 2 prior regimes x 2 annealing
policies x 7 observation counts x 5 seeds, `timesteps=100`,
`denoise_clamp=5.0` on **both** arms so the comparison is like-for-like.

The score is an *exact analytic* diffused score and the reference is the
closed-form arrowhead MAP, so there is no network error and no composition
error: every number is optimizer error. Candidates start from the **prior
mean**, not from the answer.

### Overall

| | runs | worst global err /sigma | median global err /sigma | mean network calls | pinned at clamp |
|---|---|---|---|---|---|
| `map_estimate` | 140 | **116.29** | 0.01299 | 8248 | 23 / 140 |
| `newton_map_estimate` | 140 | **0.003** | 0.00005 | **585** | **0 / 140** |

### By observation count (regime `hard`, long anneal)

widths `sigma_global=0.3, sigma_local=0.4, sigma_x=0.2`, matching
`../plot_local_vs_global_joint_map_validation.py`

| n | tweedie global | newton global | tweedie local | newton local | tweedie calls | newton calls | tweedie pinned |
|---|---|---|---|---|---|---|---|
| 2 | 0.0009 | 0.0000 | 0.0047 | 0.0001 | 3847 | 548 | 0/5 |
| 5 | 0.0023 | 0.0000 | 0.0056 | 0.0001 | 4388 | 560 | 0/5 |
| 10 | 0.0088 | 0.0000 | 0.0078 | 0.0001 | 7327 | 559 | 0/5 |
| 25 | 0.0329 | 0.0001 | 0.0091 | 0.0002 | 9999 | 567 | 0/5 |
| 50 | 0.0321 | 0.0001 | 0.0139 | 0.0002 | 9674 | 592 | 0/5 |
| 100 | **29.16** | 0.0000 | 1.4511 | 0.0002 | 9624 | 599 | 3/5 |
| 200 | **45.09** | 0.0001 | 1.4567 | 0.0002 | 9573 | 593 | 4/5 |

Independently reproduces the 43.5 sigma at n=200 recorded in
`../output/annealed_score_ascent/map_cache/exact_score_global_map_metrics.csv`
(45.1 sigma here), including the clamp-pinning mechanism.

**The error is flat in `n` for the new estimator** at ~1e-4 sigma across every
regime, anneal policy and observation count -- which was the acceptance
criterion in `PLAN.md`, not merely a smaller number. The old estimator is
acceptable to n≈50 and then diverges.

**Cost falls with accuracy, not against it**: ~585 network calls versus ~8248, a
14x reduction. The old loop never converges (0.4-1% contraction per iteration
against `convergence_tol=1e-6`) so it exhausts its 50-iteration budget at every
level; the Newton loop converges at every level and exits.

## Two findings from the test work

**The existing accuracy tests are near-vacuous.** `tests/test_hierarchical_map.py`
initializes at `expected`, the analytic MAP (`:127` and the other accuracy
tests). They verify the iteration does not walk *away* from the mode, never that
it can find it -- which is why they pass at n=50 while the tutorial records
0.73 sigma on the same n. Every test in `tests/test_newton_map.py` starts from
the prior mean.

**`denoise_clamp` is load-bearing for the old estimator.** With
`denoise_clamp=None`, `map_estimate` raises `"The stabilized uncorrected score
became non-finite"` at n=25. `newton_map_estimate` reaches 1e-4 sigma without
it, because the trust region and the positive-definite curvature do that job
instead. Pinned by
`test_tweedie_ascent_depends_on_the_denoise_clamp_but_newton_does_not`.

## Reproducing

```bash
# A/B sweep (CPU, no trained model, ~25 min)
.COMPASS/bin/python tutorials/MAP_Precision/compare_map_methods.py \
    --repeats 5 --timesteps 100
# -> tutorials/MAP_Precision/artifacts/map_method_comparison.csv

# Unit tests (CPU, ~75 s)
.COMPASS/bin/python -m pytest tests/test_newton_map.py tests/test_hierarchical_map.py -q
```

## Unrelated pre-existing bug found

Six tests in `tests/test_damping_composition.py` fail identically on unmodified
`HEAD`: `terminal_corrector_counts` is assigned only inside
`if method in ("dpm", "langevin")` (`MultiObsSampler.sample`) but read
unconditionally in `_sample_loop`, so `method="adaptive"` raises
`AttributeError`. Not touched by this work.

## Not done yet

Phases 2-4 of `PLAN.md`: multistart ranking via
`ModelTransfuser._hierarchical_candidate_scores`, the `Gauss_test` score-error
robustness gate (`FrozenErrorNet` epsilon sweep), the GPU `Partial_Pooling`
end-to-end gate, and the marginal-MAP / Neyman-Scott correction.
