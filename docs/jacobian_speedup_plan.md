# Making `gauss_jacobian_FD` cheaper (and not less accurate)

Started 2026-10-08. Goal: bring `gauss_jacobian_FD` (JAC) to or below the cost of
`gauss_hierarchical` (GH) without losing accuracy. Results are appended to
`docs/jacobian_speedup_results.md`; code lives in `src/compass/MultiObsSampler.py`
(opt-in switches, defaults unchanged until a step is validated) and the harness in
`jacobian_speedup/`.

## Starting point (measured)

- Partial pooling, 30 datasets, 100 subjects, 100 draws, 50 steps, per dataset:
  GH 79-103 s (including its pilot of 256 draws x 50 steps per subject), JAC 173-218 s.
  So JAC is ~2.2x GH.
- JAC's extra cost is the finite-difference Jacobian: D+1 copies of every row
  (D = 10 on partial pooling), refreshed every evaluation above lambda 0.3 (~2/3 of the
  log-spaced levels) and every 10th below.
- **Already in the library:** the refresh cache stores the lambda-free information
  `K_j = Lambda_j(t) - lambda^-2 I`, which equals `Sigma_0,j^-1` for a Gaussian
  posterior at every lambda. So "transport the cached Jacobian to the new noise level"
  (idea 1 as first proposed) is already done; staleness costs nothing for Gaussian
  posteriors. Whatever refresh lag costs comes from non-Gaussianity: Cov(theta_0 | y)
  changes with the state y.

## Step 1: refresh only where the Jacobian carries information that changes

With `r = lambda^2 * kappa` (kappa = information scale of the cached `K_j`, i.e. how
narrow a single-subject posterior is relative to the noise):

- `r << 1` (low noise): `Lambda_j ~ lambda^-2 I` for every j, the weights are equal
  and the rule is the plain sum. A stale K changes nothing.
- `r >> 1` (high noise, lambda far wider than the subject posterior): Cov(theta_0 | y)
  tends to the posterior covariance, independent of y, with corrections O(width/lambda).
  A stale K is again nearly exact.
- Only the band in between needs frequent refreshes.

Implementation: a three-band refresh schedule, periods (P_high, P_band, P_low), band
edges either fixed in lambda (generalising today's 0.3 switch) or adaptive in r.
Plus row-evaluation counters in the sampler so cost is measured in network rows, not
wall-clock on a shared GPU.

Test: sweep the periods and band edges; compare with refresh-every-call JAC.

## Step 2: start the reverse diffusion from a Laplace approximation at lambda_s

Above some lambda_s the diffused composed posterior p_lambda(theta | X) is close to
Gaussian. Run the existing arrow-structured Newton ascent (`newton_map_estimate`,
curvature from the same FD Jacobian) down to lambda_s only, giving the mode m and
the composed curvature blocks at that level; draw
`g ~ N(m_g, S^-1)`, `l_j | g ~ N(m_lj - H_ll^-1 H_lg (g - m_g), H_ll^-1)`, and run the
ordinary sampler from lambda_s down. The levels above lambda_s are never evaluated for
the draws. The Newton ascent touches one state, not the draws, so its cost is ~1/draws
of a level.

Risk: a posterior that is multimodal at lambda_s loses its minor modes. Test the
lambda_s at which this starts on the bimodal benchmarks.

## Step 3: secant (BFGS) updates of the cached information between refreshes

For a Gaussian subject posterior the Tweedie denoiser is `D(y) = y + lambda^2 s(y) =
mu - C s(y)` at every lambda (C = Sigma_0,j). Two evaluations of the same row at
(y1, lambda1) and (y2, lambda2) therefore satisfy the noise-level-free secant equation

    K v = u,   u = s2 - s1,   v = D1 - D2,   K = C^-1 (the cached information).

The sampler already makes two evaluations per DPM2 step, and churn moves the state
in random directions, so each step yields secant pairs per row at no extra network
cost. A BFGS update of K keeps it symmetric positive definite whenever `u.v > 0`
(skip otherwise). Between full FD refreshes the information then tracks the state.

Test: refresh periods with and without secant updates.

## Test beds and grading

1. Exact-score oracles (no network error), grading against the exact posterior
   and against refresh-every-call JAC: the lean toy (asymmetric bimodal |g|), the abs
   toy (symmetric bimodal), and the multivariate linear-Gaussian hierarchy from
   `experimental/oracles.py` (where all three steps must be exact).
2. Trained networks with exact references: lean (hierarchical attention), IK reach
   (sym/shift).
3. Partial pooling, `hierarchical_Rg_wide1.5` (D = 10, n = 100, 30 datasets), against
   the stored full-refresh run (mean r 0.928) and a reseeded copy of it as the noise floor.

Accuracy targets: within the Monte-Carlo noise floor of refresh-every-call JAC
(reseed spread), never "zero difference". Cost: network rows evaluated (sampler
counters), plus wall-clock for GH and JAC in the same session on the same GPU.
