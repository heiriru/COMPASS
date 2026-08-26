# Improving `gauss_hierarchical`, especially on the local parameters

Running log for the "make GAUSS better on non-Gaussian posteriors without
cheating" thread. **Read the "Ruled out" section before proposing anything** --
two obvious-looking fixes are already dead for structural reasons, not for lack
of trying.

Ground rules taken from the request: no oracle knowledge of the target posterior,
no per-problem hand tuning, and every candidate must be graded on the same bench
the churn work used -- the Gaussian twin (`hierarchy_gauss`, where the rules are
exact by construction), the Laplace twin (`hierarchy_laplace`, symmetric and
non-Gaussian) and the exponential original (`hierarchy`, skewed with a hard wall).

## The diagnosis, from the code

`gauss_hierarchical` and `gauss_jacobian` share **every downstream step**: both
are in `SCHUR_GAUSSIAN_CORRECTIONS` and `LOCAL_CROSS_CORRECTIONS`, so both do
arrow elimination, the Schur complement, the information projection, the composed
global score, and the linear cross-correction of each row's local score. The only
difference is where the per-observation backward precision comes from
(`_effective_global_factors`):

    gauss_hierarchical:  Lambda_j(t) = Sigma_0j^-1 + lambda^-2 I        (pilot)
    gauss_jacobian:      Sigma_t,j(y) = lambda^2 (I - lambda^2 H_j(y))  (Jacobian)
                         Lambda_j = Sigma_t,j^-1

The first line is **exact only if `p(theta_0 | x_j)` is Gaussian** -- that is the
whole of the difference. `_jacobian_joint_precision`'s own docstring says so:
if the single-observation posterior is Gaussian then `grad s_j` collapses to
`-(Sigma_0j + lambda^2 I)^-1` and the Jacobian form *becomes* the pilot form, so
the Jacobian rule "is a strict generalization of GAUSS, not an alternative
approximation to it."

Two consequences follow, and both are confirmed by measurement:

- On a Gaussian problem the two must tie. They do: Gaussian twin, exact score,
  churn eta=4 -- shared 0.022 vs 0.027, local 0.024 vs 0.023.
- On non-Gaussian problems the pilot form is wrong, and it is wrong **worst in
  the locals**, because the local cross-correction coefficient
  `R_j = -A_ll^-1 A_lg` is read off the same `Lambda_j`. Laplace: local 0.056 vs
  0.031. Exponential: local 0.097 vs 0.033. Partial pooling: 0.228 vs 0.223 at
  best, with `gauss_hierarchical`'s advantage on the shared block.

`artifacts/oracle_diagnostics/exact_curl.csv` measures the same defect from the
other side: relative Jacobian antisymmetry ~0.92 for `gauss_hierarchical` at
every noise level, against ~0.09 for `gauss_jacobian`. A constant-in-theta
`Lambda_j` composes into a field with large curl, which is not a gradient field
and therefore has no potential for a sampler to relax onto.

**So any real fix must make `Lambda_j` -- or at least its local and cross blocks
-- state-dependent, or otherwise non-Gaussian. There is no way around that.**
Churn cannot help here: it repairs transport, not the field (measured: churn
took `gauss_hierarchical` from 4.700 to 0.051 shared on the exponential bench
while its locals stayed at 0.097).

## Ruled out (do not retry)

**1. Jacobian on the local block only, pilot for the global block.**
The idea was to pay JVPs only for the blocks the failing cross-correction needs
(`A_ll` and `A_lg`) and keep the cheap pilot for the global block. *It saves
nothing.* `_row_score_jacobian` gets one **column** of every row's Jacobian per
tangent, so obtaining `A_lg` (local rows, global columns) needs one tangent per
*global* coordinate -- 7 of the 10 on partial pooling -- and `A_ll` needs the
remaining 3. That is the entire Jacobian. Dead on cost grounds.

**2. Nonparametric regression of the local on the global from pilot draws.**
The idea was to replace the constant `R_j` with a smooth `R_j(g)` fitted to the
pilot draws `(g_i, l_i) ~ p(g, l | x_j)`, at zero network cost. *It answers the
wrong question.* The quantity the composition needs at noise level `t` is a
**backward** conditional -- the regression of `theta_0` on `theta_0` given the
diffused state `theta_t` -- not the `t = 0` regression the pilot draws exhibit.
Fitting the latter and using it at level `t` conflates two different objects and
is only correct in the same Gaussian limit that already makes the pilot exact.

## Candidate under test: kernel-weighted backward covariance from the pilot

The observation that survives idea 2: the pilot draws *do* determine the backward
conditional, just not by regression. Given draws `theta_0^(i) ~ p(theta_0 | x_j)`,
Bayes' rule with the diffusion kernel gives

    p(theta_0 | theta_t, x_j) prop p(theta_0 | x_j) N(theta_t; theta_0, lambda^2 I)

so a self-normalized importance estimate of the backward covariance is

    w_i     prop exp(-||theta_t - theta_0^(i)||^2 / (2 lambda^2))
    mu      = sum_i w_i theta_0^(i)
    Sigma_t,j(theta_t) = sum_i w_i (theta_0^(i) - mu)(theta_0^(i) - mu)^T

This is exactly the object `_jacobian_joint_precision` computes from the network
Jacobian, estimated instead from the pilot the rule already pays for. Properties:

- **State-dependent**, which is the thing the pilot form lacks.
- **No Gaussian assumption** anywhere.
- **Zero extra network evaluations** -- it reuses stored pilot draws.
- **Degrades gracefully**: as `lambda` grows the weights flatten and it returns
  `Sigma_0,j`, i.e. exactly today's `gauss_hierarchical`; as `lambda -> 0` the
  weights concentrate and the covariance shrinks, which is the correct limit.
- **Same conditioning**: the estimate is converted to `H_j = (I - Sigma_t/lambda^2)
  / lambda^2`, symmetrized, and its eigenvalues clamped into
  `[0, (1 - floor)/lambda^2]` -- the identical projection the Jacobian path uses
  -- so `Lambda_j` stays bounded below by `lambda^-2 I` and the composed
  precision stays positive definite by construction rather than by repair.

Implementation is a monkeypatch on `_effective_global_factors` plus a wrapper on
`estimate_posterior_moments` to retain the pilot draws; `state=x` is already
passed to that method for every covariance-aware correction, so nothing in
`compass` needs to change.

**Known risk, stated in advance.** This is a nonparametric density-ratio estimate
and it inherits the curse of dimensionality: with `M` pilot draws the effective
sample size collapses once `lambda` is small relative to the spacing of the
draws, which will bite in the 10-dimensional partial-pooling latent block long
before it bites in the 2-dimensional toys. The toy benches will therefore
*over*-state how well it works; that is why the partial-pooling run matters and
why the effective sample size should be logged, not assumed.

### Result: REJECTED, and for a reason that also kills the whole family

It fails the *easiest* test in the suite. On the **Gaussian twin**, where both
rules are exact by construction and must therefore agree, churn eta=4, 400 draws,
30 steps:

| arm | shared W1/σ | width | local W1/σ |
|---|---|---|---|
| `gauss_hierarchical` (pilot) | **0.082** | **0.993** | **0.066** |
| `gauss_hierarchical` + kernel curvature | 0.849 | 1.643 | 0.092 |

Ten times worse on a problem where it should be identical, so it was not run on
Laplace or exponential.

**The recorded diagnostic identifies the cause precisely.** Mean effective sample
fraction of the importance weights, by noise decade, out of 512 pilot draws:

    lambda ~ 1e0 : 0.773   (~396 draws)   healthy
    lambda ~ 1e-1: 0.154   (~79 draws)    marginal
    lambda ~ 1e-2: 0.007   (~3.6 draws)   collapsed

The weights ``w_i prop exp(-||theta_t - theta_0^(i)||^2 / (2 lambda^2))``
concentrate on the single nearest pilot draw once ``lambda`` falls below the
spacing of the draws. With ``M`` draws spanning a posterior of width ``sigma`` in
``D`` dimensions that spacing is ``~ sigma M^(-1/D)``; here ``sigma ~ 0.25`` and
``M = 512`` in ``D = 2`` gives ``~0.011``, which is exactly where the collapse is
observed. Supporting the estimate down to ``lambda_min = 0.032`` needs
``M ~ (sigma/lambda)^D`` -- about ``10^3`` in 2-D, and about ``10^15`` in the
10-D partial-pooling latent block.

**Why this kills the idea rather than the implementation.** More pilot draws
would not rescue it, because the estimator is only trustworthy in exactly the
regime where it carries no information:

  * ``lambda`` large -- weights flatten, the estimate returns ``Sigma_0,j``,
    which *is* today's `gauss_hierarchical`. Reliable, and no gain.
  * ``lambda`` small -- the state-dependence that would be new information is
    precisely where the effective sample size has collapsed. Informative, and
    unusable.

Blending the two (kernel where the effective sample size is adequate, pilot
elsewhere) therefore returns today's rule everywhere. Any nonparametric estimate
of the backward covariance from ``t = 0`` draws inherits this, so the family is
dead, not just this member.

**What this implies for the original question.** The state-dependence has to come
from the network, because only the network is evaluated *at the diffused state*.
That is what makes `gauss_jacobian`'s derivative cost intrinsic rather than
incidental: it is paying for information that no reweighting of ``t = 0`` pilot
draws contains. So the search should move from "avoid the Jacobian" to "compute
the Jacobian more cheaply" -- see below.

`kernel_curvature.py` is retained because its diagnostic (effective sample
fraction per noise decade) is the reusable part, and because a future candidate
should be checked against the same Gaussian-twin identity test that caught this
one within one smoke run.

## Candidate 2: measured `Sigma_t,j` by the law of total covariance -- REJECTED

The failure of candidate 1 was a density-ratio problem, so this one avoids ratios
entirely. Law of total covariance plus Tweedie:

    Sigma_bar_t,j = E[Cov(theta_0 | theta_t)] = Sigma_0j - Cov(mu(theta_t)),
    mu(theta_t) = theta_t + lambda^2 s_j(theta_t)

Both terms are plain sample covariances of ``M`` points in ``D`` dimensions, so
it needs ``M >> D`` rather than ``M >> (sigma/lambda)^D``, and it evaluates the
network **at the diffused state**. It also reduces to `gauss_hierarchical`
*algebraically* on a Gaussian problem: ``Cov(mu) = Sigma_0j (Sigma_0j + lambda^2
I)^-1 Sigma_0j``, so the difference is ``lambda^2 Sigma_0j (Sigma_0j + lambda^2
I)^-1``, whose inverse is exactly ``Sigma_0j^-1 + lambda^-2 I``. Cost: one batched
row-score evaluation per ladder rung (24), against ~200 for the sampler.

It still fails the Gaussian-twin identity test (churn eta=4, 1,000 draws, 50
steps, 1,024 pilot draws): shared W1 **0.803** vs `gauss_hierarchical`'s 0.046,
width 1.789 vs 1.029.

**A direct unit test against the closed form -- using the exact analytic score, so
neither a wiring bug nor network error is involved -- shows the estimator itself
is the problem** (`scratchpad/test_tweedie_estimator.py`):

| lambda | \|\|Sigma_t\|\| | rel err M=1024 | rel err M=16384 | \|\|Sigma_t\|\|/\|\|Sigma_0\|\| |
|---|---|---|---|---|
| 3.0 | 0.956 | 0.032 | 0.030 | 0.907 |
| 1.0 | 0.559 | 0.086 | 0.035 | 0.530 |
| 0.3 | 0.109 | 0.379 | 0.076 | 0.103 |
| 0.1 | 0.014 | **1.117** | 0.240 | 0.013 |
| 0.03 | 0.0013 | **3.657** | 0.823 | 0.0012 |

## The unifying reason both candidates died

``Sigma_t,j = Cov(theta_0 | theta_t)`` is ``O(lambda^2)``. As ``lambda -> 0`` it
becomes a **vanishingly small fraction of ``Sigma_0j``** -- 0.12% at
``lambda = 0.03`` on this problem. Any Monte-Carlo route recovers it as a
*residual* of ``Sigma_0j``-scale quantities, so its relative error grows like

    1 / (sqrt(M) * ||Sigma_t|| / ||Sigma_0||)   ~   1 / (lambda^2 sqrt(M))

Reaching 10% relative error at ``lambda = 0.03`` needs ``M ~ 7e7`` draws **per
observation**. Catastrophic cancellation, not tuning.

The Jacobian route computes ``I - lambda^2 H_j`` to machine precision, so the
same cancellation is *benign* there. **That is why `gauss_jacobian`'s derivative
cost is irreducible by sampling-based estimators**, and it predicts that any
future candidate of this shape will fail in exactly the small-lambda regime that
matters -- which is also the regime where the composition has to be accurate,
since that is where a churned sampler does its final relaxation.

**Test both candidates would have passed and the identity test caught:** at large
lambda both are fine (3% error at lambda = 3). Grading a candidate only at
moderate noise would have produced a false positive. The Gaussian-twin identity
test caught both within one smoke run each; keep using it first.

## Still open (ranked)

1. **Stochastic-tangent (Hutchinson) Jacobian.** Estimate ``H_j`` from ``k``
   random tangents instead of one per latent coordinate, smoothed across steps by
   an EMA. Unlike the kernel estimate this evaluates the network *at the diffused
   state*, so it is estimating the right object; the only question is variance.
   ``k = 2-3`` against ``D = 10`` on partial pooling would be a 3-5x cut on top of
   `jacobian_refresh`. **This is the surviving cheap route.**
2. **`jacobian_refresh` at 5 rather than 20.** Not an accuracy improvement, but it
   is the untested confound in the partial-pooling result: refresh 20 rebuilt the
   curvature only ~5 times over 50 levels, and `gauss_jacobian` came out *worse*
   than `gauss_hierarchical` there. Until refresh 5 is run, "the pilot-free rule
   is not better on partial pooling" is not established.
3. **Symmetrizing the composed field.** The 0.92 curl is the defect; projecting
   the composed Jacobian onto its symmetric part would remove it, but obtaining
   that Jacobian is at least as expensive as the per-row one.
