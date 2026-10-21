# Why non-symmetric shared posteriors come out badly, and what fixes it

Measured 2026-09-10 on `inverse_kinematics`, frozen checkpoint
`legacy_h16d2_n10000_six_weighted_log_sigma_e10000`. New code, all outside
`src/compass/`:

| file | what |
|---|---|
| `inverse_kinematics/experiments/composition_attribution.py` | splits the composition error into a network part and a rule part |
| `inverse_kinematics/experiments/marginal_product.py` | the replacement method |
| `inverse_kinematics/experiments/evaluate_marginal_product.py` | head-to-head against the three existing rules |
| `inverse_kinematics/experiments/run_attribution_sweep.sh` | the five replicates below |

---

## 1. The diagnostic: an exact referee that contains no sampler

For one shared block `a` with independent per-observation locals, two
factorisations are exact, not approximations:

    p(a | x_1..x_n)          proportional to  p(a)^(1-n) * prod_i p(a | x_i)
    p(a, l_1..l_n | x_1..n)  =  p(a | x_1..n) * prod_i p(l_i | a, x_i)

`a` is one-dimensional here, so the first can be evaluated **numerically on a
grid**, with no composed score anywhere in it. That gives a third curve between
the ABC reference and the sampler:

* `reference_product` — the same product built from *ABC* single-observation
  posteriors. Algebraically identical to the joint ABC reference, so it is a
  **self-test of the pipeline**, not evidence. Measured error 0.023-0.051 W1/sigma
  on the four non-symmetrised cases: the pipeline is sound.
* `network_product` — the product of the **network's own** single-observation
  posteriors. What a perfect composition rule would return. Its gap to the
  reference is the network's fault; the sampler's gap to *it* is the rule's.

## 2. The verdict: the rule, not the network

`|x1| < 0.15`, n = 8, seed 99. Reference puts **0.243** of its mass on the
positive branch.

| curve | mean err/sigma | width | W1/sigma | mass > 0 |
|---|---:|---:|---:|---:|
| joint ABC reference | 0 | 1.000 | 0 | 0.243 |
| `reference_product` (self-test) | -0.046 | 0.977 | 0.047 | 0.225 |
| **`network_product`** | +0.065 | 1.059 | **0.107** | 0.288 |
| `gauss_jacobian` | +0.492 | 0.992 | **0.497** | **0.467** |
| `gauss_hierarchical` | +0.485 | 0.888 | 0.522 | **0.468** |
| `langevin_fnpse` | +0.166 | 1.017 | 0.237 | 0.339 |

The single-observation posteriors are excellent — per-observation W1/sigma
0.03-0.19, mean bias -0.002 single-obs sd — and composed exactly they land at
0.107. The Gaussian rules are **4.6x worse than the network they are built on**.

**The mechanism is the Gaussian summary.** Each rule reduces an observation's
contribution to a mean and a covariance. Unequal branch weights are exactly what
that cannot carry, so the rules return a near-symmetric answer: 0.467 against a
true 0.243. At `x1 = 0` the truth *is* 50/50, so the defect is invisible and
even reads as success. It is only visible off symmetry.

A second, independent defect shows at seed 101, where the branches happen to be
near 50/50 and the means are right (+0.003): the widths are **0.810** and
**0.718** of the reference. At n = 16 `gauss_hierarchical` falls to **0.678**.
The rules are over-confident as well as symmetrising, and both get worse with n.

**Not a KDE artifact.** Sweeping the KDE bandwidth over a 3x range moves
`network_product` only 0.091-0.138, and it *improves* as the bandwidth narrows
toward the pipeline's own 0.047 floor.

## 3. The replacement: `marginal_product`

Both factorisations above are exact and both conditionals are ones the network
is trained on directly — `p(theta | x)` is the ordinary posterior and
`p(l | g, x)` is the third entry of `SIX_CONDITIONALS`, carrying weight 7 under
the `six_weighted` menu. So the whole composed posterior comes from **two
ordinary amortized sampler calls**:

1. sample every single-observation posterior in one call; tabulate `p(a | x_i)`;
2. multiply, divide by `p(a)^(n-1)`, draw `a*` by inverting the CDF;
3. sample `p(l_i | a*, x_i)` with the shared coordinate conditioned.

No composed score, no Gaussian summary, no correction rule, no pilot covariance.

### Results, five replicates (W1/sigma against the ABC reference)

| case | `gauss_jacobian` | `gauss_hier` | `langevin_fnpse` | `marginal_product` |
|---|---:|---:|---:|---:|
| selected n=8 seed 99 (mass 0.243) | 0.497 | 0.522 | 0.237 | **0.115** |
| selected n=8 seed 101 (mass 0.467) | 0.251 | 0.347 | 0.184 | **0.036** |
| selected n=16 seed 99 (mass 0.273) | 0.436 | 0.564 | 0.261 | **0.159** |
| exact-symmetric n=8 | 0.288 | 0.333 | 0.230 | *0.541* |
| prior-predictive n=8 (sd 0.10) | 0.152 | 0.161 | 0.239 | *0.201* |
| **endpoint error, the locals** | 0.066-0.078 | 0.070-0.084 | 0.069-0.077 | **0.0136-0.0241** |

**On the locals it wins everywhere, including the two cases it loses on the
shared coordinate**, by 3-5x. Endpoint error is reference-free — the forward map
is deterministic, so a correct draw reproduces its own observation. Every
score-composition rule returns locals whose arm tips miss their own observation
by 0.066-0.084, against 0.012 for the same network sampling one observation with
no composition. Composition is degrading the locals about five-fold, and that is
not caused by the shared coordinate being wrong: at seed 101 `gauss_jacobian`
has a mean error of +0.003 and still an endpoint of 0.0669.

### Cost

Single-draw budget on selected n=8 seed 99, against `gauss_jacobian` at 15.0s /
0.497:

| single draws | 2,000 | 5,000 | 10,000 | 20,000 | 40,000 |
|---|---:|---:|---:|---:|---:|
| W1/sigma | 0.201 | 0.192 | 0.121 | 0.115 | 0.100 |
| seconds | 23.6 | 27.4 | 30.7 | 43.0 | 56.9 |

Monotone, and the cheapest setting is already 2.5x better than `gauss_jacobian`
at 1.6x its cost. Endpoint error is flat at 0.0156 across the whole range: stage
3 does not depend on the quality of the stage-2 density.

## 4. Where it does not help, and why

**Exactly symmetric data (`x1 = 0`): it is worse — 0.541 against 0.288.** The
network carries a systematic per-observation bias of **+0.028 single-obs sd**
towards the true branch. Invisible at n = 1; at n = 8 the exact product
faithfully amplifies it to a branch mass of 0.770 against a true 0.500. The
Gaussian rules' symmetrising bias, a liability everywhere else, accidentally
masks it here. The self-test also degrades on this case (0.191), because the
joint reference is symmetrised and the per-observation references are not, so
the identity no longer holds exactly — read that row as a floor, not a target.

**Sharply informative unimodal posteriors: no gain.** On the prior-predictive
dataset the composed sd is 0.10 while each single-observation posterior is 5x
wider, and KDE bias then dominates: 0.201 against `gauss_jacobian`'s 0.152.

**Dimension.** The grid is why this is practical, and it binds at roughly three
shared coordinates. The arm's shared block is one-dimensional, as is partial
pooling's global block, but this is not a general replacement for score
composition — it is the right tool for a low-dimensional shared block.

## 5. The general lesson

Composition amplifies *systematic* single-observation error. For a shared
coordinate, a systematic per-observation mean bias `e` (in single-obs sd units)
becomes roughly `sqrt(n) * e` in composed-sd units, and per-observation error
that is *random* across observations does not average away at all — it stays
roughly constant in composed units. A single-observation posterior that looks
excellent (W1/sigma 0.03) is not automatically good enough to compose eight
times. Grade the composed object, never the single-observation one.

## 6. Not done

* `marginal_product` lives in `experiments/` and is not wired into
  `run_inference.METHODS`, `run_bimodal.py` or the SBC graders.
* No SBC/coverage run: the table above is five datasets against an ABC
  reference, which is a pilot, not calibration evidence.
* The +0.028 per-observation bias found on the symmetric case is unexplained and
  is now the binding constraint on that case. It is a network-side defect and
  nothing in the sampler will fix it.
* Not tried on `Partial_Pooling`, whose global block is also low-dimensional.

---

# 7. Transplanting the idea back into score space — tried, and it fails

The obvious follow-up is to keep the sampler and borrow only the idea: compose
the *shared marginal score* exactly instead of summarising each observation by a
Gaussian. Fisher's identity supplies it without any density estimate,

    d/da log p(a | x_i) = E_{l ~ p(l | a, x_i)} [ d/da log p(a, l | x_i) ]

and the draws already in hand are, conditioned on their own `a`, draws from
`p(l | a, x_i)`. So the marginal score is a Nadaraya-Watson regression of the
network's own joint score on `a` — free, since the draws exist and it is one
extra forward pass. Implemented as `marginal_product.py --estimator score`.

**It does not work.** Selected `|x1| < 0.15`, n = 8, seed 99, against the
density product's 0.115:

| noise scale sigma | 0.03 | 0.05 | 0.08 | 0.12 |
|---|---:|---:|---:|---:|
| W1/sigma | 0.908 | 0.787 | 0.684 | 0.337 |
| width / reference | 0.123 | 0.156 | 0.370 | 0.935 |

Two separate causes, both measured.

**(a) The score network is worst exactly where the identity lives.** It predicts
`sigma_t * score`, so at `t = eps` it returns a tiny output divided by a tiny
sigma. Regressed against the numerical derivative of a KDE log-density on the
same draws, the correlation is **0.579 at sigma = 0** and **0.976-0.979 at
sigma = 0.05-0.10**. The shape is recoverable at a finite noise scale; at zero
it is not. But the product identity is only true at zero noise, so there is no
scale at which both hold.

**(b) The `(n-1)` prior subtraction is a delicate cancellation.** The composed
score is `sum_i s_i - (n-1) s_prior`, and at n = 8 that second term is seven
times a quantity the same size as the ones it is subtracted from. It cancels
only if the per-observation scores carry the prior exactly. At sigma = 0.05 the
network's score has correlation 0.976 with the truth but **slope 0.80** — right
shape, wrong scale — and 7 x 20% of a large term is what collapses the width to
0.156.

Taking the prior score from the *same* estimator, so a common multiplicative
bias would cancel (`--score-prior network`), makes it worse and unstable: W1/sigma
0.823 / 0.586 / **2.670** at sigma = 0.05 / 0.08 / 0.12, the last having flipped
onto the wrong branch entirely. It replaces one biased term with a second noisy
one inside the same fragile difference.

## What this explains

This is the same failure the Gaussian rules have, seen without the Gaussian
assumption. **Anything that composes in score space inherits the `(n-1)`
cancellation** and is therefore sensitive to a multiplicative error in the score
at a level that single-observation sampling never exposes — the terminal
denoise step uses the same score once and gets endpoint 0.012. Composition uses
it eight times with a seven-fold subtraction.

The density product avoids this because the cancellation happens in log-space
against densities anchored to actual draws, which encode the prior correctly by
construction rather than by the network reproducing it.

## Consequences

* **Do not expect a score-space rule to reach the density product on this model
  class.** The remaining Gaussian-rule work (block pooling, tempering, damping
  schedules) is tuning inside a formulation whose error floor is set by (b).
* The one transplant that *would* work is architectural, not accuracy-improving:
  take the composed marginal from the density product, diffuse it exactly by
  convolution (`q_t = q * N(0, sigma_t^2)`), and feed **that** as the shared
  score at every `t`. It is exact for the shared block at all times and drops
  into the sampler as a correction rule — but it would *match* the density
  product, not beat it, because the locals already come from `p(l | a, x)`
  exactly.
* **The binding constraint is now the network, not the rule**: the +0.028
  single-obs-sd per-observation bias that ruins the symmetric case. A score with
  slope 0.80 at sigma = 0.05 is the same defect seen a different way. That is a
  training-side problem.
