# A Gaussian-mixture per-observation summary beats both the Gaussian rules and the grid product

Measured 2026-09-11 with `inverse_kinematics/experiments/mixture_product.py`.
Saved arrays only — no model, no sampler, no GPU. Results in
`artifacts/mixture_product/mixture_product.json`.

## The idea

Every rule in `MultiObsSampler` summarises each observation's shared marginal as
**one** Gaussian, and one Gaussian cannot carry unequal branch weights. The exact
1-D density product fixes that but needs a grid, so it binds at ~3 shared
dimensions and cannot reach `Partial_Pooling`.

A Gaussian *mixture* summary removes both limits, because mixtures are closed
under the two operations composition needs:

- **Products.** `Π_j Σ_k w_jk N(μ_jk, σ_jk²)` is again a Gaussian mixture, in
  closed form, in any dimension. The shared prior here is `N(0, 0.5²)`, so the
  `prior^(1−n)` division is analytic too — one more factor with a **negative**
  precision, added like any other in natural parameters.
- **Diffusion.** A mixture convolved with `N(0, σ_t²)` is a mixture with
  `Σ + σ_t²I`, so the composed marginal can be diffused exactly and fed back as
  a shared score at every `t`.

`K = 1` is exactly what `gauss_hierarchical` does, so this is a strict
generalization of the library's current rule.

## Result 1 — `K = 1` reproduces the Gaussian rules' failure signature

This is the control, and it is the cleanest confirmation of the mechanism yet:
an analytic product with **no sampler and no rule in it** reproduces both
symptoms the rules show.

| dataset | reference mass>0 | `K=1` mass>0 | `K=1` width | `gauss_jacobian` mass>0 |
|---|---|---|---|---|
| selected n8 s99 | 0.288 | 0.462 | 0.674 | 0.467 |
| selected n8 s101 | 0.440 | 0.465 | 0.482 | 0.458 |
| selected n16 s99 | 0.214 | 0.467 | 0.393 | 0.428 |

Symmetrised to ~0.46 and over-confident, with `dip_ratio` 1.000 (i.e. unimodal),
worsening with `n` — the signature recorded in `docs/nonsymmetric_composition.md`,
now derived from the summary alone. **The rules fail because of the single
Gaussian, not because of the sampler or the score.**

## Result 2 — more components fix it, and beat the grid product

W1/σ against the same ABC reference, exact enumeration of all `K^n` components
(no pruning):

| dataset | grid product | best of 3 rules | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|---|---|
| selected n8 s99 | 0.107 | 0.237 | 0.667 | 0.188 | 0.103 | **0.066** |
| selected n8 s101 | 0.057 | 0.184 | 0.591 | 0.194 | 0.055 | **0.050** |
| selected n16 s99 | 0.180 | 0.261 | 0.814 | 0.261 | **0.172**\* | 0.220\* |
| prior n8 s3 | 0.191 | 0.152 | 1.437 | **0.095** | 0.158 | 0.191 |
| exact_zero n8 s99 | 0.526 | 0.230 | 0.479 | 0.430 | 0.599 | 0.547 |

\* beam 1024; exact enumeration at `n=16` needs `K^16` components.

Two things stand out.

**It beats the Gaussian rules by 2.5–4×** on every asymmetric case, and recovers
the branch mass almost exactly (selected n8 s99: 0.292 at `K=2` against a
reference 0.288, versus 0.467 for `gauss_jacobian`).

**It also beats the grid product it was meant to merely imitate** — 0.066 vs
0.107, 0.050 vs 0.057, 0.095 vs 0.191, 0.172 vs 0.180. The grid product
multiplies KDE densities, so it multiplies KDE noise; the mixture fit smooths
each factor before multiplying. So this is not only the dimension-free version of
the method, it is the more accurate one.

## Result 3 — pruning is safe, but the beam must scale with K

The `K^n` enumeration is only available as a check. A real implementation keeps
the `M` heaviest components after each factor. W1/σ, selected n8 s99:

| K | exact | beam 1024 | beam 256 | beam 64 |
|---|---|---|---|---|
| 1 | 0.667 | 0.667 | 0.667 | 0.667 |
| 2 | 0.188 | 0.188 | 0.188 | 0.188 |
| 3 | 0.103 | 0.103 | 0.101 | 0.137 |
| 4 | 0.066 | 0.079 | 0.254 | 0.219 |

`K = 2` is exact at beam 64. `K = 4` needs 1024 and is badly wrong at 256. The
weights do **not** concentrate enough to prune aggressively at high `K` — the
risk flagged before the test, now quantified.

**No component ever went improper.** The `prior^(1−n)` division never produced a
negative precision in any dataset at any `K`, which is what decides whether the
composed mixture can be sampled from directly. One implementation note earned
the hard way: the division must be applied **once at the end**, not spread across
the factors. Spreading it was tried first and fails outright — the
per-observation posteriors here are barely narrower than the prior (sd 0.37–0.67
against 0.5), so subtracting even `1/n` of `(n−1)` prior precisions from a single
factor drives every component improper and the product is empty before it starts.
A partial product of *normalized* densities always has positive precision, so
pruning on its masses is well defined at every step.

## Where it still loses, and why that was expected

`exact_zero` — exactly symmetric data. Every `K ≥ 2` lands at 0.43–0.60 against
`langevin_fnpse`'s 0.230, and the grid product loses there too (0.526). This is
the per-observation mean bias being amplified by composition, the failure mode
the capacity ladder in `docs/score_calibration.md` identified, and **nothing on
the rule side fixes it** — the mixture is faithfully composing a biased input.
The Gaussian rules' symmetrising bias accidentally hides it.

So the two failure modes remain distinct and the fix for each is different:
multimodality is a summary problem, now solved; per-observation bias is a network
problem, still open.

## BIC is not yet a usable selector

BIC on the per-observation draws picks `K=4` on four of five datasets. That is
right on the `selected` cases but wrong on `prior_n8_seed3`, where `K=2` scores
0.095 and `K=4` scores 0.191 — BIC is fitting structure in the draws that does
not help the product. Component selection needs a criterion tied to the composed
object, not to per-observation fit. Until then `K=2` is the safe default: it is
never worse than the rules, never needs a beam above 64, and costs 256
components at `n=8`.

## Follow-up 1 — merge, don't drop: beam 8 matches exact enumeration

Dropping the lightest components *removes* their mass, and the components it
removes are exactly the ones carrying the minority branch: they are light at
every intermediate step, long before the observations that would support them
have been multiplied in. The standard alternative is **Runnalls merging** --
replace the two components whose merge invents the least variance with the
single Gaussian preserving their combined weight, mean and variance. Mass is
conserved rather than discarded. In 1-D there is a second reason to expect a
large gain: the product depends only on *sums* of precisions and precision-means,
so many distinct index tuples land on near-identical components, and merging
deduplicates them where dropping keeps one copy and discards the rest.

W1/σ at K = 4 (`mixture_refinements.py`):

| dataset | drop b8 | drop b32 | drop b256 | **merge b8** | merge b32 | exact |
|---|---|---|---|---|---|---|
| selected n8 s99 | 0.878 | 0.098 | 0.254 | **0.050** | 0.066 | 0.066 |
| selected n8 s101 | 0.371 | 0.447 | 0.084 | **0.048** | 0.050 | 0.050 |
| prior n8 s3 | 1.364 | 1.335 | 0.323 | **0.187** | 0.193 | 0.191 |
| selected n16 s99 | 0.712 | 0.557 | 0.919 | **0.248** | 0.288 | n/a |

**Merging at beam 8 matches or beats the exact enumeration of all K^n
components** on every dataset, and is flat in the beam: beam 8, 32 and 256 agree
to three decimals. Dropping is erratic in the beam — non-monotone, because which
components survive depends on an arbitrary multiplication order.

That it sometimes *beats* exact (0.050 vs 0.066) is regularization, not magic:
K=4 exact is visibly over-peaked on the dominant branch, and merging coarsens it
back toward the reference.

The practical consequence is that **the beam stops being a tuning knob.** Fix it
at 8 and the pruning question is closed — including at `n = 16`, where exact
enumeration is 4^16 components and unavailable.

## Follow-up 2 — cost: the product is free, EM is the whole bill

Timed at n = 8, 20,000 draws per observation, 3 CPU threads:

| stage | seconds |
|---|---|
| EM fit, K=2 / K=3 / K=4 (8 observations) | 49.6 / 89.8 / 100.3 |
| product + prune, K=4, merge beam 8 | **0.009** |
| product + prune, K=4, exact (65,536 comps) | 0.048 |
| product + prune, K=4, merge beam 256 | 37.2 |

Merging is O(M²) per merge, so a *large* beam is expensive — but merging removes
any reason to use one. At beam 8 the entire composition step costs 9 ms.

So EM is the whole cost, and both of its defaults are generous. Sweeping them
(`em_budget.py`, K=3, merge beam 8):

| draws | restarts | EM s | s99 | s101 | prior |
|---|---|---|---|---|---|
| 20000 | 8 | 88.6 | 0.153 | 0.042 | 0.163 |
| 20000 | 1 | 11.1 | 0.131 | 0.047 | 0.161 |
| 5000 | 1 | **3.0** | 0.214 | 0.031 | 0.164 |
| 1000 | 1 | **0.7** | 0.201 | 0.056 | 0.233 |

**Restarts buy nothing** — 1 restart matches 8 on every dataset at 8× the speed.
Draws past ~5000 buy nothing systematic; the variation across draw counts at
fixed restarts (s101: 0.056 / 0.145 / 0.031 / 0.047) is EM fit noise, not a
budget effect, and is the honest uncertainty on every number in this document.

**Recommended setting: 5000 draws, 1 restart, K=3, merge beam 8 — about 3
seconds of CPU** against 88.6 s at the defaults, for no measured accuracy loss.

### Against `gauss_hierarchical`

Saved runs in this repository put `gauss_hierarchical` at **16–46 s** and
`gauss_jacobian` at **14–26 s** end to end on GPU, against
`marginal_product`'s **61 s** for the full two-sampler-call pipeline.

The mixture replaces that pipeline's KDE stage with EM, so it adds **~3 s of CPU**
and nothing else. What is *not* yet measured is the two sampler calls against
`gauss_hierarchical`'s composed reverse diffusion on the same GPU at the same
settings — no GPU on this host was free at the time of writing
(`time_mixture_vs_rules.py` is written and waiting). The scale is set by those
sampler calls, not by the mixture arithmetic, so that comparison is the one that
decides the headline number.

## Follow-up 3 — choosing K without knowing it: all four selectors fail

Four selectors graded against the oracle (the K actually minimizing W1 to the
reference), on five datasets:

| dataset | oracle K | BIC | AIC | holdout | stability |
|---|---|---|---|---|---|
| selected n8 s99 | 4 | 4 ✓ | 4 ✓ | 4 ✓ | 4 ✓ |
| selected n8 s101 | 4 | 4 ✓ | 4 ✓ | 4 ✓ | 4 ✓ |
| selected n16 s99 | 3 | 4 ✗ | 4 ✗ | 4 ✗ | 4 ✗ |
| prior n8 s3 | 2 | 4 ✗ | 4 ✗ | 4 ✗ | 4 ✗ |
| exact_zero n8 | 2 | 4 ✗ | 4 ✗ | 4 ✗ | 4 ✗ |

**All four give identical answers, and all four are wrong the same way**: they
pick K=4 everywhere. BIC, AIC and held-out likelihood all grade the
*per-observation* fit, so they agree by construction — more components always
describe 20,000 draws better, and the penalty never bites at that sample size.

The stability criterion has a free threshold, so it deserved a proper test rather
than one default. `stability_gaps.py` prints the consecutive composed gaps
`W1(K, K+1)/σ` and sweeps the tolerance: the best any single tolerance achieves
is **2 of 6 datasets**. There is no threshold that works, because the gaps that
must be ignored on one dataset are the same size as the gaps that must be
followed on another.

### What to do instead: don't select, fix K = 3

Worst case over the four asymmetric datasets, by fixed K: K=2 → 0.261,
**K=3 → 0.158**, K=4 → 0.288, K=5 → 0.193. A fixed K=3 has the best worst case
and is never far from the oracle, while every rule's worst case is 0.497.

This also revises the earlier "K=2 is the safe default": that was chosen because
K=2 needed no beam, and merging has since made the beam irrelevant. With merging
the cost of K=3 over K=2 is EM time alone, and K=3 is the better default.

The oracle K itself ranges over 2, 3, 4, 5 with no pattern, and the differences
between K=3, 4 and 5 are comparable to the EM fit noise above — which is the
real reason no selector works. There may be nothing stable to select.

## Follow-up 4 — K = 2 is unstable under composition; K is fixed, not detected

Two things were checked after the recommended settings were fixed, and the first
overturned an earlier recommendation in this document.

### Averaging over K instead of choosing: tested, fails

Since no selector worked and the oracle K wanders with no pattern, the standard
move is to average rather than choose -- and it is free here, because a uniform
mixture over K of composed mixtures is itself a Gaussian mixture (union the
component lists, divide the weights by the number of K values). Measured at the
recommended settings, W1/σ:

| dataset | K=2 | K=3 | K=4 | average | jacobian |
|---|---|---|---|---|---|
| selected n8 s99 | 0.232 | 0.214 | 0.167 | 0.177 | 0.497 |
| selected n8 s101 | 0.212 | 0.031 | 0.057 | 0.099 | 0.251 |
| selected n16 s99 | **2.725** | 0.240 | 0.229 | 1.032 | 0.436 |
| prior n8 s3 | 0.093 | 0.164 | 0.191 | 0.093 | 0.152 |
| exact_zero n8 | 0.412 | 0.541 | 0.537 | 0.494 | 0.288 |
| **worst** | 2.725 | 0.541 | 0.537 | 1.032 | 0.497 |

Averaging does not help: it inherits the K = 2 blow-up and lands at 1.032.

### K = 2 is bistable at n = 16

The 2.725 is not a fluke and not EM seed noise. Sweeping draws, restarts and
seed at K = 2 on `selected_n16_seed99`:

| draws | restarts | seed 0 | seed 100 | seed 200 |
|---|---|---|---|---|
| 2000 | 1 | 0.174 | 0.100 | 0.111 |
| 2000 | 4 | 0.101 | 0.101 | 0.101 |
| 5000 | 1 | **2.725** | **1.304** | **3.155** |
| 5000 | 4 | **3.246** | **3.172** | **3.220** |
| 20000 | 1 | 0.250 | 0.255 | **3.162** |
| 20000 | 4 | 0.250 | 0.250 | 0.250 |

More restarts make it *consistently wrong* at 5000 draws, so this is not
optimizer noise -- it is a genuinely different EM solution that is better in
per-observation likelihood and catastrophic after composition. This is the √n
amplification once more, now acting on the **fit choice** rather than on network
bias: at n = 16, two K = 2 fits that differ slightly per observation compose into
answers 30× apart.

The same sweep at K = 3 and K = 4 shows no such behaviour -- every one of the 18
configurations lands in 0.08–0.36, with no blow-ups at either K.

**This settles the default at K = 3 and retires the earlier "K = 2 is the safe
default" twice over.** K = 2 was first chosen because it needed no beam (merging
made that irrelevant), then kept on worst-case grounds; it is in fact the one
setting here that can fail catastrophically.

### K is fixed at inference, not detected

To be explicit, because it changes how the numbers should be read: **every
fixed-K number in this document is achievable at inference time** -- no reference
is used anywhere in the pipeline. The *oracle* column is diagnostic only and is
not reachable without ground truth.

No detection mechanism survives: BIC, AIC, held-out likelihood and composed
stability all fail (follow-up 3), and averaging fails here. So K is a fixed
hyperparameter, set to 3, and the honest cost of not being able to detect it is
the gap between the fixed-K and oracle columns -- 0.158 against 0.095 on
`prior n8 s3` being the worst case, a factor of 1.7.

## Follow-up 5 — two retractions, and why no selector can work

Investigating *why* K = 2 failed at n = 16 falsified two claims made above. Both
are wrong as written; the corrected versions follow.

### Retraction 1: K = 2 is not unstable. Beam 8 is.

The exact product over all `2^16` components at `n = 16`, K = 2 scores
**0.231** — perfectly healthy. The 3.246 came from *merge pruning at beam 8*:

| n = 16, K = 2 | exact | merge b8 | merge b16 | merge b64 | drop b8 | drop b64 |
|---|---|---|---|---|---|---|
| W1/σ | 0.231 | **3.246** | 0.231 | 0.231 | 0.275 | 0.237 |

and at K = 3, merge b8 gives 0.353 against 0.098 at b64. So **follow-up 1's
"the beam stops being a tuning knob" is wrong**: beam 8 was validated only at
n = 8. The beam must scale with the number of observations, because pruning is
applied once per factor and its damage compounds over the sequence.

### Retraction 2: merging has a failure mode dropping does not

Runnalls merging preserves the moments of the pair it collapses, so collapsing
two *distant* components manufactures one wide component. Squeezed to 8
components over 16 sequential steps that compounds, and the manufactured width
dominates the tails — where the `prior^(1−n)` division then multiplies it by
`exp(+70)` at `|a| = 1.5`, `n = 16`. **Merging can invent mass where there is
none; dropping cannot.** Dropping never puts mass outside the data's support in
any configuration measured.

Merging remains the better estimator at equal beam, but the recommended setting
becomes **merge at beam ≥ 64**, not beam 8, and the support check below should
guard it.

### Why no likelihood criterion can select K

Composition is a product, so log errors **add**:

    log q̂(a) − log q(a) = Σⱼ [ log p̂ⱼ(a) − log pⱼ(a) ]

The composed error is the *sup-norm* of the per-observation log-density error
times n, further amplified by `−(n−1) log p₀(a)`, which grows quadratically in a.

BIC, AIC and held-out likelihood all estimate `E_{pⱼ}[log p̂ⱼ]` — an average
**weighted by pⱼ itself**, which by construction cannot see a region where `pⱼ`
is small and `p̂ⱼ` is not. They estimate the wrong norm. That is not bad luck;
it is what those statistics are.

The classical fix is a tail-weighted goodness-of-fit statistic —
**Anderson–Darling**, whose weight `1/(F(1−F))` diverges at both tails. Tested
against Kolmogorov–Smirnov (its unweighted control) and BIC:

| selector | mean W1/σ | worst | picks |
|---|---|---|---|
| oracle | 0.146 | 0.412 | 5, 5, 3, 2, 2 |
| Anderson–Darling | 0.205 | 0.545 | 5 everywhere |
| Kolmogorov–Smirnov | 0.205 | 0.545 | 5 everywhere |
| BIC | 0.215 | 0.573 | 4 everywhere |

**A² also fails, and for a deeper reason than BIC did.** A² and KS decrease
monotonically in K and pick the largest available K, exactly like BIC. The
tail weighting bought nothing because the region that controls the composed
error — `|a| > 1`, where the spurious mode appears — holds an empirical mass of
`6 × 10⁻⁶`, i.e. about **0.03 draws out of 5000**. No statistic computed on
those draws can resolve it, because the information is not in the sample.

That is the real obstruction, and it is not fixable by choosing a better
statistic: **the composed posterior is sensitive to a region the
per-observation samples do not cover.**

### What does work: enforce the constraint instead of selecting

Two things follow that need no selection at all.

**A hard validity certificate.** Since `p(a|x₁..ₙ) ∝ p(a)^(1−n) Πⱼ p(a|xⱼ)`, the
composed support is the **intersection** of the per-observation supports — an
identity. Composed mass outside `∩ⱼ supp(p̂ⱼ)` is provably wrong, and the
draws observe each support directly. `support_check.py` measures it: the 3.246
configuration puts **86% of its composed mass** in a provably impossible region,
two orders of magnitude above any healthy configuration (≤ 0.005). It is a
one-sided certificate, not a ranking criterion — its rank correlation with W1
across configurations is −0.19 — but it catches exactly the failures that matter
and would have caught both retractions above.

**Bound the unmeasured evidence.** Write the composition as
`p(a) Πⱼ [pⱼ(a)/p(a)]`: each factor is the evidence observation j contributes.
Where observation j has no draws, its evidence is *unmeasured*, and the safe
default is a ratio of 1 — no evidence — not whatever a Gaussian tail
extrapolates to. Clipping `|log p̂ⱼ(a) − log p₀(a)| ≤ C` outside each
observation's sampled range bounds the composed damage by `n·C` by the additive
identity above, regardless of K. **Untested** — this is the recommended next
experiment, and it would make the method robust to K rather than dependent on it.

## Next

1. Implement `correction="mixture_hierarchical"` in `MultiObsSampler`: fit
   mixtures to the pilot draws `gauss_hierarchical` already takes, multiply,
   diffuse analytically (`Σ + σ_t²I`), use as the shared score at every `t`.
   `K=1` must reproduce `gauss_hierarchical` bit-for-bit as the regression test.
2. Grade it on `Partial_Pooling`, whose shared block is multi-dimensional — the
   case the grid product cannot reach and the reason this was worth doing.
3. A composed-object criterion for `K`.

## Not done

Multi-dimensional shared blocks (everything here is the 1-D `angle_1`, so the
dimension-free claim is structural, not yet measured); sampling from the composed
mixture rather than tabulating it for grading; locals (the mixture only touches
the shared marginal — the `p(l|a,x)` stage is unchanged from `marginal_product`).
