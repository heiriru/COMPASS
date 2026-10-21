# Inference with K inferred: three benchmarks against the three rules

Run 2026-09-13 on GPU 4 (idle, exclusive). Scripts:
`inverse_kinematics/experiments/{mixture_inference,run_mixture_ik,plot_mixture_inference}.py`
and `tutorials/.../New_Attempt/run_mixture_shared_local.py`.
Artifacts in `inverse_kinematics/artifacts/mixture_inference/` and
`tutorials/.../New_Attempt/mixture_inference/`.

**K is never supplied and never swept.** `mixture_inference.choose_and_fit`
selects it per observation by BIC at inference time. The composition is guarded
against the near-cancellation failure (`docs/mixture_product.md`, follow-up 5):
no merge may produce a partial-product component with precision below
`4·(n−1)/σ₀²`, the condition for the `prior^(1−n)` division to leave a
non-degenerate component.

## What K the method chose on its own

| benchmark | per-observation posterior | K chosen |
|---|---|---|
| inverse kinematics, n = 8 | bimodal, unequal branches | **4–5** (`[5,4,4,4,5,5,4,5]`) |
| shared/local Gaussian mixture, n = 30 | unimodal Gaussian | **1** for all 30 |
| shared/local Gaussian, noise sweep, n = 30 | unimodal Gaussian | **1** for all 30, at every noise level |

This is the dispatcher working without being told anything: K = 1 reduces the
method exactly to `gauss_hierarchical`'s summary, and that is what it picks
wherever a single Gaussian is sufficient.

## Global posterior, W1/σ against each benchmark's own reference

| benchmark | gauss_hier | gauss_jac | Langevin F-NPSE | **mixture (K inferred)** |
|---|---|---|---|---|
| inverse kinematics n = 8 | 0.522 | 0.497 | 0.237 | **0.169** |
| Gaussian mixture hierarchy | **0.064** | 0.165 | 0.362 | 0.203 |
| obs noise 0.05 | 0.116 | 0.129 | 0.507 | **0.027** |
| obs noise 0.1135 | 0.078 | 0.117 | 0.475 | **0.023** |
| obs noise 0.171 | **0.042** | 0.046 | 0.368 | 0.074 |
| obs noise 0.5848 | **0.011** | 0.041 | 0.341 | 0.050 |
| obs noise 2.0 | 0.246 | 0.245 | 0.378 | **0.127** |

Reference is the ABC posterior for inverse kinematics and the **exact
closed-form Gaussian** for the shared/local hierarchy.

Mixture wins 4 of 7, loses 3, and its worst case (0.203) is better than
`gauss_jacobian`'s (0.497) and `langevin_fnpse`'s (0.507). Where it loses it
loses by 0.03–0.14, never by a factor.

## Locals

Inverse kinematics, endpoint error (reference-free — the forward map is
deterministic, so this is the distance from a drawn arm's tip to the
observation it was conditioned on):

| method | endpoint mean |
|---|---|
| gauss_hierarchical | 0.0855 |
| gauss_jacobian | 0.0798 |
| langevin_fnpse | 0.0821 |
| **mixture product** | **0.0216** |

**Four times better than every rule**, and this is the metric a method cannot
game by getting the shared coordinate right while returning locals that do not
reproduce the data. On the shared/local hierarchy the local mean errors are
within 0.005 of each other for the three non-F-NPSE methods — that hierarchy's
locals are easy and nothing separates them.

## Cost

| benchmark | gauss_hier | gauss_jac | F-NPSE | mixture |
|---|---|---|---|---|
| inverse kinematics (4000 draws, n = 8) | 10.0 s | 11.4 s | 17.0 s | **35.9 s** |
| shared/local (3000 draws, n = 30) | 29 s | 33 s | 55 s | **100 s** |

**Roughly 3× `gauss_hierarchical`, 2× F-NPSE.** The cost is not the mixture
arithmetic — the product plus guarded pruning is ~10 ms. It is that the method
makes *two* ordinary batched sampler calls (the per-observation posteriors, then
the locals conditioned on the drawn shared value) where a composition rule makes
one composed pass. The EM stage is ~3 s at these settings.

That is the honest trade: 2–3× wall clock for a method that is 3× better on the
bimodal benchmark's shared coordinate, 4× better on its locals, and reduces to
the existing rule's summary by itself wherever the extra components are not
needed.

## Caveats

- One dataset and one seed per benchmark; the EM fit noise measured in
  `docs/mixture_product.md` (follow-up 2) is ±0.05 in W1/σ, so differences
  below that are not resolved.
- The `outside_support` diagnostic reads 1.1e-2 on the inverse-kinematics run —
  small, but not zero, so a little composed mass sits outside the intersection of
  the per-observation sampled ranges. Worth watching, not currently alarming.
- Every shared block here is one-dimensional. The dimension-free claim for the
  mixture product remains structural and unmeasured.
