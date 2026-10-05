# Would JAX make gauss_jacobian's score Jacobian cheap?

A self-contained prototype, measured on 2026-10-05. Nothing in `src/compass` is changed.
The question was whether porting COMPASS to JAX would make the per-row score Jacobian
that `gauss_jacobian` needs fast enough to speed up inference.

**Short answer: JAX helps, by about 1.3x over our finite-difference (FD) path, and it gives
exact derivatives at FD cost. The real win is something the measurement turned up along
the way. For the trial-token network the latent Jacobian is exactly diagonal, so one
JVP, or one FD copy, gives all of it. That makes the Jacobian 5-7x cheaper than the
library's FD path, with no JAX needed.**

## Setup

- Network: trial-token H64/d2 (`sde_joint-tok-trial-h64d2x8m4-sub100000-train-400000`), the
  model the gauss_jacobian ladder runs used. VESDE, theta (10 dims) latent, 30 trials observed.
- Inputs: subjects of test dataset 0, normalized; theta noised at the given lambda.
- Reference: float64 PyTorch, 10 forward-mode JVPs (exactly `_row_score_jacobian`).
- JAX port (`jax_model.py`): inference-only, reads the PyTorch `state_dict`. It matches
  PyTorch to 1e-10 in float64, for both scores and Jacobians (`artifacts/jax_results_x64.json`).
- Timing: one RTX 2080 Ti picked by autocvd, float32, median of 30 calls after warm-up,
  at lambda = 0.3. PyTorch 2.12 (eager, plus `torch.compile`); JAX 0.11.2 in a separate venv at
  `/export/data/rheinric/jax_jacobian_env`.

## Results (20 rows x 256 draws = 5,120 network rows; full tables in `artifacts/summary.md`)

| Jacobian | framework | exact? | ms | vs library FD |
|---|---|---|---|---|
| 10 JVPs in a loop (`gauss_jacobian` today) | torch | yes | 1272 | 0.29x |
| 11 stacked copies (`gauss_jacobian_FD` today) | torch | 1e-3 rel. | 363 | 1x |
| `vmap(jvp)`, 10 tangents | torch | yes | OOM (185 ms at 1,280 rows) | — |
| `jacfwd` = `vmap(jvp)` | jax | yes | OOM (70 ms at 1,280 rows, ~280 if chunked) | ~1.3x |
| 10 JVPs, `lax.scan` | jax | yes | 527 | 0.69x |
| **one FD copy, all coords (diagonal)** | **torch** | 1e-3 rel. | **67** | **5.4x** |
| one JVP, all coords (diagonal) | torch | yes | 129 | 2.8x |
| **one JVP, all coords (diagonal)** | **jax** | **yes** | **52** | **7.0x** |
| for scale: one forward pass | torch / jax | | 33 / 28 | |

Every exact variant agrees with the float64 reference to about 1e-6. FD to 1e-3 to 5e-3,
for both frameworks.

## What the numbers say

1. **JAX's AD is not what makes the difference.** `jacfwd` is the same maths as batching our
   10 JVPs. On this GPU it costs about what the library's stacked FD costs (10 tangent passes
   against 11 copies). The predicted floor holds: D tangent columns cost about D forward
   passes in any framework.
2. **JAX is somewhat faster per pass.** The forward pass is 1.2x faster at 5,120 rows and
   4.7x faster at 120 rows, where PyTorch is bound by Python and launch overhead. Forward-mode
   AD through attention costs about 1.8 forward passes in JAX, against about 4 in PyTorch,
   where `jvp` has to run on the MATH attention backend.
3. **`torch.compile` changed nothing** at realistic batches (forward 33 ms either way). It
   helped only the 120-row forward pass (4.7 to 2.7 ms).
4. **The Jacobian of this network is exactly diagonal** (max off-diagonal/diagonal = 0.0 at
   every lambda). That follows from the trunk's construction: latent tokens are never
   attention keys (key-padding mask `1 - c`), the LayerNorm is per token (`LayerNorm(H)`),
   and each theta scalar is its own token. So theta_i's output depends only on theta_i and
   the observed trials. A tangent of all ones on the theta block therefore returns the whole
   Jacobian in one pass, and FD needs one shifted copy instead of ten.

## End-to-end estimate (not measured)

The 2026-09-29 ladder run took 7,499 s with FD (refresh 1 above lambda 0.3, 10 below),
against 4,976 s for `gauss_hierarchical`. So roughly 2,500 s of it was the Jacobian. A
5.4x cheaper Jacobian puts gauss_jacobian at about 5,400 s, within about 10% of
`gauss_hierarchical`. A full JAX port would add perhaps 1.2x more from the faster forward
pass, at the cost of maintaining a second implementation.

## Caveats

- The diagonal structure is a property of the **tokenized `shared` trunk with atomic theta
  tokens**. It does **not** hold for:
  - the legacy `ConditionTransformer`, whose `LayerNorm((N, H))` couples all tokens;
  - any model in which latents attend to latents (role-ordered or hierarchical attention);
  - grouped or set-encoder trunks.

  A library shortcut should be gated on the architecture, or check off-diagonals once on
  the first evaluation.
- A diagonal Jacobian also means that gauss_jacobian's Schur reduction and local
  cross-correction have nothing to act on for this network: no global-local coupling enters
  through the curvature. That deserves a look on its own (see the memory note on the
  mean-field tokenized trunk).
- Batches above about 1,300 rows with 10 stacked tangents or copies run out of the GPU's 11 GB
  in both frameworks. The library already chunks FD (`FD_JACOBIAN_MAX_ROWS`).

## Files

- `common.py`: constants, VESDE helpers, the CPU and GPU guard (3 threads, autocvd).
- `export_reference.py`: weights to npz, float64 reference (torch, CPU).
- `torch_variants.py`, `run_torch.py [--compile]`: PyTorch variants and their timing.
- `jax_model.py`, `run_jax.py [--x64 --cpu]`: JAX port, variants and timing.
- `timing.py`, `summarize.py`: helpers and tables; results are in `artifacts/*.json` and `artifacts/summary.md`.

Re-run with:

```
../../COMPASS/.COMPASS/bin/python export_reference.py
../../COMPASS/.COMPASS/bin/python run_torch.py; ../../COMPASS/.COMPASS/bin/python run_torch.py --compile
XLA_PYTHON_CLIENT_PREALLOCATE=false /export/data/rheinric/jax_jacobian_env/bin/python run_jax.py
JAX_PLATFORMS=cpu /export/data/rheinric/jax_jacobian_env/bin/python run_jax.py --x64 --cpu
/export/data/rheinric/jax_jacobian_env/bin/python summarize.py > artifacts/summary.md
```
