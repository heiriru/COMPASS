# Test-suite baseline before the Group/Sequence work

`python -m pytest tests/ -q` on 2026-09-04, before any source change:
**272 passed, 18 failed**, 341 s.

All 18 failures are **pre-existing and unrelated** to the schema/tokenized code.
Verified by: no source file was modified to produce them (the one temporary edit
to `TokenizedTransformer.py`, a deliberate negative control, was reverted and the
file confirmed byte-identical to its backup), and every failure names an API that
does not exist anywhere in `src/compass/`.

## The 18, by cause

| cause | count | files |
|---|---:|---|
| `equation=` kwarg (PF-ODE as a *sampler*) | 8 | `test_pfode_vpsde_analytic.py` (5), `test_multiobs_analytic.py` (3) |
| `divergence_head` / `instantaneous_divergence` | 7 | `test_instantaneous_divergence.py` |
| stale `SimpleNamespace` mock lacking `.schema` | 1 | `test_memory_bounded_attention.py` |
| CUDA/CPU device mismatch | 1 | `test_multiobs_analytic.py::test_gauss_correction_estimated_precision` |
| numeric quality assertion (1.152 < 0.5) | 1 | `test_multiobs_analytic.py::test_fnpe_langevin` |

`grep -rn "equation" src/compass/` returns nothing, and `ScoreBasedInferenceModel.load`
still carries an explicit tolerance for `"divergence_head."`-prefixed state-dict keys
as a "legacy leftover". Both features were dropped by `6bbb57c Restore pre-divergence
upstream source` while the tests that exercise them were kept by `17fb465 Restore
preserved tutorials and tests`.

## What this means for the plan

* **Tier 3.4 is viable.** The pieces it needs all pass: `test_pfode_logprob_vesde`,
  `test_pfode_logprob_vesde_hutchinson`, `test_pfode_logprob_vpsde` (the analytic
  linear-Gaussian log-density checks, exact *and* Hutchinson), and
  `test_gauss_correction_analytic_precision` plus `test_hierarchical_shared_and_local`
  (the analytic composition checks). Only the PF-ODE-*sampler* variants are broken.
* **`tutorials/Population_Dynamics.py` is currently broken**, for the same reason: its
  `COMPARISON_MODES` pass `equation="reverse_sde"` / `"probability_flow_ode"` into
  `compare()`, which no longer accepts them. It is the plan's chosen `Sequence`
  model-comparison vehicle (step 6), so either `equation=` is restored first or another
  vehicle is chosen. This does **not** block the `Group` work (steps 1-5).

## Tier-0 guard tests, all green (91 passed)

`test_legacy_arch_regression.py`, `test_tokenized_arch_regression.py` (new),
`test_schema_api.py`, `test_conditional_menu.py`, `test_budget_and_heads.py`,
`test_tokenized_stack_integration.py`.
