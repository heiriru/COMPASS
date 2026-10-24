# Integration branch `integration-2026-10-24`

One branch that carries every line of COMPASS work in progress on 2026-10-24, stacked on top of
each other so that they are tested together.

| layer | source | what it brings |
|---|---|---|
| base | `lambda-s-pd-study` @ 65ea8a2 (pushed) | Λ_S PD strategies, metric clamp, Cholesky screen and eigh precision for the gauss_jacobian path, PD study rounds 1–2 |
| merged | `library-roadmap-2x` | roadmap 2.1–2.6 library changes, menu + EDM `auto` defaults, `ScorePrior`, latent attention in the library, the experimental Fisher penalty and x0 parameterisation (`docs/roadmap2.md`) |
| merged | `jac-extensions` | JAC plan items 1.8–1.12, all opt-in: N-level JAC/GH, covariance head, observation mini-batched JAC (`jacobian_row_batch`), conservative score, mode-aware JAC (`jac_extensions/README.md`) |
| merged | `jax-jacobian-prototype` | JAX vs PyTorch score-Jacobian prototype (`jax_jacobian/`) |
| snapshot | main tree's **uncommitted** work on `lambda-s-pd-study` (base 65ea8a2, taken 2026-10-24 13:51) | compact `HierarchicalState` sampler state (`src/compass/hierarchical_state.py`), WAIC weight experiment 5, PD-study screen updates, partial-pooling fair-comparison scripts, the GCE application code (without the vendored Chempy, 14 MB, which stays local in `GCE/vendor/`) |

The snapshot was work in progress when it was taken. The agent working in the main tree may
still change it. Its patch and file archive are kept at
`/export/data/rheinric/roadmap2/snapshot_maintree_20261024_1351/`.

## Conflicts resolved

* `MultiObsSampler.py`, `jac-extensions` vs PD study: both appended methods at the same place
  (the covariance head vs the Cholesky screen and eigh precision). Both were kept.
* `MultiObsSampler._get_score`, `jac-extensions` vs the snapshot: both changed the start of the
  function. The merged function takes the snapshot's signature (`parts=False`) and its
  `HierarchicalState` path. It then dispatches to `jacobian_row_batch` for dense input.
  `jacobian_row_batch` together with `parts=True` raises `NotImplementedError` instead of
  returning the wrong shape.

## Adding later work

Each line of work keeps its own branch. To fold in new commits:

```bash
git checkout integration-2026-10-24
git merge --no-ff <branch>          # e.g. library-roadmap-2x, jac-extensions
PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests --ignore=tests/test_multiobs_analytic.py
```

Once the main-tree work is committed on `lambda-s-pd-study`, merge that branch. Git will see the
snapshot's identical hunks as already applied. Re-check `MultiObsSampler._get_score` wherever the
snapshot and the final version differ.

## Default clash fixed after merging

`jac-extensions` was written when `jacobian_refresh` defaulted to 1, and `jacobian_row_batch`
refused any other value. The PD-study line made 9 the default, so every row-batched call failed.
`jacobian_row_batch` now accepts the default period (it replaces it) and still refuses explicit
other periods. It also runs on the compact `HierarchicalState` through the dense path that
subclasses use. Its cost test now compares against `jacobian_refresh=1`, the reference it was
written for.

## Test status (CPU, without `test_multiobs_analytic.py`)

* 16 failed, 501 passed, 8 skipped before the fix above; **14 failed, 503 passed, 8 skipped** after it (full suite, 2026-10-24).
* After the fix, the three new failures are down to one:
  `test_pd_strategies.py::test_consistent_gate_switches_to_the_default_above_the_threshold`.
  It fails identically in the main tree. It is the in-progress PD-study work: the test expects
  `pd_consistent_below`, which is not written yet.
* The other 13 failures predate all of this:
  * `test_instantaneous_divergence.py` ×7 and `test_pfode_vpsde_analytic.py` ×5 target features
    that live on other branches.
  * `test_memory_bounded_attention.py` ×1 is an outdated test stub.

## Library defaults on this branch (2026-10-24)

**Sampling** (`SBIm.sample`, `MultiObsSampler.sample`; the MAP estimators keep `jacobian_refresh=1`):

* Composition rule `gauss_jacobian_FD` with `jacobian_refresh=9`.
* Compact `HierarchicalState` (`set_state_layout`) with the `legacy` noise stream, so seeded
  runs reproduce the dense sampler.
* Elimination Schur route with the tiny-matrix kernels (`set_schur_mode`), and the reused FD
  base point.
* Per-row eigendecompositions in float32, in chunks of 1,024 (`set_eigh_precision`).
* Young-generation garbage collection after every network call (`set_network_gc`); no row
  budget (`set_network_row_budget`).
* `eps` is forwarded to the samplers. Non-Gaussian priors are passed as
  `prior=ScorePrior(...)`.

**Training** (`SBIm.train`):

* `mask_menu="auto"` and `time_loss_weight="auto"`: menu + EDM whenever `mask_roles` or a
  `mask_sampler` is given; Bernoulli and the plain ε-loss otherwise.
* `validation_noise_seed=20250719`, `early_stopping_patience=100`, `restore="final"`.
* Latent attention is a constructor option: `SBIm(latent_attention="hierarchical",
  latent_roles=...)`.

**Opt-in:**

* `time_eps`
* global-only TSM (`global_target_fn`)
* `jacobian_row_batch`
* the covariance head
* the PD strategies (`set_pd_strategy`) and the metric clamp (`set_clamp_mode`)
* everything under `compass.experimental`, plus `compass.consistency` and
  `compass.parameterization`
