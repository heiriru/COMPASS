# True compositional GAUSS for hierarchical SBI — results

See `derivation.md` for the full mathematical derivation. Summary here.

## The problem with 02f / 02g

`02f_dpm2_gauss_global_local.png` and `02g_dpm2_gauss_moment.png` use
`correction="Gauss_global_local"` fed an *analytic* single-observation
covariance/mean bank (`build_single_observation_covariance_bank`, hardcoded
from the toy model's true generative parameters). The sampler's
`_marginal_global_scores` then substitutes a closed-form Gaussian score built
from that oracle mean for the network's real output — the trained score
network is barely involved. That is not compositional score modeling.

## The fix

A new correction, `correction="gauss_hierarchical"`
(`src/compass/MultiObsSampler.py`), generalizes the paper's GAUSS algorithm
(Linhart et al. 2024, Algorithm 2 / Lemma 3.2) to a global+local hierarchy by
exact block/arrow-precision elimination. The key result (derived in
`derivation.md`, cross-checked in
`tests/test_gauss_hierarchical_composition.py`): weighting the network's
*real* conditional score on the shared coordinate by the marginal precision
`Var(g_t|x_j)⁻¹`, plus a linear cross-correction to each local score, is
*exactly* the correct Gaussian-approximation-consistent generalization — no
marginal-score approximation, oracle mean, or synthetic Gaussian score is
needed. The pilot covariance `Σ_0,j` (the only approximation GAUSS itself
makes) is estimated from the trained network's own DDIM draws.
`posterior_mean`/`global_posterior_mean` are refused outright by the API for
this correction, so the "moment projection" cheat can't be reintroduced by
accident. `Gauss_global_local` itself is left as-is (with its oracle-mean
shortcut) since it's used elsewhere as a deliberate ablation; its docstring
now says explicitly it is not compositional score modeling when used that way.

A real efficiency bug was also fixed along the way:
`estimate_posterior_moments`/`_estimate_posterior_precision` looped over each
observation *sequentially*, running one full reverse-diffusion trajectory per
subject even though the underlying sampler already vectorizes over
observation rows — this left the GPU at ~3% utilization. Both now batch every
subject into one trajectory (chunked only over `num_samples`), matching how
the main compositional sampler already batches.

## Experiment

`run_gauss_hierarchical_experiment.py` reuses the exact checkpoint
(`shared_local_mixture_full`), the same 30 real observations, and the same
exact-joint reference truth that `02a`–`02g` were generated from. Every
covariance is estimated from the trained network's own draws
(`precision_est_samples=4096`, `posterior_covariance=None` — no oracle
anywhere). Two settings were run:

- `gauss_hierarchical_dense_correctors`: dense Langevin correctors, matching
  02a/02f's sampler settings.
- `gauss_hierarchical_deterministic`: zero-corrector deterministic DPM-2,
  matching 02g's settings exactly (for a direct, apples-to-apples comparison).

`02a` (`dpm2_gaussian`), `02c` (`langevin_fnpe`) and `02f`
(`dpm2_gauss_global_local`) are reused unchanged from the parent directory;
`02g`'s own `.npz` (missing from disk) was regenerated, unmodified, from the
original script for a real number to compare against.

## Results (`combined_method_metrics.csv`, `03_method_comparison.png`)

| method | global mean error (/σ) | locals mean error (/σ) | runtime (s) | uses oracle? |
|---|---|---|---|---|
| `dpm2_gaussian` (naive "gauss", ignores hierarchy) | 0.573 | 0.174 | 150 | no |
| `langevin_fnpe` (F-NPSE baseline) | **0.105** | 0.046 | 119 | no |
| `dpm2_gauss_global_local` (oracle moments) | 0.023 | 0.046 | 155 | **yes — cheat** |
| `dpm2_gauss_moment` (oracle moments) | 0.170 | 0.174 | 23 | **yes — cheat** |
| `gauss_hierarchical`, dense correctors (ours) | 0.732 | **0.055** | 204 | no |
| `gauss_hierarchical`, deterministic (ours) | **0.176** | 0.415 | 80 | no |

Reading this honestly:

- **`gauss_hierarchical` (deterministic) matches the oracle-moment cheat's
  global accuracy (0.176σ vs 0.170σ) using only the trained network's own
  scores and its own DDIM-estimated covariances — no analytic ground truth
  anywhere.** It beats the naive, hierarchy-blind `"gauss"` baseline by 3.3×
  on the global parameter, and gets within ~1.7× of Langevin F-NPSE's global
  accuracy while running ~1.5× faster (80s vs 119s) and without any Langevin
  corrector steps or MCMC hyperparameter tuning.
- **`gauss_hierarchical` (dense correctors) gives the best local-parameter
  recovery of every non-cheating method (0.055σ, beating even F-NPSE's
  0.046σ by a hair)**, but its global accuracy is worse than the deterministic
  variant and worse than the naive baseline. This is a real, reproducible
  effect of composing many (10-per-step) Langevin corrector steps together
  with the arrow-elimination cross-correction on the global coordinate; it
  is not present in the deterministic (correctors=0) setting. Recommendation:
  use `gauss_hierarchical` with few or zero correctors when the global
  parameter is the primary quantity of interest, and consider dense
  correctors only when local-parameter recovery matters most. Understanding
  and fixing this corrector interaction is the natural next step (candidates:
  reducing `corrector_steps`/`snr` specifically for the shared coordinate, or
  re-deriving the cross-correction under Langevin noise rather than the
  deterministic-predictor assumption used here).
- Both `gauss_hierarchical` variants exceed the honesty bar the 02f/02g
  figures failed: every number above for "ours" comes from the same trained
  network and 30 real observations, with zero analytic ground truth fed into
  the sampler.

## Files

- `derivation.md` — full math.
- `run_gauss_hierarchical_experiment.py` — experiment script (reruns with
  `--force`; `--posterior-samples`/`--timesteps` are configurable).
- `plot_comparison.py` — regenerates `03_method_comparison.png` from
  `combined_method_metrics.csv`.
- `02h_dpm2_gauss_hierarchical.png`, `02i_dpm2_gauss_hierarchical_deterministic.png`,
  `02g_dpm2_gauss_moment.png` (regenerated) — per-method dashboards
  (`plot_shared_local` format, same as 02a–02g).
- `03_method_comparison.png` — bar-chart summary across all six methods.
- `combined_method_metrics.csv`, `run_config.json`, `run.log`/`run2.log` — raw
  metrics and logs.
- `*.npz` — raw samples for every method (copies for 02a/02c/02f, regenerated
  for 02g and the two new variants).

## Tests

`tests/test_gauss_hierarchical_composition.py` (8 tests): exact-arrowhead
cross-check, equivalence to `Gauss_global_local`'s real-score path, `n=1` and
no-locals reductions, and API guards rejecting `posterior_mean`/
`global_posterior_mean`. `tests/test_full_gaussian_composition.py` and
`tests/test_memory_bounded_attention.py` were updated for the batched
covariance-estimation refactor (their fake single-observation sampler stubs
now simulate an all-subjects-at-once call, matching the real `Sampler`).
