# Hierarchical Partial-Pooling Benchmark

This directory contains a reproducible native-COMPASS benchmark for hierarchical
partial pooling. It has no BayesFlow dependency. The scientific DDM equations,
priors, transforms, constants, and plotting palette were copied and adapted from
[`case_study4`](https://github.com/bayesflow-org/diffusion-experiments/tree/main/case_study4)
at revision `363186e485add4f062b2cf33f356ffa215f07056`; the source URL and revision are
also written to every stage manifest.

## Scientific model

The hierarchical parameter order is

`(mu_nu, mu_log_alpha, mu_log_t0, log_sigma_nu, log_sigma_log_alpha,
log_sigma_log_t0, beta_raw, nu, log_alpha, log_t0)`.

The first seven coordinates are shared. Physical subject parameters are

`nu`, `alpha = exp(log_alpha)`, and `t0 = exp(log_t0)` are subject-specific.
The shared starting bias is
`beta = Beta(50,50)^-1(Phi(beta_raw))`. Subject parameters are drawn from the
three group means and exponentiated group log standard deviations.

The raw SDE tensor is `(instances, subjects, trials, 3)` with fields
`(choice, reaction_time, censored)`. Reported RT contains `tau`, while the
10-second censoring limit applies only to decision time. COMPASS receives a
deterministic trial-major flattening of `(choice, log_rt, censored)`.

The adapted simulator retains a direct single-trial Euler--Maruyama reference and
a batched active-mask implementation. All randomness uses explicit named NumPy
generators derived from one root seed. Training, validation, test, dataset,
subject, posterior, and mini-batch namespaces therefore remain independent.

## Model and inference

The focused pipeline trains one joint SDE checkpoint with seven shared global
coordinates and three subject-local coordinates. Inference composes only indices
`0..6`, verifies that the sampler keeps those draws synchronized across subjects,
and compares posterior means, intervals, and coverage directly with simulator
truth. Global and local recovery are written separately.

The backward-compatible composition names are:

- `legacy_mean`: the historical supplied-prior score plus mean subject score
  (with the required ordinary-score boundary at `R=1`);
- `prior_corrected_sum`: `(1-R) s_prior,t + sum_r s_r`;
- `damped_sum`: the prior-corrected sum multiplied by the constant `1/R`;
- `minibatch_damped`: the same constant damping with the unbiased
  `(R/M) sum_(r in B) s_r` estimator and default `M=min(3,R)`.

The noised Gaussian prior has mean `alpha(t) mu` and variance
`alpha(t)^2 sigma_prior^2 + sigma(t)^2`. VE and VP samplers share the existing
`alpha_t`/`lambda_t` abstraction. The joint model evaluates every selected subject
row so each local state receives an update while the seven global states remain
synchronized.

## Reproduction

Every executable applies a hard Linux affinity cap of at most 6% of host CPUs
and calls `autocvd(num_gpus=1, interval=1)` before importing numerical packages.
The selected physical GPU is exposed to PyTorch as `cuda:0`. From the repository
root, no external `autocvd` shell command is needed:

```bash
python Partial_Pooling/create_training_data.py --preset full
python Partial_Pooling/create_test_data.py --preset full
python Partial_Pooling/train_models.py --preset full --device cuda
python Partial_Pooling/infer_partial_pooling.py --preset full --device cuda
```

To train and evaluate the same benchmark with the default VPSDE schedule
`beta_min=0.1`, `beta_max=20`, reuse the existing simulations and select VPSDE
for both model training and inference:

```bash
python Partial_Pooling/train_models.py \
  --preset full --sde-type vpsde --device cuda
python Partial_Pooling/infer_partial_pooling.py \
  --preset full --sde-type vpsde --inference-method dpm2_gaussian --device cuda
```

Use `--beta-min` and `--beta-max` on both commands for another schedule. VE keeps
the legacy checkpoint names; VP checkpoints include the beta schedule in their
names, so the two model families cannot overwrite one another. Simulation data
and normalizers are SDE-independent and are shared after compatibility checks.

To retrain only the joint partial-pooling model with 100,000 training and 5,000
validation simulations, use the isolated `large` artifact namespace:

```bash
python Partial_Pooling/create_training_data.py --preset large
python Partial_Pooling/train_models.py --preset large --model sde_joint --device cuda
python Partial_Pooling/create_test_data.py --preset large

# Primary global inference: compositional DPM-Solver-2 + Gaussian
python Partial_Pooling/infer_partial_pooling.py \
  --preset large --inference-method dpm2_gaussian \
  --gaussian-precision-batch-size 128 --device cuda

# Full-covariance variant: compositional DPM-Solver-2 + full Gaussian
python Partial_Pooling/infer_partial_pooling.py \
  --preset large --inference-method dpm2_full_gaussian \
  --gaussian-precision-batch-size 128 --device cuda

# Global/local moment variant: composed scores with posterior mean + covariance
python Partial_Pooling/infer_partial_pooling.py \
  --preset large --inference-method dpm2_gauss_global_local_moment \
  --gaussian-precision-batch-size 128 --device cuda

# Reference global inference: compositional Langevin + F-NPSE
python Partial_Pooling/infer_partial_pooling.py \
  --preset large --inference-method langevin_fnpse --device cuda
```

`train_models.py` contains only the `sde_joint` specification, so this sequence
does not train flat, ancestral, choice-only, or full-observation alternatives.
The existing `full` checkpoint and results are left untouched.

`infer_partial_pooling.py` is the only inference entry point. Its primary
profile is `dpm2_gaussian`: compositional score modeling with DPM-Solver order 2
and the Gaussian correction. The Gaussian single-observation precision is estimated
once per dataset, reused across posterior batches, saved, and reused for hierarchical
MAP refinement. The `dpm2_full_gaussian` profile uses the same inference and MAP
pipeline but estimates and reuses a full single-observation posterior covariance,
retaining correlations between global coordinates. The
`dpm2_gauss_global_local_moment` profile estimates the full ten-dimensional
single-subject posterior mean and covariance once per dataset, supplies both to
`Gauss_global_local` for every posterior batch, saves them in each dataset
artifact, and reuses them during hierarchical MAP refinement. The comparison profile is
`langevin_fnpse`: annealed Langevin
sampling with F-NPSE, ten updates per noise level, and SNR 0.1 by default. Because
F-NPSE bridging scores do not define a reverse-diffusion score-MAP objective, its
reported global mode is the joint KDE mode of its posterior global samples.

Practical defaults are five datasets, 20 subjects, 256 posterior draws, and 50
sampler levels. One COMPASS observation is one subject's vector of 30 trials. By
default the script also evaluates 1, 2, 4, 8, 16, and 20 subject observations; use
`--skip-observation-sweep` to disable that experiment. `--estimate-only` reports
the sampling workload without reserving a GPU. Each method has a distinct artifact
signature, and every dataset/count is saved independently for safe restart.

`--root` selects an artifact directory, `--seed` changes the root seed, and
`--force` explicitly replaces matching recovery artifacts. Matching training data,
test data, checkpoints, and completed recovery datasets are reused automatically;
mismatched configurations fail closed.

The smoke preset contains 2,048 training simulations, 256 validation simulations,
10 test datasets, 20 subjects, 30 trials, 128 posterior draws, and 20 diffusion
steps. The full preset contains 32,768/4,096 training/validation simulations,
100 test datasets, 100 subjects, 30 trials, up to 1,000 draws, and paper-scale
early-stopped training. The large preset retains the full model and test settings
but uses 100,000/5,000 training/validation simulations.

Artifacts are standard `.pt`, CSV, and JSON files under `artifacts/`.
Normalization is fitted only on training shards. Each normalizer records its
training provenance, per-feature center/scale, degenerate mask, and the explicit
unit-scale policy for degenerate features. Writes use a temporary sibling followed
by an atomic replace; recovery reports are refreshed after every completed dataset.

Artifact names expose their scientific method and scale. For example, the full
training set is indexed by `partial_pooling-train-32768.json`, its shards live in
`data/partial_pooling-train-32768/`, and its normalizers are stored in
`partial_pooling-train-32768-normalization.pt`. A current joint checkpoint is
stored as
`checkpoints/full/sde_joint-train-32768/sde_joint-train-32768-checkpoint.pt`.
The shared simulator test input is method-neutral but records the associated
training scale and its own size, for example
`partial_pooling-train-32768-test-100.pt`, because every inference method consumes
the same datasets.

Rename artifacts produced by an older checkout with the idempotent migration:

```bash
python Partial_Pooling/migrate_artifact_names.py
```

This is also safe to rerun after a previously started inference process finishes.

GPU memory is bounded by the configured model batch size. Validation uses the
training batch size without autograd, posterior draws are chunked according to the
number of subject rows, and the transformer uses the intended compact per-example
key-padding mask instead of materializing a
`(batch * heads, nodes, nodes)` attention mask.

Outputs are stored below
`artifacts/partial_pooling_recovery/<preset>/<inference-method>-<signature>/`.
Every recovery filename also carries the method, such as
`dpm2_gaussian-dataset-0000.pt`, `dpm2_gaussian-global_recovery.csv`, and
`dpm2_gaussian-summary.json`. `dpm2_gauss_global_local_moment` writes a
separate signed run directory and therefore never overwrites
`dpm2_gaussian`. Observation-count artifacts use the same method
prefix under `observation_sweep/subjects-XXXX/`. Tensors retained from the older
multi-method pipeline follow the same rule, for example
`posteriors/full/ancestral/ancestral-dataset-0000.pt`.

At the end of inference, nine figures are written to
`artifacts/figures/partial_pooling/<preset>/<inference-method>-<signature>/`.
The parity,
residual, shrinkage, and error-distribution figures use the joint MAP. The former
global forest filename now contains a 7-by-4 global posterior-density grid for
dataset 0 at 1, 4, 8, and 20 subjects, including simulator-truth, joint-MAP, and
posterior-median lines. Robust shared limits and KDE bandwidths prevent a minority
of finite solver excursions from flattening every panel; each panel explicitly
reports how many draws lie outside its displayed central range.
`error_vs_observations.png` shows training-normalized global and local MAP
RMSE, with the mean and a transparent +/-1 standard-deviation band across the five
datasets. The mock-data overview shows choices, reaction times, censoring, and true
global/local parameters. Every image uses a constrained layout and external legend.
A `manifest.json` records the estimator, observation unit/counts, and filenames.

After all inference methods have completed, regenerate comparison figures without
running posterior sampling or MAP estimation again. Pass the legacy damped-sum run
first so the unsuffixed combined figures are written alongside its existing figures:

```bash
python Partial_Pooling/plot_partial_pooling_comparison.py \
  --preset full \
  --run-signature a68895612490acc7 \
  --run-signature 48d770a95427da1c \
  --run-signature 46d973c007c4dc86 \
  --output-signature a68895612490acc7
```

The signatures above are, respectively, DPM-Solver-2 with damped-sum composition,
DPM-Solver-2 with Gaussian correction, and Langevin with F-NPSE for the documented
full-preset defaults. The command validates that every method has the same datasets
and complete observation counts before plotting. It creates method-suffixed parity,
residual, local-shrinkage, error-distribution, and global-density files. The
unsuffixed `error_distributions.png`, `error_vs_observations.png`, and
`global_forest.png` contain all methods together. `comparison_manifest.json`
records the exact input signatures and generated filenames.
