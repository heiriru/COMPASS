# Arruda et al. (ICLR 2026) benchmarks, run with COMPASS composition rules

Paper: `Papers/Compositional_score_dumping.pdf` (arXiv:2505.14429, "Compositional
amortized inference for large-scale hierarchical Bayesian models"). Their code:
github.com/bayesflow-org/hierarchical-abi. Simulators and priors here were checked
against it.

| Paper experiment | What it is | Runnable quickly? | Script |
|---|---|---|---|
| 1. Gaussian toy | eta in R^10, Y_j ~ N(eta, 0.1^2 I), prior N(0, 0.1^2 I), J = 1..100k, analytic posterior | **yes** | `gaussian_toy.py` |
| 2. Hierarchical AR(1) grid | globals (alpha, beta, log sigma), one local per grid cell, T=5, grids 4x4..128x128 | **yes up to 16x16**, 64x64 costs ~16 min/dataset/rule, 128x128 not quick | `ar1_grid.py` |
| 3. FLI microscopy | real data, >750k parameters | no (real data and a 7.6 h/model training in the paper) | - |

## Things that had to be fixed before any number meant anything

1. **Diffusion floor.** `Trainer` hard-codes eps=1e-3 (lambda_min ~ 0.032), but
   the toy posterior sd is 0.032 at J=1000 and 0.010 at J=10k. Both scripts
   train and sample with eps=1e-5 (patched per run, library untouched).
2. **Network parameterization.** The plain network had relative score error
   10x at lambda=0.003, 2.4x at 0.01 and slope 0.38 at 0.1. Every rule was then
   1.7-2.8 sd off at J=10. `common.py` wraps it in
   `ModeTree/experiments/residual_score.PreconditionedScore` (Gaussian-prior
   residual, EDM input scaling, log-lambda time). After that the error is 3% at
   lambda=0.1, 6% at 0.03 and 13% at 0.01 (`gaussian_toy.py --probe`).
3. **Checkpoint selection.** The denoising validation loss is dominated by
   noise, so the best-validation checkpoint was epoch ~3. Both scripts train a
   fixed number of epochs and load `Model_final.pt`.
4. **gauss_jacobian OOM.** Its JVPs run under `torch.enable_grad()`. With
   trainable parameters each one records a full autograd graph (4.4 GB at 2,000
   rows). `common.freeze` fixes this at 1.2 GB.

## Arruda's method: `bayesflow_arruda` (`bayesflow_arruda.py`, `run_bayesflow.sh`)

Their estimator as shipped in BayesFlow 2.0.13 (vendored in `Arruda/.vendor`),
`CompositionalApproximator.compositional_sample`, with the paper's settings:

- Network: 5x256 MLP, dropout 0.1, cosine VP schedule, velocity prediction,
  likelihood weighting, AdamW 5e-4 with cosine decay, 1000 epochs, batch 128.
- Data: toy uses 10k fresh simulations per epoch in paper units; AR(1) uses a
  fixed 10k simulations.
- Sampling: `two_step_adaptive` with at most 10k steps, mini-batches of 10% of
  J, bridge d0=1.
- Prior score: passed without time, so BayesFlow applies the paper's (1-t)
  weighting.
- Tuning (as in the paper): d1 in {1, 0.1, 0.01, 1e-3, 1e-5} x shift s in
  {0, 2, 5, 10}. The pair is chosen by RMSE + calibration error on separate
  tuning datasets, then scored on the same test datasets and references as
  every other row.
- AR(1): BayesFlow composes only the globals, so the locals come from a
  second approximator for theta_j | eta, y_j. This is the paper's Eq. 12
  two-stage sampler.
- Caveat: the paper's own runs used `hierarchical-abi`, not this package.

## Results (`python gaussian_toy.py --report`, `python ar1_grid.py --report`)

Toy, standardized units, 10 datasets x 500 draws. `exact` is the Monte-Carlo
floor: exact posterior draws scored the same way.

| J | method | KL | cal. err | mean err / sd | width ratio |
|---|---|---|---|---|---|
| 10 | exact | 0.071 | 0.130 | 0.04 | 1.001 |
| 10 | gauss / full_gaussian = gauss_hierarchical | 0.093 / 0.095 | 0.13 | 0.07 | 1.01 |
| 10 | gauss_jacobian | 0.086 | 0.136 | 0.06 | 1.006 |
| 10 | gauss_jacobian_FD | 0.086 | 0.135 | 0.06 | 1.005 |
| 10 | gauss_jacobian_FD_r5 | 0.086 | 0.136 | 0.06 | 1.006 |
| 10 | gauss_jacobian_FD_r9 / r9_big | 0.085 / 0.085 | 0.135 | 0.06 | 1.005 |
| 10 | langevin_fnpe | 0.096 | 0.134 | 0.06 | 1.038 |
| 10 | bayesflow_arruda (d1=1, s=0) | 1.22 | 0.162 | 0.21 | 0.729 |
| 100 | exact | 0.071 | 0.074 | 0.04 | 1.001 |
| 100 | gauss | 0.26 | 0.089 | 0.17 | 1.02 |
| 100 | gauss_hierarchical | 0.39 | 0.084 | 0.20 | 1.02 |
| 100 | gauss_jacobian | 0.37 | 0.089 | 0.20 | 1.008 |
| 100 | gauss_jacobian_FD | 0.37 | 0.089 | 0.20 | 1.007 |
| 100 | gauss_jacobian_FD_r5 | 0.37 | 0.089 | 0.20 | 1.008 |
| 100 | gauss_jacobian_FD_r9 / r9_big | 0.37 / 0.38 | 0.089 / 0.079 | 0.20 | 1.007 / 1.013 |
| 100 | langevin_fnpe | 0.24 | 0.076 | 0.15 | 1.035 |
| 100 | bayesflow_arruda (d1=1e-3, s=0) | 1.75 | 0.130 | 0.42 | 0.828 |
| 1000 | exact (5 datasets x 200 draws) | 0.18 | 0.118 | 0.06 | 0.996 |
| 1000 | gauss | 6.7 | 0.186 | 0.95 | 1.021 |
| 1000 | gauss_hierarchical | 14.7 | 0.277 | 1.49 | 0.962 |
| 1000 | gauss_jacobian | 14.8 | 0.286 | 1.49 | 0.939 |
| 1000 | gauss_jacobian_FD | 14.8 | 0.288 | 1.49 | 0.936 |
| 1000 | gauss_jacobian_FD_r5 | 14.8 | 0.286 | 1.49 | 0.939 |
| 1000 | gauss_jacobian_FD_r9 / r9_big | 14.7 / 14.7 | 0.284 / 0.287 | 1.48 / 1.50 | 0.942 / 0.931 |
| 1000 | langevin_fnpe | 1.7 | 0.131 | 0.48 | 1.028 |
| 1000 | bayesflow_arruda (d1=1e-3, s=2) | 229 | 0.435 | 5.96 | 0.938 |

AR(1), 10k training simulations (the paper's budget), 100 draws. Global RMSE is
normalized by the range of the true values, as in the paper's Table 2 (their
4x4: NUTS 0.07, theirs 0.08; 16x16: NUTS 0.03, theirs 0.08).

| grid | method | global RMSE | contraction | mean err / ref sd (alpha, beta, log sigma) | local RMSE |
|---|---|---|---|---|---|
| 4x4 | reference | 0.062 | 0.935 | 0.04, 0.07, 0.04 | 0.145 |
| 4x4 | gauss_hierarchical | 0.055 | 0.928 | 0.44, 0.46, 0.71 | 0.159 |
| 4x4 | gauss_jacobian | 0.060 | 0.922 | 0.42, 0.55, 0.59 | 0.161 |
| 4x4 | gauss_jacobian_FD | 0.061 | 0.924 | 0.42, 0.49, 0.64 | 0.157 |
| 4x4 | gauss_jacobian_FD_r5 | 0.060 | 0.922 | 0.42, 0.55, 0.59 | 0.161 |
| 4x4 | gauss_jacobian_FD_r9 (= r9_big) | 0.064 | 0.916 | 0.48, 0.55, 0.59 | 0.164 |
| 4x4 | langevin_fnpe | 0.069 | 0.904 | 0.53, 0.49, 0.85 | 0.205 |
| 4x4 | bayesflow_arruda (d1=0.01, s=0) | 0.097 | 0.849 | 0.60, 1.13, 0.82 | 0.164 |
| 16x16 | reference | 0.016 | 0.997 | 0.03, 0.03, 0.03 | 0.062 |
| 16x16 | gauss_hierarchical | 0.037 | 0.993 | 1.39, 1.52, 1.00 | 0.075 |
| 16x16 | gauss_jacobian | 0.040 | 0.989 | 0.81, 1.97, 2.19 | 0.065 |
| 16x16 | gauss_jacobian_FD | 0.041 | 0.989 | 0.77, 1.93, 2.27 | 0.065 |
| 16x16 | gauss_jacobian_FD_r5 | 0.040 | 0.989 | 0.81, 1.99, 2.19 | 0.065 |
| 16x16 | gauss_jacobian_FD_r9 / r9_big | 0.043 / 0.041 | 0.988 / 0.991 | 0.68, 1.79, 2.20 / 0.78, 1.69, 2.18 | 0.066 / 0.065 |
| 16x16 | langevin_fnpe | 0.026 | 0.995 | 1.39, 1.62, 1.24 | 0.077 |
| 16x16 | bayesflow_arruda (d1=0.01, s=5) | 0.087 | 0.923 | 1.43, 2.28, 4.45 | 0.072 |
| 64x64 | reference (3 datasets) | 0.009 | 1.000 | 0.04, 0.05, 0.07 | 0.089 |
| 64x64 | gauss_jacobian (3 datasets) | 0.100 | 0.998 | 12.3, 6.0, 20.9 | 0.097 |
| 64x64 | gauss_jacobian_FD (3 datasets) | 0.099 | 0.999 | 11.7, 5.8, 20.8 | 0.097 |
| 64x64 | gauss_jacobian_FD_r5 (3 datasets) | 0.100 | 0.998 | 12.3, 6.0, 20.9 | 0.097 |
| 64x64 | gauss_jacobian_FD_r9 / r9_big (3 datasets) | 0.100 / 0.100 | 0.998 | 11.5, 5.9, 20.9 / 11.8, 6.6, 20.7 | 0.098 |
| 64x64 | gauss_hierarchical (3 datasets) | 0.122 | 0.999 | 14.3, 6.8, 23.4 | 0.098 |
| 64x64 | langevin_fnpe (3 datasets) | 0.149 | 1.000 | 2.1, 17.1, 25.1 | 0.110 |
| 64x64 | bayesflow_arruda (d1=0.1, s=2; 3 datasets) | 0.236 | 1.000 | 57.5, 13.6, 38.5 | 0.098 |

`gauss_jacobian_FD` runs with refresh 1 at high noise and 10 below lambda 0.3,
the Partial_Pooling ladder setting. `gauss_jacobian` runs with refresh 5
throughout. The two agree within Monte-Carlo noise everywhere. On this shared
GPU the FD rule was slower in wall-clock time except at toy J=100.
Per-dataset times: 94 vs 45 s (toy J=10), 366 vs 442 s (J=100),
2415 vs 1691 s (J=1000), 37 vs 9 s (4x4), 192 vs 60 s (16x16), and
3312 vs 974 s (64x64).
`gauss_jacobian_FD_r5` (FD with refresh 5, the same schedule as
`gauss_jacobian`) reproduces `gauss_jacobian` to the printed digits everywhere.
Its per-dataset times are 26 s (J=10), 305 s (J=100), 896 s (J=1000), 13 s (4x4),
56 s (16x16) and 1236 s (64x64). The FD slowdown above came from the refresh-1
schedule, not from the finite differences.

`gauss_jacobian_FD_r9` (refresh 9) and `gauss_jacobian_FD_r9_big` (refresh 9,
row cap 96k instead of 6k; `FD_MAX_ROWS`) stay within Monte-Carlo noise of r5.
Seconds per dataset, r5 / r9 / r9_big / gauss_hierarchical:

| | toy 10 | toy 100 | toy 1000 | 4x4 | 16x16 | 64x64 |
|---|---|---|---|---|---|---|
| r5 | 26 | 305 | 896 | 13 | 56 | 1236 |
| r9 | 18 | 174 | 658 | 4.7 | 36 | 650 |
| r9_big | 18 | 160 | 620 | 3.9 | 28 | 446 |
| gauss_hierarchical | 14 | 103 | 2322 | 8.7 | 102 | 4248 |

gauss_hierarchical needs a single chunk only at toy 10/100 and 4x4. There it
re-estimates its 500-draw pilot per chunk, so its times at toy 1000, 16x16 and
64x64 are inflated. The FD cost grows with the latent dimension: D+1 = 11
copies on the toy and 5 on AR(1).

The AR(1) reference needs no Stan. The likelihood is Gaussian in
(alpha, theta_j) with five sufficient statistics per group, so each group's
marginal likelihood is a 1-D grid integral. The 3-D global posterior is sampled
by importance sampling from a Student-t Laplace proposal (ESS ~3000/4000), and
two disjoint halves of it agree to 0.03-0.07 sd.

All queued runs are complete. Not run: toy J=10,000/100,000 and AR(1) 128x128
(many GPU-hours each with every row evaluated per step).
