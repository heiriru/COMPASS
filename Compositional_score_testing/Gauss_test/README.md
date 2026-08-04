# Tall-data Figures 1–4 in COMPASS

This directory recreates Figures 1–4 of *Diffusion posterior sampling for
simulation-based inference in tall data settings* with the three paper-comparison methods below. Figure 1 additionally includes
the posterior-moment variant:

- **F-NPSE Langevin**: the paper's baseline (400 diffusion levels, 5 ULA steps
  per level, `tau=0.5`, and the published step-size schedule).
- **COMPASS GAUSS (diagonal)**: the existing `MultiObsSampler` correction with
  `correction="gauss"` and marginal posterior precisions.
- **COMPASS full Gaussian**: the paper's full-covariance Algorithm 2 correction
  with `correction="full_gaussian"` and matrix-valued covariance estimates.
- **COMPASS global/local Gaussian + posterior moments (Figure 1)**: the
  covariance-aware `correction="Gauss_global_local"` supplied with the exact
  single-observation posterior mean and covariance. This reconstructs each
  Gaussian score before compositional summation.

All COMPASS methods use second-order DPM sampling without Langevin corrector
steps. The ordinary Figure 1 commands sample and plot all four curves; Figures 2--4
retain their original three-method comparison.

No JAC, clipping, deterministic Geffner sampler, NPE, NLE, or NRE baselines are
included by default. Pass `--clip` only if you also want the paper's clipped
Langevin variant.

## What each command creates

| Figure | Recreated experiment | Main settings |
|---|---|---|
| 1 | Gaussian posterior concentration | `n = 2, 16, 64`; analytic truth plus the three original samplers and global/local posterior moments |
| 2 | Analytic Gaussian and GMM robustness to score error | `m=10`; `n = 2,4,8,16,32,64,90`; epsilon `0,1e-3,1e-2,1e-1`; 5 seeds |
| 3 | Learned-score SLCP, Lotka–Volterra, and SIR benchmark | `Ntrain = 1k,3k,10k,30k`; `n = 1,8,14,22,30`; 25 test parameters |
| 4 | 3D Jansen–Rit NMM concentration | 50,000 training simulations; fixed `(C,mu,sigma)=(135,220,2000)`; 30 single-observation posteriors and `n = 1,8,14,22,30` |

Figures and intermediate artifacts are written below `artifacts/figure<N>/`.
The posterior-moment Figure 1 is deliberately isolated under
`artifacts/figure1_gauss_global_local_moment/`, so it does not overwrite the
original paper-reproduction artifact.
Stages are resumable: `prepare` creates simulations/reference posterior draws,
`train` trains COMPASS models, `sample` runs both Gaussian methods and Langevin, and `plot`
only reads saved artifacts.

The benchmark parameters are transformed to a latent `z` with an exact
standard-normal prior before training. Uniform priors use a probit transform;
log-normal priors use their underlying normal variable. This is the natural
COMPASS version of the experiment because the current Gaussian correction
expects a Gaussian prior. Samples are transformed back to scientific parameter
units before metrics and plots are computed.

## Environment

Run from the repository root with the COMPASS virtual environment:

```bash
cd /export/home/rheinric/COMPASS
source .COMPASS/bin/activate
```

Figures 1–3 use dependencies already declared by COMPASS. The exact Figure 4
simulator additionally needs R, `rpy2`, and `sdbmsABC`:

```bash
conda install -c conda-forge r-devtools rpy2 r-bh
Rscript -e "devtools::install_github('massimilianotamborrino/sdbmpABC')"
```

If that R backend is unavailable, `--jrnnm-backend torch` selects the included
vectorized Euler–Maruyama implementation of the same SDE. Use the default exact
backend for a paper-faithful Figure 4.

Every non-plot command calls `autocvd(num_gpus=1, interval=1)` internally before
using a GPU, in accordance with this workspace's shared-GPU policy.

## Execute Figure 1

```bash
python -m Compositional_score_testing.Gauss_test.run --figure 1 --stage sample --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 1 --stage plot
```

These commands include the posterior-mean `Gauss_global_local` curve and write
`artifacts/figure1_gauss_global_local_moment/figure1.png`.

## Execute Figure 2

This is a large analytic-score sweep but requires no model training:

```bash
python -m Compositional_score_testing.Gauss_test.run --figure 2 --stage sample --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 2 --stage plot
```

## Execute Figure 3

Run the stages separately so expensive simulation, training, and inference are
restartable:

```bash
python -m Compositional_score_testing.Gauss_test.run --figure 3 --stage prepare --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 3 --stage train --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 3 --stage sample --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 3 --stage plot
```

You can shard this sweep safely by task and training size, for example:

```bash
python -m Compositional_score_testing.Gauss_test.run --figure 3 --stage train \
  --tasks slcp --n-train 1000 3000 --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 3 --stage sample \
  --tasks slcp --n-train 1000 3000 \
  --methods gauss full_gaussian langevin --device cuda
```

## Execute Figure 4

```bash
python -m Compositional_score_testing.Gauss_test.run --figure 4 --stage prepare \
  --jrnnm-backend exact --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 4 --stage train --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 4 --stage sample --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 4 --stage plot
```

The paper uses 10,000 posterior samples in Figure 4. Reduce
`--figure4-samples` only for a smoke run.

## Lightweight smoke configuration

These commands exercise the pipeline with much smaller workloads; they do not
reproduce the paper's numerical results:

```bash
python -m Compositional_score_testing.Gauss_test.run --figure 2 --stage sample \
  --num-samples 64 --repeats 1 --device cuda
python -m Compositional_score_testing.Gauss_test.run --figure 3 --stage train \
  --tasks slcp --n-train 1000 --epochs 2 --device cuda
```

Use `python -m Compositional_score_testing.Gauss_test.run --help` for all
selection and resource options.
