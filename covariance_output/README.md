# Covariance output on the score network (branch `covariance-output`, off `integration-2026-10-24`)

The question: can the network *predict* each row's backward covariance and replace both the score
Jacobian (`gauss_jacobian_FD`) and the pilot run (`gauss_hierarchical`)?

## What was implemented (`src/compass/covariance_output.py`, opt-in)

- **Head.** It reads the trunk's final tokens, captured by a pre-hook on `final_layer`, so it works
  with every trunk: legacy, tokenized, latent attention, and behind PreconditionedScore or the
  precondition wrappers. Each node i gets a readout `(v_i, a_i)`, and from those
  `K = S^-1 (V V^T + diag softplus a) S^-1`, which is SPD by construction.
- **Objective.** Second-order DSM: the Gaussian NLL of the detached denoising residual relative to
  `K = 0`. It enters through `Trainer.covariance_objective` and `SBIm.train(covariance_objective=...)`,
  on training passes only, so validation stays plain DSM. `detach_trunk=True` trains only the
  readout and leaves the score training unchanged.
- **Sampler.** `NetworkCovarianceHead` plugs into `MultiObsSampler.set_covariance_head`. It costs one
  forward pass per refresh instead of D+1 FD copies, and needs no pilot run. `SBIm.load`
  re-attaches the output when the checkpoint has one.
- **Default paths are unchanged.** `jac_extensions/golden_all.py check` gives ALL IDENTICAL against
  `integration-2026-10-24`. `tests/test_covariance_output.py` has 7 tests.

**Calibration finding:** the target is the information of the *network's own* residual, not the true
posterior's. With an imperfect score it is smaller than `C^-1` (measured: the output equals the
empirical residual information within 10-15%). That is the right weight for a composition.

## Results

All comparisons are within the same network, against the library defaults. Arms:
- `cov`: joint training (readout and trunk).
- `cov_det`: readout only; the score training is unchanged.

### Covariance accuracy vs exact (GM, oracle Tweedie information, median relative error)

The network output beats the FD Jacobian in 23 of 24 (net, seed, lambda) cells:

| noise level | network output | FD Jacobian |
|---|---|---|
| lambda 0.03 | 0.23-0.36 | 0.40-0.66 |
| lambda 1 | 0.05-0.08 | 0.31-0.33 |
| lambda 3 | 0.04-0.18 | 0.67-1.6 |

At lambda=3 the FD Jacobian suffers the Tweedie cancellation.

### GM composition (exact posterior, 2 seeds x 8 configs, means)

| net | method | W1/sd | abs branch err | width | local W1 | s |
|---|---|---|---|---|---|---|
| cov | GH pilot | 0.90 | 0.43 | 0.29 | 1.72 | 103 |
| cov | JAC-FD r9 | 1.14 | 0.31 | 1.47 | 1.73 | 116 |
| cov | **JAC-netcov** | **0.93** | **0.27** | 1.61 | **1.59** | 180 |
| cov_det | GH pilot | 0.85 | 0.36 | 0.25 | 1.72 | 62 |
| cov_det | JAC-FD r9 | 0.98 | 0.26 | 1.62 | 1.72 | 67 |
| cov_det | **JAC-netcov** | 1.04 | **0.18** | 1.92 | **1.55** | 103 |

- **Local posteriors:** clearly better with the network output, in all 16 configs (1.55-1.59 vs
  1.72). No other rule moves them.
- **Branch masses:** better too.
- **Global W1:** better on the joint net, slightly worse on the detached one.
- **GH:** lowest W1, but 4x overconfident (width 0.25-0.29) and the worst branch masses.
- **Cost:** with refresh 1 the network output is the slowest here, because D=2 makes FD cheap.

### Partial pooling (30 datasets x 100 subjects x 100 draws, hierarchical + Rg + wide1.5, 65,536 sims)

| net | method | mean r | global RMSE | cover90 | z_rms | local RMSE | s/dataset |
|---|---|---|---|---|---|---|---|
| cov | GH pilot | 0.939 | 0.344 | 0.85 | 1.03 | 0.156 | 169 |
| cov | JAC-FD r9 | 0.945 | 0.333 | 0.81 | 1.21 | 0.153 | 126 |
| cov | JAC-netcov | 0.946 | 0.338 | 0.83 | 1.19 | 0.153 | 118 |
| cov_det | GH pilot | 0.938 | 0.339 | 0.88 | 0.98 | 0.156 | 229 |
| cov_det | JAC-FD r9 | 0.949 | 0.316 | 0.85 | 1.07 | 0.153 | 118 |
| cov_det | JAC-netcov | 0.947 | 0.355 | 0.87 | 1.05 | 0.156 | 114 |

- **Network output vs FD Jacobian:** tied within seed spread (per-seed mean r 0.938-0.954). It is
  slightly cheaper and needs no FD copies.
- **Against GH:** both are 1.4-2x faster than GH and have higher mean r.
- **Joint training:** does not hurt the score (cov vs cov_det mean r 0.946 vs 0.947).
- **This branch's recipe:** all arms reach mean r 0.94-0.95, above the earlier 0.904 / 0.928
  records.

### Gaussian toy (Arruda, exact posterior)

| J | KL (all three rules) | width ratio |
|---|---|---|
| 10 | 0.10-0.11 | — |
| 100 | 0.27-0.31 | 1.004 (network output) vs 1.016 (others) |

All three rules tie here, as they must: every rule is exact for a Gaussian posterior, so this
benchmark can only show "no harm". J = 1000 was still running when this was written.

## Verdict

The network covariance output is a viable replacement for both the Jacobian and the pilot run:
- It equals JAC accuracy on partial pooling, at equal or lower cost and with no pilot.
- It improves local posteriors and branch masses on GM.
- Its information is far closer to the exact one than the FD Jacobian's.

It is not a clear accuracy gain on partial pooling.

Open:
- refresh 9 for the network output (it only used refresh 1);
- more seeds;
- a rank sweep.

## Running

- `experiments_roadmap2/gm.py`: arms `cov` and `cov_det`, action `info`, methods `*_netcov`.
- `covariance_output/toy.py`.
- `covariance_output/launch_pp.py`: runs the main-tree `pp.py` with `COVOUT_TRAIN` / `COVOUT_NETCOV`.
- Queues: `run_gm*.sh`, `run_toy.sh`, `run_pp.sh`.
- Tables: `summarize.py` -> `RESULTS.md`.
- Outputs:
  - GM: `/export/data/rheinric/covout/gm`
  - PP: `experiments_arruda_schur/artifacts/pp/hierarchical_Rg_wide1.5_covout_*` in the main tree
  - logs: `/export/data/rheinric/covout/logs`
