# Paper benchmark of the final COMPASS method

This folder runs the full benchmark for the paper on one GPU: the final method on every problem we
have, against the baselines, and one-component-at-a-time ablations that show why each default was
chosen. Results: `RESULTS.md` (written by `summarize.py`).

## The final method

**Network and training**:
- tokenized transformer with hierarchical latent attention;
- the library's training defaults on this branch:
  - mask menu + EDM loss weight (`auto`);
  - `log_sigma` time sampling;
  - seeded validation noise;
  - patience 100;
  - final weights;
- a covariance output trained jointly by second-order DSM (`compass.covariance_output`).

Problem-level choices are kept where a benchmark has them:
- partial pooling:
  - residual output on the globals (`Rg`);
  - a 1.5× wide training prior, divided out at inference;
- GM: the training floor `time_eps` 1e-5.

**Sampling**: the library default, `MultiObsSampler.sample` as it ships. Its parts:
- `gauss_jacobian_FD` with `jacobian_refresh=9` at every noise level;
- float32 eigendecompositions in 1,024-row chunks;
- compact `HierarchicalState`;
- elimination Schur;
- per-row information projection, then composed-level repair;
- coordinate denoise clamp.

The network-covariance variant (`netcov`) replaces the FD Jacobian by the covariance output; it is
benchmarked next to it.

The drivers pin older refresh schedules so that their archived numbers keep their meaning. The final
arm therefore overrides exactly those arguments back to the library default (`jobs.FINAL_KW`).

## What a paper has to show, and which benchmark shows it

| claim / open question | benchmark | reference |
|---|---|---|
| The rule is exact when the scores are exact; the error budget splits into rule vs network | toys `abs`, `lean` (oracle_joint); latent-attention GM (oracle) | Gibbs / quadrature |
| Correct on a Gaussian problem at every J (no harm), and how error grows with J | Arruda Exp. 1, Gaussian toy, J = 10, 100, 1000 | analytic |
| Multimodal globals: branch masses and widths, Gaussian vs non-Gaussian prior interface | GM, n = 1–64, symmetric and asymmetric | exact MC |
| Symmetric bimodality from the likelihood (no prior asymmetry) | IK reach-only, sym / shift, n = 8, 32, 128 | quadrature |
| Non-log-concave, folded, curved locals | toys `abs`, `lean`, `square`, `curved` | Gibbs |
| Large numbers of groups, locals with non-Gaussian links | AR(1) grid 4×4, 16×16, 64×64 | importance sampling |
| A real hierarchical model against BayesFlow, on BayesFlow's own network and protocol | partial-pooling DDM, upstream SetTransformer + DiffusionModel trained on the joint s(g, l \| x_j); n = 4, 25, 100; inference and wide training prior | upstream's metrics; truth recovery, ECE, coverage |
| The same model with the COMPASS network, 3 training seeds, 100 datasets | partial-pooling DDM, COMPASS transformer | truth recovery, coverage, z-scores |
| Why each default | ablations below | paired against the final arm |
| Cost | every job logs seconds; fair n-sweep; PP s/dataset | — |

## Ablations

### Sampler

Each variant changes one component and keeps everything else final (`jobs.SAMPLER_VARIANTS`):

| variant | change | question |
|---|---|---|
| `r1` | Jacobian refreshed at every call | does refresh 9 cost accuracy? |
| `r5`, `r10`, `r19` | other periods (fair only) | why an odd period, why 9 |
| `fp64` | float64 eigh, 2,048-row chunks | does float32 cost accuracy? |
| `dense` | dense state + Cholesky Schur (the pre-refactor route) | identical draws, speed, memory |
| `noproj` | no per-row information projection | is the projection needed? |
| `nocurv` | no lower curvature clip | is the clip needed? |
| `consistent` | consistent row-projection numerator | PD study round 1 |
| `gate` | consistent only below λ 0.5, with the row-mean guard | PD study round 3 |
| `gloeckler` | Gloeckler et al. 2025 PD fix (nugget 0.1) | the published alternative |
| `metric` | metric (conservative) denoise clamp | clamp study |
| `churn0`, `t25`, `t100` | solver: no churn, 25 / 100 steps | solver sensitivity |
| `netcov_r1`, `netcov_r9` | network covariance output instead of the FD Jacobian | does it replace the Jacobian? |
| `bestval` | best-validation weights instead of final | checkpoint choice |

Baselines on the same networks:
- `gauss_hierarchical` (pilot covariances);
- `arruda_direct` (the plain sum of Arruda et al.);
- `langevin_fnpse` (Geffner et al.);
- on partial pooling, upstream BayesFlow's own `compositional_sample`.

### Training

Partial pooling, COMPASS network, 30 paired datasets. Each arm changes one component of the final
recipe:

| arm | change | seeds |
|---|---|---|
| `nocov` | no covariance output | 7, 8 |
| `noedm` | plain ε-loss | 7, 8 |
| `unseeded` | fresh validation noise per epoch (early stopping on noise) | 7, 8 |
| `tsm` | + target score matching on the six globals | 7, 8 |
| `norg` | no residual output on the globals | 7 |
| `nowide` | inference prior as training prior | 7 |
| `full` | full latent attention instead of hierarchical | 7 |

The final recipe has three seeds: `covout_cov_s7` and `covout_cov_s8` from the covariance-output
study (same commit line, same recipe), and `paper_final_s9`.

## Known limits that the benchmark states rather than hides

- **Network score bias summed over many observations.** Arruda's Gaussian toy at J = 1000 and the
  AR(1) grid at 64×64 already failed every GAUSS-type rule in the archived runs (mean error > 1 sd).
  F-NPSE degrades less there. The rerun shows whether that still holds with the final recipe.
- **Seeds.**
  - BayesFlow networks: one training seed each (8 h per net). The inference seed is replicated
    instead.
  - Architecture ablations: one seed.
- **Multi-level (L > 2) composition, `jacobian_row_batch` and the conservative score are
  experimental.** They are not part of the final method and not benchmarked here.

## Running

```bash
python benchmarks/paper/setup_tree.py         # worktree at HEAD + copies of the data/checkpoints
nohup python benchmarks/paper/jobqueue.py --gpu 7 > /export/data/rheinric/paper_bench/queue.out 2>&1 &
python benchmarks/paper/jobqueue.py --list       # status
python benchmarks/paper/summarize.py          # RESULTS.md
```

- Everything runs from `/export/data/rheinric/paper_bench/tree`, a detached worktree of the
  benchmarked commit. No running job can see later edits.
- The data and checkpoints are copied into the tree, so no job writes into another worktree.
- The hook (`hook/sitecustomize.py`) applies the overrides and pins every driver's autocvd call to
  the queue's GPU.
