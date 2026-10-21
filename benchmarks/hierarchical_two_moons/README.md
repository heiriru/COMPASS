# Hierarchical Two Moons (hsbibm) — all composition methods vs an exact reference

Task: `hierarchical_two_moons` from [mrc-ide/hsbibm](https://github.com/mrc-ide/hsbibm)
(vendored in `external/hsbibm`, commit `2c674ee`, with its light dependencies pyro/pandas in
`.vendor`). Every simulation — training and test — comes from hsbibm itself:

```python
import sbibm                      # works as-is via ./run.sh, or with PYTHONPATH=.vendor:external/hsbibm
task = sbibm.get_task("hierarchical_two_moons", n_l=10)
theta = task.get_prior()(num_samples=1000)      # (1000, 4 + 2*10)
x = task.get_simulator()(theta)                 # (1000, 2*10)
```

Globals g = (μ_0, μ_1, σ_0, σ_1) with μ ~ U(-1,1), σ ~ U(0.1,3); locals θ_i ~ TN(μ, σ²; [-1,1]²);
x_i = two-moons map of θ_i (a 0.01-wide crescent, two branches through the |z_0| fold).

## Run it

```bash
./run.sh all                          # reference -> train -> run -> report (resumable, skip-if-exists)
./queue_full.sh                       # the same, one process at a time (what produced the results)
./run.sh run --methods gauss_hierarchical attn_hier:arruda_direct --Ns 10 25
./run.sh run --family bayesflow --Ns 10
./run.sh report                       # artifacts/TABLES.md + artifacts/figures/
```

Defaults: N ∈ {1, 5, 10, 25, 50, 100}, 10 test datasets per N (truth drawn from the prior),
500 draws per method, 100k single-group training simulations for every network.
Each process pins itself to 3 CPU cores (CLAUDE.md rule); `--device cpu` works for everything,
GPU is picked by `autocvd` unless `CUDA_VISIBLE_DEVICES` is set. `--out DIR` sends results
elsewhere (used for the smoke test).

## The five questions and their metrics (`metrics.py`)

| question | metrics |
|---|---|
| Q1 local posterior p(θ_i \| x_1:N), incl. crescent / bimodality | `local_c2st` (kNN C2ST per local), `local_w1`, `branch_err` (mass on each moon), `crescent_dev` (reference-free: |r−0.1|/0.01 of θ_i mapped back through its own x_i, exact ≈ 0.67), `off_support` (θ_i that cannot have produced x_i), local mean error / width ratio |
| Q2 population mean μ_g | mean error / ref sd, width ratio, W1/sd, 90% coverage of the truth |
| Q3 population spread σ_g | same, plus the joint 4-D global C2ST |
| Q4 global–local dependence | `dep_corr_err` = mean \|corr(g_k, θ_i,d) − corr_ref\|, `joint_c2st` on (g, θ_i), and `joint_c2st_indep` — the same after shuffling θ_i across draws, i.e. what a method that ignores the dependence would score |
| Q5 behaviour in N | everything above per N: tables + `figures/scaling_q*.png` |

The `reference` row is 500 *independent* reference draws scored identically: the
Monte-Carlo floor of every metric.

## Reference: exact collapsed SMC-MCMC (`reference.py`), not hsbibm's NUTS

hsbibm's `_likelihood` is not the simulator's likelihood. It assigns `log L = 1e-10` (L ≈ 1)
to θ that **cannot** produce x, only 3.6 nats below the true θ, while a valid θ 36 sd off
gets −654 (checked in this folder). It also drops the polar Jacobian. So its NUTS reference
would keep real mass on impossible θ. Instead:

* every θ_i reaching x_i is θ(a, r, s) for one angle a, radius r and branch s, and the
  Jacobian cancels the density's 1/r, so ∫ f(θ) p(x_i|θ) dθ = Σ_s E_{a,r}[f(θ(a,r,s))]. This gives
  p(x_i | g) by quadrature (256 a × 3 r × 2 s), and p(θ_i | g, x_i) is drawn exactly by
  importance resampling;
* the 4-D collapsed global posterior (probit coordinates, prior N(0, I)) is sampled by
  likelihood-tempered SMC with adaptive random-walk Metropolis moves, then the particles
  run on as MCMC chains.

Checks (`checks/`): the quadrature matches a 2M-sample Monte-Carlo density of hsbibm's own
simulator to within MC noise (ratios 0.95–1.09); SBC passes for globals and locals at N=5
(300 datasets) and N=25 (200 datasets), all χ²(9) < 16.9.

## Methods

COMPASS (`compass_arms.py`). Every network uses the recipe of `Arruda/paper_benchmarks`:
VESDE σ=8, eps floor 1e-5, `PreconditionedScore` with the exact N(0, I) prior of the probit
globals as residual baseline, six_weighted mask menu, log_sigma time sampling, 200 fixed epochs,
final weights, h64 d4. All rules use DPM-2 with 100 steps and churn on (the library default),
except F-NPSE, which uses Langevin.

| method | network | rule |
|---|---|---|
| `uncorrected`, `gauss`, `gauss_hierarchical`, `gauss_jacobian`, `gauss_jacobian_FD`, `langevin_fnpe` | production per-node (`legacy`) | production corrections |
| `attn_hier:arruda_direct` / `:arruda_gauss_direct` / `:gauss_jacobian_FD` | tokenized, `hierarchical` latent attention (g output = marginal g score) | plain Arruda sum, precision-weighted marginal, joint-state Jacobian |
| `attn_full:gauss_hierarchical` / `:gauss_jacobian_FD` / `:arruda_schur` / `:arruda_laplace` | tokenized, `full` latent attention (joint score) | production joint-state rules, Schur / Laplace marginals |

BayesFlow / Arruda et al. (`bayesflow_arm.py`): BayesFlow 2.0.13
`compositional_sample` with the paper's network and sampler. Globals are composed, then
θ_i ~ q(θ_i | g, x_i) (their Eq. 12). d1 ∈ {1, .1, .01, .001} × shift ∈ {0, 2, 5} are tuned per N
on 5 separate tuning datasets by the paper's criterion. Training uses the same 100k
simulations and the same number of optimizer steps (100 epochs at batch 128).
`bayesflow_arruda` passes the time-free prior score (the paper's (1−t) weighting), and
`bayesflow_arruda_diffprior` passes the exactly diffused prior.

## Results

See `RESULTS.md` (written once the runs finish) and `artifacts/TABLES.md`.
