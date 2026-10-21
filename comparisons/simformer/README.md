# Simformer comparison

> **The paper is Gloeckler, Deistler, Weilbach, Wood & Macke,
> *All-in-one simulation-based inference*, ICML 2024 (arXiv:2404.09636v3).**
> "Simformer" is the name that paper gives its own method; this directory and
> that paper are the same thing.

Reproducing the experiments of **Gloeckler, Deistler, Weilbach, Wood, Macke,
*All-in-one simulation-based inference*, ICML 2024** (arXiv:2404.09636v3; local
copy at `~/Papers/Gloeckler et al. - 2024 - All-in-one simulation-based
inference.pdf`; code at <https://github.com/mackelab/simformer>) with COMPASS,
one directory per experiment.

**Nothing here has been trained or sampled yet.** This is the scaffold: task
definitions, the paper's protocol constants, reference-sampling machinery,
metrics COMPASS did not previously have, and staged `run.py` entry points.

---

## Why this comparison and not another

Simformer is the closest published neighbour to COMPASS. Both train a
score-based diffusion model on the **joint** `p(θ, x)` with a transformer score
network and a per-variable condition mask, and both can therefore sample
arbitrary conditionals — posterior, likelihood, and anything between. The
architectures differ in ways that are measurable rather than cosmetic (see
*Known mismatches*), and COMPASS additionally owns a compositional / tall-data
layer that Simformer mentions in one sentence (§3.3, citing Geffner et al.) and
never benchmarks.

So the comparison has two halves, and only the first is in this directory:

1. **Their turf.** Can COMPASS match Simformer under the paper's own protocol
   and metric? This needs C2ST, which does not exist anywhere else in this
   repository — COMPASS scores itself with SBC/coverage, standardised RMSE,
   sliced-Wasserstein and endpoint error, none of which is comparable to a
   published C2ST number.
2. **Our turf** — n-observation composition, hierarchical partial pooling, MAP
   estimation, model comparison. That lives in `Partial_Pooling/`,
   `Compositional_score_testing/` and `inverse_kinematics/`, and is where the
   COMPASS-specific claims belong.

## What the paper actually claims

Worth keeping in view while reading any result produced here, because the
headline is narrower than it is usually quoted as:

- Simformer beats **NPE** in C2ST on all four sbibm tasks at every budget, with
  one stated exception (Gaussian Linear at 10k simulations, §4.1).
- Averaged across tasks and observations it needs **~10× fewer simulations than
  NPE** — an average, concentrated in the tasks with exploitable sparsity, and
  it is the *structured attention mask* arms that carry it.
- Training on all conditionals is **free**: "Simformer (posterior only)" ties the
  full menu (Fig. A5/A6). Their NPSE baseline — a conditional-MLP score model,
  posterior only — lands in the same place, which makes NPSE the honest ablation
  and it is not discussed in the main text.
- On **NLL** they concede NPE/NLE sometimes win (Fig. A8), and explain it by the
  SDE-sampling / ODE-density discrepancy.
- Everything after §4.1 (Lotka-Volterra, SIRD, Hodgkin-Huxley) is **demonstration
  with no baseline**. The only quantitative panel is Fig. 5c.

## The three arms

Defined in `common/model.py`; every experiment uses the same three.

| arm | what it is | answers |
|---|---|---|
| `paper_matched` | COMPASS under their SDE constants, token width/depth/heads, five-way per-batch mask law, batch 1000, Euler-Maruyama at 500 steps | how does COMPASS do on their protocol |
| `compass_native` | COMPASS's own measured recipe: `log_sigma` time sampling, posterior-weighted structured menu, DPM-2 + churn at ~50 steps | how good is COMPASS actually |
| `posterior_only` | the paper's own control, trained on the posterior mask alone | settles the free-lunch question below |

The gap between the first two is a **recipe** effect, not an architecture
effect, and must be reported as such.

### The free-lunch question this directory is built to settle

Their ablation says posterior-only ties the full conditional menu. Our own
measurements say the weighting *within* an all-conditional menu is worth 2×:
on inverse kinematics at 10k simulations, with `log_sigma` held fixed, mean
posterior-predictive endpoint distance is 0.0420 (Bernoulli), 0.0410 (uniform
six) and **0.0213** (six weighted `[7,7,7,1,1,1]`) — see
`inverse_kinematics/artifacts/all_local_10k_*/metrics.json` and
`docs/followup_results.md` §1. Both cannot be regime-independent. Candidate
resolutions, in order of plausibility:

1. **Capacity.** Theirs is not binding, ours is. `sixw200k` (0.0232) is no
   better than `sixw10k` (0.0213) — twenty times the data changes nothing, so
   the network is at its representational limit and the conditionals compete.
2. **Loss share.** Under a uniform menu the observation-latent conditionals
   carry 87.2% of the loss at partial-pooling scale (`docs/followup_results.md`),
   where `dim(x)/dim(θ) = 9`. The paper's four tasks have `dim(θ) ≈ dim(x)`, so
   a uniform menu is already balanced for them.
3. **Metric headroom.** C2ST on a 2-D posterior at 10 observations is blunt
   compared with p90 endpoint distance or cov95.

The `posterior_only` arm in e01 is the direct test.

## Known mismatches — read before quoting any number

Enumerated in `common/paper.py::MISMATCHES`, and restated in every artifact
manifest. The four that matter:

- **Attention scope.** Simformer attends densely over all tokens and uses the
  edge mask `M_E` only to *restrict*. COMPASS sets `key_padding_mask = 1 - c`
  (`src/compass/ConditionTransformer.py:167`,
  `src/compass/TokenizedTransformer.py:205`), so only **conditioned** tokens are
  keys and latent tokens never attend to each other. This is the largest
  architectural difference in the comparison and e03 is built to measure it.
- **The joint mask.** Their five-way menu includes `M_C = 0`, which under
  COMPASS's key-padding convention leaves the attention with no keys at all and
  survives only because `add_bias_kv=True`. `compass.ConditionalMenu` refuses
  such a pattern outright, so `common/masks.py` implements their law directly.
- **Head width.** Their token dim is 50 with 4 heads of attention size 10 — the
  attention width is decoupled from the token width, which
  `nn.MultiheadAttention` cannot express, and 50 is not divisible by 4 anyway.
  We use `hidden_size = 52` (nearest multiple of 4 to their token dim);
  `hidden_size = 40` would instead match their total attention width.
- **Edge masks.** COMPASS has no `M_E`, so their headline simulation-efficiency
  arms cannot be reproduced without a trunk change (e02).

Two further unstated-in-the-paper settings are ours and are recorded as such:
the learning rate, epoch cap and early-stopping patience.

## Experiment map

| dir | paper ref | what it measures | status |
|---|---|---|---|
| `e01_benchmark_posterior` | Fig. 4a, A5a | posterior C2ST on 4 sbibm tasks × 3 budgets | ready to run |
| `e02_attention_masks` | Fig. 4a, A4, A1.1 | structured-attention simulation efficiency | **blocked** — no `M_E` in COMPASS |
| `e03_all_conditionals` | Fig. 4b, A5b | C2ST over 100 arbitrary conditionals (Tree, HMM, TM, SLCP) | ready to run |
| `e04_sde_variants` | Fig. A6 | VESDE vs VPSDE | ready (drives e01) |
| `e05_nfe_ablation` | Fig. A7 | C2ST vs reverse-SDE steps | ready to run |
| `e06_nll` | Fig. A8 | average NLL of the true value, posterior and likelihood | ready to run |
| `e07_calibration` | Fig. A9–A13 | expected coverage | ready to run |
| `e08_lotka_volterra` | Fig. 5 | unstructured/irregular observations, full time series | simulator to write |
| `e09_sird_functional` | Fig. 6, A13 | ∞-dimensional parameters (GP contact rate) | **blocked** — no index embedding / subsampling |
| `e10_hodgkin_huxley_guidance` | Fig. 7, A15–A16 | interval conditioning via guided diffusion | **blocked** — no constraint guidance |
| `e11_gravitational_waves` | A3.2 | embedding net condensing data to one token | deferred |
| `e12_posthoc_score_surgery` | A1.3, A1.4, Fig. A3 | post-hoc prior/likelihood change by affine score surgery | ready to write |

## Protocol constants

All in `common/paper.py`, cited to the section they come from. Budgets
1k/10k/100k; 10 ground-truth observations per task; 100 random conditionals for
the all-conditional tasks; 10,000 draws per C2ST side; 5,000 joint draws for
NLL; VESDE `σ_max = 15`, `σ_min = 1e-4`; VPSDE `β ∈ [0.01, 10]`; time interval
`[1e-5, 1]`; batch size 1000.

## Layout and running order

```
common/       runtime, paper constants, tasks, mask law, metrics, MCMC references,
              model construction, shared pipeline stages, paths + manifests
data/         shared simulations, observations and reference posteriors (git-ignored)
eNN_*/        one experiment: README.md, run.py, artifacts/
tests/        cheap correctness checks — simulators against their own densities
```

```bash
cd /export/home/rheinric/COMPASS
source .COMPASS/bin/activate          # torch 2.12.0+cu130; NOT .bayesflow

# 0. shared data (CPU, do once per task; --skip-reference for a fast dry run)
python comparisons/simformer/prepare_data.py --all

# 1..4. per experiment — see each directory's README
python comparisons/simformer/e01_benchmark_posterior/run.py train \
    --task two_moons --arm paper_matched --budget 10000 --device cuda
```

Conventions this directory inherits from the rest of the repository, and does
not get to opt out of:

- **Three CPU threads, set before numeric imports** (`common/runtime.py`, per the
  repository `CLAUDE.md`).
- **GPU selection via `autocvd` only.** This host is shared; never pin a device
  by hand.
- **A manifest beside every artifact** (`common/paths.py::write_manifest`):
  seed, arm, budget, data digest, git revision. An artifact without one cannot
  later be proven comparable to another.
- **Frozen evaluation sets.** Observations and the e03 mask set are written once
  and then read-only. Re-drawing them invalidates every score built on them.
- **Seed repeats before believing a difference.** Validation curves in COMPASS
  correlate strongly across architectures, so a single-seed gap is not a result;
  state the tolerance relative to the estimator's own Monte-Carlo noise.

## Before the first real run

1. `tests/` passes (simulator-vs-density checks are cheap and catch a
   transcription error in a task definition, which would silently corrupt every
   downstream C2ST).
2. The MCMC reference recipes are validated the way the paper validates them:
   C2ST ≈ 0.5 against an independent reference on a task where one exists
   (Gaussian Linear has an analytic posterior; the paper checked against sbibm).
   An unvalidated reference set is a silent way to publish a wrong number.
3. `scikit-learn` is **not** installed in `.COMPASS`, so `common/metrics.py`
   falls back to a torch C2ST with the same architecture and protocol. Install
   it if the canonical sbibm path is wanted; either way the backend is recorded
   in every result.
