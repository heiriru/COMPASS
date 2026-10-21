# Simformer benchmark vs COMPASS — plan

Paper: Gloeckler, Deistler, Weilbach, Wood & Macke, *All-in-one simulation-based
inference*, ICML 2024 (`~/Papers/Gloeckler et al. - 2024 - All-in-one simulation-based inference.pdf`).
Official code: `simformer_official/` (clone of github.com/mackelab/simformer).

## Principle: one harness, two methods

Their numbers come from their `scoresbibm` harness (Hydra + JAX). Instead of
re-implementing tasks, reference posteriors and C2ST, **COMPASS is plugged into
that same harness as one more method** (`bridge/`). For a given
`(task, budget, seed)` both methods therefore see

* the identical training simulations (`task.get_data(N, rng)` with the same key),
* the identical observations and sbibm reference posteriors (Fig. 4a),
* the identical conditional masks, observed values and MCMC reference samples
  (Fig. 4b — the eval RNG stream is independent of the method),
* the identical metric code (`sbi.analysis.c2st`, random-forest classifier, 5 folds).

COMPASS itself runs unchanged in its own env (`.COMPASS`, torch 2.12/CUDA) as a
persistent worker process; the harness talks to it over a pipe. No COMPASS
source is modified.

Environments: `.simformer/` (py3.10, jax 0.4.23 cuda12, torch 2.1 cpu, sbi 0.22,
sbibm 1.1) and the repo's `.COMPASS/`.

## What is run (in this order, strictly sequential, one GPU chosen by autocvd)

| # | comparison | paper ref | tasks | budgets | metric |
|---|---|---|---|---|---|
| 0 | pilot: pick the COMPASS arm | — | two_moons | 10k | C2ST posterior |
| 1 | posterior benchmark | Fig. 4a | gaussian_linear, gaussian_mixture, two_moons, slcp | 1k, 10k, 100k | C2ST vs sbibm reference, 10 obs |
| 2 | all-conditionals benchmark | Fig. 4b | tree, hmm (markov chain), two_moons, slcp (`*_all_cond`) | 1k, 10k, 100k | C2ST over random conditionals vs MCMC reference |
| 3 | NPE baseline (their reference line in Fig. 4a) | Fig. 4a | the 4 sbibm tasks | 1k, 10k, 100k | C2ST posterior (CPU) |

Methods per cell:

* **Simformer** — their `score_transformer` (dense attention, VESDE, structured
  random mask law, Euler–Maruyama 500 steps), exactly their configs.
* **COMPASS** — our setup, trained on the same data under the *same* all-conditional
  mask law (so both models can answer every conditional; neither is posterior-only):
  - architecture/recipe chosen in the pilot between the `legacy`
    (ConditionTransformer, dense read-out) and `tokenized` trunks — the tokenized
    trunk is known to be mean-field over latent tokens, which a correlated
    posterior (Two Moons, SLCP) will expose;
  - COMPASS-native recipe: per-dimension z-scoring, `log_sigma` time sampling,
    schedule-free AdamW, best-validation checkpoint, DPM-2 + churn sampler;
  - the same gradient-step cap as Simformer (5k / 30k / 100k steps for 1k / 10k /
    100k simulations) and the same batch size 1000.

The paper's structured-attention arms (directed/undirected `M_E`) are **not** in
the main grid: COMPASS has no edge mask, so they are not a like-for-like
comparison. They can be added later as a Simformer-only reference.

## Reporting (after every finished cell, not at the end)

`report.py` is called by the driver after each run and regenerates

* `results/tables/*.md|csv` — mean ± s.e.m. C2ST per (task, budget, method), and
  the COMPASS − Simformer difference,
* `results/figures/fig4a_posterior_c2st.png`, `fig4b_allcond_c2st.png` — the
  paper's layout (C2ST vs number of simulations, one panel per task),
* `RESULTS.md` — running summary with the current verdict per task.

## Caveats stated up front

* One seed per cell (as in the paper's figure); differences smaller than the
  spread over the 10 observations are not claimed as wins.
* C2ST uses their random-forest classifier (sbi 0.22 `c2st(..., classifier="rf")`),
  not sbibm's MLP; numbers are comparable to our Simformer run, and only
  approximately to the published figure.
* Fig. 4b conditionals: their released configs use 50 evaluations × 2000 samples
  by default (the paper figure: 100 × 5000). We use the default unless time
  allows the larger setting; both methods always share the setting.
