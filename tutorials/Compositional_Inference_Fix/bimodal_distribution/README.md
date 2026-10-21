# Bimodal composition: can COMPASS *build* a bimodality, not just keep one?

> **Sampler and training-recipe change.** Churn now lives in `compass.churn` and
> is the sampler default (`churn_eta=2.0`, corrector-free); `time_sampling` and
> the training condition-mask law are chosen through the COMPASS API. The
> defaults used by the commands below are no longer the ones the tables in this
> file were produced under. See [`docs/unified_method.md`](../../../docs/unified_method.md)
> for what changed, what is pinned to reproduce the old numbers, and which
> checkpoints are now stale.


Every hierarchical benchmark in `../New_Attempt/` has a unimodal posterior, so
none of them can say what a composition rule does when the target has two modes
and a deep low-density gap between them. This directory adds two that do, and
runs the existing samplers on both — twice each, once with a trained score
network and once with the **exact closed-form row score**, so that a failure can
be attributed to the composition rule rather than to the network.

## The two benchmarks

### `sign_model.py` — bimodality *constructed* by composition (the interesting one)

    g          ~ Normal(0, sigma_g^2)                  sigma_g = 1.5
    l_j | g    ~ Normal(g, tau^2)                      tau     = 0.5
    s_j        ~ Uniform{-1, +1}                       (never observed)
    y_j | l_j  ~ Normal(s_j l_j, sigma_y^2)            sigma_y = 0.25

Integrating the sign out gives an emission that is even in `y`:

    p(y_j | l_j) = 1/2 N(y_j; l_j, sigma_y^2) + 1/2 N(y_j; -l_j, sigma_y^2)

so **one** observation fixes the magnitude of `l_j` and says nothing at all
about its sign: the row posterior `p(g, l_j | y_j)` is an exactly symmetric
two-component Gaussian mixture. But the hierarchy couples the signs
ferromagnetically — after integrating `g` out,

    w_s  ~  exp[ sigma_g^2 (sum_j s_j y_j)^2 / (2 v (v + n sigma_g^2)) ],
    v = tau^2 + sigma_y^2

which is maximised by the two configurations in which every row points the same
way. So `n` factors that are *individually* ambiguous in the same way have to be
composed into one *globally coordinated* two-moded posterior — not into the
`2^n`-moded product that treating the rows as independent would give.

At the shipped settings and `n = 8` the two coherent configurations carry
**96.4%** of the posterior mass, the modes sit at **g = ±1.14**, and the valley
at `g = 0` is **11.5 nats** deep.

### `control_model.py` — bimodality *given* in the prior

    z ~ Bernoulli(1/2);  g | z ~ Normal(±a, sigma_0^2);  l_j | g ~ Normal(g, tau^2)
    y_j | l_j ~ Normal(l_j, sigma_y^2)                   -- no sign ambiguity

Here each row already prefers one lobe, and a rule only has to avoid destroying
a bimodality it was handed. Running both separates *"can COMPASS preserve
bimodality?"* from *"can COMPASS construct the correct bimodality from ambiguous
rows?"*

Two things about the control are worth stating plainly, because they are results
rather than implementation details:

- **It cannot use the sign model's parameters.** With locals that tight, eight
  observations pin `g` to ~0.1 and the prior's two lobes stop being a posterior
  question — the data simply says which lobe. The prior gap only survives into
  the posterior while `sqrt(v/n)` is comparable to `a`, so the control runs at
  `tau = 3`, `a = 2`, `sigma_0 = 0.4`.
- **Its dataset is chosen; the sign model's is not.** Most control datasets
  resolve the lobe outright and leave a unimodal posterior, so a seed whose
  exact posterior does retain both lobes (weights 0.36 / 0.64) is used. In the
  sign model *every* dataset is bimodal, because the ambiguity is in the
  likelihood rather than in the prior.

The control also carries one approximation the sign model does not:
`MultiObsSampler` takes a **Gaussian** prior for its `(1 - n)` prior-score term,
and the control's prior over `g` is not Gaussian, so every arm is given the
moment-matched `Normal(0, a^2 + sigma_0^2)`. In the sign model the prior really
is `Normal(0, sigma_g^2)` and the rule is graded with nothing else mis-specified.

## Why these two problems are usable as gold standards

Conditional on the discrete latent (the sign configuration, or the prior
component) both models are linear Gaussian with an arrow precision whose inverse
`C` **does not depend on that latent**. So

    p(g, l_1..l_n | y) = sum_k w_k N(m_k, C)      exactly,

and because the components share one covariance, the VE kernel adds `lam^2 I` to
all of them at once:

    p_lam(X | y)              = sum_k w_k N(m_k, C + lam^2 I)
    grad_X log p_lam(X | y)   = (C + lam^2 I)^-1 [ sum_k r_k(X) m_k - X ]

with `r_k` the responsibilities. That is the **exact composed score at every
noise level**, in closed form, with no quadrature and no Monte Carlo — which is
what a composition rule has to reproduce. For `n = 8` the sign model's mixture
has 256 components and the whole reference costs milliseconds.

The same algebra applies to a single row, which is what makes
`row_scores.ExactRowScoreNetwork` possible: a drop-in for `SBIm.model` returning
the exact score of `p_lam(g_t, l_t | y_j)` — i.e. precisely what the network is
*trained* to output. Running every arm with it as well as with the network
separates "the rule is wrong at finite noise" from "the network is wrong", which
on a bimodal target is exactly the ambiguity worth removing: a collapsed mode
could be either.

## The arms

| key | sampler | rule |
|---|---|---|
| `dpm2_churn8_gauss_hierarchical` | DPM-2, Karras churn `eta = 8`, corrector-free | `gauss_hierarchical` |
| `dpm2_churn8_gauss_jacobian` | same | `gauss_jacobian` |
| `langevin_fnpe` | annealed Langevin, 10 correctors, snr 0.2 | `fnpe` |
| `dpm2_gauss_hierarchical` | DPM-2, 10 correctors per level, snr 0.2 | `gauss_hierarchical` |
| `dpm2_gauss_jacobian` | same | `gauss_jacobian` |

The last two are the no-churn baseline the churned arms are read against; the
first three are the requested comparison. `eta = 8` is the strongest churn the
predictor can express — the per-step noise-up ratio is capped at `sqrt(2) - 1`,
so past `eta ~ 4` on this schedule every step is already at the cap.

Every arm is run against **three** score sources, not two:

| key | row score |
|---|---|
| `learned` | the small `h32d3` network, 552K parameters |
| `learned_large` | the stock `h128d6` backbone, 5.2M parameters |
| `analytic` | the closed form — no network at all |

The middle rung is there because the small network's row score is only accurate
to ~17% relative RMS on average, and to 44% at the smallest probed noise level,
so a learned-arm failure could be capacity rather than composition. The stock
backbone reaches ~6% on the same grade. With all three in the table, "is this
the network or the rule?" stops being an argument and becomes a row.

## Training recipe: `log_sigma`, not the COMPASS default

`train.py` ships `DEFAULT_TIME_SAMPLING = "log_sigma"`, overriding COMPASS's own
`"mixture"`. That is a measured decision, not a preference: `"mixture"` spends
half its budget uniformly in `t`, which puts almost nothing near small `lambda`
— exactly where the sampler needs the score and where essentially all of the
error was. On the 552K network the mean relative row-score error drops from
0.166 to 0.054, and at `lambda = 0.05` from 0.440 to 0.088, which beats the 5.2M
network trained the `"mixture"` way. It costs nothing at inference. Do *not* also
standardize the data on top of it; that combination is worse (see `RESULTS.md`).

**The main tables in `RESULTS.md` predate this and were produced with
`"mixture"`.** Pass `--time-sampling mixture` to reproduce them exactly. Every
checkpoint directory is tagged with its recipe except the legacy `"mixture"` ones,
which keep their original untagged paths, so no directory can ever be re-pointed
at weights trained under the other recipe.

## Running it

    python validate_model.py       # grade every closed form against a second derivation
    python row_scores.py           # grade the exact row score, incl. under torch.func.jvp
    python train_models.py                          # the small 552K network per benchmark
    python train_models.py --config h128d6          # the 5.2M rung
    python run_experiment.py                        # every arm, all three score sources
    python plot_model.py           # figures 1-2 (no network, no sampler needed)
    python plot_methods.py         # figures 3-6
    python make_tables.py          # the CSVs as Markdown, for RESULTS.md

`run_experiment.py` caches each arm's `.npz`; pass `--force` to redo one.
Every script caps itself at 3 CPU threads per the repository's `CLAUDE.md`, and
`run_experiment.py` / `train_models.py` call `autocvd` before touching a GPU.

## What is measured

Two layers, both against exact references.

**The composed score itself** (`05_*_score_error.png`): relative RMS error of
`grad_g log p_lam` against the closed form, at the noise levels the sampler
queries, on states drawn from the exact diffused posterior — so the error is
weighted by where the sampler really is. This is the rule with the sampler
removed entirely.

**The draws** (`03`, `04`, `06`): W1 against the exact marginals in units of
their own standard deviation; how many modes survived; how the mass split
between them; how deep the recovered valley is; and whether the local signs
stayed coordinated with the shared one (a rule can keep both modes and still get
the coupling wrong — that shows up in `04_*_joint.png` and nowhere else).

One methodological note that mattered: **every kernel density estimate here uses
an explicit absolute bandwidth**, tied to the width of the exact posterior's own
modes. Scott's rule sets the bandwidth from the sample standard deviation, which
on a two-moded posterior is dominated by the *gap between the modes* — it
flattens the peaks by nearly half and fills in the valley that is the entire
object of the experiment. Mode counts and valley depths are compared against the
exact posterior seen through the *same* kernel, so the KDE's own floor is never
read as the sampler's failure.

## Figures

Each is written twice, once per benchmark, under `artifacts/<name>/`.

| figure | what it shows |
|---|---|
| `01_<name>_model.png` | the forward law, the dataset, the n ambiguous row posteriors, and the one posterior they compose into |
| `02_<name>_emergence.png` | the mechanism: which discrete configurations hold the mass, what adding rows does, how it depends on pooling, and how a row's own two modes merge under diffusion |
| `03_<name>_marginals.png` | the shared marginal every arm produced, against the exact one, one row per score source |
| `04_<name>_joint.png` | the same draws in the `(g, l_1)` plane, where a rule that keeps both modes but bridges them is visible and the marginal alone would not show it |
| `05_<name>_score_error.png` | the composed score's own error against the exact diffused score, per noise level — the rule with the sampler removed |
| `06_<name>_summary.png` | six numbers per arm: distance to the exact marginal, leakage into the gap, cost, the mass split, mode width, and shared/local coupling |

## Files

| file | what it is |
|---|---|
| `sign_model.py`, `control_model.py` | the two generative models and their exact mixtures |
| `mixture_reference.py` | every reference that follows from `(weights, means, covariance)` |
| `row_scores.py` | the exact single-observation score, as a drop-in for `SBIm.model` |
| `churn.py` | the corrector-free churned DPM-2 predictor (copied from `../New_Attempt/exponential_score_comparison/`) |
| `train.py`, `train_models.py` | the training recipe and the network-fidelity grade |
| `make_tables.py` | renders the result CSVs as the Markdown tables in `RESULTS.md` |
| `run_experiment.py` | the sweep: every arm, both score sources, both benchmarks |
| `validate_model.py` | grades every closed form against an independent derivation |
| `style.py` | one visual language: colour = composition rule, line style = sampler |
| `plot_model.py`, `plot_methods.py` | figures 1-2 and 3-6 |
| `artifacts/` | figures, per-arm `.npz`, `method_metrics.csv`, `network_fidelity.csv` |

## Results

See `RESULTS.md`.
