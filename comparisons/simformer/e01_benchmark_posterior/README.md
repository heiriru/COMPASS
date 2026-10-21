# E01 — Posterior C2ST on the sbibm benchmark tasks

**Paper:** §4.1, Fig. 4a; extended baselines in Fig. A5a (VESDE) and A6a (VPSDE).

## What they did

Four tasks — Gaussian Linear (θ, x ∈ R¹⁰), Gaussian Mixture (R²), Two Moons
(R²), SLCP (θ ∈ R⁵, x ∈ R⁸ as four i.i.d. 2-vectors) — at simulation budgets
10³, 10⁴, 10⁵. Ten ground-truth observations per task from sbibm; C2ST between
approximate and reference posterior samples, averaged over the ten.

Baselines: NPE in the main figure; NLE, NRE, NPSE and "Simformer (posterior
only)" in the appendix. NPE/NLE use the `sbi` library with a neural spline flow.

Reported: Simformer beats NPE everywhere except Gaussian Linear at 10⁴.

## What we do

The three arms of `common/model.py` on the same four tasks and budgets, scored
with the same metric on the same observations. Task definitions are transcribed
from A2.2 into `common/tasks.py` with exact joint densities, so the Gaussian
Linear reference posterior is analytic and the rest come from MCMC
(`common/reference.py`).

`posterior_only` is not an afterthought here — it is the arm that tests whether
the paper's free-lunch result survives in COMPASS's capacity regime. See the
main README.

## Running

```bash
python prepare_data.py --task gaussian_linear --task gaussian_mixture \
                       --task two_moons --task slcp          # once, CPU

python e01_benchmark_posterior/run.py train  --task two_moons --arm paper_matched --budget 10000 --device cuda
python e01_benchmark_posterior/run.py sample --task two_moons --arm paper_matched --budget 10000
python e01_benchmark_posterior/run.py score  --task two_moons --arm paper_matched --budget 10000
python e01_benchmark_posterior/run.py plot
```

36 training runs for the full grid (4 tasks × 3 budgets × 3 arms) per SDE and
architecture. Start with one task at 10⁴ and confirm the pipeline end to end
before launching the grid.

## Watch out for

- **`--arch`.** `tokenized` is the closer analogue of Simformer (shared trunk,
  per-variable identity embedding). But `inverse_kinematics/README.md` measured
  the `legacy` per-node trunk as **7× better** on endpoint error at 6
  heterogeneous nodes (0.097 vs 0.676), with capacity ruled out. These tasks are
  the same shape. Run both; a single-arch table would be misleading in either
  direction.
- **Reference quality is the floor on the metric.** A C2ST of 0.52 means nothing
  if the reference chains have not mixed. Validate first (see main README).
- **Sampler cost is not held equal across arms** by construction —
  `paper_matched` uses 500 Euler-Maruyama steps, `compass_native` ~50 DPM-2
  steps with churn. Always report the step count beside the C2ST; e05 is where
  that trade-off is measured properly.
- **Seeds.** One seed is not a result. The `--seed` flag threads through the
  data split, the mask draws, the sampler and the C2ST classifier.
