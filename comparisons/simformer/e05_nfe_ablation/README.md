# E05 — C2ST vs number of reverse-SDE evaluation steps

**Paper:** A3.1, Fig. A7. Steps 5, 10, 50, 100, 250, 500, 750, 1000 under both
SDEs. Finding: a sharp transition rather than a gradual improvement — above ~50
steps performance is at its best for every task except Two Moons under VPSDE.
They use 500 by default. They note this compares favourably with NLE/NRE, which
need an MCMC run on top.

## What we do

Reuse e01's checkpoints and only re-sample. Two questions:

1. Does COMPASS show the same sharp transition and the same threshold?
2. **The cheap win.** Their 500 steps are Euler–Maruyama. COMPASS ships DPM-2
   with churn, which `docs/unified_method.md` measured as the better predictor
   (corrector-free shared-marginal W1 of 0.049σ at η=2 against 2.598σ at η=0 on
   the exactly-solvable exponential hierarchy). If `compass_native` reaches the
   same C2ST at 20–50 function evaluations, that is a straight compute result on
   their own benchmark, obtained without retraining anything.

```bash
python e05_nfe_ablation/run.py sweep --task two_moons --arm compass_native \
    --budget 100000 --device cuda
python e05_nfe_ablation/run.py plot
```

## Watch out for

- **Churn and Langevin correctors are alternatives, not complements** — COMPASS
  raises if both are requested. `sampling_kwargs` sets `corrector_steps=0` for
  the churned arm and `churn_eta=0.0` for the paper-matched one.
- **η=8 is not reachable** at a geometric σ schedule
  (`docs/followup_results.md` §2); do not add it to the sweep.
- The x-axis is *function evaluations*, and DPM-2 costs two score calls per
  step. Plot NFE, not solver steps, or the comparison flatters COMPASS by 2×.
