# E06 — Average negative log-likelihood of the true value

**Paper:** A3.1, Fig. A8. Averaged over 5,000 draws from the joint, for the
posterior *and* the likelihood, against NPE and NLE.

They are careful here, and the care is worth preserving: NPE and NLE are trained
to minimise exactly this quantity and Simformer is not, so the comparison is
stacked against them; and Simformer's density comes from the probability-flow
ODE while its samples come from the SDE, which is why NLL and C2ST sometimes
disagree. In some cases NPE/NLE do win.

## What we do

COMPASS needs no new machinery: `ScoreBasedInferenceModel.log_prob` already
solves the PF-ODE with an exact divergence. This is the cheapest experiment in
the directory and it exercises a COMPASS code path that C2ST never touches.

```bash
python e06_nll/run.py evaluate --task slcp --arm paper_matched \
    --budget 100000 --target posterior --device cuda
python e06_nll/run.py evaluate --task slcp --arm paper_matched \
    --budget 100000 --target likelihood --device cuda
```

## Watch out for

- **`divergence="exact"` scales with dimension.** For the 20-node tasks
  (Gaussian Linear, HMM) consider `divergence="hutchinson"` for a pilot, but
  report the exact number — a stochastic trace estimator adds variance to a
  metric whose differences are small.
- **Non-finite log-probabilities are information, not noise.** `average_nll`
  counts them separately; a task with many is telling you the ODE is leaving the
  support (Two Moons and SLCP have bounded priors), not that the model is bad.
- The same PF-ODE / SDE discrepancy they flag applies to COMPASS. Say so rather
  than reconciling it.
