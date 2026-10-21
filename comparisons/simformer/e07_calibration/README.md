# E07 — Expected coverage / calibration

**Paper:** A3.1, Fig. A9–A13. Expected-coverage analysis (Hermans et al. 2022)
for the posterior and the likelihood on every benchmark task, with NPE as the
reference, plus a coverage run on the SIRD task at 20 random time points
(Fig. A13). Reported conclusion: well calibrated, and like NPE tending to the
conservative side (curves above the diagonal).

## What we do

Run their protocol on their tasks. This is the one metric where COMPASS is
already fluent — `inverse_kinematics/sbc_reference.py`,
`inverse_kinematics/evaluate_composed.py` and the partial-pooling cov95 tables
all do this — so the work is protocol translation, not new machinery.

```bash
python e07_calibration/run.py evaluate --task slcp --arm paper_matched \
    --budget 100000 --target posterior --device cuda
```

## Watch out for

- **The HDR proxy is wrong for multimodal posteriors.**
  `common/metrics.py::expected_coverage` ranks draws by whitened distance to the
  posterior mean, which is exact only for Gaussian posteriors. For Two Moons,
  Tree and HMM this will *look* miscalibrated when it is the proxy failing. Use
  SBC ranks there — the repository already has that code.
- **Coverage is noisy at small case counts.** `compass-precision-calibration-decoupled`
  is the standing lesson here: an n=128 run looked broken and n=512 said it
  mostly was not. Establish the noise floor and vary the test seed before
  believing a coverage deficit.
- Conservative coverage is not a win. Both a well-calibrated and an
  over-dispersed posterior sit above the diagonal; pair this with C2ST.
