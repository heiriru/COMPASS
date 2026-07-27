# Learned global/local score: compositional-method comparison

These three runs use the same `shared_local_mixture_full` score backbone, the same saved 30 observations and analytic posterior, 3,000 posterior samples, 100 noise levels, and seed 1208.

- `02a_dpm2_gaussian.png`: second-order probability-flow predictor, Gaussian composition, 10 Langevin corrector steps at every level, 3 final corrector steps, SNR 0.2.
- `02b_pfode_gaussian.png`: the same second-order probability-flow integration with Gaussian composition, but no Langevin correctors. The Gaussian endpoint denoising step remains enabled.
- `02c_langevin_fnpe.png`: pure annealed Langevin sampling of the F-NPSE bridging densities, 10 steps per noise level, SNR 0.2.

| Method | Global mean error / exact σ | Global width ratio | Mean local error / exact σ | Mean local width ratio | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| DPM2 + Gaussian | 0.575 | 1.424 | 0.174 | 1.276 | 146.9 |
| PF-ODE + Gaussian | 0.573 | 1.450 | 0.725 | 1.555 | 23.6 |
| Langevin + F-NPSE | 0.105 | 1.086 | 0.046 | 0.905 | 118.8 |

These are results for one fixed dataset and seed, not a repeated-dataset uncertainty study. Machine-readable details are in `method_metrics.csv` and `run_config.json`.

Regenerate or resume with:

```bash
python tutorials/compare_shared_local_composition_methods.py
```

Use `--force` to intentionally replace all saved samples, or `--methods METHOD_KEY` to recalculate one method.
