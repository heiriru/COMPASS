# COMPASS global/local inference validation

This folder collects three complementary checks.

1. `01_exact_score_global_validation.png` isolates composition and sampling error by replacing the neural network with the exact diffused score for
   `g ~ Normal(-2.5, 0.3²)`, `l_i ~ Normal(0, 0.4²)`, `x_i = g + l_i + Normal(0, 0.2²)`.
2. `02_learned_score_global_local_validation.png` checks a trained score model against the exact joint posterior for
   `g ~ Normal(0,1)`, `l_i ~ Normal(0,1)`, `x_i = g + l_i + Normal(0, 0.5²)`.
3. `03_hierarchy_validation_dashboard.png` summarizes the linear/quadratic hierarchy stress test, including global pooling, local reconstruction, sharing enforcement, and BIC model identification.

## Reproduce the exact-score metrics

Run from the repository root after reserving one GPU with `autocvd`:

```bash
python tests/test_multiobs_analytic.py \
  --hierarchical-only \
  --output-csv tutorials/output/compositional_inference_local_vs_global/analytic_exact_score_metrics.csv
```

## Retrain and rerun the learned-score experiment

```bash
python tutorials/Compositional_Score/Compositional_Inference.py \
  --experiments hierarchy
```

A normal run uses the high-quality `shared_local_mixture_hq_v1_full` checkpoint: 200,000 training simulations, 20,000 validation simulations, a 128-wide six-block transformer, mixture diffusion-time sampling, up to 300 epochs, and early-stopping patience 40. Delete or move only that versioned checkpoint if an entirely fresh training run is required. `--quick` is a wiring smoke test and must not be used for scientific conclusions.

## Rerun the hierarchy stress test

```bash
python tutorials/Hierarchical_Linear_Quadratic.py
```

Use `--quick` only for a wiring check. The full run writes its detailed table to `tutorials/output/compositional_inference/08_miniexperiment/`.

## Regenerate all figures without training or inference

```bash
python tutorials/plot_local_vs_global_validation.py
```
