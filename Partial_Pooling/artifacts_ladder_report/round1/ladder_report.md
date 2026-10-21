# Partial-pooling ladder

Baseline: `legacy-5.96M/100k`. Errors are MAP errors standardized by the prior standard deviation of each parameter (1.0 = no better than the prior). Calibration is a constraint, not a score: a regression is declared when mean |coverage - 0.95| exceeds the baseline's by more than 0.05.

### dpm2_gauss_jacobian_newton

| model | params | tokens | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 | sampling s | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| legacy-5.96M/400k | 5,957,984 | 100 | 400,000 | 0.8441 | 0.3361 | 0.901 | 0.947 | 7,579 | global x1.056, local x0.975 |
| legacy-5.96M/100k | 5,957,984 | 100 | 100,000 | 0.7996 | 0.3446 | 0.907 | 0.946 | 11,455 | baseline |
| trial-h16d2-98k | 97,604 | 40 | 100,000 | 1.0561 | 0.4481 | 0.836 | 0.979 | 12,827 | global x1.321, local x1.300, **calibration regression** |
| trial-h16d4-153k | 153,444 | 40 | 100,000 | 0.8872 | 0.4425 | 0.879 | 0.983 | 19,870 | global x1.110, local x1.284 |
| trial-h32d2-207k | 207,236 | 40 | 100,000 | 0.8193 | 0.4314 | 0.810 | 0.984 | 16,091 | global x1.025, local x1.252, **calibration regression** |
| trial-h48d2-329k | 329,156 | 40 | 100,000 | 0.7341 | 0.3806 | 0.896 | 0.990 | 16,098 | global x0.918, local x1.105 |
| trial-h48d4-534k | 533,540 | 40 | 100,000 | 0.7132 | 0.3422 | 0.849 | 0.985 | 20,130 | global x0.892, local x0.993, **calibration regression** |
| scalar-h64d4-772k | 771,521 | 100 | 100,000 | 0.8228 | 0.3926 | 0.864 | 0.984 | 19,768 | global x1.029, local x1.140 |
