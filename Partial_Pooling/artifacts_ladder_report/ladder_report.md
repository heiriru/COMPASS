# Partial-pooling ladder

Baseline: `legacy-5.96M/400k`. Errors are MAP errors standardized by the prior standard deviation of each parameter (1.0 = no better than the prior). Calibration is a constraint, not a score: a regression is declared when mean |coverage - 0.95| exceeds the baseline's by more than 0.05.

### dpm2_gauss_jacobian_newton

| model | params | tokens | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 | sampling s | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| legacy-5.96M/400k | 5,957,984 | 100 | 400,000 | 0.8441 | 0.3361 | 0.901 | 0.947 | 7,579 | baseline |
| legacy-5.96M/100k | 5,957,984 | 100 | 100,000 | 0.7996 | 0.3446 | 0.907 | 0.946 | 11,455 | global x0.947, local x1.025 |
| tokenized-scalar-100k | 100,369 | 100 | 100,000 | 1.2853 | 0.5613 | 0.887 | 0.978 | 12,383 | global x1.523, local x1.670 |
| iid-trial-98k | 97,604 | 40 | 100,000 | 1.0561 | 0.4481 | 0.836 | 0.979 | 12,827 | global x1.251, local x1.333, **calibration regression** |

### dpm2_gauss_hierarchical

| model | params | tokens | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 | sampling s | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| legacy-5.96M/400k | 5,957,984 | 100 | 400,000 | 1.0102 | 0.3544 | 0.920 | 0.941 | 3,734 | baseline |
| legacy-5.96M/100k | 5,957,984 | 100 | 100,000 | 0.8250 | 0.3431 | 0.871 | 0.945 | 5,199 | global x0.817, local x0.968 |
| tokenized-scalar-100k | 100,369 | 100 | 100,000 | 1.3099 | 0.5613 | 0.869 | 0.979 | 6,024 | global x1.297, local x1.584, **calibration regression** |
| iid-trial-98k | 97,604 | 40 | 100,000 | 1.0670 | 0.4490 | 0.826 | 0.981 | 5,773 | global x1.056, local x1.267, **calibration regression** |

### langevin_fnpse

| model | params | tokens | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 | sampling s | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| legacy-5.96M/400k | 5,957,984 | 100 | 400,000 | 0.7805 | 0.3411 | 0.786 | 0.936 | 8,293 | baseline |
| legacy-5.96M/100k | 5,957,984 | 100 | 100,000 | 0.7918 | 0.3487 | 0.774 | 0.945 | 9,454 | global x1.015, local x1.022 |
| tokenized-scalar-100k | 100,369 | 100 | 100,000 | 0.9510 | 0.5604 | 0.710 | 0.982 | 12,684 | global x1.219, local x1.643, **calibration regression** |
| iid-trial-98k | 97,604 | 40 | 100,000 | 0.9081 | 0.4503 | 0.734 | 0.983 | 14,796 | global x1.163, local x1.320, **calibration regression** |
