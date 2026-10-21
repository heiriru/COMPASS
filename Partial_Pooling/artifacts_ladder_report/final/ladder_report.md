# Partial-pooling ladder

Baseline: `legacy-5.96M/100k`. Errors are MAP errors standardized by the prior standard deviation of each parameter (1.0 = no better than the prior). Calibration is a constraint, not a score: a regression is declared when mean |coverage - 0.95| exceeds the baseline's by more than 0.05.

### dpm2_gauss_jacobian_newton

| model | params | tokens | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 | sampling s | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| legacy-5.96M/400k | 5,957,984 | 100 | 400,000 | 0.8441 | 0.3361 | 0.901 | 0.947 | 7,579 | global x1.056, local x0.975 |
| legacy-5.96M/100k | 5,957,984 | 100 | 100,000 | 0.7996 | 0.3446 | 0.907 | 0.946 | 11,455 | baseline |
| scalar-h16d2-100k | 100,369 | 100 | 100,000 | 1.2853 | 0.5613 | 0.887 | 0.978 | 12,383 | global x1.608, local x1.629 |
| scalar-h64d4-772k | 771,521 | 100 | 100,000 | 0.8228 | 0.3926 | 0.864 | 0.984 | 19,768 | global x1.029, local x1.140 |
| trial-h16d2-98k | 97,604 | 40 | 100,000 | 1.0561 | 0.4481 | 0.836 | 0.979 | 12,827 | global x1.321, local x1.300, **calibration regression** |
| trial-h16d4-153k | 153,444 | 40 | 100,000 | 0.8872 | 0.4425 | 0.879 | 0.983 | 19,870 | global x1.110, local x1.284 |
| trial-h32d2-207k | 207,236 | 40 | 100,000 | 0.8193 | 0.4314 | 0.810 | 0.984 | 16,091 | global x1.025, local x1.252, **calibration regression** |
| trial-h32d3-269k | 269,220 | 40 | 100,000 | 0.8714 | 0.3768 | 0.813 | 0.985 | 10,805 | global x1.090, local x1.093, **calibration regression** |
| trial-h48d2-329k | 329,156 | 40 | 100,000 | 0.7341 | 0.3806 | 0.896 | 0.990 | 16,098 | global x0.918, local x1.105 |
| trial-h48d3-431k | 431,348 | 40 | 100,000 | 0.7738 | 0.3565 | 0.927 | 0.990 | 13,918 | global x0.968, local x1.035 |
| trial-h64d2-463k | 463,364 | 40 | 100,000 | 0.7376 | 0.3149 | 0.857 | 0.981 | 12,152 | global x0.923, local x0.914 |
| trial-h48d4-534k | 533,540 | 40 | 100,000 | 0.7132 | 0.3422 | 0.849 | 0.985 | 20,130 | global x0.892, local x0.993, **calibration regression** |
| trial-h64d3-612k | 611,908 | 40 | 100,000 | 0.7451 | 0.3749 | 0.843 | 0.982 | 10,828 | global x0.932, local x1.088, **calibration regression** |

### dpm2_gauss_hierarchical

| model | params | tokens | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 | sampling s | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| legacy-5.96M/400k | 5,957,984 | 100 | 400,000 | 1.0102 | 0.3544 | 0.920 | 0.941 | 3,734 | global x1.225, local x1.033 |
| legacy-5.96M/100k | 5,957,984 | 100 | 100,000 | 0.8250 | 0.3431 | 0.871 | 0.945 | 5,199 | baseline |
| scalar-h16d2-100k | 100,369 | 100 | 100,000 | 1.3099 | 0.5613 | 0.869 | 0.979 | 6,024 | global x1.588, local x1.636 |
| trial-h16d2-98k | 97,604 | 40 | 100,000 | 1.0670 | 0.4490 | 0.826 | 0.981 | 5,773 | global x1.293, local x1.309 |
| trial-h48d2-329k | 329,156 | 40 | 100,000 | 0.7656 | 0.3827 | 0.886 | 0.989 | 6,095 | global x0.928, local x1.115 |
| trial-h64d2-463k | 463,364 | 40 | 100,000 | 0.7460 | 0.3203 | 0.853 | 0.979 | 4,976 | global x0.904, local x0.934 |
| trial-h48d4-534k | 533,540 | 40 | 100,000 | 0.7274 | 0.3425 | 0.831 | 0.983 | 7,661 | global x0.882, local x0.998 |

### langevin_fnpse

| model | params | tokens | train | global std-RMSE | local std-RMSE | global cov95 | local cov95 | sampling s | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| legacy-5.96M/400k | 5,957,984 | 100 | 400,000 | 0.7805 | 0.3411 | 0.786 | 0.936 | 8,293 | global x0.986, local x0.978 |
| legacy-5.96M/100k | 5,957,984 | 100 | 100,000 | 0.7918 | 0.3487 | 0.774 | 0.945 | 9,454 | baseline |
| scalar-h16d2-100k | 100,369 | 100 | 100,000 | 0.9510 | 0.5604 | 0.710 | 0.982 | 12,684 | global x1.201, local x1.607, **calibration regression** |
| trial-h16d2-98k | 97,604 | 40 | 100,000 | 0.9081 | 0.4503 | 0.734 | 0.983 | 14,796 | global x1.147, local x1.291 |
| trial-h48d2-329k | 329,156 | 40 | 100,000 | 0.7692 | 0.3961 | 0.801 | 0.994 | 13,062 | global x0.971, local x1.136 |
| trial-h64d2-463k | 463,364 | 40 | 100,000 | 0.7512 | 0.3624 | 0.786 | 0.987 | 10,018 | global x0.949, local x1.039 |
| trial-h48d4-534k | 533,540 | 40 | 100,000 | 0.7329 | 0.3739 | 0.794 | 0.990 | 14,334 | global x0.926, local x1.072 |
