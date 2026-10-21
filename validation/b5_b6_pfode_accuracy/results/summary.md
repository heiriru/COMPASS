## Error per family and budget (100 steps, exact divergence, eps 1e-3)

| family | budget | seeds | bias (nats/row) | RMSE (isotropic endpoint, default) | RMSE fitted-Gaussian endpoint | RMSE with exact endpoint only | oracle floor RMSE |
|---|---|---|---|---|---|---|---|
| bimodal_nu0.5 | 4096 | 3 | -0.285 | 0.678 ± 0.013 | 0.724 | 0.589 | 0.0453 |
| bimodal_nu0.5 | 16384 | 3 | -0.082 | 0.404 ± 0.074 | 0.538 | 0.372 | 0.0453 |
| bimodal_nu0.5 | 65536 | 3 | -0.038 | 0.188 ± 0.035 | 0.236 | 0.166 | 0.0453 |
| gauss_nu0.05 | 65536 | 1 | -0.017 | 0.146 ± 0.000 | 0.167 | 0.126 | nan |
| gauss_nu0.1 | 65536 | 1 | -0.020 | 0.104 ± 0.000 | 0.143 | 0.101 | nan |
| gauss_nu0.2 | 65536 | 1 | -0.028 | 0.118 ± 0.000 | 0.137 | 0.104 | nan |
| gauss_nu0.5 | 4096 | 3 | -0.164 | 0.387 ± 0.030 | 0.555 | 0.311 | 0.0123 |
| gauss_nu0.5 | 16384 | 3 | -0.081 | 0.203 ± 0.050 | 0.298 | 0.151 | 0.0123 |
| gauss_nu0.5 | 65536 | 3 | -0.032 | 0.111 ± 0.002 | 0.156 | 0.094 | 0.0123 |
| student_nu0.5 | 4096 | 3 | -0.273 | 0.582 ± 0.015 | 1.765 | 0.462 | 0.0317 |
| student_nu0.5 | 16384 | 3 | -0.089 | 0.288 ± 0.031 | 1.113 | 0.251 | 0.0317 |
| student_nu0.5 | 65536 | 3 | -0.045 | 0.168 ± 0.011 | 0.417 | 0.143 | 0.0317 |

## B6.3 model differences

| pair | data | budget | endpoint | exact Δ sd/row | Δ error mean | Δ error sd | N=16 sum: sign agreement |
|---|---|---|---|---|---|---|---|
| gauss vs student | gauss | 4096 | gaussian | 0.383 | -0.031 | 0.490 | 0.854 |
| gauss vs student | gauss | 4096 | isotropic | 0.383 | +0.054 | 0.399 | 0.927 |
| gauss vs student | gauss | 16384 | gaussian | 0.383 | -0.040 | 0.280 | 0.906 |
| gauss vs student | gauss | 16384 | isotropic | 0.383 | -0.014 | 0.220 | 0.938 |
| gauss vs student | gauss | 65536 | gaussian | 0.383 | -0.018 | 0.155 | 0.948 |
| gauss vs student | gauss | 65536 | isotropic | 0.383 | -0.003 | 0.125 | 0.948 |
| gauss vs student | student | 4096 | gaussian | 22.802 | +1.826 | 20.769 | 0.958 |
| gauss vs student | student | 4096 | isotropic | 22.802 | +2.633 | 22.565 | 0.906 |
| gauss vs student | student | 16384 | gaussian | 22.802 | +1.349 | 19.445 | 0.969 |
| gauss vs student | student | 16384 | isotropic | 22.802 | +2.356 | 22.391 | 0.948 |
| gauss vs student | student | 65536 | gaussian | 22.802 | +1.318 | 18.867 | 0.969 |
| gauss vs student | student | 65536 | isotropic | 22.802 | +2.243 | 22.211 | 0.979 |
| gauss vs bimodal | gauss | 4096 | gaussian | 1.896 | -0.826 | 0.726 | 1.000 |
| gauss vs bimodal | gauss | 4096 | isotropic | 1.896 | -0.796 | 0.711 | 1.000 |
| gauss vs bimodal | gauss | 16384 | gaussian | 1.896 | +1.970 | 87.544 | 1.000 |
| gauss vs bimodal | gauss | 16384 | isotropic | 1.896 | -0.276 | 0.671 | 1.000 |
| gauss vs bimodal | gauss | 65536 | gaussian | 1.896 | +5.012 | 202.440 | 1.000 |
| gauss vs bimodal | gauss | 65536 | isotropic | 1.896 | -0.167 | 0.271 | 1.000 |
| gauss vs bimodal | bimodal | 4096 | gaussian | 3.288 | -0.484 | 2.016 | 1.000 |
| gauss vs bimodal | bimodal | 4096 | isotropic | 3.288 | +1.287 | 2.182 | 1.000 |
| gauss vs bimodal | bimodal | 16384 | gaussian | 3.288 | -0.669 | 1.642 | 1.000 |
| gauss vs bimodal | bimodal | 16384 | isotropic | 3.288 | +0.882 | 1.901 | 1.000 |
| gauss vs bimodal | bimodal | 65536 | gaussian | 3.288 | -0.258 | 0.978 | 1.000 |
| gauss vs bimodal | bimodal | 65536 | isotropic | 3.288 | +0.783 | 1.516 | 1.000 |

## A2 'too good MLE' check

| family | budget | iters | PF value − exact at returned θ (overestimate) | exact at returned θ − exact max |
|---|---|---|---|---|
| bimodal_nu0.5 | 4096 | 5 | -0.120 | -0.291 |
| bimodal_nu0.5 | 4096 | 20 | -0.030 | -0.263 |
| bimodal_nu0.5 | 4096 | 100 | -0.017 | -0.259 |
| bimodal_nu0.5 | 16384 | 5 | -0.024 | -0.083 |
| bimodal_nu0.5 | 16384 | 20 | +0.021 | -0.049 |
| bimodal_nu0.5 | 16384 | 100 | +0.036 | -0.054 |
| bimodal_nu0.5 | 65536 | 5 | -0.031 | -0.033 |
| bimodal_nu0.5 | 65536 | 20 | -0.024 | +0.017 |
| bimodal_nu0.5 | 65536 | 100 | -0.024 | +0.017 |
| gauss_nu0.5 | 4096 | 5 | +0.013 | -0.078 |
| gauss_nu0.5 | 4096 | 20 | +0.162 | -0.070 |
| gauss_nu0.5 | 4096 | 100 | +0.218 | -0.074 |
| gauss_nu0.5 | 16384 | 5 | +0.026 | -0.025 |
| gauss_nu0.5 | 16384 | 20 | +0.066 | -0.023 |
| gauss_nu0.5 | 16384 | 100 | +0.076 | -0.022 |
| gauss_nu0.5 | 65536 | 5 | +0.012 | -0.012 |
| gauss_nu0.5 | 65536 | 20 | +0.026 | -0.011 |
| gauss_nu0.5 | 65536 | 100 | +0.028 | -0.012 |
| student_nu0.5 | 4096 | 5 | -0.086 | -0.251 |
| student_nu0.5 | 4096 | 20 | +0.093 | -0.199 |
| student_nu0.5 | 4096 | 100 | +0.182 | -0.186 |
| student_nu0.5 | 16384 | 5 | -0.030 | -0.120 |
| student_nu0.5 | 16384 | 20 | +0.055 | -0.086 |
| student_nu0.5 | 16384 | 100 | +0.082 | -0.084 |
| student_nu0.5 | 65536 | 5 | +0.009 | -0.067 |
| student_nu0.5 | 65536 | 20 | +0.042 | -0.051 |
| student_nu0.5 | 65536 | 100 | +0.057 | -0.049 |
