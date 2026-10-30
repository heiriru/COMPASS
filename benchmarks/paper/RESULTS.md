# Paper benchmark: results

Benchmarked commit: `8ebb7982 (189 jobs, 2026-10-26..28); c9a7e8c8, d758b4ba (reruns + fairtest; driver/harness fixes only); 9fe457c4 (fixed-500-epoch arm); 10df7566 (best combination: EDM + netcov)`. Written by `summarize.py`; design in `README.md`.

## 1. Partial pooling on BayesFlow's own network (fair benchmark)

`bf_*` = upstream's `compositional_sample` on its global net (+ its local net); `paper-*` = COMPASS on the joint net with the same architecture and simulations. 100 test datasets.

```
                                  --------- globals ---------         --- locals ---
arm                         mean r  NRMSE    ECE  cov90      r  NRMSE    ECE    min  per-global r (mu_nu, mu_log_alpha, mu_log_t0, log_sigma_nu, log_sigma_log_alpha, log_sigma_log_t0, beta)
bf_global_n100               0.899  0.324  0.247   0.57  0.907  0.270  0.098    0.2  0.844 0.890 0.978 0.917 0.813 0.880 0.970
bf_global_n25                0.852  0.473  0.110   0.79  0.905  0.312  0.077    0.2  0.796 0.858 0.952 0.874 0.754 0.797 0.934
bf_global_n4                 0.616  1.096  0.133   0.99  0.896  0.461  0.053    0.1  0.514 0.550 0.808 0.709 0.520 0.378 0.836
bf_global_wide_n100          0.926  0.290  0.106   0.79  0.907  0.275  0.082    0.3  0.901 0.925 0.984 0.932 0.821 0.933 0.983
bf_global_wide_n25           0.872  0.508  0.035   0.92  0.907  0.308  0.064    0.4  0.839 0.862 0.953 0.882 0.770 0.850 0.946
bf_global_wide_n4            0.621  1.241  0.157   1.00  0.899  0.475  0.066    0.2  0.553 0.576 0.797 0.729 0.573 0.298 0.823
paper-churn0_joint_wide_n100   0.938  0.434  0.060   0.93  0.942  0.283  0.024    5.2  0.905 0.940 0.986 0.948 0.843 0.962 0.984
paper-consistent_joint_n100   0.917  0.298  0.188   0.67  0.933  0.280  0.026   12.6  0.906 0.905 0.979 0.935 0.807 0.901 0.983
paper-consistent_joint_wide_n100   0.844  0.885  0.077   0.93  0.936  0.307  0.036   10.5  0.782 0.872 0.972 0.902 0.836 0.585 0.961
paper-dense_joint_wide_n100   0.926  0.303  0.038   0.87  0.942  0.280  0.026   12.7  0.921 0.931 0.985 0.948 0.764 0.951 0.982
paper-final-seed1_joint_wide_n100   0.933  0.313  0.037   0.90  0.942  0.280  0.025    7.5  0.925 0.936 0.985 0.949 0.800 0.953 0.981
paper-final-seed2_joint_wide_n100   0.926  0.316  0.042   0.88  0.942  0.280  0.026    8.5  0.924 0.934 0.985 0.947 0.760 0.948 0.981
paper-final_joint_n100       0.886  0.429  0.031   0.89  0.933  0.279  0.029    4.9  0.880 0.891 0.978 0.848 0.715 0.915 0.975
paper-final_joint_n25        0.871  0.492  0.029   0.91  0.922  0.300  0.024    2.5  0.832 0.851 0.954 0.892 0.733 0.893 0.939
paper-final_joint_n4         0.659  0.779  0.024   0.91  0.902  0.342  0.030    0.4  0.560 0.614 0.810 0.766 0.553 0.478 0.830
paper-final_joint_wide_n100   0.926  0.303  0.038   0.87  0.942  0.280  0.026    3.4  0.921 0.931 0.985 0.948 0.764 0.951 0.982
paper-final_joint_wide_n25   0.891  0.462  0.038   0.91  0.927  0.307  0.025    2.1  0.860 0.882 0.956 0.891 0.801 0.898 0.946
paper-final_joint_wide_n4    0.661  0.762  0.023   0.91  0.912  0.355  0.032    1.6  0.585 0.612 0.803 0.768 0.572 0.441 0.845
paper-fp64_joint_wide_n100   0.926  0.303  0.038   0.87  0.942  0.279  0.026    9.2  0.921 0.931 0.985 0.948 0.764 0.951 0.982
paper-gate_joint_wide_n100   0.926  0.282  0.170   0.65  0.940  0.265  0.035    9.6  0.907 0.934 0.986 0.930 0.839 0.902 0.982
paper-gloeckler_joint_wide_n100   0.925  0.309  0.036   0.88  0.942  0.280  0.027    7.6  0.919 0.932 0.985 0.947 0.761 0.950 0.982
paper-metric_joint_n100      0.887  0.431  0.033   0.89  0.933  0.278  0.029   11.5  0.885 0.885 0.977 0.847 0.720 0.918 0.975
paper-metric_joint_wide_n100   0.926  0.305  0.039   0.87  0.942  0.279  0.026    9.7  0.921 0.931 0.985 0.948 0.763 0.951 0.982
paper-nocurv_joint_wide_n100   0.926  0.305  0.040   0.87  0.942  0.279  0.026    8.2  0.921 0.931 0.985 0.949 0.765 0.951 0.982
paper-noproj_joint_n100      0.847  0.736  0.046   0.95  0.931  0.315  0.027    9.0  0.811 0.831 0.968 0.851 0.636 0.861 0.970
paper-noproj_joint_wide_n100   0.892  0.326  0.039   0.88  0.942  0.284  0.029    9.1  0.914 0.878 0.981 0.949 0.596 0.943 0.980
paper-r10_joint_wide_n100    0.933  0.302  0.037   0.87  0.942  0.281  0.025    7.8  0.919 0.932 0.985 0.948 0.811 0.952 0.982
paper-r19_joint_wide_n100    0.936  0.311  0.034   0.88  0.942  0.280  0.026    4.9  0.919 0.937 0.985 0.948 0.830 0.949 0.982
paper-r1_joint_n100          0.880  0.460  0.029   0.89  0.932  0.282  0.031   21.5  0.886 0.879 0.977 0.844 0.697 0.901 0.973
paper-r1_joint_wide_n100     0.919  0.311  0.038   0.88  0.942  0.280  0.026   21.2  0.924 0.916 0.985 0.946 0.728 0.951 0.981
paper-r5_joint_wide_n100     0.923  0.300  0.039   0.88  0.942  0.279  0.026    8.4  0.923 0.921 0.985 0.947 0.756 0.950 0.981
paper-t100_joint_wide_n100   0.923  0.313  0.042   0.88  0.942  0.278  0.027   13.9  0.927 0.919 0.984 0.947 0.754 0.949 0.982
paper-t25_joint_wide_n100    0.932  0.326  0.035   0.90  0.942  0.287  0.031    3.4  0.921 0.935 0.985 0.951 0.801 0.949 0.982
```

### 1b. Head-to-head on the same 100 test datasets: BayesFlow vs the full COMPASS method

`compass-*` = the COMPASS network and sampler (3 training seeds) on upstream's test datasets (`fairtest.py`), scored by the same `fair.py score`. `min` is sampling minutes for all 100 datasets with the GPU shared by 3 jobs; with the GPU alone the COMPASS network takes 47 s per dataset (pp_fairtest_timing_s7).

```
arm                         mean r  NRMSE    ECE  cov90      r  NRMSE    ECE    min  per-global r (mu_nu, mu_log_alpha, mu_log_t0, log_sigma_nu, log_sigma_log_alpha, log_sigma_log_t0, beta)
bf_global_n100               0.899  0.325  0.247   0.57  0.907  0.270  0.098    0.2  0.844 0.890 0.978 0.917 0.813 0.880 0.970
bf_global_wide_n100          0.926  0.293  0.106   0.79  0.907  0.275  0.082    0.3  0.901 0.925 0.984 0.932 0.821 0.933 0.983
compass-covout_cov_s7_n100   0.940  0.299  0.086   0.80  0.943  0.279  0.023  258.1  0.933 0.947 0.986 0.939 0.831 0.960 0.986
compass-covout_cov_s8_n100   0.938  0.274  0.072   0.83  0.943  0.277  0.018  253.0  0.925 0.948 0.987 0.948 0.829 0.950 0.980
compass-paper_e500_s7_n100   0.940  0.304  0.086   0.80  0.943  0.278  0.023  257.9  0.933 0.947 0.986 0.939 0.831 0.960 0.986
compass-paper_e500_s8_n100   0.938  0.279  0.072   0.83  0.943  0.277  0.018  250.8  0.925 0.948 0.987 0.948 0.829 0.950 0.980
compass-paper_e500_s9_n100   0.935  0.305  0.078   0.83  0.942  0.269  0.034  248.1  0.914 0.930 0.986 0.950 0.829 0.949 0.986
compass-paper_final_s9_n100   0.933  0.282  0.059   0.85  0.943  0.279  0.016  223.1  0.915 0.919 0.986 0.950 0.825 0.949 0.985
paper-final-seed1_joint_wide_n100   0.933  0.308  0.037   0.90  0.942  0.280  0.025    7.5  0.925 0.936 0.985 0.949 0.800 0.953 0.981
paper-final-seed2_joint_wide_n100   0.926  0.314  0.042   0.88  0.942  0.280  0.026    8.5  0.924 0.934 0.985 0.947 0.760 0.948 0.981
paper-final_joint_wide_n100   0.926  0.303  0.038   0.87  0.942  0.279  0.026    3.4  0.921 0.931 0.985 0.948 0.764 0.951 0.982
```

## 2. Partial pooling, COMPASS network

### Final method, 3 training seeds x 100 datasets

| arm | seeds | datasets | mean r | global RMSE | cover90 | z_rms | local RMSE | collapses | s/dataset |
|---|---|---|---|---|---|---|---|---|---|
| final: JAC-FD r9 | 3 | 100 | 0.940 (0.936–0.947) | 0.359 | 0.81 | 1.20 | 0.346 | 0 | 127 |
| final net, netcov r9 | 3 | 100 | 0.941 (0.936–0.948) | 0.357 | 0.84 | 1.13 | 0.345 | 0 | 110 |
| final recipe, fixed 500 epochs (no early stopping) | 3 | 100 | 0.941 (0.936–0.947) | 0.356 | 0.80 | 1.21 | 0.347 | 0 | 152 |

### Sampler ablations and baselines (seed-7 final net, datasets 0-29, paired)

| arm | seeds | datasets | mean r | global RMSE | cover90 | z_rms | local RMSE | collapses | s/dataset |
|---|---|---|---|---|---|---|---|---|---|
| final: JAC-FD r9 | 1 | 30 | 0.938 | 0.365 | 0.79 | 1.16 | 0.344 | 0 | 136 |
| r1 | 1 | 30 | 0.939 | 0.364 | 0.80 | 1.17 | 0.344 | 0 | 475 |
| fp64 | 0 | — | — | — | — | — | — | — | — |
| dense | 0 | — | — | — | — | — | — | — | — |
| noproj | 1 | 30 | 0.937 | 0.369 | 0.78 | 1.19 | 0.345 | 0 | 163 |
| nocurv | 1 | 30 | 0.938 | 0.367 | 0.80 | 1.17 | 0.344 | 0 | 109 |
| consistent | 1 | 30 | 0.783 | 0.702 | 0.90 | 0.73 | 0.375 | 0 | 103 |
| gate | 1 | 30 | 0.928 | 0.430 | 0.65 | 1.79 | 0.348 | 0 | 105 |
| gloeckler | 1 | 30 | 0.940 | 0.366 | 0.83 | 1.03 | 0.345 | 0 | 99 |
| metric | 1 | 30 | 0.938 | 0.365 | 0.79 | 1.16 | 0.345 | 0 | 172 |
| netcov r9 | 1 | 30 | 0.938 | 0.368 | 0.81 | 1.12 | 0.345 | 0 | 88 |
| netcov r1 | 1 | 30 | 0.938 | 0.368 | 0.81 | 1.13 | 0.344 | 0 | 169 |
| gauss_hierarchical (pilot) | 1 | 30 | 0.934 | 0.374 | 0.82 | 1.03 | 0.349 | 0 | 207 |
| arruda_direct (plain sum) | 1 | 30 | 0.633 | 3.736 | 0.44 | 17.73 | 1.101 | 26 | 123 |
| no churn | 1 | 30 | 0.940 | 0.364 | 0.86 | 1.05 | 0.353 | 0 | 146 |
| best-validation weights | 1 | 30 | 0.938 | 0.365 | 0.79 | 1.16 | 0.344 | 0 | 103 |
| nowide net: JAC-FD r9 | 1 | 30 | 0.944 | 0.369 | 0.78 | 1.29 | 0.350 | 0 | 99 |
| nowide net: F-NPSE (Langevin) | 1 | 30 | 0.904 | 0.565 | 0.54 | 2.68 | 0.384 | 0 | 312 |
| 25 steps | 1 | 30 | 0.938 | 0.364 | 0.81 | 1.11 | 0.345 | 0 | 81 |
| 100 steps | 1 | 30 | 0.934 | 0.375 | 0.76 | 1.23 | 0.346 | 0 | 199 |

Same draws? fp64 eigh and the dense / Cholesky route against the final arm, datasets 0-4 (max |difference| over all global draws, in units of the posterior sd; seconds per dataset):

| variant | max diff / sd | median diff / sd | s/dataset | final s/dataset |
|---|---|---|---|---|
| fp64 | 4.79e-03 | 4.38e-05 | 172 | 120 |
| dense | 0.00e+00 | 0.00e+00 | 173 | 120 |

seed-8 cross-check of the two PD choices closest to the default:

| arm | seeds | datasets | mean r | global RMSE | cover90 | z_rms | local RMSE | collapses | s/dataset |
|---|---|---|---|---|---|---|---|---|---|
| final (seed 8) | 1 | 30 | 0.952 | 0.343 | 0.82 | 1.26 | 0.346 | 0 | 136 |
| consistent (seed 8) | 1 | 30 | 0.827 | 0.692 | 0.85 | 0.85 | 0.379 | 0 | 103 |
| gate (seed 8) | 1 | 30 | 0.940 | 0.429 | 0.67 | 1.92 | 0.349 | 0 | 52 |

### Number of subjects (seed-7 final net, datasets 0-29)

| arm | seeds | datasets | mean r | global RMSE | cover90 | z_rms | local RMSE | collapses | s/dataset |
|---|---|---|---|---|---|---|---|---|---|
| n = 10 | 1 | 30 | 0.808 | 0.613 | 0.86 | 1.09 | 0.456 | 0 | 51 |
| n = 25 | 1 | 30 | 0.890 | 0.474 | 0.84 | 1.06 | 0.390 | 0 | 60 |
| n = 100 | 1 | 30 | 0.938 | 0.365 | 0.79 | 1.16 | 0.344 | 0 | 136 |

### Training ablations (one component of the final recipe changed; datasets 0-29)

| arm | seeds | datasets | mean r | global RMSE | cover90 | z_rms | local RMSE | collapses | s/dataset |
|---|---|---|---|---|---|---|---|---|---|
| final recipe | 2 | 30 | 0.945 (0.938–0.952) | 0.354 | 0.81 | 1.21 | 0.345 | 0 | 136 |
| nocov | 2 | 30 | 0.949 (0.944–0.954) | 0.342 | 0.87 | 1.02 | 0.344 | 0 | 103 |
| noedm | 2 | 30 | 0.957 (0.956–0.958) | 0.317 | 0.89 | 0.95 | 0.342 | 0 | 102 |
| unseeded | 2 | 30 | 0.902 (0.885–0.919) | 0.481 | 0.81 | 1.40 | 0.352 | 0 | 102 |
| tsm | 2 | 30 | 0.948 (0.946–0.950) | 0.342 | 0.83 | 1.14 | 0.346 | 0 | 102 |
| norg | 1 | 30 | 0.942 | 0.412 | 0.59 | 1.92 | 0.351 | 0 | 103 |
| nowide | 1 | 30 | 0.944 | 0.369 | 0.78 | 1.29 | 0.350 | 0 | 99 |
| full | 1 | 30 | 0.941 | 0.393 | 0.80 | 1.31 | 0.345 | 0 | 102 |

Training length (epochs run, best-validation epoch, minutes):

| network | epochs | best epoch | minutes |
|---|---|---|---|
| full_Rg_wide1.5_paper_full_s7 | 269 | 248 | 76 |
| hierarchical_Rg_paper_nowide_s7 | 403 | 382 | 113 |
| hierarchical_Rg_wide1.5_paper_e500_s7 | 500 | 499 | 148 |
| hierarchical_Rg_wide1.5_paper_e500_s8 | 500 | 499 | 159 |
| hierarchical_Rg_wide1.5_paper_e500_s9 | 500 | 419 | 134 |
| hierarchical_Rg_wide1.5_paper_final_s9 | 440 | 419 | 147 |
| hierarchical_Rg_wide1.5_paper_nocov_s7 | 500 | 499 | 120 |
| hierarchical_Rg_wide1.5_paper_nocov_s8 | 500 | 499 | 120 |
| hierarchical_Rg_wide1.5_paper_noedm_s7 | 500 | 499 | 150 |
| hierarchical_Rg_wide1.5_paper_noedm_s8 | 500 | 499 | 151 |
| hierarchical_Rg_wide1.5_paper_tsm_s7 | 476 | 455 | 154 |
| hierarchical_Rg_wide1.5_paper_tsm_s8 | 500 | 499 | 159 |
| hierarchical_Rg_wide1.5_paper_unseeded_s7 | 150 | 129 | 46 |
| hierarchical_Rg_wide1.5_paper_unseeded_s8 | 119 | 98 | 37 |
| hierarchical_wide1.5_paper_norg_s7 | 470 | 449 | 155 |
| hierarchical_Rg_wide1.5_covout_cov_s7 | 500 | 499 | 201 |
| hierarchical_Rg_wide1.5_covout_cov_s8 | 500 | 499 | 161 |

## 3. Exact-reference problems (toys, IK, latent-attention GM)

### Final method vs baselines (W1 / exact sd; |P(g>0) error| in brackets)

| benchmark | arm | n | P exact | final (JAC-FD r9) | gauss_hierarchical | arruda_direct | arruda_gauss_direct | netcov |
|---|---|---|---|---|---|---|---|---|
| GM asymmetric | hierarchical | 8 | 0.75 | 0.56 (0.21) | — | 1.30 (0.75) | 0.63 (0.24) | — |
| GM asymmetric | hierarchical | 16 | 0.75 | 1.30 (0.45) | — | 1.70 (0.75) | 1.34 (0.47) | — |
| GM asymmetric | hierarchical | 32 | 0.74 | 2.66 (0.62) | — | 3.22 (0.74) | 2.81 (0.64) | — |
| GM asymmetric | hierarchical | 64 | 0.76 | 5.08 (0.72) | — | 5.79 (0.76) | 5.25 (0.73) | — |
| GM asymmetric | oracle | 8 | 0.75 | 0.65 (0.09) | — | 0.50 (0.22) | — | — |
| GM asymmetric | oracle | 16 | 0.75 | 1.19 (0.06) | — | 0.42 (0.18) | — | — |
| GM asymmetric | oracle | 32 | 0.74 | 2.40 (0.11) | — | 0.46 (0.18) | — | — |
| GM asymmetric | oracle | 64 | 0.76 | 4.48 (0.14) | — | 0.47 (0.21) | — | — |
| GM symmetric | hierarchical | 8 | 0.51 | 0.36 (0.14) | — | 0.87 (0.51) | 0.39 (0.16) | — |
| GM symmetric | hierarchical | 16 | 0.48 | 0.81 (0.26) | — | 0.99 (0.48) | 0.88 (0.30) | — |
| GM symmetric | hierarchical | 32 | 0.49 | 1.93 (0.41) | — | 2.30 (0.49) | 2.04 (0.42) | — |
| GM symmetric | hierarchical | 64 | 0.52 | 4.02 (0.49) | — | 4.59 (0.52) | 4.17 (0.50) | — |
| GM symmetric | oracle | 8 | 0.51 | 0.52 (0.06) | — | 0.19 (0.07) | — | — |
| GM symmetric | oracle | 16 | 0.48 | 1.10 (0.12) | — | 0.18 (0.03) | — | — |
| GM symmetric | oracle | 32 | 0.49 | 2.11 (0.07) | — | 0.26 (0.01) | — | — |
| GM symmetric | oracle | 64 | 0.52 | 3.99 (0.08) | — | 0.08 (0.02) | — | — |
| IK absx1 | hierarchical attn | 8 | 0.50 | 0.14 (0.07) | — | — | — | — |
| IK absx1 | hierarchical attn | 32 | 0.50 | 0.16 (0.04) | — | — | — | — |
| IK absx1 | hierarchical attn | 128 | 0.50 | 0.09 (0.01) | — | — | — | — |
| IK flat_fixed | hierarchical attn | 8 | 0.52 | 0.08 (0.03) | — | — | — | — |
| IK flat_fixed | hierarchical attn | 32 | 0.32 | 0.48 (0.22) | — | — | — | — |
| IK flat_fixed | hierarchical attn | 128 | 0.25 | 0.64 (0.27) | — | — | — | — |
| IK flat_global | hierarchical attn | 8 | 0.50 | 0.17 (0.06) | — | — | — | — |
| IK flat_global | hierarchical attn | 32 | 0.50 | 0.27 (0.08) | — | — | — | — |
| IK flat_global | hierarchical attn | 128 | 0.09 | 3.57 (0.49) | — | — | — | — |
| IK reach_shift | hierarchical attn | 8 | 0.67 | 0.06 (0.00) | — | 0.09 (0.04) | — | — |
| IK reach_shift | hierarchical attn | 32 | 0.68 | 0.08 (0.01) | — | 0.33 (0.15) | — | — |
| IK reach_shift | hierarchical attn | 128 | 0.65 | 0.21 (0.06) | — | 0.59 (0.25) | — | — |
| IK reach_shift (final recipe) | hierarchical attn | 8 | 0.67 | 0.05 (0.00) | — | 0.06 (0.02) | — | 0.12 (0.01) |
| IK reach_shift (final recipe) | hierarchical attn | 32 | 0.68 | 0.05 (0.01) | — | 0.25 (0.11) | — | 0.18 (0.01) |
| IK reach_shift (final recipe) | hierarchical attn | 128 | 0.65 | 0.15 (0.06) | — | 0.26 (0.10) | — | 0.28 (0.06) |
| IK reach_sym | hierarchical attn | 8 | 0.50 | 0.08 (0.01) | — | 0.07 (0.03) | — | — |
| IK reach_sym | hierarchical attn | 32 | 0.50 | 0.09 (0.02) | — | 0.21 (0.10) | — | — |
| IK reach_sym | hierarchical attn | 128 | 0.50 | 0.13 (0.02) | — | 0.57 (0.26) | — | — |
| IK reach_sym (final recipe) | hierarchical attn | 8 | 0.50 | 0.08 (0.01) | — | 0.02 (0.00) | — | 0.14 (0.01) |
| IK reach_sym (final recipe) | hierarchical attn | 32 | 0.50 | 0.05 (0.00) | — | 0.03 (0.00) | — | 0.18 (0.01) |
| IK reach_sym (final recipe) | hierarchical attn | 128 | 0.50 | 0.13 (0.04) | — | 0.20 (0.05) | — | 0.25 (0.04) |
| IK reach_sym_bernoulli | hierarchical attn | 8 | 0.50 | 0.06 (0.00) | — | — | — | — |
| IK reach_sym_bernoulli | hierarchical attn | 32 | 0.50 | 0.07 (0.01) | — | — | — | — |
| IK reach_sym_bernoulli | hierarchical attn | 128 | 0.50 | 0.15 (0.03) | — | — | — | — |
| toy abs | hierarchical | 4 | 0.50 | 0.05 (0.02) | 0.04 (0.01) | 0.08 (0.02) | — | — |
| toy abs | hierarchical | 16 | 0.50 | 0.05 (0.01) | 0.08 (0.03) | 0.08 (0.02) | — | — |
| toy abs | hierarchical | 64 | 0.50 | 0.06 (0.03) | 0.07 (0.03) | 0.35 (0.18) | — | — |
| toy abs | hierarchical | 128 | 0.50 | 0.06 (0.03) | 0.11 (0.04) | 0.49 (0.24) | — | — |
| toy abs | oracle_joint | 4 | 0.50 | 0.01 (0.00) | 0.03 (0.02) | — | — | — |
| toy abs | oracle_joint | 16 | 0.50 | 0.01 (0.00) | 0.01 (0.00) | — | — | — |
| toy abs | oracle_joint | 64 | 0.50 | 0.04 (0.01) | 0.04 (0.00) | — | — | — |
| toy abs | oracle_joint | 128 | 0.50 | 0.05 (0.01) | 0.06 (0.01) | — | — | — |
| toy curved | hierarchical | 4 | 0.62 | 0.04 (0.02) | 0.08 (0.04) | — | — | — |
| toy curved | hierarchical | 16 | 0.63 | 0.11 (0.05) | 0.13 (0.06) | — | — | — |
| toy curved | hierarchical | 64 | 0.62 | 0.07 (0.03) | 0.05 (0.01) | — | — | — |
| toy curved | hierarchical | 128 | 0.62 | 0.04 (0.01) | 0.04 (0.01) | — | — | — |
| toy lean | hierarchical | 4 | 0.74 | 0.05 (0.01) | 0.12 (0.04) | — | — | — |
| toy lean | hierarchical | 16 | 0.78 | 0.09 (0.04) | 0.27 (0.11) | — | — | — |
| toy lean | hierarchical | 64 | 0.74 | 0.10 (0.02) | 0.36 (0.13) | — | — | — |
| toy lean | hierarchical | 128 | 0.72 | 0.11 (0.00) | 0.31 (0.10) | — | — | — |
| toy lean | oracle_joint | 4 | 0.74 | 0.06 (0.02) | 0.11 (0.04) | — | — | — |
| toy lean | oracle_joint | 16 | 0.78 | 0.27 (0.11) | 0.36 (0.14) | — | — | — |
| toy lean | oracle_joint | 64 | 0.74 | 0.20 (0.09) | 0.37 (0.16) | — | — | — |
| toy lean | oracle_joint | 128 | 0.72 | 0.18 (0.08) | 0.28 (0.12) | — | — | — |
| toy square | hierarchical | 4 | 0.50 | 0.04 (0.01) | 0.07 (0.03) | — | — | — |
| toy square | hierarchical | 16 | 0.50 | 0.06 (0.02) | 0.09 (0.04) | — | — | — |
| toy square | hierarchical | 64 | 0.50 | 0.09 (0.04) | 0.10 (0.04) | — | — | — |
| toy square | hierarchical | 128 | 0.50 | 0.10 (0.05) | 0.08 (0.03) | — | — | — |

### Sampler variants (JAC-FD rule): W1 / exact sd

| benchmark | arm | n | final | r1 | fp64 | dense | noproj | nocurv | consistent | gate | gloeckler | metric |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM asymmetric | hierarchical | 8 | 0.56 | — | — | — | 0.54 | — | 0.56 | 0.56 | 0.55 | 0.56 |
| GM asymmetric | hierarchical | 16 | 1.30 | — | — | — | 1.29 | — | 1.53 | 1.30 | 1.30 | 1.30 |
| GM asymmetric | hierarchical | 32 | 2.66 | — | — | — | 2.68 | — | 3.08 | 2.66 | 2.65 | 2.66 |
| GM asymmetric | hierarchical | 64 | 5.08 | — | — | — | 5.08 | — | 5.52 | 5.08 | 5.04 | 5.07 |
| GM asymmetric | oracle | 8 | 0.65 | — | — | — | — | — | 0.65 | — | — | 0.65 |
| GM asymmetric | oracle | 16 | 1.19 | — | — | — | — | — | 1.19 | — | — | 1.19 |
| GM asymmetric | oracle | 32 | 2.40 | — | — | — | — | — | 2.40 | — | — | 2.40 |
| GM asymmetric | oracle | 64 | 4.48 | — | — | — | — | — | 4.49 | — | — | 4.48 |
| GM symmetric | hierarchical | 8 | 0.36 | — | — | — | 0.36 | — | 0.46 | 0.36 | 0.36 | 0.36 |
| GM symmetric | hierarchical | 16 | 0.81 | — | — | — | 0.80 | — | 1.14 | 0.81 | 0.79 | 0.81 |
| GM symmetric | hierarchical | 32 | 1.93 | — | — | — | 1.95 | — | 2.32 | 1.94 | 1.94 | 1.94 |
| GM symmetric | hierarchical | 64 | 4.02 | — | — | — | 4.01 | — | 4.34 | 4.02 | 4.01 | 4.02 |
| GM symmetric | oracle | 8 | 0.52 | — | — | — | — | — | 0.51 | — | — | 0.52 |
| GM symmetric | oracle | 16 | 1.10 | — | — | — | — | — | 1.10 | — | — | 1.10 |
| GM symmetric | oracle | 32 | 2.11 | — | — | — | — | — | 2.11 | — | — | 2.11 |
| GM symmetric | oracle | 64 | 3.99 | — | — | — | — | — | 4.00 | — | — | 3.99 |
| IK absx1 | hierarchical attn | 8 | 0.14 | 0.15 | — | — | 0.19 | — | 0.25 | 0.21 | 0.18 | 0.14 |
| IK absx1 | hierarchical attn | 32 | 0.16 | 0.16 | — | — | 0.23 | — | 0.23 | 0.20 | 0.18 | 0.16 |
| IK absx1 | hierarchical attn | 128 | 0.09 | 0.10 | — | — | 0.13 | — | 0.20 | 0.12 | 0.10 | 0.09 |
| IK flat_fixed | hierarchical attn | 8 | 0.08 | 0.09 | — | — | 0.03 | — | 0.08 | 0.08 | 0.08 | 0.08 |
| IK flat_fixed | hierarchical attn | 32 | 0.48 | 0.50 | — | — | 0.49 | — | 0.48 | 0.48 | 0.45 | 0.48 |
| IK flat_fixed | hierarchical attn | 128 | 0.64 | 0.73 | — | — | 0.65 | — | 0.89 | 0.65 | 0.64 | 0.64 |
| IK flat_global | hierarchical attn | 8 | 0.17 | 0.20 | — | — | 1.63 | — | 0.06 | 0.16 | 1.22 | 0.17 |
| IK flat_global | hierarchical attn | 32 | 0.27 | 0.29 | — | — | 1.72 | — | 0.05 | 0.17 | 1.10 | 0.24 |
| IK flat_global | hierarchical attn | 128 | 3.57 | 3.67 | — | — | 5.73 | — | 1.95 | 2.29 | 4.80 | 2.50 |
| IK reach_shift | hierarchical attn | 8 | 0.06 | 0.05 | — | — | 0.13 | — | 0.46 | 0.34 | 0.05 | 0.06 |
| IK reach_shift | hierarchical attn | 32 | 0.08 | 0.07 | — | — | 0.26 | — | 0.52 | 0.40 | 0.11 | 0.08 |
| IK reach_shift | hierarchical attn | 128 | 0.21 | 0.21 | — | — | 0.13 | — | 0.56 | 0.45 | 0.15 | 0.21 |
| IK reach_shift (final recipe) | hierarchical attn | 8 | 0.05 | — | — | — | — | — | — | — | — | — |
| IK reach_shift (final recipe) | hierarchical attn | 32 | 0.05 | — | — | — | — | — | — | — | — | — |
| IK reach_shift (final recipe) | hierarchical attn | 128 | 0.15 | — | — | — | — | — | — | — | — | — |
| IK reach_sym | hierarchical attn | 8 | 0.08 | 0.07 | — | — | 0.23 | — | 0.12 | 0.11 | 0.12 | 0.08 |
| IK reach_sym | hierarchical attn | 32 | 0.09 | 0.09 | — | — | 0.35 | — | 0.08 | 0.11 | 0.19 | 0.09 |
| IK reach_sym | hierarchical attn | 128 | 0.13 | 0.12 | — | — | 0.15 | — | 0.13 | 0.21 | 0.13 | 0.13 |
| IK reach_sym (final recipe) | hierarchical attn | 8 | 0.08 | — | — | — | — | — | — | — | — | — |
| IK reach_sym (final recipe) | hierarchical attn | 32 | 0.05 | — | — | — | — | — | — | — | — | — |
| IK reach_sym (final recipe) | hierarchical attn | 128 | 0.13 | — | — | — | — | — | — | — | — | — |
| IK reach_sym_bernoulli | hierarchical attn | 8 | 0.06 | 0.06 | — | — | 0.14 | — | 0.13 | 0.11 | 0.07 | 0.06 |
| IK reach_sym_bernoulli | hierarchical attn | 32 | 0.07 | 0.07 | — | — | 0.23 | — | 0.27 | 0.15 | 0.09 | 0.07 |
| IK reach_sym_bernoulli | hierarchical attn | 128 | 0.15 | 0.14 | — | — | 0.14 | — | 0.19 | 0.13 | 0.15 | 0.15 |
| toy abs | hierarchical | 4 | 0.05 | 0.05 | — | — | 0.04 | — | 0.18 | 0.05 | 0.06 | 0.05 |
| toy abs | hierarchical | 16 | 0.05 | 0.05 | — | — | 0.07 | — | 0.18 | 0.05 | 0.09 | 0.05 |
| toy abs | hierarchical | 64 | 0.06 | 0.02 | — | — | 0.27 | — | 0.28 | 0.07 | 0.11 | 0.06 |
| toy abs | hierarchical | 128 | 0.06 | 0.03 | — | — | 0.15 | — | 0.26 | 0.07 | 0.09 | 0.06 |
| toy abs | oracle_joint | 4 | 0.01 | 0.01 | — | — | 0.03 | — | 0.02 | 0.01 | 0.02 | 0.01 |
| toy abs | oracle_joint | 16 | 0.01 | 0.02 | — | — | 0.01 | — | 0.01 | 0.01 | 0.01 | 0.01 |
| toy abs | oracle_joint | 64 | 0.04 | 0.02 | — | — | 0.04 | — | 0.02 | 0.04 | 0.03 | 0.04 |
| toy abs | oracle_joint | 128 | 0.05 | 0.04 | — | — | 0.04 | — | 0.02 | 0.04 | 0.04 | 0.05 |
| toy curved | hierarchical | 4 | 0.04 | 0.07 | — | — | 0.16 | — | 0.40 | 0.04 | 0.10 | 0.04 |
| toy curved | hierarchical | 16 | 0.11 | 0.18 | — | — | 0.17 | — | 0.42 | 0.10 | 0.11 | 0.11 |
| toy curved | hierarchical | 64 | 0.07 | 0.08 | — | — | 0.08 | — | 0.39 | 0.05 | 0.03 | 0.07 |
| toy curved | hierarchical | 128 | 0.04 | 0.05 | — | — | 0.03 | — | 0.39 | 0.05 | 0.06 | 0.04 |
| toy lean | hierarchical | 4 | 0.05 | 0.06 | — | — | 0.13 | — | 0.42 | 0.05 | 0.03 | 0.05 |
| toy lean | hierarchical | 16 | 0.09 | 0.08 | — | — | 0.12 | — | 0.39 | 0.09 | 0.05 | 0.09 |
| toy lean | hierarchical | 64 | 0.10 | 0.10 | — | — | 0.09 | — | 0.44 | 0.10 | 0.10 | 0.10 |
| toy lean | hierarchical | 128 | 0.11 | 0.12 | — | — | 0.14 | — | 0.49 | 0.10 | 0.12 | 0.10 |
| toy lean | oracle_joint | 4 | 0.06 | 0.09 | — | — | 0.31 | — | 0.46 | 0.06 | 0.11 | 0.06 |
| toy lean | oracle_joint | 16 | 0.27 | 0.34 | — | — | 0.32 | — | 0.48 | 0.27 | 0.18 | 0.27 |
| toy lean | oracle_joint | 64 | 0.20 | 0.29 | — | — | 0.28 | — | 0.52 | 0.20 | 0.13 | 0.20 |
| toy lean | oracle_joint | 128 | 0.18 | 0.23 | — | — | 0.35 | — | 0.55 | 0.18 | 0.15 | 0.18 |
| toy square | hierarchical | 4 | 0.04 | 0.03 | — | — | 0.04 | — | 0.15 | 0.07 | 0.04 | 0.04 |
| toy square | hierarchical | 16 | 0.06 | 0.05 | — | — | 0.07 | — | 0.09 | 0.11 | 0.08 | 0.06 |
| toy square | hierarchical | 64 | 0.09 | 0.08 | — | — | 0.12 | — | 0.08 | 0.12 | 0.12 | 0.09 |
| toy square | hierarchical | 128 | 0.10 | 0.10 | — | — | 0.15 | — | 0.13 | 0.12 | 0.13 | 0.10 |

### Sampler variants, summary over every exact-reference case (JAC-FD rule)

Ratio = (W1_variant + 0.02) / (W1_final + 0.02); the 0.02 keeps near-exact cases from dominating. Better / worse: ratio below 0.87 / above 1.15 (the ±15% band of the earlier safeguard comparisons).

| variant | cases | geometric-mean ratio | worst ratio | better | same | worse |
|---|---|---|---|---|---|---|
| r1 | 42 | 1.03 | 1.56 | 4 | 29 | 9 |
| fp64 | 0 | — | — | — | — | — |
| dense | 0 | — | — | — | — | — |
| noproj | 50 | 1.45 | 8.73 | 4 | 17 | 29 |
| nocurv | 0 | — | — | — | — | — |
| consistent | 58 | 1.66 | 7.12 | 6 | 15 | 37 |
| gate | 50 | 1.15 | 4.73 | 3 | 31 | 16 |
| gloeckler | 50 | 1.14 | 6.57 | 8 | 24 | 18 |
| metric | 58 | 0.99 | 1.02 | 1 | 57 | 0 |

## 4. GM with the covariance-output networks (exact MC; 2 seeds x 2 replicates)

| arm | n | regime | method | W1/sd | branch err | width | local W1 | s | k |
|---|---|---|---|---|---|---|---|---|---|
| cov | 1 | asymmetric | GH pilot, Gaussian prior | 0.046 | 0.017 | 1.02 | 0.034 | 25 | 4 |
| cov | 1 | asymmetric | GH pilot, mixture prior | 0.046 | 0.017 | 1.02 | 0.034 | 26 | 4 |
| cov | 1 | asymmetric | JAC-FD r9, Gaussian prior | 0.037 | 0.013 | 1.01 | 0.047 | 12 | 4 |
| cov | 1 | asymmetric | JAC-FD r9, mixture prior | 0.037 | 0.013 | 1.01 | 0.047 | 11 | 4 |
| cov | 1 | asymmetric | netcov, mixture prior (r1) | 0.037 | 0.013 | 1.01 | 0.047 | 13 | 4 |
| cov | 1 | asymmetric | netcov, mixture prior (r9) | 0.037 | 0.013 | 1.01 | 0.047 | 11 | 4 |
| cov | 1 | asymmetric | netcov, Gaussian prior (r1) | 0.037 | 0.013 | 1.01 | 0.047 | 13 | 4 |
| cov | 1 | asymmetric | netcov, Gaussian prior (r9) | 0.037 | 0.013 | 1.01 | 0.047 | 11 | 4 |
| cov | 1 | asymmetric | F-NPSE (Langevin) | 0.054 | 0.020 | 1.03 | 0.061 | 20 | 4 |
| cov | 1 | symmetric | GH pilot, Gaussian prior | 0.028 | 0.012 | 0.99 | 0.031 | 31 | 4 |
| cov | 1 | symmetric | GH pilot, mixture prior | 0.028 | 0.012 | 0.99 | 0.031 | 31 | 4 |
| cov | 1 | symmetric | JAC-FD r9, Gaussian prior | 0.025 | 0.010 | 1.00 | 0.042 | 16 | 4 |
| cov | 1 | symmetric | JAC-FD r9, mixture prior | 0.025 | 0.010 | 1.00 | 0.042 | 16 | 4 |
| cov | 1 | symmetric | netcov, mixture prior (r1) | 0.025 | 0.010 | 1.00 | 0.042 | 10 | 4 |
| cov | 1 | symmetric | netcov, mixture prior (r9) | 0.025 | 0.010 | 1.00 | 0.042 | 15 | 4 |
| cov | 1 | symmetric | netcov, Gaussian prior (r1) | 0.025 | 0.010 | 1.00 | 0.042 | 10 | 4 |
| cov | 1 | symmetric | netcov, Gaussian prior (r9) | 0.025 | 0.010 | 1.00 | 0.042 | 15 | 4 |
| cov | 1 | symmetric | F-NPSE (Langevin) | 0.027 | 0.010 | 1.00 | 0.042 | 24 | 4 |
| cov | 8 | asymmetric | GH pilot, Gaussian prior | 1.258 | 0.136 | 1.33 | 1.348 | 32 | 4 |
| cov | 8 | asymmetric | GH pilot, mixture prior | 0.665 | 0.300 | 0.60 | 1.349 | 35 | 4 |
| cov | 8 | asymmetric | JAC-FD r9, Gaussian prior | 1.072 | 0.048 | 1.81 | 1.362 | 21 | 4 |
| cov | 8 | asymmetric | JAC-FD r9, mixture prior | 0.393 | 0.076 | 1.13 | 1.362 | 22 | 4 |
| cov | 8 | asymmetric | netcov, mixture prior (r1) | 0.357 | 0.071 | 1.22 | 1.255 | 37 | 4 |
| cov | 8 | asymmetric | netcov, mixture prior (r9) | 0.361 | 0.072 | 1.18 | 1.240 | 22 | 4 |
| cov | 8 | asymmetric | netcov, Gaussian prior (r1) | 1.066 | 0.045 | 1.91 | 1.274 | 34 | 4 |
| cov | 8 | asymmetric | netcov, Gaussian prior (r9) | 1.077 | 0.053 | 1.86 | 1.261 | 21 | 4 |
| cov | 8 | asymmetric | F-NPSE (Langevin) | 1.551 | 0.307 | 2.12 | 1.345 | 46 | 4 |
| cov | 8 | symmetric | GH pilot, Gaussian prior | 0.968 | 0.063 | 1.85 | 1.286 | 29 | 4 |
| cov | 8 | symmetric | GH pilot, mixture prior | 0.606 | 0.157 | 0.56 | 1.286 | 30 | 4 |
| cov | 8 | symmetric | JAC-FD r9, Gaussian prior | 1.051 | 0.114 | 1.80 | 1.287 | 21 | 4 |
| cov | 8 | symmetric | JAC-FD r9, mixture prior | 0.354 | 0.073 | 1.12 | 1.287 | 24 | 4 |
| cov | 8 | symmetric | netcov, mixture prior (r1) | 0.291 | 0.041 | 1.17 | 1.178 | 21 | 4 |
| cov | 8 | symmetric | netcov, mixture prior (r9) | 0.297 | 0.046 | 1.16 | 1.163 | 19 | 4 |
| cov | 8 | symmetric | netcov, Gaussian prior (r1) | 0.997 | 0.083 | 1.84 | 1.204 | 23 | 4 |
| cov | 8 | symmetric | netcov, Gaussian prior (r9) | 0.992 | 0.080 | 1.84 | 1.193 | 21 | 4 |
| cov | 8 | symmetric | F-NPSE (Langevin) | 1.210 | 0.208 | 1.68 | 1.280 | 44 | 4 |
| cov | 16 | asymmetric | GH pilot, Gaussian prior | 2.379 | 0.178 | 3.16 | 1.667 | 38 | 4 |
| cov | 16 | asymmetric | GH pilot, mixture prior | 1.079 | 0.738 | 0.31 | 1.669 | 38 | 4 |
| cov | 16 | asymmetric | JAC-FD r9, Gaussian prior | 2.397 | 0.181 | 3.11 | 1.660 | 33 | 4 |
| cov | 16 | asymmetric | JAC-FD r9, mixture prior | 0.918 | 0.378 | 1.44 | 1.660 | 36 | 4 |
| cov | 16 | asymmetric | netcov, mixture prior (r1) | 0.812 | 0.332 | 1.52 | 1.539 | 47 | 4 |
| cov | 16 | asymmetric | netcov, mixture prior (r9) | 0.818 | 0.339 | 1.50 | 1.525 | 32 | 4 |
| cov | 16 | asymmetric | netcov, Gaussian prior (r1) | 2.405 | 0.193 | 3.16 | 1.574 | 51 | 4 |
| cov | 16 | asymmetric | netcov, Gaussian prior (r9) | 2.401 | 0.190 | 3.15 | 1.564 | 36 | 4 |
| cov | 16 | asymmetric | F-NPSE (Langevin) | 2.553 | 0.253 | 2.18 | 1.658 | 81 | 4 |
| cov | 16 | symmetric | GH pilot, Gaussian prior | 1.964 | 0.086 | 2.78 | 1.515 | 38 | 4 |
| cov | 16 | symmetric | GH pilot, mixture prior | 0.859 | 0.331 | 0.19 | 1.515 | 40 | 4 |
| cov | 16 | symmetric | JAC-FD r9, Gaussian prior | 1.948 | 0.079 | 2.79 | 1.516 | 37 | 4 |
| cov | 16 | symmetric | JAC-FD r9, mixture prior | 0.620 | 0.108 | 1.25 | 1.516 | 38 | 4 |
| cov | 16 | symmetric | netcov, mixture prior (r1) | 0.564 | 0.097 | 1.34 | 1.401 | 42 | 4 |
| cov | 16 | symmetric | netcov, mixture prior (r9) | 0.570 | 0.105 | 1.31 | 1.387 | 34 | 4 |
| cov | 16 | symmetric | netcov, Gaussian prior (r1) | 1.913 | 0.045 | 2.81 | 1.435 | 43 | 4 |
| cov | 16 | symmetric | netcov, Gaussian prior (r9) | 1.924 | 0.056 | 2.80 | 1.426 | 34 | 4 |
| cov | 16 | symmetric | F-NPSE (Langevin) | 2.079 | 0.163 | 2.60 | 1.509 | 81 | 4 |
| cov | 32 | asymmetric | GH pilot, Gaussian prior | 4.427 | 0.180 | 5.13 | 1.910 | 56 | 4 |
| cov | 32 | asymmetric | GH pilot, mixture prior | 1.091 | 0.717 | 0.10 | 1.911 | 59 | 4 |
| cov | 32 | asymmetric | JAC-FD r9, Gaussian prior | 4.426 | 0.186 | 5.17 | 1.913 | 59 | 4 |
| cov | 32 | asymmetric | JAC-FD r9, mixture prior | 1.440 | 0.539 | 1.69 | 1.913 | 61 | 4 |
| cov | 32 | asymmetric | netcov, mixture prior (r1) | 1.229 | 0.480 | 1.82 | 1.775 | 91 | 4 |
| cov | 32 | asymmetric | netcov, mixture prior (r9) | 1.221 | 0.470 | 1.83 | 1.763 | 58 | 4 |
| cov | 32 | asymmetric | netcov, Gaussian prior (r1) | 4.383 | 0.164 | 5.21 | 1.821 | 83 | 4 |
| cov | 32 | asymmetric | netcov, Gaussian prior (r9) | 4.383 | 0.165 | 5.20 | 1.812 | 59 | 4 |
| cov | 32 | asymmetric | F-NPSE (Langevin) | 4.328 | 0.135 | 5.06 | 1.909 | 150 | 4 |
| cov | 32 | symmetric | GH pilot, Gaussian prior | 3.800 | 0.072 | 4.62 | 1.918 | 60 | 4 |
| cov | 32 | symmetric | GH pilot, mixture prior | 0.892 | 0.288 | 0.26 | 1.919 | 59 | 4 |
| cov | 32 | symmetric | JAC-FD r9, Gaussian prior | 3.824 | 0.092 | 4.57 | 1.920 | 59 | 4 |
| cov | 32 | symmetric | JAC-FD r9, mixture prior | 1.186 | 0.308 | 1.35 | 1.920 | 59 | 4 |
| cov | 32 | symmetric | netcov, mixture prior (r1) | 0.974 | 0.224 | 1.52 | 1.777 | 75 | 4 |
| cov | 32 | symmetric | netcov, mixture prior (r9) | 0.980 | 0.217 | 1.52 | 1.763 | 60 | 4 |
| cov | 32 | symmetric | netcov, Gaussian prior (r1) | 3.771 | 0.051 | 4.65 | 1.822 | 75 | 4 |
| cov | 32 | symmetric | netcov, Gaussian prior (r9) | 3.774 | 0.053 | 4.65 | 1.811 | 61 | 4 |
| cov | 32 | symmetric | F-NPSE (Langevin) | 4.231 | 0.328 | 3.14 | 1.906 | 150 | 4 |
| cov | 64 | asymmetric | GH pilot, Gaussian prior | 8.118 | 0.158 | 8.84 | 2.072 | 83 | 4 |
| cov | 64 | asymmetric | GH pilot, mixture prior | 1.168 | 0.614 | 0.13 | 2.072 | 88 | 4 |
| cov | 64 | asymmetric | JAC-FD r9, Gaussian prior | 8.254 | 0.240 | 8.85 | 2.074 | 94 | 4 |
| cov | 64 | asymmetric | JAC-FD r9, mixture prior | 2.394 | 0.640 | 2.08 | 2.074 | 89 | 4 |
| cov | 64 | asymmetric | netcov, mixture prior (r1) | 1.830 | 0.562 | 2.32 | 1.916 | 121 | 4 |
| cov | 64 | asymmetric | netcov, mixture prior (r9) | 1.829 | 0.546 | 2.35 | 1.902 | 83 | 4 |
| cov | 64 | asymmetric | netcov, Gaussian prior (r1) | 8.164 | 0.192 | 8.96 | 1.974 | 151 | 4 |
| cov | 64 | asymmetric | netcov, Gaussian prior (r9) | 8.159 | 0.188 | 8.95 | 1.963 | 87 | 4 |
| cov | 64 | asymmetric | F-NPSE (Langevin) | 8.092 | 0.124 | 7.29 | 2.070 | 270 | 4 |
| cov | 64 | symmetric | GH pilot, Gaussian prior | 7.240 | 0.087 | 7.96 | 2.069 | 107 | 4 |
| cov | 64 | symmetric | GH pilot, mixture prior | 0.864 | 0.413 | 0.15 | 2.069 | 102 | 4 |
| cov | 64 | symmetric | JAC-FD r9, Gaussian prior | 7.258 | 0.102 | 7.85 | 2.068 | 119 | 4 |
| cov | 64 | symmetric | JAC-FD r9, mixture prior | 1.799 | 0.409 | 1.70 | 2.068 | 127 | 4 |
| cov | 64 | symmetric | netcov, mixture prior (r1) | 1.391 | 0.372 | 1.97 | 1.913 | 139 | 4 |
| cov | 64 | symmetric | netcov, mixture prior (r9) | 1.421 | 0.361 | 1.99 | 1.900 | 109 | 4 |
| cov | 64 | symmetric | netcov, Gaussian prior (r1) | 7.213 | 0.056 | 8.04 | 1.969 | 151 | 4 |
| cov | 64 | symmetric | netcov, Gaussian prior (r9) | 7.217 | 0.060 | 8.03 | 1.957 | 115 | 4 |
| cov | 64 | symmetric | F-NPSE (Langevin) | 7.844 | 0.422 | 3.59 | 2.067 | 307 | 4 |
| cov_lo | 1 | asymmetric | GH pilot, Gaussian prior | 0.040 | 0.017 | 1.02 | 0.043 | 38 | 4 |
| cov_lo | 1 | asymmetric | GH pilot, mixture prior | 0.040 | 0.017 | 1.02 | 0.043 | 41 | 4 |
| cov_lo | 1 | asymmetric | JAC-FD r9, Gaussian prior | 0.041 | 0.017 | 1.02 | 0.059 | 23 | 4 |
| cov_lo | 1 | asymmetric | JAC-FD r9, mixture prior | 0.041 | 0.017 | 1.02 | 0.059 | 21 | 4 |
| cov_lo | 1 | asymmetric | netcov, mixture prior (r1) | 0.041 | 0.017 | 1.02 | 0.059 | 20 | 4 |
| cov_lo | 1 | asymmetric | netcov, mixture prior (r9) | 0.041 | 0.017 | 1.02 | 0.059 | 20 | 4 |
| cov_lo | 1 | asymmetric | netcov, Gaussian prior (r1) | 0.041 | 0.017 | 1.02 | 0.059 | 22 | 4 |
| cov_lo | 1 | asymmetric | netcov, Gaussian prior (r9) | 0.041 | 0.017 | 1.02 | 0.059 | 20 | 4 |
| cov_lo | 1 | asymmetric | F-NPSE (Langevin) | 0.086 | 0.035 | 1.05 | 0.067 | 37 | 4 |
| cov_lo | 1 | symmetric | GH pilot, Gaussian prior | 0.025 | 0.012 | 1.00 | 0.065 | 43 | 4 |
| cov_lo | 1 | symmetric | GH pilot, mixture prior | 0.025 | 0.012 | 1.00 | 0.065 | 42 | 4 |
| cov_lo | 1 | symmetric | JAC-FD r9, Gaussian prior | 0.025 | 0.011 | 1.00 | 0.067 | 22 | 4 |
| cov_lo | 1 | symmetric | JAC-FD r9, mixture prior | 0.025 | 0.011 | 1.00 | 0.067 | 21 | 4 |
| cov_lo | 1 | symmetric | netcov, mixture prior (r1) | 0.025 | 0.011 | 1.00 | 0.067 | 23 | 4 |
| cov_lo | 1 | symmetric | netcov, mixture prior (r9) | 0.025 | 0.011 | 1.00 | 0.067 | 22 | 4 |
| cov_lo | 1 | symmetric | netcov, Gaussian prior (r1) | 0.025 | 0.011 | 1.00 | 0.067 | 21 | 4 |
| cov_lo | 1 | symmetric | netcov, Gaussian prior (r9) | 0.025 | 0.011 | 1.00 | 0.067 | 20 | 4 |
| cov_lo | 1 | symmetric | F-NPSE (Langevin) | 0.054 | 0.024 | 1.00 | 0.079 | 34 | 4 |
| cov_lo | 8 | asymmetric | GH pilot, Gaussian prior | 1.172 | 0.097 | 1.48 | 1.341 | 41 | 4 |
| cov_lo | 8 | asymmetric | GH pilot, mixture prior | 0.583 | 0.245 | 0.60 | 1.342 | 42 | 4 |
| cov_lo | 8 | asymmetric | JAC-FD r9, Gaussian prior | 1.029 | 0.029 | 1.83 | 1.358 | 38 | 4 |
| cov_lo | 8 | asymmetric | JAC-FD r9, mixture prior | 0.310 | 0.056 | 1.18 | 1.358 | 38 | 4 |
| cov_lo | 8 | asymmetric | netcov, mixture prior (r1) | 0.351 | 0.078 | 1.24 | 1.254 | 48 | 4 |
| cov_lo | 8 | asymmetric | netcov, mixture prior (r9) | 0.322 | 0.060 | 1.20 | 1.231 | 30 | 4 |
| cov_lo | 8 | asymmetric | netcov, Gaussian prior (r1) | 1.058 | 0.050 | 1.91 | 1.263 | 47 | 4 |
| cov_lo | 8 | asymmetric | netcov, Gaussian prior (r9) | 1.050 | 0.043 | 1.86 | 1.244 | 31 | 4 |
| cov_lo | 8 | asymmetric | F-NPSE (Langevin) | 1.632 | 0.352 | 2.07 | 1.318 | 65 | 4 |
| cov_lo | 8 | symmetric | GH pilot, Gaussian prior | 0.988 | 0.089 | 1.81 | 1.299 | 45 | 4 |
| cov_lo | 8 | symmetric | GH pilot, mixture prior | 0.505 | 0.070 | 0.65 | 1.298 | 45 | 4 |
| cov_lo | 8 | symmetric | JAC-FD r9, Gaussian prior | 0.982 | 0.086 | 1.82 | 1.295 | 33 | 4 |
| cov_lo | 8 | symmetric | JAC-FD r9, mixture prior | 0.262 | 0.056 | 1.14 | 1.295 | 34 | 4 |
| cov_lo | 8 | symmetric | netcov, mixture prior (r1) | 0.250 | 0.043 | 1.16 | 1.184 | 45 | 4 |
| cov_lo | 8 | symmetric | netcov, mixture prior (r9) | 0.264 | 0.058 | 1.13 | 1.160 | 35 | 4 |
| cov_lo | 8 | symmetric | netcov, Gaussian prior (r1) | 0.963 | 0.075 | 1.82 | 1.202 | 48 | 4 |
| cov_lo | 8 | symmetric | netcov, Gaussian prior (r9) | 0.961 | 0.073 | 1.83 | 1.187 | 33 | 4 |
| cov_lo | 8 | symmetric | F-NPSE (Langevin) | 1.241 | 0.237 | 1.61 | 1.279 | 64 | 4 |
| cov_lo | 16 | asymmetric | GH pilot, Gaussian prior | 2.291 | 0.129 | 2.97 | 1.667 | 55 | 4 |
| cov_lo | 16 | asymmetric | GH pilot, mixture prior | 1.056 | 0.393 | 0.15 | 1.668 | 56 | 4 |
| cov_lo | 16 | asymmetric | JAC-FD r9, Gaussian prior | 2.375 | 0.189 | 3.09 | 1.662 | 43 | 4 |
| cov_lo | 16 | asymmetric | JAC-FD r9, mixture prior | 0.651 | 0.297 | 1.27 | 1.662 | 43 | 4 |
| cov_lo | 16 | asymmetric | netcov, mixture prior (r1) | 0.629 | 0.296 | 1.28 | 1.546 | 71 | 4 |
| cov_lo | 16 | asymmetric | netcov, mixture prior (r9) | 0.619 | 0.293 | 1.25 | 1.519 | 48 | 4 |
| cov_lo | 16 | asymmetric | netcov, Gaussian prior (r1) | 2.394 | 0.202 | 3.12 | 1.567 | 74 | 4 |
| cov_lo | 16 | asymmetric | netcov, Gaussian prior (r9) | 2.391 | 0.200 | 3.11 | 1.551 | 46 | 4 |
| cov_lo | 16 | asymmetric | F-NPSE (Langevin) | 2.538 | 0.263 | 2.33 | 1.657 | 102 | 4 |
| cov_lo | 16 | symmetric | GH pilot, Gaussian prior | 2.047 | 0.155 | 2.61 | 1.519 | 55 | 4 |
| cov_lo | 16 | symmetric | GH pilot, mixture prior | 0.908 | 0.261 | 0.15 | 1.519 | 52 | 4 |
| cov_lo | 16 | symmetric | JAC-FD r9, Gaussian prior | 1.895 | 0.053 | 2.78 | 1.523 | 55 | 4 |
| cov_lo | 16 | symmetric | JAC-FD r9, mixture prior | 0.430 | 0.077 | 1.12 | 1.523 | 58 | 4 |
| cov_lo | 16 | symmetric | netcov, mixture prior (r1) | 0.413 | 0.068 | 1.14 | 1.417 | 57 | 4 |
| cov_lo | 16 | symmetric | netcov, mixture prior (r9) | 0.432 | 0.087 | 1.11 | 1.390 | 52 | 4 |
| cov_lo | 16 | symmetric | netcov, Gaussian prior (r1) | 1.908 | 0.067 | 2.77 | 1.436 | 59 | 4 |
| cov_lo | 16 | symmetric | netcov, Gaussian prior (r9) | 1.919 | 0.074 | 2.77 | 1.423 | 52 | 4 |
| cov_lo | 16 | symmetric | F-NPSE (Langevin) | 2.004 | 0.123 | 2.63 | 1.511 | 100 | 4 |
| cov_lo | 32 | asymmetric | GH pilot, Gaussian prior | 4.324 | 0.141 | 5.03 | 1.916 | 77 | 4 |
| cov_lo | 32 | asymmetric | GH pilot, mixture prior | 1.040 | 0.466 | 0.03 | 1.916 | 77 | 4 |
| cov_lo | 32 | asymmetric | JAC-FD r9, Gaussian prior | 4.401 | 0.201 | 5.17 | 1.920 | 73 | 4 |
| cov_lo | 32 | asymmetric | JAC-FD r9, mixture prior | 0.883 | 0.407 | 0.76 | 1.920 | 76 | 4 |
| cov_lo | 32 | asymmetric | netcov, mixture prior (r1) | 0.840 | 0.413 | 0.77 | 1.794 | 113 | 4 |
| cov_lo | 32 | asymmetric | netcov, mixture prior (r9) | 0.837 | 0.410 | 0.79 | 1.765 | 77 | 4 |
| cov_lo | 32 | asymmetric | netcov, Gaussian prior (r1) | 4.352 | 0.174 | 5.16 | 1.819 | 112 | 4 |
| cov_lo | 32 | asymmetric | netcov, Gaussian prior (r9) | 4.345 | 0.169 | 5.15 | 1.804 | 74 | 4 |
| cov_lo | 32 | asymmetric | F-NPSE (Langevin) | 4.381 | 0.197 | 5.19 | 1.904 | 168 | 4 |
| cov_lo | 32 | symmetric | GH pilot, Gaussian prior | 3.745 | 0.073 | 4.53 | 1.928 | 74 | 4 |
| cov_lo | 32 | symmetric | GH pilot, mixture prior | 0.914 | 0.385 | 0.03 | 1.929 | 75 | 4 |
| cov_lo | 32 | symmetric | JAC-FD r9, Gaussian prior | 3.705 | 0.042 | 4.60 | 1.927 | 86 | 4 |
| cov_lo | 32 | symmetric | JAC-FD r9, mixture prior | 0.781 | 0.308 | 0.64 | 1.927 | 81 | 4 |
| cov_lo | 32 | symmetric | netcov, mixture prior (r1) | 0.765 | 0.328 | 0.63 | 1.791 | 108 | 4 |
| cov_lo | 32 | symmetric | netcov, mixture prior (r9) | 0.765 | 0.325 | 0.65 | 1.760 | 77 | 4 |
| cov_lo | 32 | symmetric | netcov, Gaussian prior (r1) | 3.710 | 0.048 | 4.60 | 1.818 | 109 | 4 |
| cov_lo | 32 | symmetric | netcov, Gaussian prior (r9) | 3.721 | 0.057 | 4.59 | 1.801 | 74 | 4 |
| cov_lo | 32 | symmetric | F-NPSE (Langevin) | 4.146 | 0.318 | 3.47 | 1.921 | 184 | 4 |
| cov_lo | 64 | asymmetric | GH pilot, Gaussian prior | 7.999 | 0.135 | 8.65 | 2.084 | 106 | 4 |
| cov_lo | 64 | asymmetric | GH pilot, mixture prior | 0.953 | 0.499 | 0.05 | 2.083 | 107 | 4 |
| cov_lo | 64 | asymmetric | JAC-FD r9, Gaussian prior | 8.155 | 0.265 | 8.87 | 2.085 | 132 | 4 |
| cov_lo | 64 | asymmetric | JAC-FD r9, mixture prior | 0.947 | 0.496 | 0.23 | 2.085 | 151 | 4 |
| cov_lo | 64 | asymmetric | netcov, mixture prior (r1) | 0.935 | 0.497 | 0.24 | 1.943 | 205 | 4 |
| cov_lo | 64 | asymmetric | netcov, mixture prior (r9) | 0.934 | 0.497 | 0.25 | 1.911 | 122 | 4 |
| cov_lo | 64 | asymmetric | netcov, Gaussian prior (r1) | 8.059 | 0.195 | 8.83 | 1.973 | 220 | 4 |
| cov_lo | 64 | asymmetric | netcov, Gaussian prior (r9) | 8.047 | 0.188 | 8.81 | 1.957 | 127 | 4 |
| cov_lo | 64 | asymmetric | F-NPSE (Langevin) | 8.002 | 0.127 | 8.35 | 2.070 | 313 | 4 |
| cov_lo | 64 | symmetric | GH pilot, Gaussian prior | 7.181 | 0.138 | 7.66 | 2.079 | 107 | 4 |
| cov_lo | 64 | symmetric | GH pilot, mixture prior | 0.874 | 0.443 | 0.04 | 2.079 | 107 | 4 |
| cov_lo | 64 | symmetric | JAC-FD r9, Gaussian prior | 7.129 | 0.079 | 7.82 | 2.078 | 135 | 4 |
| cov_lo | 64 | symmetric | JAC-FD r9, mixture prior | 0.870 | 0.444 | 0.20 | 2.078 | 141 | 4 |
| cov_lo | 64 | symmetric | netcov, mixture prior (r1) | 0.862 | 0.443 | 0.19 | 1.937 | 231 | 4 |
| cov_lo | 64 | symmetric | netcov, mixture prior (r9) | 0.862 | 0.443 | 0.18 | 1.907 | 130 | 4 |
| cov_lo | 64 | symmetric | netcov, Gaussian prior (r1) | 7.101 | 0.052 | 7.93 | 1.965 | 194 | 4 |
| cov_lo | 64 | symmetric | netcov, Gaussian prior (r9) | 7.095 | 0.054 | 7.92 | 1.949 | 143 | 4 |
| cov_lo | 64 | symmetric | F-NPSE (Langevin) | 7.690 | 0.386 | 4.22 | 2.072 | 299 | 4 |

## 5. Gaussian toy (Arruda Exp. 1, analytic posterior; covariance-output net)

| J | method | datasets | KL | mean err / sd | width ratio | s/dataset |
|---|---|---|---|---|---|---|
| 10 | gh | 5 | 0.108 | 0.075 | 1.010 | 13 |
| 10 | jac_r9 | 5 | 0.105 | 0.076 | 1.007 | 14 |
| 10 | netcov_r9 | 5 | 0.106 | 0.076 | 1.005 | 9 |
| 100 | gh | 5 | 0.344 | 0.193 | 1.018 | 304 |
| 100 | jac_r9 | 5 | 0.331 | 0.190 | 1.012 | 124 |
| 100 | netcov_r9 | 5 | 0.36 | 0.191 | 0.996 | 77 |
| 1000 | jac_r9 | 3 | 11.9 | 1.185 | 0.989 | 2202 |
| 1000 | netcov_r9 | 3 | 12.3 | 1.205 | 0.968 | 1100 |

## 6. AR(1) grid (Arruda Exp. 2; `paper_final` = this commit, the rest archived)

| grid | method | global RMSE (norm) | contraction | |mean err|/ref sd (α, β, logσ) | width ratio (α, β, logσ) | local RMSE θ | local |mean err|/sd | failed | s/dataset |
|---|---|---|---|---|---|---|---|---|---|
| 4x4 | reference | 0.062 | 0.935 | 0.04, 0.07, 0.04 | 1.00, 1.00, 1.00 | 0.145 | 0.09 | 0/10 | 0.0 |
| 4x4 | gauss_hierarchical | 0.055 | 0.928 | 0.44, 0.46, 0.71 | 1.00, 0.87, 1.02 | 0.159 | 0.37 | 0/10 | 8.7 |
| 4x4 | gauss_jacobian_FD_r9 | 0.064 | 0.916 | 0.48, 0.55, 0.59 | 1.14, 0.99, 1.10 | 0.164 | 0.24 | 0/10 | 4.7 |
| 4x4 | langevin_fnpe | 0.069 | 0.904 | 0.53, 0.49, 0.85 | 1.06, 0.92, 1.29 | 0.205 | 0.45 | 0/10 | 7.7 |
| 4x4 | paper_final | 0.064 | 0.916 | 0.48, 0.55, 0.59 | 1.14, 0.99, 1.10 | 0.164 | 0.24 | 0/10 | 8.4 |
| 4x4 | bayesflow_arruda | 0.097 | 0.849 | 0.60, 1.13, 0.82 | 1.19, 1.23, 1.22 | 0.164 | 0.30 | 0/10 | 17.2 |
| 16x16 | reference | 0.016 | 0.997 | 0.03, 0.03, 0.03 | 1.00, 1.00, 1.01 | 0.062 | 0.09 | 0/10 | 0.0 |
| 16x16 | gauss_hierarchical | 0.037 | 0.993 | 1.39, 1.52, 1.00 | 1.05, 1.17, 1.27 | 0.075 | 0.33 | 0/10 | 101.7 |
| 16x16 | gauss_jacobian_FD_r9 | 0.043 | 0.988 | 0.68, 1.79, 2.20 | 1.35, 1.71, 1.71 | 0.066 | 0.29 | 0/10 | 35.7 |
| 16x16 | langevin_fnpe | 0.026 | 0.995 | 1.39, 1.62, 1.24 | 0.99, 1.01, 1.57 | 0.077 | 0.48 | 0/10 | 85.8 |
| 16x16 | paper_final | 0.041 | 0.991 | 0.78, 1.69, 2.18 | 1.29, 1.60, 1.36 | 0.065 | 0.31 | 0/10 | 54.9 |
| 16x16 | bayesflow_arruda | 0.087 | 0.923 | 1.43, 2.28, 4.45 | 1.86, 4.77, 5.70 | 0.072 | 0.23 | 0/10 | 8.1 |
| 64x64 | reference | 0.009 | 1.000 | 0.04, 0.05, 0.07 | 1.01, 1.00, 1.06 | 0.089 | 0.09 | 0/3 | 0.0 |
| 64x64 | gauss_hierarchical | 0.122 | 0.999 | 14.31, 6.80, 23.41 | 2.41, 6.61, 1.96 | 0.098 | 0.45 | 0/3 | 4247.6 |
| 64x64 | gauss_jacobian_FD_r9 | 0.100 | 0.998 | 11.46, 5.86, 20.93 | 3.18, 3.84, 2.17 | 0.098 | 0.40 | 0/3 | 650.2 |
| 64x64 | langevin_fnpe | 0.149 | 1.000 | 2.06, 17.13, 25.07 | 0.96, 1.24, 1.26 | 0.110 | 0.44 | 0/3 | 1333.4 |
| 64x64 | paper_final | 0.100 | 0.998 | 11.74, 6.57, 20.73 | 3.28, 3.65, 2.18 | 0.098 | 0.40 | 0/3 | 660.5 |
| 64x64 | bayesflow_arruda | 0.236 | 1.000 | 57.45, 13.58, 38.48 | 5.77, 0.70, 0.39 | 0.098 | 0.92 | 0/3 | 184.2 |

