# Hierarchical Two Moons — results

Medians over datasets (coverage: mean). `reference` = 500 independent reference draws scored the same way: the Monte-Carlo floor.

## Q1 local posterior p(θ_i | x_1:N)

**local C2ST (0.5 best)** (`local_c2st`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.494 |
| uncorrected | – | 0.973 |
| gauss | – | 0.976 |
| gauss_hierarchical | – | 0.971 |
| gauss_jacobian | – | 0.973 |
| gauss_jacobian_FD | – | 0.974 |
| langevin_fnpe | – | 0.964 |
| attn_hier:arruda_direct | – | 0.979 |
| attn_hier:arruda_gauss_direct | – | 0.98 |
| attn_hier:gauss_jacobian_FD | – | 0.979 |
| attn_full:gauss_hierarchical | – | 0.977 |
| attn_full:gauss_jacobian_FD | – | 0.977 |
| attn_full:arruda_schur | – | 0.979 |
| attn_full:arruda_laplace | – | 0.979 |
| bayesflow_arruda | 0.938 | 0.954 |
| bayesflow_arruda_diffprior | 0.933 | 0.956 |

**local W1/sd** (`local_w1`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.0542 |
| uncorrected | – | 2.66 |
| gauss | – | 3.66 |
| gauss_hierarchical | – | 2.68 |
| gauss_jacobian | – | 2.74 |
| gauss_jacobian_FD | – | 2.73 |
| langevin_fnpe | – | 1.32 |
| attn_hier:arruda_direct | – | 7 |
| attn_hier:arruda_gauss_direct | – | 6.99 |
| attn_hier:gauss_jacobian_FD | – | 3.72 |
| attn_full:gauss_hierarchical | – | 3.76 |
| attn_full:gauss_jacobian_FD | – | 3.71 |
| attn_full:arruda_schur | – | 7.01 |
| attn_full:arruda_laplace | – | 7.06 |
| bayesflow_arruda | 1.15 | 0.903 |
| bayesflow_arruda_diffprior | 1.14 | 1.06 |

**branch-mass err** (`branch_err`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.0248 |
| uncorrected | – | 0.104 |
| gauss | – | 0.0764 |
| gauss_hierarchical | – | 0.116 |
| gauss_jacobian | – | 0.144 |
| gauss_jacobian_FD | – | 0.143 |
| langevin_fnpe | – | 0.178 |
| attn_hier:arruda_direct | – | 0.0686 |
| attn_hier:arruda_gauss_direct | – | 0.071 |
| attn_hier:gauss_jacobian_FD | – | 0.0314 |
| attn_full:gauss_hierarchical | – | 0.0378 |
| attn_full:gauss_jacobian_FD | – | 0.0386 |
| attn_full:arruda_schur | – | 0.0562 |
| attn_full:arruda_laplace | – | 0.0598 |
| bayesflow_arruda | 0.102 | 0.174 |
| bayesflow_arruda_diffprior | 0.108 | 0.151 |

**crescent |r-0.1|/0.01 (exact 0.67)** (`crescent_dev`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.717 |
| uncorrected | – | 97.8 |
| gauss | – | 146 |
| gauss_hierarchical | – | 99.3 |
| gauss_jacobian | – | 103 |
| gauss_jacobian_FD | – | 102 |
| langevin_fnpe | – | 49.1 |
| attn_hier:arruda_direct | – | 275 |
| attn_hier:arruda_gauss_direct | – | 279 |
| attn_hier:gauss_jacobian_FD | – | 183 |
| attn_full:gauss_hierarchical | – | 181 |
| attn_full:gauss_jacobian_FD | – | 181 |
| attn_full:arruda_schur | – | 281 |
| attn_full:arruda_laplace | – | 281 |
| bayesflow_arruda | 26.8 | 33.4 |
| bayesflow_arruda_diffprior | 27.5 | 39.8 |

**off-support frac** (`off_support`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0 |
| uncorrected | – | 0.821 |
| gauss | – | 0.886 |
| gauss_hierarchical | – | 0.827 |
| gauss_jacobian | – | 0.839 |
| gauss_jacobian_FD | – | 0.835 |
| langevin_fnpe | – | 0.754 |
| attn_hier:arruda_direct | – | 0.917 |
| attn_hier:arruda_gauss_direct | – | 0.917 |
| attn_hier:gauss_jacobian_FD | – | 0.902 |
| attn_full:gauss_hierarchical | – | 0.904 |
| attn_full:gauss_jacobian_FD | – | 0.899 |
| attn_full:arruda_schur | – | 0.918 |
| attn_full:arruda_laplace | – | 0.916 |
| bayesflow_arruda | 0.388 | 0.601 |
| bayesflow_arruda_diffprior | 0.376 | 0.62 |

**local |mean err|/sd** (`local_mean_err`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.0517 |
| uncorrected | – | 0.434 |
| gauss | – | 0.347 |
| gauss_hierarchical | – | 0.544 |
| gauss_jacobian | – | 0.862 |
| gauss_jacobian_FD | – | 0.845 |
| langevin_fnpe | – | 0.612 |
| attn_hier:arruda_direct | – | 2.29 |
| attn_hier:arruda_gauss_direct | – | 2.3 |
| attn_hier:gauss_jacobian_FD | – | 1.08 |
| attn_full:gauss_hierarchical | – | 1.09 |
| attn_full:gauss_jacobian_FD | – | 1.05 |
| attn_full:arruda_schur | – | 2.3 |
| attn_full:arruda_laplace | – | 2.31 |
| bayesflow_arruda | 0.433 | 0.489 |
| bayesflow_arruda_diffprior | 0.513 | 0.4 |

**local width ratio** (`local_width`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 1 |
| uncorrected | – | 2.81 |
| gauss | – | 3.79 |
| gauss_hierarchical | – | 2.82 |
| gauss_jacobian | – | 2.79 |
| gauss_jacobian_FD | – | 2.78 |
| langevin_fnpe | – | 1.55 |
| attn_hier:arruda_direct | – | 8.58 |
| attn_hier:arruda_gauss_direct | – | 8.56 |
| attn_hier:gauss_jacobian_FD | – | 3.83 |
| attn_full:gauss_hierarchical | – | 3.77 |
| attn_full:gauss_jacobian_FD | – | 3.79 |
| attn_full:arruda_schur | – | 8.55 |
| attn_full:arruda_laplace | – | 8.58 |
| bayesflow_arruda | 2.38 | 1.25 |
| bayesflow_arruda_diffprior | 2.34 | 1.34 |

## Q2 population mean μ_g

**|mean err|/sd** (`mu_mean_err`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.0596 |
| uncorrected | – | 0.643 |
| gauss | – | 0.522 |
| gauss_hierarchical | – | 0.627 |
| gauss_jacobian | – | 0.676 |
| gauss_jacobian_FD | – | 0.677 |
| langevin_fnpe | – | 0.627 |
| attn_hier:arruda_direct | – | 0.222 |
| attn_hier:arruda_gauss_direct | – | 0.264 |
| attn_hier:gauss_jacobian_FD | – | 0.258 |
| attn_full:gauss_hierarchical | – | 0.2 |
| attn_full:gauss_jacobian_FD | – | 0.251 |
| attn_full:arruda_schur | – | 0.652 |
| attn_full:arruda_laplace | – | 0.138 |
| bayesflow_arruda | 0.154 | 0.247 |
| bayesflow_arruda_diffprior | 0.125 | 0.087 |

**width ratio** (`mu_width`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.993 |
| uncorrected | – | 0.973 |
| gauss | – | 1.01 |
| gauss_hierarchical | – | 0.971 |
| gauss_jacobian | – | 0.967 |
| gauss_jacobian_FD | – | 0.966 |
| langevin_fnpe | – | 0.924 |
| attn_hier:arruda_direct | – | 1.25 |
| attn_hier:arruda_gauss_direct | – | 1.25 |
| attn_hier:gauss_jacobian_FD | – | 1.25 |
| attn_full:gauss_hierarchical | – | 1.25 |
| attn_full:gauss_jacobian_FD | – | 1.26 |
| attn_full:arruda_schur | – | 1.19 |
| attn_full:arruda_laplace | – | 1.37 |
| bayesflow_arruda | 1.03 | 1.07 |
| bayesflow_arruda_diffprior | 1.03 | 1.4 |

**W1/sd** (`mu_w1`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.0711 |
| uncorrected | – | 0.646 |
| gauss | – | 0.524 |
| gauss_hierarchical | – | 0.629 |
| gauss_jacobian | – | 0.678 |
| gauss_jacobian_FD | – | 0.679 |
| langevin_fnpe | – | 0.63 |
| attn_hier:arruda_direct | – | 0.547 |
| attn_hier:arruda_gauss_direct | – | 0.539 |
| attn_hier:gauss_jacobian_FD | – | 0.537 |
| attn_full:gauss_hierarchical | – | 0.535 |
| attn_full:gauss_jacobian_FD | – | 0.521 |
| attn_full:arruda_schur | – | 0.659 |
| attn_full:arruda_laplace | – | 0.437 |
| bayesflow_arruda | 0.16 | 0.265 |
| bayesflow_arruda_diffprior | 0.167 | 0.451 |

**90% coverage** (`mu_cover90`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 1 |
| uncorrected | – | 1 |
| gauss | – | 1 |
| gauss_hierarchical | – | 1 |
| gauss_jacobian | – | 1 |
| gauss_jacobian_FD | – | 1 |
| langevin_fnpe | – | 1 |
| attn_hier:arruda_direct | – | 1 |
| attn_hier:arruda_gauss_direct | – | 1 |
| attn_hier:gauss_jacobian_FD | – | 1 |
| attn_full:gauss_hierarchical | – | 1 |
| attn_full:gauss_jacobian_FD | – | 1 |
| attn_full:arruda_schur | – | 1 |
| attn_full:arruda_laplace | – | 1 |
| bayesflow_arruda | 1 | 1 |
| bayesflow_arruda_diffprior | 1 | 1 |

## Q3 population spread σ_g

**|mean err|/sd** (`sigma_mean_err`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.0785 |
| uncorrected | – | 0.177 |
| gauss | – | 0.161 |
| gauss_hierarchical | – | 0.166 |
| gauss_jacobian | – | 0.189 |
| gauss_jacobian_FD | – | 0.19 |
| langevin_fnpe | – | 0.238 |
| attn_hier:arruda_direct | – | 1.02 |
| attn_hier:arruda_gauss_direct | – | 0.866 |
| attn_hier:gauss_jacobian_FD | – | 0.866 |
| attn_full:gauss_hierarchical | – | 0.992 |
| attn_full:gauss_jacobian_FD | – | 0.821 |
| attn_full:arruda_schur | – | 1.02 |
| attn_full:arruda_laplace | – | 0.906 |
| bayesflow_arruda | 0.211 | 0.519 |
| bayesflow_arruda_diffprior | 0.259 | 0.32 |

**width ratio** (`sigma_width`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.984 |
| uncorrected | – | 1.06 |
| gauss | – | 1.03 |
| gauss_hierarchical | – | 1.03 |
| gauss_jacobian | – | 1.06 |
| gauss_jacobian_FD | – | 1.06 |
| langevin_fnpe | – | 0.972 |
| attn_hier:arruda_direct | – | 0.813 |
| attn_hier:arruda_gauss_direct | – | 0.937 |
| attn_hier:gauss_jacobian_FD | – | 0.935 |
| attn_full:gauss_hierarchical | – | 0.847 |
| attn_full:gauss_jacobian_FD | – | 0.983 |
| attn_full:arruda_schur | – | 1.07 |
| attn_full:arruda_laplace | – | 1.04 |
| bayesflow_arruda | 1.05 | 1.03 |
| bayesflow_arruda_diffprior | 1.03 | 1.36 |

**W1/sd** (`sigma_w1`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.0909 |
| uncorrected | – | 0.177 |
| gauss | – | 0.163 |
| gauss_hierarchical | – | 0.167 |
| gauss_jacobian | – | 0.19 |
| gauss_jacobian_FD | – | 0.19 |
| langevin_fnpe | – | 0.239 |
| attn_hier:arruda_direct | – | 1.08 |
| attn_hier:arruda_gauss_direct | – | 0.949 |
| attn_hier:gauss_jacobian_FD | – | 0.948 |
| attn_full:gauss_hierarchical | – | 1.04 |
| attn_full:gauss_jacobian_FD | – | 0.9 |
| attn_full:arruda_schur | – | 1.09 |
| attn_full:arruda_laplace | – | 1.01 |
| bayesflow_arruda | 0.266 | 0.524 |
| bayesflow_arruda_diffprior | 0.299 | 0.534 |

**90% coverage** (`sigma_cover90`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 1 |
| uncorrected | – | 1 |
| gauss | – | 1 |
| gauss_hierarchical | – | 1 |
| gauss_jacobian | – | 1 |
| gauss_jacobian_FD | – | 1 |
| langevin_fnpe | – | 1 |
| attn_hier:arruda_direct | – | 1 |
| attn_hier:arruda_gauss_direct | – | 1 |
| attn_hier:gauss_jacobian_FD | – | 1 |
| attn_full:gauss_hierarchical | – | 1 |
| attn_full:gauss_jacobian_FD | – | 1 |
| attn_full:arruda_schur | – | 1 |
| attn_full:arruda_laplace | – | 1 |
| bayesflow_arruda | 1 | 1 |
| bayesflow_arruda_diffprior | 1 | 1 |

**joint global C2ST** (`global_c2st`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.466 |
| uncorrected | – | 0.64 |
| gauss | – | 0.644 |
| gauss_hierarchical | – | 0.659 |
| gauss_jacobian | – | 0.661 |
| gauss_jacobian_FD | – | 0.666 |
| langevin_fnpe | – | 0.68 |
| attn_hier:arruda_direct | – | 0.92 |
| attn_hier:arruda_gauss_direct | – | 0.891 |
| attn_hier:gauss_jacobian_FD | – | 0.891 |
| attn_full:gauss_hierarchical | – | 0.886 |
| attn_full:gauss_jacobian_FD | – | 0.863 |
| attn_full:arruda_schur | – | 0.909 |
| attn_full:arruda_laplace | – | 0.869 |
| bayesflow_arruda | 0.603 | 0.758 |
| bayesflow_arruda_diffprior | 0.601 | 0.788 |

## Q4 global–local dependence

**|Δ corr(g, θ_i)|** (`dep_corr_err`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.0476 |
| uncorrected | – | 0.0941 |
| gauss | – | 0.11 |
| gauss_hierarchical | – | 0.098 |
| gauss_jacobian | – | 0.0912 |
| gauss_jacobian_FD | – | 0.0913 |
| langevin_fnpe | – | 0.07 |
| attn_hier:arruda_direct | – | 0.145 |
| attn_hier:arruda_gauss_direct | – | 0.145 |
| attn_hier:gauss_jacobian_FD | – | 0.15 |
| attn_full:gauss_hierarchical | – | 0.124 |
| attn_full:gauss_jacobian_FD | – | 0.149 |
| attn_full:arruda_schur | – | 0.166 |
| attn_full:arruda_laplace | – | 0.145 |
| bayesflow_arruda | 0.0776 | 0.14 |
| bayesflow_arruda_diffprior | 0.0521 | 0.144 |

**ref |corr| (signal)** (`dep_corr_ref`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.138 |
| uncorrected | – | 0.138 |
| gauss | – | 0.138 |
| gauss_hierarchical | – | 0.138 |
| gauss_jacobian | – | 0.138 |
| gauss_jacobian_FD | – | 0.138 |
| langevin_fnpe | – | 0.138 |
| attn_hier:arruda_direct | – | 0.138 |
| attn_hier:arruda_gauss_direct | – | 0.138 |
| attn_hier:gauss_jacobian_FD | – | 0.138 |
| attn_full:gauss_hierarchical | – | 0.138 |
| attn_full:gauss_jacobian_FD | – | 0.138 |
| attn_full:arruda_schur | – | 0.138 |
| attn_full:arruda_laplace | – | 0.138 |
| bayesflow_arruda | 0.0665 | 0.138 |
| bayesflow_arruda_diffprior | 0.0665 | 0.138 |

**joint (g, θ_i) C2ST** (`joint_c2st`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.489 |
| uncorrected | – | 0.819 |
| gauss | – | 0.803 |
| gauss_hierarchical | – | 0.813 |
| gauss_jacobian | – | 0.82 |
| gauss_jacobian_FD | – | 0.822 |
| langevin_fnpe | – | 0.873 |
| attn_hier:arruda_direct | – | 0.932 |
| attn_hier:arruda_gauss_direct | – | 0.913 |
| attn_hier:gauss_jacobian_FD | – | 0.918 |
| attn_full:gauss_hierarchical | – | 0.929 |
| attn_full:gauss_jacobian_FD | – | 0.909 |
| attn_full:arruda_schur | – | 0.915 |
| attn_full:arruda_laplace | – | 0.91 |
| bayesflow_arruda | 0.771 | 0.834 |
| bayesflow_arruda_diffprior | 0.768 | 0.835 |

**same, dependence shuffled out** (`joint_c2st_indep`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0.548 |
| uncorrected | – | 0.812 |
| gauss | – | 0.805 |
| gauss_hierarchical | – | 0.814 |
| gauss_jacobian | – | 0.813 |
| gauss_jacobian_FD | – | 0.813 |
| langevin_fnpe | – | 0.863 |
| attn_hier:arruda_direct | – | 0.927 |
| attn_hier:arruda_gauss_direct | – | 0.908 |
| attn_hier:gauss_jacobian_FD | – | 0.923 |
| attn_full:gauss_hierarchical | – | 0.924 |
| attn_full:gauss_jacobian_FD | – | 0.908 |
| attn_full:arruda_schur | – | 0.914 |
| attn_full:arruda_laplace | – | 0.902 |
| bayesflow_arruda | 0.77 | 0.836 |
| bayesflow_arruda_diffprior | 0.769 | 0.836 |

## Cost

**s / dataset** (`seconds`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 24.4 |
| uncorrected | – | 26.7 |
| gauss | – | 50.8 |
| gauss_hierarchical | – | 50.4 |
| gauss_jacobian | – | 134 |
| gauss_jacobian_FD | – | 131 |
| langevin_fnpe | – | 115 |
| attn_hier:arruda_direct | – | 50.3 |
| attn_hier:arruda_gauss_direct | – | 301 |
| attn_hier:gauss_jacobian_FD | – | 135 |
| attn_full:gauss_hierarchical | – | 45.7 |
| attn_full:gauss_jacobian_FD | – | 126 |
| attn_full:arruda_schur | – | 293 |
| attn_full:arruda_laplace | – | 510 |
| bayesflow_arruda | 6.93 | 15.3 |
| bayesflow_arruda_diffprior | 6.91 | 15.4 |

**non-finite frac** (`nonfinite`)

| method | N=1 | N=5 |
|---|---|---|
| reference | – | 0 |
| uncorrected | – | 0 |
| gauss | – | 0 |
| gauss_hierarchical | – | 0 |
| gauss_jacobian | – | 0 |
| gauss_jacobian_FD | – | 0 |
| langevin_fnpe | – | 0 |
| attn_hier:arruda_direct | – | 0 |
| attn_hier:arruda_gauss_direct | – | 0 |
| attn_hier:gauss_jacobian_FD | – | 0 |
| attn_full:gauss_hierarchical | – | 0 |
| attn_full:gauss_jacobian_FD | – | 0 |
| attn_full:arruda_schur | – | 0 |
| attn_full:arruda_laplace | – | 0 |
| bayesflow_arruda | 0 | 0 |
| bayesflow_arruda_diffprior | 0 | 0 |

## BayesFlow tuned (d1, shift)

| method | N | d1 | shift |
|---|---|---|---|
| bayesflow_arruda | 1 | 1 | 0 |
| bayesflow_arruda | 5 | 0.1 | 0 |
| bayesflow_arruda_diffprior | 1 | 1 | 0 |
| bayesflow_arruda_diffprior | 5 | 0.1 | 0 |
