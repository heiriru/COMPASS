# Hierarchical Two Moons — results

Medians over datasets (coverage: mean). `reference` = 500 independent reference draws scored the same way: the Monte-Carlo floor.

## Q1 local posterior p(θ_i | x_1:N)

**local C2ST (0.5 best)** (`local_c2st`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.498 | 0.494 | 0.5 | 0.497 | 0.497 | 0.498 |
| uncorrected | 0.51 | 0.52 | 0.514 | 0.523 | 0.523 | 0.529 |
| gauss | 0.956 | 0.961 | 0.957 | 0.953 | 0.944 | 0.932 |
| gauss_hierarchical | 0.515 | 0.526 | 0.524 | 0.521 | 0.52 | 0.518 |
| gauss_jacobian | 0.51 | 0.519 | 0.517 | 0.522 | 0.524 | 0.524 |
| gauss_jacobian_FD | 0.51 | 0.523 | 0.518 | 0.521 | 0.521 | 0.525 |
| langevin_fnpe | 0.516 | 0.517 | 0.523 | 0.527 | 0.526 | 0.525 |
| attn_hier:arruda_direct | 0.601 | 0.615 | 0.609 | 0.62 | 0.615 | 0.615 |
| attn_hier:arruda_gauss_direct | 0.601 | 0.615 | 0.612 | 0.619 | 0.616 | 0.614 |
| attn_hier:gauss_jacobian_FD | 0.512 | 0.511 | 0.516 | 0.52 | 0.515 | 0.519 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.508 | 0.517 | 0.518 | 0.522 | 0.521 | 0.519 |
| attn_full:gauss_hierarchical | 0.521 | 0.517 | 0.52 | 0.523 | 0.517 | 0.515 |
| attn_full:gauss_jacobian_FD | 0.494 | 0.509 | 0.515 | 0.52 | 0.519 | 0.518 |
| attn_full:arruda_schur | 0.611 | 0.617 | 0.628 | 0.639 | 0.638 | 0.648 |
| attn_full:arruda_laplace | 0.617 | 0.615 | 0.615 | 0.618 | 0.623 | 0.621 |
| bayesflow_arruda | 0.506 | 0.52 | 0.52 | 0.516 | 0.512 | 0.513 |
| bayesflow_arruda_diffprior | 0.51 | 0.514 | 0.514 | 0.518 | 0.516 | 0.516 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.506 | 0.511 | 0.514 | 0.518 | 0.512 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.508 | 0.516 | 0.517 | 0.524 | 0.52 | 0.52 |
| legacy_w1.5:gauss_hierarchical | 0.514 | 0.516 | 0.522 | 0.526 | 0.518 | 0.515 |

**local W1/sd** (`local_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0377 | 0.0483 | 0.0481 | 0.0472 | 0.0465 | 0.0466 |
| uncorrected | 0.0431 | 0.054 | 0.0568 | 0.0612 | 0.0579 | 0.0741 |
| gauss | 1.61 | 1.28 | 1.43 | 1.4 | 0.962 | 0.735 |
| gauss_hierarchical | 0.0484 | 0.0594 | 0.0632 | 0.0665 | 0.0578 | 0.0615 |
| gauss_jacobian | 0.0431 | 0.0541 | 0.058 | 0.061 | 0.0617 | 0.0615 |
| gauss_jacobian_FD | 0.0431 | 0.0552 | 0.0574 | 0.0609 | 0.0618 | 0.0611 |
| langevin_fnpe | 0.0509 | 0.0552 | 0.0646 | 0.0728 | 0.0716 | 0.0732 |
| attn_hier:arruda_direct | 0.0587 | 0.0567 | 0.0735 | 0.0812 | 0.0717 | 0.0973 |
| attn_hier:arruda_gauss_direct | 0.0587 | 0.0569 | 0.0751 | 0.0733 | 0.0688 | 0.0915 |
| attn_hier:gauss_jacobian_FD | 0.0377 | 0.0503 | 0.062 | 0.0608 | 0.062 | 0.0623 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0524 | 0.0575 | 0.0609 | 0.0631 | 0.0605 | 0.0668 |
| attn_full:gauss_hierarchical | 0.0358 | 0.0589 | 0.0675 | 0.0621 | 0.056 | 0.0591 |
| attn_full:gauss_jacobian_FD | 0.0306 | 0.0498 | 0.0642 | 0.0569 | 0.06 | 0.0601 |
| attn_full:arruda_schur | 0.0578 | 0.0663 | 0.154 | 0.281 | 0.269 | 0.367 |
| attn_full:arruda_laplace | 0.0542 | 0.0634 | 0.0739 | 0.111 | 0.136 | 0.158 |
| bayesflow_arruda | 0.0602 | 0.0599 | 0.0622 | 0.0606 | 0.0479 | 0.0548 |
| bayesflow_arruda_diffprior | 0.0433 | 0.0605 | 0.0622 | 0.0534 | 0.0507 | 0.0575 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0407 | 0.0529 | 0.0612 | 0.0538 | 0.057 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0345 | 0.0579 | 0.0575 | 0.0609 | 0.0628 | 0.0608 |
| legacy_w1.5:gauss_hierarchical | 0.0435 | 0.0635 | 0.0616 | 0.0639 | 0.0622 | 0.0601 |

**branch-mass err** (`branch_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.015 | 0.0184 | 0.0213 | 0.0199 | 0.0202 | 0.0198 |
| uncorrected | 0.018 | 0.0193 | 0.0237 | 0.022 | 0.0208 | 0.0269 |
| gauss | 0.0135 | 0.0155 | 0.0194 | 0.0193 | 0.0202 | 0.0206 |
| gauss_hierarchical | 0.0213 | 0.0218 | 0.0222 | 0.0232 | 0.0216 | 0.023 |
| gauss_jacobian | 0.018 | 0.0185 | 0.0225 | 0.0225 | 0.024 | 0.0217 |
| gauss_jacobian_FD | 0.018 | 0.0196 | 0.0224 | 0.0226 | 0.0241 | 0.022 |
| langevin_fnpe | 0.01 | 0.019 | 0.0235 | 0.029 | 0.0286 | 0.0285 |
| attn_hier:arruda_direct | 0.018 | 0.0186 | 0.0239 | 0.0304 | 0.0245 | 0.0367 |
| attn_hier:arruda_gauss_direct | 0.018 | 0.0186 | 0.0225 | 0.0254 | 0.0235 | 0.0329 |
| attn_hier:gauss_jacobian_FD | 0.013 | 0.0192 | 0.0215 | 0.024 | 0.0255 | 0.024 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0233 | 0.0222 | 0.0244 | 0.0252 | 0.0248 | 0.0264 |
| attn_full:gauss_hierarchical | 0.0118 | 0.0244 | 0.0267 | 0.0228 | 0.023 | 0.0235 |
| attn_full:gauss_jacobian_FD | 0.00825 | 0.0182 | 0.0233 | 0.0243 | 0.0234 | 0.0231 |
| attn_full:arruda_schur | 0.0253 | 0.0243 | 0.0718 | 0.132 | 0.13 | 0.18 |
| attn_full:arruda_laplace | 0.0193 | 0.0229 | 0.0241 | 0.0421 | 0.0566 | 0.0694 |
| bayesflow_arruda | 0.0305 | 0.0231 | 0.0232 | 0.0262 | 0.0191 | 0.0226 |
| bayesflow_arruda_diffprior | 0.0218 | 0.023 | 0.0242 | 0.0218 | 0.0199 | 0.0238 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0127 | 0.0193 | 0.0224 | 0.0231 | 0.0238 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0103 | 0.0196 | 0.0186 | 0.0225 | 0.0239 | 0.0227 |
| legacy_w1.5:gauss_hierarchical | 0.0193 | 0.0226 | 0.0248 | 0.0254 | 0.0228 | 0.0228 |

**crescent |r-0.1|/0.01 (exact 0.67)** (`crescent_dev`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.701 | 0.682 | 0.674 | 0.678 | 0.675 | 0.673 |
| uncorrected | 0.756 | 0.765 | 0.778 | 0.78 | 0.791 | 0.819 |
| gauss | 49 | 45.8 | 37.1 | 33.1 | 25.3 | 17.4 |
| gauss_hierarchical | 0.757 | 0.754 | 0.772 | 0.774 | 0.774 | 0.777 |
| gauss_jacobian | 0.756 | 0.765 | 0.776 | 0.784 | 0.788 | 0.807 |
| gauss_jacobian_FD | 0.756 | 0.765 | 0.774 | 0.783 | 0.789 | 0.81 |
| langevin_fnpe | 0.762 | 0.773 | 0.773 | 0.787 | 0.781 | 0.778 |
| attn_hier:arruda_direct | 1.41 | 1.4 | 1.39 | 1.41 | 1.4 | 1.4 |
| attn_hier:arruda_gauss_direct | 1.41 | 1.4 | 1.39 | 1.41 | 1.4 | 1.4 |
| attn_hier:gauss_jacobian_FD | 0.754 | 0.755 | 0.757 | 0.766 | 0.765 | 0.765 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.755 | 0.765 | 0.758 | 0.768 | 0.765 | 0.767 |
| attn_full:gauss_hierarchical | 0.742 | 0.747 | 0.753 | 0.762 | 0.76 | 0.76 |
| attn_full:gauss_jacobian_FD | 0.753 | 0.746 | 0.76 | 0.769 | 0.765 | 0.763 |
| attn_full:arruda_schur | 1.41 | 1.43 | 1.44 | 1.47 | 1.48 | 1.49 |
| attn_full:arruda_laplace | 1.43 | 1.4 | 1.41 | 1.42 | 1.42 | 1.43 |
| bayesflow_arruda | 0.742 | 0.711 | 0.711 | 0.728 | 0.725 | 0.726 |
| bayesflow_arruda_diffprior | 0.682 | 0.74 | 0.741 | 0.747 | 0.741 | 0.752 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.76 | 0.752 | 0.754 | 0.763 | 0.762 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.785 | 0.77 | 0.766 | 0.775 | 0.777 | 0.78 |
| legacy_w1.5:gauss_hierarchical | 0.767 | 0.768 | 0.775 | 0.776 | 0.777 | 0.776 |

**off-support frac** (`off_support`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0 | 0 | 0 | 0 | 0 | 0 |
| uncorrected | 0.019 | 0.0616 | 0.0375 | 0.0422 | 0.053 | 0.0464 |
| gauss | 0.448 | 0.578 | 0.503 | 0.553 | 0.521 | 0.449 |
| gauss_hierarchical | 0.02 | 0.0632 | 0.0371 | 0.0443 | 0.0498 | 0.043 |
| gauss_jacobian | 0.019 | 0.0618 | 0.0383 | 0.0448 | 0.0517 | 0.0462 |
| gauss_jacobian_FD | 0.019 | 0.0616 | 0.0383 | 0.045 | 0.0513 | 0.0467 |
| langevin_fnpe | 0.022 | 0.0598 | 0.0326 | 0.0436 | 0.0475 | 0.0409 |
| attn_hier:arruda_direct | 0.052 | 0.0764 | 0.0582 | 0.065 | 0.0718 | 0.0642 |
| attn_hier:arruda_gauss_direct | 0.052 | 0.0766 | 0.0579 | 0.0649 | 0.0709 | 0.0643 |
| attn_hier:gauss_jacobian_FD | 0.02 | 0.0504 | 0.0312 | 0.0335 | 0.0389 | 0.0329 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.02 | 0.046 | 0.0303 | 0.0301 | 0.0384 | 0.031 |
| attn_full:gauss_hierarchical | 0.018 | 0.048 | 0.0331 | 0.0326 | 0.0385 | 0.0337 |
| attn_full:gauss_jacobian_FD | 0.019 | 0.0492 | 0.0319 | 0.0337 | 0.0388 | 0.0336 |
| attn_full:arruda_schur | 0.046 | 0.0674 | 0.0566 | 0.065 | 0.068 | 0.0642 |
| attn_full:arruda_laplace | 0.045 | 0.0752 | 0.0578 | 0.0626 | 0.0681 | 0.063 |
| bayesflow_arruda | 0.018 | 0.0376 | 0.0205 | 0.0294 | 0.0359 | 0.03 |
| bayesflow_arruda_diffprior | 0.014 | 0.0364 | 0.0259 | 0.0307 | 0.0368 | 0.0301 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.017 | 0.0396 | 0.0256 | 0.0291 | 0.0342 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.017 | 0.057 | 0.0381 | 0.0416 | 0.0481 | 0.0404 |
| legacy_w1.5:gauss_hierarchical | 0.022 | 0.0598 | 0.0364 | 0.0407 | 0.0481 | 0.0406 |

**local |mean err|/sd** (`local_mean_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0288 | 0.0336 | 0.0317 | 0.0355 | 0.0324 | 0.0315 |
| uncorrected | 0.0327 | 0.0437 | 0.0349 | 0.0348 | 0.0369 | 0.0491 |
| gauss | 0.0645 | 0.069 | 0.0773 | 0.0573 | 0.0542 | 0.0517 |
| gauss_hierarchical | 0.0429 | 0.0422 | 0.0417 | 0.0485 | 0.0413 | 0.0409 |
| gauss_jacobian | 0.0327 | 0.0426 | 0.0352 | 0.0391 | 0.042 | 0.0384 |
| gauss_jacobian_FD | 0.0327 | 0.0451 | 0.0334 | 0.0415 | 0.0422 | 0.039 |
| langevin_fnpe | 0.0229 | 0.0341 | 0.0446 | 0.047 | 0.0505 | 0.0514 |
| attn_hier:arruda_direct | 0.0283 | 0.0307 | 0.044 | 0.0571 | 0.0407 | 0.0677 |
| attn_hier:arruda_gauss_direct | 0.0283 | 0.0368 | 0.0449 | 0.0458 | 0.043 | 0.0599 |
| attn_hier:gauss_jacobian_FD | 0.0242 | 0.0365 | 0.0429 | 0.0451 | 0.0448 | 0.042 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0426 | 0.0387 | 0.0455 | 0.0472 | 0.044 | 0.0476 |
| attn_full:gauss_hierarchical | 0.029 | 0.0448 | 0.0479 | 0.0435 | 0.039 | 0.042 |
| attn_full:gauss_jacobian_FD | 0.022 | 0.0364 | 0.0428 | 0.0389 | 0.039 | 0.0434 |
| attn_full:arruda_schur | 0.0421 | 0.0488 | 0.133 | 0.283 | 0.261 | 0.374 |
| attn_full:arruda_laplace | 0.0355 | 0.0433 | 0.0396 | 0.0853 | 0.121 | 0.148 |
| bayesflow_arruda | 0.0569 | 0.0491 | 0.0416 | 0.0445 | 0.0327 | 0.0395 |
| bayesflow_arruda_diffprior | 0.0301 | 0.048 | 0.0541 | 0.0384 | 0.0317 | 0.0425 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0202 | 0.0452 | 0.0387 | 0.0374 | 0.0414 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0198 | 0.0451 | 0.0275 | 0.0419 | 0.0439 | 0.0393 |
| legacy_w1.5:gauss_hierarchical | 0.0301 | 0.0544 | 0.0437 | 0.0491 | 0.0425 | 0.0421 |

**local width ratio** (`local_width`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 1 | 0.999 | 1 | 1 | 1 | 1 |
| uncorrected | 1 | 0.999 | 0.997 | 0.999 | 0.998 | 0.998 |
| gauss | 3.2 | 1.99 | 2.08 | 1.64 | 1.47 | 1.28 |
| gauss_hierarchical | 0.998 | 1 | 0.998 | 0.999 | 0.998 | 0.999 |
| gauss_jacobian | 1 | 0.999 | 0.997 | 0.999 | 0.998 | 0.999 |
| gauss_jacobian_FD | 1 | 0.999 | 0.998 | 0.999 | 0.998 | 0.999 |
| langevin_fnpe | 0.997 | 0.999 | 1 | 1 | 0.998 | 0.998 |
| attn_hier:arruda_direct | 1.01 | 1 | 1.01 | 1 | 1 | 1 |
| attn_hier:arruda_gauss_direct | 1.01 | 1 | 1.01 | 1 | 1 | 1 |
| attn_hier:gauss_jacobian_FD | 1 | 1 | 1 | 1 | 1 | 1 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 1 | 0.999 | 1 | 1 | 1 | 1 |
| attn_full:gauss_hierarchical | 1 | 1 | 1 | 1 | 1 | 1 |
| attn_full:gauss_jacobian_FD | 1 | 1 | 1 | 1 | 1 | 1 |
| attn_full:arruda_schur | 1.01 | 1 | 1 | 0.967 | 0.971 | 0.935 |
| attn_full:arruda_laplace | 1.01 | 1 | 1 | 1 | 0.998 | 0.994 |
| bayesflow_arruda | 1 | 1 | 1 | 1 | 1 | 1 |
| bayesflow_arruda_diffprior | 1 | 1 | 1 | 1 | 1 | 1 |
| attn_hier_w1.5:gauss_jacobian_FD | 1 | 1 | 1 | 1 | 1 | – |
| legacy_w1.5:gauss_jacobian_FD | 1 | 0.998 | 0.999 | 0.999 | 0.997 | 0.997 |
| legacy_w1.5:gauss_hierarchical | 0.997 | 0.998 | 0.999 | 0.997 | 0.997 | 0.998 |

## Q2 population mean μ_g

**|mean err|/sd** (`mu_mean_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0348 | 0.04 | 0.0523 | 0.0316 | 0.0374 | 0.0479 |
| uncorrected | 0.0556 | 0.0573 | 0.102 | 0.213 | 0.201 | 0.452 |
| gauss | 0.0391 | 0.0541 | 0.108 | 0.158 | 0.139 | 0.23 |
| gauss_hierarchical | 0.0367 | 0.0344 | 0.0739 | 0.121 | 0.118 | 0.124 |
| gauss_jacobian | 0.0556 | 0.0411 | 0.0707 | 0.136 | 0.138 | 0.228 |
| gauss_jacobian_FD | 0.0556 | 0.0414 | 0.0705 | 0.137 | 0.136 | 0.237 |
| langevin_fnpe | 0.039 | 0.0622 | 0.0788 | 0.238 | 0.123 | 0.179 |
| attn_hier:arruda_direct | 0.0463 | 0.09 | 0.139 | 0.298 | 0.223 | 0.366 |
| attn_hier:arruda_gauss_direct | 0.0463 | 0.0737 | 0.0535 | 0.0961 | 0.0924 | 0.221 |
| attn_hier:gauss_jacobian_FD | 0.0463 | 0.0722 | 0.0544 | 0.0768 | 0.0779 | 0.212 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0438 | 0.0685 | 0.0699 | 0.109 | 0.139 | 0.287 |
| attn_full:gauss_hierarchical | 0.0294 | 0.0469 | 0.065 | 0.157 | 0.119 | 0.177 |
| attn_full:gauss_jacobian_FD | 0.0502 | 0.0333 | 0.0517 | 0.109 | 0.149 | 0.208 |
| attn_full:arruda_schur | 0.116 | 0.454 | 0.629 | 0.59 | 0.602 | 1.11 |
| attn_full:arruda_laplace | 0.0551 | 0.0627 | 0.0578 | 0.299 | 0.471 | 0.55 |
| bayesflow_arruda | 0.027 | 0.0668 | 0.302 | 0.226 | 0.193 | 0.268 |
| bayesflow_arruda_diffprior | 0.0383 | 0.0618 | 0.136 | 0.231 | 0.12 | 0.217 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.042 | 0.0633 | 0.0779 | 0.135 | 0.115 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0528 | 0.0295 | 0.0817 | 0.138 | 0.16 | 0.213 |
| legacy_w1.5:gauss_hierarchical | 0.0281 | 0.0403 | 0.0985 | 0.0941 | 0.116 | 0.157 |

**width ratio** (`mu_width`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.992 | 1.01 | 1.01 | 1 | 1.01 | 0.997 |
| uncorrected | 1.01 | 1 | 0.987 | 0.936 | 1.02 | 1.09 |
| gauss | 1.01 | 1.02 | 1.04 | 1.01 | 0.986 | 1.01 |
| gauss_hierarchical | 1 | 1 | 1.04 | 0.973 | 1 | 1.04 |
| gauss_jacobian | 1.01 | 1.01 | 1.07 | 1.04 | 1.08 | 1.14 |
| gauss_jacobian_FD | 1.01 | 1.02 | 1.07 | 1.04 | 1.09 | 1.15 |
| langevin_fnpe | 1.02 | 0.96 | 0.937 | 0.713 | 0.539 | 0.413 |
| attn_hier:arruda_direct | 1.01 | 1 | 1.01 | 0.888 | 1.01 | 1.11 |
| attn_hier:arruda_gauss_direct | 1.01 | 1.02 | 1.03 | 1.04 | 1.11 | 1.2 |
| attn_hier:gauss_jacobian_FD | 1.01 | 1.02 | 1.03 | 1.02 | 1.12 | 1.19 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 1.01 | 1.02 | 1.03 | 1.02 | 1.14 | 1.23 |
| attn_full:gauss_hierarchical | 1 | 0.999 | 1.01 | 0.958 | 0.993 | 1.01 |
| attn_full:gauss_jacobian_FD | 1.01 | 1.02 | 1.02 | 1.01 | 1.06 | 1.09 |
| attn_full:arruda_schur | 1.15 | 1.37 | 1.63 | 1.61 | 1.51 | 1.44 |
| attn_full:arruda_laplace | 1.02 | 1.08 | 1.13 | 1.23 | 1.33 | 1.47 |
| bayesflow_arruda | 0.997 | 0.699 | 0.455 | 0.518 | 1.05 | 0.771 |
| bayesflow_arruda_diffprior | 1.01 | 1 | 0.989 | 0.952 | 1.01 | 1.11 |
| attn_hier_w1.5:gauss_jacobian_FD | 1.01 | 1.04 | 1.1 | 1.08 | 1.13 | – |
| legacy_w1.5:gauss_jacobian_FD | 1.01 | 1.03 | 1.11 | 1.07 | 1.09 | 1.09 |
| legacy_w1.5:gauss_hierarchical | 1.01 | 1.01 | 1.09 | 1.05 | 1.08 | 1.11 |

**W1/sd** (`mu_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0475 | 0.0521 | 0.0646 | 0.0523 | 0.0513 | 0.0665 |
| uncorrected | 0.0654 | 0.0874 | 0.131 | 0.22 | 0.238 | 0.466 |
| gauss | 0.0563 | 0.107 | 0.185 | 0.197 | 0.142 | 0.259 |
| gauss_hierarchical | 0.055 | 0.0705 | 0.115 | 0.132 | 0.134 | 0.158 |
| gauss_jacobian | 0.0654 | 0.0849 | 0.139 | 0.171 | 0.168 | 0.257 |
| gauss_jacobian_FD | 0.0654 | 0.085 | 0.136 | 0.171 | 0.168 | 0.269 |
| langevin_fnpe | 0.0531 | 0.0961 | 0.201 | 0.362 | 0.451 | 0.576 |
| attn_hier:arruda_direct | 0.0616 | 0.108 | 0.17 | 0.332 | 0.238 | 0.448 |
| attn_hier:arruda_gauss_direct | 0.0616 | 0.0945 | 0.132 | 0.134 | 0.16 | 0.32 |
| attn_hier:gauss_jacobian_FD | 0.0616 | 0.095 | 0.131 | 0.122 | 0.159 | 0.303 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0593 | 0.0971 | 0.131 | 0.152 | 0.196 | 0.364 |
| attn_full:gauss_hierarchical | 0.0527 | 0.0709 | 0.0997 | 0.169 | 0.132 | 0.198 |
| attn_full:gauss_jacobian_FD | 0.0612 | 0.0779 | 0.125 | 0.142 | 0.189 | 0.276 |
| attn_full:arruda_schur | 0.19 | 0.561 | 0.823 | 0.92 | 0.845 | 1.23 |
| attn_full:arruda_laplace | 0.0731 | 0.138 | 0.175 | 0.367 | 0.583 | 0.692 |
| bayesflow_arruda | 0.0444 | 0.291 | 0.601 | 0.52 | 0.193 | 0.379 |
| bayesflow_arruda_diffprior | 0.055 | 0.0803 | 0.158 | 0.236 | 0.14 | 0.264 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0614 | 0.0892 | 0.133 | 0.191 | 0.181 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0643 | 0.0726 | 0.164 | 0.18 | 0.176 | 0.27 |
| legacy_w1.5:gauss_hierarchical | 0.0517 | 0.0562 | 0.108 | 0.155 | 0.14 | 0.211 |

**90% coverage** (`mu_cover90`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.9 | 1 | 0.95 | 0.85 | 0.85 | 0.95 |
| uncorrected | 0.9 | 1 | 1 | 0.9 | 0.95 | 0.9 |
| gauss | 0.9 | 1 | 1 | 0.95 | 0.95 | 0.9 |
| gauss_hierarchical | 0.9 | 1 | 1 | 0.85 | 0.9 | 0.9 |
| gauss_jacobian | 0.9 | 1 | 1 | 0.85 | 0.95 | 1 |
| gauss_jacobian_FD | 0.9 | 1 | 1 | 0.85 | 0.95 | 1 |
| langevin_fnpe | 0.95 | 1 | 0.9 | 0.75 | 0.55 | 0.65 |
| attn_hier:arruda_direct | 0.9 | 1 | 1 | 0.8 | 0.95 | 0.95 |
| attn_hier:arruda_gauss_direct | 0.9 | 1 | 1 | 0.75 | 1 | 1 |
| attn_hier:gauss_jacobian_FD | 0.9 | 1 | 1 | 0.75 | 0.9 | 1 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.9 | 1 | 1 | 0.8 | 0.95 | 0.95 |
| attn_full:gauss_hierarchical | 0.9 | 1 | 1 | 0.85 | 0.95 | 0.95 |
| attn_full:gauss_jacobian_FD | 0.9 | 1 | 1 | 0.85 | 1 | 0.95 |
| attn_full:arruda_schur | 0.95 | 1 | 1 | 1 | 1 | 1 |
| attn_full:arruda_laplace | 0.9 | 1 | 1 | 0.9 | 1 | 1 |
| bayesflow_arruda | 0.9 | 0.75 | 0.45 | 0.65 | 0.85 | 0.8 |
| bayesflow_arruda_diffprior | 0.9 | 0.95 | 1 | 0.85 | 0.9 | 0.95 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.9 | 1 | 1 | 0.8 | 0.95 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.9 | 1 | 1 | 0.85 | 0.85 | 0.95 |
| legacy_w1.5:gauss_hierarchical | 0.9 | 1 | 1 | 0.85 | 0.85 | 0.95 |

## Q3 population spread σ_g

**|mean err|/sd** (`sigma_mean_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.035 | 0.0501 | 0.0632 | 0.0352 | 0.0386 | 0.0293 |
| uncorrected | 0.0452 | 0.139 | 0.177 | 0.094 | 0.125 | 0.261 |
| gauss | 0.0243 | 0.124 | 0.194 | 0.109 | 0.101 | 0.0664 |
| gauss_hierarchical | 0.0502 | 0.0962 | 0.152 | 0.0782 | 0.13 | 0.187 |
| gauss_jacobian | 0.0452 | 0.0768 | 0.225 | 0.0992 | 0.105 | 0.17 |
| gauss_jacobian_FD | 0.0452 | 0.075 | 0.223 | 0.1 | 0.104 | 0.167 |
| langevin_fnpe | 0.0334 | 0.0922 | 0.276 | 0.253 | 0.679 | 0.709 |
| attn_hier:arruda_direct | 0.0464 | 0.146 | 0.205 | 0.184 | 0.288 | 0.56 |
| attn_hier:arruda_gauss_direct | 0.0464 | 0.11 | 0.174 | 0.157 | 0.245 | 0.461 |
| attn_hier:gauss_jacobian_FD | 0.0464 | 0.11 | 0.173 | 0.167 | 0.23 | 0.451 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0412 | 0.117 | 0.172 | 0.162 | 0.221 | 0.487 |
| attn_full:gauss_hierarchical | 0.0445 | 0.0876 | 0.166 | 0.0983 | 0.124 | 0.186 |
| attn_full:gauss_jacobian_FD | 0.0501 | 0.0738 | 0.209 | 0.168 | 0.121 | 0.263 |
| attn_full:arruda_schur | 0.162 | 0.585 | 0.358 | 0.991 | 1.58 | 1.52 |
| attn_full:arruda_laplace | 0.049 | 0.221 | 0.267 | 0.557 | 0.873 | 0.874 |
| bayesflow_arruda | 0.0309 | 0.115 | 0.38 | 0.143 | 0.222 | 0.496 |
| bayesflow_arruda_diffprior | 0.0483 | 0.1 | 0.177 | 0.115 | 0.11 | 0.339 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0492 | 0.0952 | 0.217 | 0.17 | 0.146 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.052 | 0.0764 | 0.243 | 0.17 | 0.111 | 0.255 |
| legacy_w1.5:gauss_hierarchical | 0.0665 | 0.0676 | 0.249 | 0.144 | 0.113 | 0.228 |

**width ratio** (`sigma_width`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.993 | 1 | 1.01 | 0.996 | 1.01 | 0.995 |
| uncorrected | 0.997 | 1.01 | 1.01 | 1.01 | 1.12 | 1.23 |
| gauss | 0.993 | 1.04 | 1.07 | 1.1 | 1.1 | 1.2 |
| gauss_hierarchical | 0.994 | 1.01 | 1.01 | 1.02 | 1.09 | 1.14 |
| gauss_jacobian | 0.997 | 0.984 | 0.995 | 0.994 | 1.1 | 1.17 |
| gauss_jacobian_FD | 0.997 | 0.983 | 0.993 | 0.993 | 1.11 | 1.18 |
| langevin_fnpe | 0.991 | 0.957 | 0.886 | 0.752 | 0.659 | 0.513 |
| attn_hier:arruda_direct | 0.998 | 1.04 | 1.08 | 1.09 | 1.24 | 1.43 |
| attn_hier:arruda_gauss_direct | 0.998 | 1.02 | 1.06 | 1.07 | 1.19 | 1.34 |
| attn_hier:gauss_jacobian_FD | 0.998 | 1.02 | 1.06 | 1.06 | 1.2 | 1.34 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.995 | 1.03 | 1.06 | 1.07 | 1.22 | 1.39 |
| attn_full:gauss_hierarchical | 0.991 | 1 | 0.976 | 0.957 | 1.01 | 1.04 |
| attn_full:gauss_jacobian_FD | 0.996 | 0.974 | 0.969 | 0.942 | 1.02 | 1.09 |
| attn_full:arruda_schur | 1.11 | 1.21 | 1.28 | 1.44 | 1.67 | 1.61 |
| attn_full:arruda_laplace | 1.01 | 1.1 | 1.14 | 1.28 | 1.56 | 1.72 |
| bayesflow_arruda | 0.998 | 0.699 | 0.409 | 0.511 | 1.07 | 0.858 |
| bayesflow_arruda_diffprior | 0.995 | 1.04 | 1.03 | 1.03 | 0.999 | 0.997 |
| attn_hier_w1.5:gauss_jacobian_FD | 1 | 1.04 | 1.08 | 1.08 | 1.16 | – |
| legacy_w1.5:gauss_jacobian_FD | 1 | 0.997 | 0.992 | 0.974 | 1.03 | 1.05 |
| legacy_w1.5:gauss_hierarchical | 0.989 | 1.02 | 1.02 | 1.01 | 1.05 | 1.08 |

**W1/sd** (`sigma_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0543 | 0.0568 | 0.0679 | 0.0567 | 0.0514 | 0.0503 |
| uncorrected | 0.0616 | 0.146 | 0.181 | 0.133 | 0.208 | 0.347 |
| gauss | 0.0449 | 0.125 | 0.2 | 0.185 | 0.145 | 0.251 |
| gauss_hierarchical | 0.055 | 0.0991 | 0.156 | 0.102 | 0.17 | 0.229 |
| gauss_jacobian | 0.0616 | 0.0902 | 0.228 | 0.129 | 0.153 | 0.266 |
| gauss_jacobian_FD | 0.0616 | 0.0885 | 0.225 | 0.127 | 0.155 | 0.271 |
| langevin_fnpe | 0.0462 | 0.111 | 0.283 | 0.367 | 0.69 | 0.79 |
| attn_hier:arruda_direct | 0.0616 | 0.151 | 0.208 | 0.236 | 0.34 | 0.699 |
| attn_hier:arruda_gauss_direct | 0.0616 | 0.118 | 0.177 | 0.205 | 0.301 | 0.556 |
| attn_hier:gauss_jacobian_FD | 0.0616 | 0.118 | 0.177 | 0.192 | 0.301 | 0.531 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.057 | 0.122 | 0.173 | 0.184 | 0.295 | 0.61 |
| attn_full:gauss_hierarchical | 0.0523 | 0.0896 | 0.171 | 0.141 | 0.146 | 0.202 |
| attn_full:gauss_jacobian_FD | 0.0632 | 0.0821 | 0.217 | 0.17 | 0.139 | 0.276 |
| attn_full:arruda_schur | 0.176 | 0.596 | 0.511 | 1.03 | 1.63 | 1.82 |
| attn_full:arruda_laplace | 0.0614 | 0.232 | 0.298 | 0.591 | 0.92 | 0.984 |
| bayesflow_arruda | 0.0478 | 0.302 | 0.679 | 0.469 | 0.223 | 0.537 |
| bayesflow_arruda_diffprior | 0.057 | 0.11 | 0.186 | 0.148 | 0.155 | 0.349 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0644 | 0.109 | 0.224 | 0.182 | 0.211 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.061 | 0.0886 | 0.247 | 0.201 | 0.13 | 0.265 |
| legacy_w1.5:gauss_hierarchical | 0.0755 | 0.0721 | 0.25 | 0.178 | 0.158 | 0.282 |

**90% coverage** (`sigma_cover90`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.95 | 0.9 | 0.95 | 0.9 | 0.85 | 0.9 |
| uncorrected | 0.95 | 0.9 | 0.9 | 0.8 | 0.85 | 0.8 |
| gauss | 0.95 | 0.9 | 0.9 | 0.9 | 0.9 | 0.95 |
| gauss_hierarchical | 0.95 | 0.9 | 0.85 | 0.8 | 0.85 | 0.9 |
| gauss_jacobian | 0.95 | 0.9 | 0.8 | 0.85 | 0.85 | 0.9 |
| gauss_jacobian_FD | 0.95 | 0.9 | 0.8 | 0.85 | 0.85 | 0.9 |
| langevin_fnpe | 0.95 | 0.9 | 0.8 | 0.7 | 0.65 | 0.45 |
| attn_hier:arruda_direct | 0.95 | 0.9 | 0.95 | 0.85 | 0.9 | 0.95 |
| attn_hier:arruda_gauss_direct | 0.95 | 0.9 | 0.9 | 0.9 | 0.85 | 0.95 |
| attn_hier:gauss_jacobian_FD | 0.95 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.95 | 0.9 | 0.95 | 0.9 | 0.9 | 0.95 |
| attn_full:gauss_hierarchical | 0.95 | 0.9 | 0.85 | 0.8 | 0.8 | 0.85 |
| attn_full:gauss_jacobian_FD | 0.95 | 0.9 | 0.8 | 0.7 | 0.8 | 0.85 |
| attn_full:arruda_schur | 0.95 | 0.95 | 1 | 0.95 | 1 | 0.95 |
| attn_full:arruda_laplace | 0.95 | 0.9 | 0.95 | 0.95 | 0.95 | 1 |
| bayesflow_arruda | 0.95 | 0.8 | 0.25 | 0.6 | 0.9 | 0.7 |
| bayesflow_arruda_diffprior | 0.95 | 0.9 | 0.95 | 0.95 | 0.8 | 0.95 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.95 | 0.9 | 0.95 | 0.95 | 0.9 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.95 | 0.9 | 0.8 | 0.8 | 0.8 | 0.9 |
| legacy_w1.5:gauss_hierarchical | 0.95 | 0.9 | 0.85 | 0.9 | 0.8 | 0.9 |

**joint global C2ST** (`global_c2st`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.51 | 0.497 | 0.499 | 0.496 | 0.498 | 0.487 |
| uncorrected | 0.513 | 0.522 | 0.572 | 0.614 | 0.679 | 0.819 |
| gauss | 0.496 | 0.536 | 0.672 | 0.581 | 0.568 | 0.681 |
| gauss_hierarchical | 0.509 | 0.52 | 0.57 | 0.565 | 0.589 | 0.698 |
| gauss_jacobian | 0.513 | 0.519 | 0.615 | 0.554 | 0.604 | 0.73 |
| gauss_jacobian_FD | 0.513 | 0.523 | 0.595 | 0.566 | 0.601 | 0.728 |
| langevin_fnpe | 0.504 | 0.528 | 0.651 | 0.728 | 0.821 | 0.911 |
| attn_hier:arruda_direct | 0.509 | 0.541 | 0.606 | 0.65 | 0.695 | 0.855 |
| attn_hier:arruda_gauss_direct | 0.509 | 0.542 | 0.613 | 0.587 | 0.657 | 0.798 |
| attn_hier:gauss_jacobian_FD | 0.509 | 0.536 | 0.607 | 0.595 | 0.655 | 0.783 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.518 | 0.554 | 0.6 | 0.599 | 0.663 | 0.802 |
| attn_full:gauss_hierarchical | 0.503 | 0.513 | 0.575 | 0.56 | 0.579 | 0.654 |
| attn_full:gauss_jacobian_FD | 0.517 | 0.521 | 0.597 | 0.583 | 0.583 | 0.678 |
| attn_full:arruda_schur | 0.579 | 0.756 | 0.869 | 0.909 | 0.931 | 0.952 |
| attn_full:arruda_laplace | 0.517 | 0.565 | 0.655 | 0.759 | 0.838 | 0.891 |
| bayesflow_arruda | 0.495 | 0.72 | 0.94 | 0.867 | 0.589 | 0.744 |
| bayesflow_arruda_diffprior | 0.492 | 0.52 | 0.582 | 0.6 | 0.596 | 0.719 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.519 | 0.514 | 0.584 | 0.558 | 0.603 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.515 | 0.518 | 0.61 | 0.567 | 0.58 | 0.692 |
| legacy_w1.5:gauss_hierarchical | 0.504 | 0.517 | 0.574 | 0.559 | 0.576 | 0.693 |

## Q2+Q3 globals in mirror-adapted coordinates

**P(u>0) (exact 0.5)** (`mode_balance`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| attn_hier_w1.5:gauss_jacobian_FD | 0.499 | 0.489 | 0.51 | 0.472 | 0.483 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.502 | 0.509 | 0.525 | 0.508 | 0.577 | 0.582 |
| legacy_w1.5:gauss_hierarchical | 0.496 | 0.497 | 0.518 | 0.508 | 0.534 | 0.546 |

**|P(u>0) - ref|** (`mode_balance_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| attn_hier_w1.5:gauss_jacobian_FD | 0.015 | 0.0297 | 0.0185 | 0.0285 | 0.0297 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0143 | 0.0223 | 0.0323 | 0.0145 | 0.0642 | 0.084 |
| legacy_w1.5:gauss_hierarchical | 0.0113 | 0.00975 | 0.033 | 0.0405 | 0.0248 | 0.0703 |

**u=(μ0+μ1)/√2 W1/sd (bimodal)** (`sym_u_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| attn_hier_w1.5:gauss_jacobian_FD | 0.0522 | 0.0848 | 0.119 | 0.167 | 0.182 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0593 | 0.0714 | 0.12 | 0.113 | 0.158 | 0.247 |
| legacy_w1.5:gauss_hierarchical | 0.0465 | 0.0588 | 0.112 | 0.108 | 0.095 | 0.194 |

**w=σ0-σ1 W1/sd (bimodal)** (`sym_w_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| attn_hier_w1.5:gauss_jacobian_FD | 0.0797 | 0.0834 | 0.138 | 0.156 | 0.178 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0672 | 0.0691 | 0.116 | 0.172 | 0.167 | 0.191 |
| legacy_w1.5:gauss_hierarchical | 0.0613 | 0.0652 | 0.138 | 0.127 | 0.133 | 0.228 |

**v=(μ1-μ0)/√2 W1/sd** (`sym_v_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| attn_hier_w1.5:gauss_jacobian_FD | 0.0649 | 0.0801 | 0.123 | 0.217 | 0.122 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0744 | 0.0668 | 0.164 | 0.237 | 0.179 | 0.305 |
| legacy_w1.5:gauss_hierarchical | 0.0726 | 0.0752 | 0.156 | 0.164 | 0.179 | 0.257 |

**s=σ0+σ1 W1/sd** (`sym_s_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| attn_hier_w1.5:gauss_jacobian_FD | 0.0662 | 0.137 | 0.309 | 0.261 | 0.228 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0535 | 0.115 | 0.419 | 0.274 | 0.141 | 0.386 |
| legacy_w1.5:gauss_hierarchical | 0.0919 | 0.108 | 0.435 | 0.291 | 0.13 | 0.346 |

**mirror-invariant W1/sd** (`sym_inv_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| attn_hier_w1.5:gauss_jacobian_FD | 0.0651 | 0.106 | 0.193 | 0.226 | 0.221 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.068 | 0.0866 | 0.258 | 0.219 | 0.142 | 0.286 |
| legacy_w1.5:gauss_hierarchical | 0.0681 | 0.0797 | 0.219 | 0.189 | 0.123 | 0.244 |

## Q4 global–local dependence

**|Δ corr(g, θ_i)|** (`dep_corr_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0355 | 0.0391 | 0.0372 | 0.0366 | 0.0404 | 0.0386 |
| uncorrected | 0.0332 | 0.0514 | 0.0834 | 0.0643 | 0.0555 | 0.0664 |
| gauss | 0.0539 | 0.0658 | 0.107 | 0.0772 | 0.045 | 0.0586 |
| gauss_hierarchical | 0.0344 | 0.0533 | 0.0905 | 0.069 | 0.0438 | 0.0592 |
| gauss_jacobian | 0.0332 | 0.0523 | 0.0839 | 0.0702 | 0.045 | 0.0623 |
| gauss_jacobian_FD | 0.0332 | 0.0531 | 0.0825 | 0.0705 | 0.0449 | 0.0627 |
| langevin_fnpe | 0.0405 | 0.0623 | 0.0993 | 0.0909 | 0.0497 | 0.074 |
| attn_hier:arruda_direct | 0.0412 | 0.0501 | 0.0677 | 0.0633 | 0.0635 | 0.103 |
| attn_hier:arruda_gauss_direct | 0.0412 | 0.0551 | 0.0775 | 0.0605 | 0.0632 | 0.0867 |
| attn_hier:gauss_jacobian_FD | 0.0504 | 0.088 | 0.106 | 0.0945 | 0.0533 | 0.0692 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0482 | 0.0923 | 0.115 | 0.0999 | 0.0557 | 0.072 |
| attn_full:gauss_hierarchical | 0.0354 | 0.0526 | 0.0904 | 0.0744 | 0.0456 | 0.0652 |
| attn_full:gauss_jacobian_FD | 0.0347 | 0.0523 | 0.0851 | 0.0767 | 0.0486 | 0.0681 |
| attn_full:arruda_schur | 0.052 | 0.103 | 0.137 | 0.161 | 0.169 | 0.195 |
| attn_full:arruda_laplace | 0.0401 | 0.0785 | 0.0945 | 0.126 | 0.154 | 0.162 |
| bayesflow_arruda | 0.037 | 0.0819 | 0.137 | 0.099 | 0.0429 | 0.0603 |
| bayesflow_arruda_diffprior | 0.0427 | 0.0513 | 0.0518 | 0.053 | 0.0452 | 0.048 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0517 | 0.0803 | 0.0962 | 0.0817 | 0.0488 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0403 | 0.0477 | 0.0795 | 0.067 | 0.0451 | 0.0624 |
| legacy_w1.5:gauss_hierarchical | 0.0374 | 0.051 | 0.0781 | 0.0675 | 0.0434 | 0.0609 |

**ref |corr| (signal)** (`dep_corr_ref`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| uncorrected | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| gauss | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| gauss_hierarchical | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| gauss_jacobian | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| gauss_jacobian_FD | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| langevin_fnpe | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_hier:arruda_direct | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_hier:arruda_gauss_direct | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_hier:gauss_jacobian_FD | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_full:gauss_hierarchical | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_full:gauss_jacobian_FD | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_full:arruda_schur | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_full:arruda_laplace | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| bayesflow_arruda | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| bayesflow_arruda_diffprior | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |
| legacy_w1.5:gauss_hierarchical | 0.0595 | 0.141 | 0.145 | 0.116 | 0.0583 | 0.0845 |

**joint (g, θ_i) C2ST** (`joint_c2st`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.504 | 0.5 | 0.495 | 0.494 | 0.494 | 0.498 |
| uncorrected | 0.515 | 0.527 | 0.562 | 0.603 | 0.66 | 0.786 |
| gauss | 0.772 | 0.736 | 0.764 | 0.722 | 0.702 | 0.737 |
| gauss_hierarchical | 0.506 | 0.518 | 0.552 | 0.562 | 0.585 | 0.665 |
| gauss_jacobian | 0.515 | 0.517 | 0.586 | 0.561 | 0.588 | 0.69 |
| gauss_jacobian_FD | 0.515 | 0.518 | 0.585 | 0.563 | 0.59 | 0.694 |
| langevin_fnpe | 0.513 | 0.533 | 0.653 | 0.712 | 0.813 | 0.897 |
| attn_hier:arruda_direct | 0.513 | 0.536 | 0.589 | 0.64 | 0.672 | 0.832 |
| attn_hier:arruda_gauss_direct | 0.513 | 0.525 | 0.58 | 0.576 | 0.624 | 0.767 |
| attn_hier:gauss_jacobian_FD | 0.517 | 0.544 | 0.584 | 0.59 | 0.633 | 0.761 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.528 | 0.554 | 0.598 | 0.586 | 0.64 | 0.791 |
| attn_full:gauss_hierarchical | 0.506 | 0.521 | 0.558 | 0.554 | 0.564 | 0.643 |
| attn_full:gauss_jacobian_FD | 0.5 | 0.518 | 0.578 | 0.57 | 0.575 | 0.668 |
| attn_full:arruda_schur | 0.573 | 0.749 | 0.841 | 0.887 | 0.91 | 0.943 |
| attn_full:arruda_laplace | 0.512 | 0.559 | 0.618 | 0.741 | 0.818 | 0.866 |
| bayesflow_arruda | 0.496 | 0.686 | 0.925 | 0.85 | 0.576 | 0.736 |
| bayesflow_arruda_diffprior | 0.489 | 0.515 | 0.574 | 0.582 | 0.585 | 0.699 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.518 | 0.538 | 0.571 | 0.567 | 0.587 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.506 | 0.512 | 0.595 | 0.564 | 0.574 | 0.668 |
| legacy_w1.5:gauss_hierarchical | 0.499 | 0.512 | 0.567 | 0.551 | 0.562 | 0.667 |

**same, dependence shuffled out** (`joint_c2st_indep`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.518 | 0.538 | 0.544 | 0.519 | 0.503 | 0.511 |
| uncorrected | 0.536 | 0.547 | 0.586 | 0.604 | 0.662 | 0.788 |
| gauss | 0.778 | 0.741 | 0.775 | 0.728 | 0.697 | 0.738 |
| gauss_hierarchical | 0.529 | 0.552 | 0.579 | 0.568 | 0.586 | 0.666 |
| gauss_jacobian | 0.536 | 0.537 | 0.611 | 0.57 | 0.591 | 0.691 |
| gauss_jacobian_FD | 0.536 | 0.54 | 0.608 | 0.571 | 0.591 | 0.695 |
| langevin_fnpe | 0.514 | 0.565 | 0.663 | 0.715 | 0.813 | 0.898 |
| attn_hier:arruda_direct | 0.52 | 0.561 | 0.614 | 0.649 | 0.673 | 0.835 |
| attn_hier:arruda_gauss_direct | 0.52 | 0.559 | 0.608 | 0.589 | 0.629 | 0.769 |
| attn_hier:gauss_jacobian_FD | 0.534 | 0.554 | 0.604 | 0.591 | 0.632 | 0.762 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.536 | 0.565 | 0.606 | 0.588 | 0.643 | 0.79 |
| attn_full:gauss_hierarchical | 0.526 | 0.549 | 0.576 | 0.561 | 0.566 | 0.646 |
| attn_full:gauss_jacobian_FD | 0.538 | 0.546 | 0.608 | 0.578 | 0.577 | 0.671 |
| attn_full:arruda_schur | 0.585 | 0.752 | 0.843 | 0.885 | 0.907 | 0.933 |
| attn_full:arruda_laplace | 0.53 | 0.588 | 0.644 | 0.742 | 0.819 | 0.865 |
| bayesflow_arruda | 0.524 | 0.701 | 0.925 | 0.851 | 0.58 | 0.739 |
| bayesflow_arruda_diffprior | 0.52 | 0.552 | 0.602 | 0.596 | 0.589 | 0.701 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.531 | 0.547 | 0.609 | 0.576 | 0.591 | – |
| legacy_w1.5:gauss_jacobian_FD | 0.537 | 0.536 | 0.611 | 0.574 | 0.579 | 0.666 |
| legacy_w1.5:gauss_hierarchical | 0.517 | 0.543 | 0.585 | 0.56 | 0.566 | 0.669 |

## Cost

**s / dataset** (`seconds`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 13.9 | 25.1 | 0.61 | 0.639 | 0.987 | 2.41 |
| uncorrected | 1.27 | 2.27 | 4.49 | 10.6 | 18.7 | 41.8 |
| gauss | 5.56 | 6.49 | 9.02 | 21.5 | 43.3 | 83.2 |
| gauss_hierarchical | 6.18 | 7.07 | 9.11 | 22.5 | 65.4 | 82.8 |
| gauss_jacobian | 10.8 | 12.4 | 17.6 | 50.7 | 198 | 176 |
| gauss_jacobian_FD | 3.22 | 9.71 | 18.8 | 47.7 | 183 | 189 |
| langevin_fnpe | 6.38 | 11.4 | 22.6 | 52.5 | 134 | 223 |
| attn_hier:arruda_direct | 2.36 | 5.72 | 10.7 | 26.3 | 55.7 | 102 |
| attn_hier:arruda_gauss_direct | 6.15 | 23.5 | 46.3 | 111 | 221 | 440 |
| attn_hier:gauss_jacobian_FD | 3.41 | 10.6 | 21.4 | 58.1 | 107 | 208 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 18.5 | 20.8 | 31.7 | 64.7 | 118 | 265 |
| attn_full:gauss_hierarchical | 6.15 | 7.32 | 10.6 | 26.4 | 51 | 96.4 |
| attn_full:gauss_jacobian_FD | 3.41 | 10.8 | 21.2 | 59.5 | 106 | 212 |
| attn_full:arruda_schur | 5.38 | 22.7 | 44.5 | 137 | 215 | 439 |
| attn_full:arruda_laplace | 8.61 | 39.1 | 75.9 | 217 | 370 | 755 |
| bayesflow_arruda | 5.43 | 14.1 | 15.1 | 19.6 | 20.4 | 17.9 |
| bayesflow_arruda_diffprior | 5.36 | 18.9 | 25.8 | 23.8 | 28 | 32.9 |
| attn_hier_w1.5:gauss_jacobian_FD | 19.3 | 50.6 | 104 | 384 | 313 | – |
| legacy_w1.5:gauss_jacobian_FD | 18.8 | 52.1 | 103 | 327 | 663 | 610 |
| legacy_w1.5:gauss_hierarchical | 31.1 | 56.8 | 95.5 | 134 | 211 | 387 |

**non-finite frac** (`nonfinite`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0 | 0 | 0 | 0 | 0 | 0 |
| uncorrected | 0 | 0 | 0 | 0 | 0 | 0 |
| gauss | 0 | 0 | 0 | 0 | 0 | 0 |
| gauss_hierarchical | 0 | 0 | 0 | 0 | 0 | 0 |
| gauss_jacobian | 0 | 0 | 0 | 0 | 0 | 0 |
| gauss_jacobian_FD | 0 | 0 | 0 | 0 | 0 | 0 |
| langevin_fnpe | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_hier:arruda_direct | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_hier:arruda_gauss_direct | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_hier:gauss_jacobian_FD | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_full:gauss_hierarchical | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_full:gauss_jacobian_FD | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_full:arruda_schur | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_full:arruda_laplace | 0 | 0 | 0 | 0 | 0 | 0 |
| bayesflow_arruda | 0 | 0 | 0 | 0 | 0 | 0 |
| bayesflow_arruda_diffprior | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_hier_w1.5:gauss_jacobian_FD | 0 | 0 | 0 | 0 | 0 | – |
| legacy_w1.5:gauss_jacobian_FD | 0 | 0 | 0 | 0 | 0 | 0 |
| legacy_w1.5:gauss_hierarchical | 0 | 0 | 0 | 0 | 0 | 0 |

## BayesFlow tuned (d1, shift)

| method | N | d1 | shift |
|---|---|---|---|
| bayesflow_arruda | 1 | 1 | 0 |
| bayesflow_arruda | 5 | 1 | 0 |
| bayesflow_arruda | 10 | 1 | 2 |
| bayesflow_arruda | 25 | 1 | 0 |
| bayesflow_arruda | 50 | 0.001 | 0 |
| bayesflow_arruda | 100 | 0.01 | 2 |
| bayesflow_arruda_diffprior | 1 | 1 | 0 |
| bayesflow_arruda_diffprior | 5 | 1 | 0 |
| bayesflow_arruda_diffprior | 10 | 1 | 0 |
| bayesflow_arruda_diffprior | 25 | 1 | 2 |
| bayesflow_arruda_diffprior | 50 | 1 | 0 |
| bayesflow_arruda_diffprior | 100 | 1 | 2 |
