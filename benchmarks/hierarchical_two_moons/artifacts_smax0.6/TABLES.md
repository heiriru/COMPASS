# Hierarchical Two Moons — results

Medians over datasets (coverage: mean). `reference` = 500 independent reference draws scored the same way: the Monte-Carlo floor.

## Q1 local posterior p(θ_i | x_1:N)

**local C2ST (0.5 best)** (`local_c2st`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.493 | 0.496 | 0.502 | 0.5 | 0.499 | 0.498 |
| uncorrected | 0.518 | 0.519 | 0.521 | 0.553 | 0.54 | 0.58 |
| gauss | 0.968 | 0.959 | 0.956 | 0.952 | 0.944 | 0.937 |
| gauss_hierarchical | 0.519 | 0.509 | 0.516 | 0.536 | 0.529 | 0.543 |
| gauss_jacobian | 0.518 | 0.525 | 0.519 | 0.54 | 0.526 | 0.559 |
| gauss_jacobian_FD | 0.518 | 0.525 | 0.523 | 0.543 | 0.53 | 0.564 |
| langevin_fnpe | 0.515 | 0.52 | 0.528 | 0.537 | 0.53 | 0.537 |
| attn_hier:arruda_direct | 0.617 | 0.612 | 0.617 | 0.631 | 0.618 | 0.637 |
| attn_hier:arruda_gauss_direct | 0.617 | 0.608 | 0.616 | 0.629 | 0.618 | 0.629 |
| attn_hier:gauss_jacobian_FD | 0.502 | 0.517 | 0.52 | 0.527 | 0.521 | 0.543 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.516 | 0.515 | 0.523 | 0.532 | 0.523 | 0.548 |
| attn_full:gauss_hierarchical | 0.522 | 0.516 | 0.52 | 0.531 | 0.524 | 0.533 |
| attn_full:gauss_jacobian_FD | 0.522 | 0.515 | 0.518 | 0.53 | 0.526 | 0.555 |
| attn_full:arruda_schur | 0.614 | 0.62 | 0.626 | 0.648 | 0.646 | 0.655 |
| attn_full:arruda_laplace | 0.611 | 0.611 | 0.618 | 0.637 | 0.625 | 0.644 |
| bayesflow_arruda | 0.502 | 0.516 | 0.524 | 0.53 | 0.529 | 0.529 |
| bayesflow_arruda_diffprior | 0.508 | 0.514 | 0.521 | 0.541 | 0.53 | 0.551 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.514 | 0.513 | 0.521 | 0.524 | 0.521 | 0.526 |
| legacy_w1.5:gauss_jacobian_FD | 0.53 | 0.517 | 0.522 | 0.53 | 0.519 | 0.529 |
| legacy_w1.5:gauss_hierarchical | 0.514 | 0.514 | 0.521 | 0.528 | 0.522 | 0.529 |
| attn_hier_tsm:gauss_jacobian_FD | 0.512 | 0.511 | 0.521 | 0.524 | 0.522 | 0.531 |
| attn_hier_tsm:arruda_direct | 0.613 | 0.61 | 0.617 | 0.628 | 0.626 | 0.631 |

**local W1/sd** (`local_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0405 | 0.0513 | 0.0525 | 0.048 | 0.0457 | 0.0475 |
| uncorrected | 0.0615 | 0.0603 | 0.0686 | 0.121 | 0.135 | 0.191 |
| gauss | 0.808 | 1.35 | 1.24 | 0.971 | 1.01 | 1.05 |
| gauss_hierarchical | 0.0757 | 0.0742 | 0.0652 | 0.102 | 0.0762 | 0.128 |
| gauss_jacobian | 0.0615 | 0.0685 | 0.0697 | 0.096 | 0.0672 | 0.109 |
| gauss_jacobian_FD | 0.0615 | 0.0688 | 0.0705 | 0.0933 | 0.0689 | 0.114 |
| langevin_fnpe | 0.085 | 0.0652 | 0.082 | 0.0885 | 0.0971 | 0.0986 |
| attn_hier:arruda_direct | 0.0663 | 0.071 | 0.086 | 0.146 | 0.135 | 0.21 |
| attn_hier:arruda_gauss_direct | 0.0663 | 0.0707 | 0.0754 | 0.142 | 0.116 | 0.18 |
| attn_hier:gauss_jacobian_FD | 0.0565 | 0.0756 | 0.0712 | 0.0988 | 0.0917 | 0.143 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.063 | 0.0699 | 0.0729 | 0.104 | 0.0963 | 0.134 |
| attn_full:gauss_hierarchical | 0.0736 | 0.0626 | 0.0699 | 0.0884 | 0.101 | 0.108 |
| attn_full:gauss_jacobian_FD | 0.0483 | 0.0666 | 0.0691 | 0.106 | 0.122 | 0.229 |
| attn_full:arruda_schur | 0.0857 | 0.142 | 0.178 | 0.25 | 0.274 | 0.366 |
| attn_full:arruda_laplace | 0.0776 | 0.076 | 0.0861 | 0.228 | 0.188 | 0.243 |
| bayesflow_arruda | 0.066 | 0.0652 | 0.0637 | 0.0829 | 0.116 | 0.0942 |
| bayesflow_arruda_diffprior | 0.0654 | 0.0625 | 0.0744 | 0.109 | 0.117 | 0.165 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0408 | 0.0745 | 0.0671 | 0.089 | 0.0845 | 0.111 |
| legacy_w1.5:gauss_jacobian_FD | 0.0579 | 0.0669 | 0.0666 | 0.0718 | 0.0725 | 0.0831 |
| legacy_w1.5:gauss_hierarchical | 0.0786 | 0.0583 | 0.0576 | 0.0826 | 0.0798 | 0.0777 |
| attn_hier_tsm:gauss_jacobian_FD | 0.051 | 0.0708 | 0.0665 | 0.0873 | 0.0773 | 0.111 |
| attn_hier_tsm:arruda_direct | 0.0729 | 0.0637 | 0.0745 | 0.164 | 0.235 | 0.223 |

**branch-mass err** (`branch_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0153 | 0.0224 | 0.0221 | 0.0224 | 0.0195 | 0.0188 |
| uncorrected | 0.0102 | 0.0199 | 0.0212 | 0.0468 | 0.0503 | 0.0926 |
| gauss | 0.0123 | 0.0172 | 0.0201 | 0.024 | 0.0232 | 0.0537 |
| gauss_hierarchical | 0.026 | 0.026 | 0.0221 | 0.0328 | 0.0322 | 0.0544 |
| gauss_jacobian | 0.0102 | 0.0201 | 0.0222 | 0.0356 | 0.0248 | 0.0394 |
| gauss_jacobian_FD | 0.0102 | 0.0203 | 0.0215 | 0.0359 | 0.0263 | 0.0427 |
| langevin_fnpe | 0.0172 | 0.0257 | 0.0299 | 0.0247 | 0.0387 | 0.0376 |
| attn_hier:arruda_direct | 0.0193 | 0.0201 | 0.026 | 0.0547 | 0.0544 | 0.0813 |
| attn_hier:arruda_gauss_direct | 0.0193 | 0.021 | 0.0215 | 0.0512 | 0.0396 | 0.0574 |
| attn_hier:gauss_jacobian_FD | 0.0175 | 0.0201 | 0.0222 | 0.0284 | 0.0367 | 0.0465 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.026 | 0.0232 | 0.0229 | 0.0319 | 0.0384 | 0.044 |
| attn_full:gauss_hierarchical | 0.0225 | 0.0242 | 0.022 | 0.0269 | 0.0396 | 0.0365 |
| attn_full:gauss_jacobian_FD | 0.00775 | 0.0204 | 0.0224 | 0.0362 | 0.0527 | 0.0951 |
| attn_full:arruda_schur | 0.0192 | 0.0614 | 0.0822 | 0.111 | 0.122 | 0.167 |
| attn_full:arruda_laplace | 0.0173 | 0.0206 | 0.0282 | 0.106 | 0.0693 | 0.11 |
| bayesflow_arruda | 0.0133 | 0.0251 | 0.0236 | 0.0268 | 0.0473 | 0.0344 |
| bayesflow_arruda_diffprior | 0.0227 | 0.024 | 0.0278 | 0.045 | 0.054 | 0.0759 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0102 | 0.0208 | 0.0204 | 0.0325 | 0.0345 | 0.0439 |
| legacy_w1.5:gauss_jacobian_FD | 0.0178 | 0.023 | 0.0198 | 0.023 | 0.0272 | 0.0279 |
| legacy_w1.5:gauss_hierarchical | 0.0297 | 0.0183 | 0.0179 | 0.0252 | 0.0288 | 0.0268 |
| attn_hier_tsm:gauss_jacobian_FD | 0.015 | 0.0202 | 0.0173 | 0.0267 | 0.0333 | 0.0437 |
| attn_hier_tsm:arruda_direct | 0.0148 | 0.0234 | 0.026 | 0.0618 | 0.0993 | 0.0893 |

**crescent |r-0.1|/0.01 (exact 0.67)** (`crescent_dev`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.694 | 0.657 | 0.671 | 0.673 | 0.672 | 0.674 |
| uncorrected | 0.793 | 0.773 | 0.778 | 0.815 | 0.811 | 0.913 |
| gauss | 48.8 | 46.5 | 35.4 | 30 | 24.2 | 19.1 |
| gauss_hierarchical | 0.763 | 0.788 | 0.766 | 0.788 | 0.789 | 0.826 |
| gauss_jacobian | 0.793 | 0.779 | 0.776 | 0.803 | 0.796 | 0.853 |
| gauss_jacobian_FD | 0.793 | 0.779 | 0.777 | 0.804 | 0.797 | 0.86 |
| langevin_fnpe | 0.769 | 0.783 | 0.765 | 0.791 | 0.778 | 0.799 |
| attn_hier:arruda_direct | 1.39 | 1.38 | 1.39 | 1.42 | 1.41 | 1.42 |
| attn_hier:arruda_gauss_direct | 1.39 | 1.39 | 1.4 | 1.42 | 1.4 | 1.42 |
| attn_hier:gauss_jacobian_FD | 0.757 | 0.765 | 0.758 | 0.781 | 0.766 | 0.804 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.752 | 0.776 | 0.758 | 0.783 | 0.769 | 0.804 |
| attn_full:gauss_hierarchical | 0.775 | 0.778 | 0.753 | 0.767 | 0.764 | 0.796 |
| attn_full:gauss_jacobian_FD | 0.785 | 0.75 | 0.756 | 0.777 | 0.765 | 0.801 |
| attn_full:arruda_schur | 1.41 | 1.43 | 1.44 | 1.47 | 1.47 | 1.51 |
| attn_full:arruda_laplace | 1.43 | 1.4 | 1.41 | 1.44 | 1.43 | 1.47 |
| bayesflow_arruda | 0.735 | 0.718 | 0.713 | 0.732 | 0.728 | 0.757 |
| bayesflow_arruda_diffprior | 0.8 | 0.755 | 0.735 | 0.8 | 0.798 | 0.836 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.758 | 0.748 | 0.756 | 0.764 | 0.761 | 0.776 |
| legacy_w1.5:gauss_jacobian_FD | 0.763 | 0.782 | 0.776 | 0.77 | 0.774 | 0.785 |
| legacy_w1.5:gauss_hierarchical | 0.767 | 0.781 | 0.762 | 0.775 | 0.769 | 0.78 |
| attn_hier_tsm:gauss_jacobian_FD | 0.748 | 0.766 | 0.755 | 0.769 | 0.759 | 0.782 |
| attn_hier_tsm:arruda_direct | 1.37 | 1.38 | 1.4 | 1.41 | 1.4 | 1.41 |

**off-support frac** (`off_support`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0 | 0 | 0 | 0 | 0 | 0 |
| uncorrected | 0.016 | 0.0316 | 0.023 | 0.0724 | 0.0442 | 0.0751 |
| gauss | 0.738 | 0.646 | 0.542 | 0.554 | 0.526 | 0.482 |
| gauss_hierarchical | 0.019 | 0.0254 | 0.023 | 0.0683 | 0.0412 | 0.0546 |
| gauss_jacobian | 0.016 | 0.0302 | 0.0231 | 0.0665 | 0.0481 | 0.0593 |
| gauss_jacobian_FD | 0.016 | 0.03 | 0.023 | 0.0662 | 0.0478 | 0.0599 |
| langevin_fnpe | 0.019 | 0.0302 | 0.0253 | 0.0492 | 0.0319 | 0.0482 |
| attn_hier:arruda_direct | 0.05 | 0.0582 | 0.0564 | 0.0763 | 0.0566 | 0.0737 |
| attn_hier:arruda_gauss_direct | 0.05 | 0.0558 | 0.0563 | 0.0778 | 0.0553 | 0.0746 |
| attn_hier:gauss_jacobian_FD | 0.02 | 0.0266 | 0.0247 | 0.0418 | 0.0283 | 0.0416 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.019 | 0.0224 | 0.0233 | 0.0373 | 0.0272 | 0.0389 |
| attn_full:gauss_hierarchical | 0.016 | 0.019 | 0.022 | 0.0408 | 0.0283 | 0.0417 |
| attn_full:gauss_jacobian_FD | 0.02 | 0.0242 | 0.0235 | 0.0422 | 0.0312 | 0.0421 |
| attn_full:arruda_schur | 0.053 | 0.0548 | 0.0548 | 0.0689 | 0.0565 | 0.0687 |
| attn_full:arruda_laplace | 0.049 | 0.0562 | 0.056 | 0.0771 | 0.0575 | 0.0732 |
| bayesflow_arruda | 0.014 | 0.02 | 0.0187 | 0.0345 | 0.0273 | 0.0399 |
| bayesflow_arruda_diffprior | 0.016 | 0.0202 | 0.0215 | 0.0393 | 0.0279 | 0.0391 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.019 | 0.0268 | 0.0215 | 0.0336 | 0.0297 | 0.0372 |
| legacy_w1.5:gauss_jacobian_FD | 0.018 | 0.0264 | 0.0219 | 0.0509 | 0.0358 | 0.0485 |
| legacy_w1.5:gauss_hierarchical | 0.019 | 0.0234 | 0.0226 | 0.0475 | 0.0335 | 0.0495 |
| attn_hier_tsm:gauss_jacobian_FD | 0.018 | 0.0264 | 0.0247 | 0.0382 | 0.0329 | 0.0419 |
| attn_hier_tsm:arruda_direct | 0.051 | 0.0584 | 0.0569 | 0.0772 | 0.0639 | 0.0775 |

**local |mean err|/sd** (`local_mean_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0328 | 0.0351 | 0.0405 | 0.0354 | 0.0335 | 0.0321 |
| uncorrected | 0.0424 | 0.0478 | 0.0427 | 0.0872 | 0.122 | 0.173 |
| gauss | 0.0515 | 0.0812 | 0.0773 | 0.105 | 0.0664 | 0.171 |
| gauss_hierarchical | 0.0745 | 0.0563 | 0.0422 | 0.0728 | 0.0655 | 0.0937 |
| gauss_jacobian | 0.0424 | 0.0561 | 0.0424 | 0.0642 | 0.0477 | 0.0804 |
| gauss_jacobian_FD | 0.0424 | 0.0555 | 0.0432 | 0.0643 | 0.0467 | 0.0877 |
| langevin_fnpe | 0.0753 | 0.0489 | 0.0536 | 0.0629 | 0.0732 | 0.0656 |
| attn_hier:arruda_direct | 0.0409 | 0.0505 | 0.0524 | 0.107 | 0.104 | 0.16 |
| attn_hier:arruda_gauss_direct | 0.0409 | 0.0476 | 0.047 | 0.105 | 0.0773 | 0.123 |
| attn_hier:gauss_jacobian_FD | 0.0446 | 0.0566 | 0.046 | 0.0644 | 0.0726 | 0.113 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0468 | 0.0527 | 0.0565 | 0.071 | 0.0731 | 0.0951 |
| attn_full:gauss_hierarchical | 0.0671 | 0.0472 | 0.0386 | 0.0577 | 0.082 | 0.0711 |
| attn_full:gauss_jacobian_FD | 0.0375 | 0.0553 | 0.0533 | 0.0755 | 0.109 | 0.175 |
| attn_full:arruda_schur | 0.0466 | 0.103 | 0.157 | 0.242 | 0.266 | 0.346 |
| attn_full:arruda_laplace | 0.0393 | 0.0388 | 0.045 | 0.227 | 0.15 | 0.216 |
| bayesflow_arruda | 0.038 | 0.0447 | 0.0433 | 0.067 | 0.0958 | 0.0695 |
| bayesflow_arruda_diffprior | 0.0521 | 0.0507 | 0.0672 | 0.0905 | 0.096 | 0.146 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0297 | 0.0624 | 0.0401 | 0.0728 | 0.0722 | 0.0911 |
| legacy_w1.5:gauss_jacobian_FD | 0.0468 | 0.0488 | 0.0454 | 0.0514 | 0.0503 | 0.0545 |
| legacy_w1.5:gauss_hierarchical | 0.0677 | 0.0454 | 0.0373 | 0.0557 | 0.0559 | 0.0529 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0421 | 0.0541 | 0.0446 | 0.0658 | 0.0642 | 0.08 |
| attn_hier_tsm:arruda_direct | 0.0372 | 0.0475 | 0.0486 | 0.133 | 0.216 | 0.173 |

**local width ratio** (`local_width`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.999 | 1 | 0.999 | 1 | 1 | 1 |
| uncorrected | 1 | 0.997 | 1 | 0.997 | 0.988 | 0.983 |
| gauss | 1.95 | 2.31 | 2.11 | 1.69 | 1.61 | 1.45 |
| gauss_hierarchical | 1 | 0.998 | 1 | 0.998 | 0.996 | 0.994 |
| gauss_jacobian | 1 | 0.997 | 0.999 | 0.997 | 0.997 | 0.993 |
| gauss_jacobian_FD | 1 | 0.998 | 0.999 | 0.997 | 0.996 | 0.992 |
| langevin_fnpe | 1 | 0.995 | 1 | 0.999 | 0.999 | 0.999 |
| attn_hier:arruda_direct | 1.01 | 1 | 1 | 0.997 | 0.999 | 0.994 |
| attn_hier:arruda_gauss_direct | 1.01 | 1 | 1 | 0.999 | 1 | 0.997 |
| attn_hier:gauss_jacobian_FD | 1 | 1 | 1 | 0.999 | 1 | 0.997 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 1 | 0.999 | 1 | 0.999 | 1 | 1 |
| attn_full:gauss_hierarchical | 1.01 | 1 | 1 | 1 | 1 | 1 |
| attn_full:gauss_jacobian_FD | 1.01 | 1 | 1 | 1 | 0.999 | 0.996 |
| attn_full:arruda_schur | 1 | 1 | 0.995 | 0.994 | 0.974 | 0.966 |
| attn_full:arruda_laplace | 1.01 | 1 | 1 | 0.977 | 0.998 | 0.987 |
| bayesflow_arruda | 1 | 1 | 1 | 1 | 0.996 | 0.996 |
| bayesflow_arruda_diffprior | 1 | 1 | 1 | 0.998 | 1 | 0.991 |
| attn_hier_w1.5:gauss_jacobian_FD | 1 | 0.997 | 1 | 0.997 | 0.999 | 0.995 |
| legacy_w1.5:gauss_jacobian_FD | 1 | 0.996 | 0.997 | 0.996 | 0.996 | 0.995 |
| legacy_w1.5:gauss_hierarchical | 1 | 0.995 | 0.998 | 0.995 | 0.995 | 0.996 |
| attn_hier_tsm:gauss_jacobian_FD | 1 | 0.999 | 1 | 1 | 1 | 0.995 |
| attn_hier_tsm:arruda_direct | 1.01 | 1 | 1 | 0.993 | 0.991 | 0.987 |

## Q2 population mean μ_g

**|mean err|/sd** (`mu_mean_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0455 | 0.0379 | 0.0552 | 0.0322 | 0.0374 | 0.0472 |
| uncorrected | 0.0462 | 0.108 | 0.11 | 0.415 | 0.48 | 0.8 |
| gauss | 0.0487 | 0.0604 | 0.0877 | 0.296 | 0.69 | 0.936 |
| gauss_hierarchical | 0.0414 | 0.105 | 0.0754 | 0.337 | 0.323 | 0.787 |
| gauss_jacobian | 0.0462 | 0.161 | 0.0697 | 0.322 | 0.259 | 0.515 |
| gauss_jacobian_FD | 0.0462 | 0.161 | 0.0698 | 0.317 | 0.265 | 0.515 |
| langevin_fnpe | 0.0412 | 0.207 | 0.184 | 0.222 | 0.408 | 0.318 |
| attn_hier:arruda_direct | 0.0564 | 0.0902 | 0.197 | 0.222 | 0.368 | 0.695 |
| attn_hier:arruda_gauss_direct | 0.0564 | 0.128 | 0.1 | 0.177 | 0.198 | 0.516 |
| attn_hier:gauss_jacobian_FD | 0.0564 | 0.129 | 0.101 | 0.206 | 0.181 | 0.539 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0591 | 0.11 | 0.109 | 0.205 | 0.284 | 0.537 |
| attn_full:gauss_hierarchical | 0.0436 | 0.0828 | 0.0688 | 0.325 | 0.311 | 0.604 |
| attn_full:gauss_jacobian_FD | 0.0575 | 0.169 | 0.0507 | 0.404 | 0.412 | 0.614 |
| attn_full:arruda_schur | 0.111 | 0.483 | 0.887 | 2.74 | 3.01 | 3.97 |
| attn_full:arruda_laplace | 0.0582 | 0.105 | 0.0833 | 0.374 | 0.658 | 0.836 |
| bayesflow_arruda | 0.045 | 0.418 | 0.255 | 0.159 | 0.257 | 0.327 |
| bayesflow_arruda_diffprior | 0.0274 | 0.0621 | 0.19 | 0.399 | 0.528 | 0.678 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0471 | 0.188 | 0.166 | 0.138 | 0.239 | 0.59 |
| legacy_w1.5:gauss_jacobian_FD | 0.0481 | 0.191 | 0.0857 | 0.215 | 0.185 | 0.366 |
| legacy_w1.5:gauss_hierarchical | 0.054 | 0.127 | 0.0555 | 0.201 | 0.141 | 0.276 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0508 | 0.164 | 0.0838 | 0.278 | 0.311 | 0.433 |
| attn_hier_tsm:arruda_direct | 0.0508 | 0.115 | 0.0818 | 0.262 | 0.585 | 0.857 |

**width ratio** (`mu_width`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 1.01 | 0.996 | 1.01 | 1.01 | 0.991 | 0.992 |
| uncorrected | 1.01 | 1.09 | 1.09 | 0.869 | 1.04 | 0.834 |
| gauss | 1 | 1.11 | 1.28 | 1.59 | 1.98 | 1.88 |
| gauss_hierarchical | 1 | 1.09 | 1.22 | 1.13 | 1.13 | 1.09 |
| gauss_jacobian | 1.01 | 1.11 | 1.21 | 1.1 | 1.29 | 1.25 |
| gauss_jacobian_FD | 1.01 | 1.11 | 1.2 | 1.1 | 1.33 | 1.24 |
| langevin_fnpe | 1.02 | 1.07 | 1.09 | 1.12 | 1.01 | 0.745 |
| attn_hier:arruda_direct | 1.01 | 1.07 | 1.05 | 0.992 | 0.992 | 1.03 |
| attn_hier:arruda_gauss_direct | 1.01 | 1.1 | 1.11 | 1 | 1.09 | 1.34 |
| attn_hier:gauss_jacobian_FD | 1.01 | 1.1 | 1.11 | 1.01 | 1.07 | 1.29 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 1.01 | 1.09 | 1.11 | 0.995 | 1.13 | 1.33 |
| attn_full:gauss_hierarchical | 1 | 1.07 | 1.17 | 0.987 | 1.02 | 0.932 |
| attn_full:gauss_jacobian_FD | 1.01 | 1.09 | 1.14 | 0.932 | 1 | 0.923 |
| attn_full:arruda_schur | 1.13 | 1.51 | 1.79 | 2.64 | 4.09 | 3.87 |
| attn_full:arruda_laplace | 1.02 | 1.14 | 1.34 | 1.35 | 2.12 | 2 |
| bayesflow_arruda | 1.01 | 0.535 | 0.762 | 0.747 | 0.529 | 0.741 |
| bayesflow_arruda_diffprior | 1.01 | 1.1 | 1.09 | 0.901 | 1.1 | 0.931 |
| attn_hier_w1.5:gauss_jacobian_FD | 1.01 | 1.16 | 1.27 | 1.14 | 1.18 | 1.15 |
| legacy_w1.5:gauss_jacobian_FD | 1.02 | 1.13 | 1.26 | 1.05 | 1 | 1.15 |
| legacy_w1.5:gauss_hierarchical | 1 | 1.11 | 1.28 | 1.25 | 1.31 | 1.3 |
| attn_hier_tsm:gauss_jacobian_FD | 1.01 | 1.12 | 1.13 | 0.948 | 0.941 | 1.03 |
| attn_hier_tsm:arruda_direct | 1.01 | 1.11 | 1.06 | 0.853 | 0.726 | 0.487 |

**W1/sd** (`mu_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0581 | 0.0471 | 0.0688 | 0.0431 | 0.0544 | 0.0662 |
| uncorrected | 0.062 | 0.179 | 0.252 | 0.65 | 0.611 | 1.16 |
| gauss | 0.0625 | 0.143 | 0.344 | 0.772 | 1.17 | 1.08 |
| gauss_hierarchical | 0.0611 | 0.173 | 0.268 | 0.509 | 0.473 | 0.942 |
| gauss_jacobian | 0.062 | 0.211 | 0.267 | 0.479 | 0.498 | 0.754 |
| gauss_jacobian_FD | 0.062 | 0.211 | 0.264 | 0.467 | 0.497 | 0.76 |
| langevin_fnpe | 0.0576 | 0.268 | 0.308 | 0.557 | 0.478 | 0.606 |
| attn_hier:arruda_direct | 0.0675 | 0.157 | 0.256 | 0.417 | 0.503 | 0.855 |
| attn_hier:arruda_gauss_direct | 0.0675 | 0.194 | 0.239 | 0.376 | 0.381 | 0.769 |
| attn_hier:gauss_jacobian_FD | 0.0675 | 0.194 | 0.239 | 0.402 | 0.379 | 0.78 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0711 | 0.173 | 0.25 | 0.523 | 0.523 | 0.786 |
| attn_full:gauss_hierarchical | 0.0624 | 0.141 | 0.243 | 0.618 | 0.448 | 0.72 |
| attn_full:gauss_jacobian_FD | 0.0693 | 0.194 | 0.242 | 0.548 | 0.472 | 0.687 |
| attn_full:arruda_schur | 0.183 | 0.685 | 1.18 | 3.04 | 3.55 | 4.46 |
| attn_full:arruda_laplace | 0.0717 | 0.188 | 0.373 | 0.782 | 1.15 | 1.55 |
| bayesflow_arruda | 0.0589 | 0.701 | 0.435 | 0.59 | 0.583 | 0.548 |
| bayesflow_arruda_diffprior | 0.0479 | 0.143 | 0.244 | 0.517 | 0.542 | 0.71 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0638 | 0.216 | 0.256 | 0.291 | 0.334 | 0.688 |
| legacy_w1.5:gauss_jacobian_FD | 0.0646 | 0.203 | 0.263 | 0.347 | 0.361 | 0.521 |
| legacy_w1.5:gauss_hierarchical | 0.0685 | 0.158 | 0.261 | 0.414 | 0.397 | 0.581 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0668 | 0.196 | 0.218 | 0.519 | 0.479 | 0.67 |
| attn_hier_tsm:arruda_direct | 0.0668 | 0.175 | 0.21 | 0.648 | 0.729 | 0.959 |

**90% coverage** (`mu_cover90`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.95 | 0.95 | 1 | 0.8 | 0.9 | 0.85 |
| uncorrected | 0.95 | 0.9 | 1 | 0.65 | 0.75 | 0.5 |
| gauss | 0.95 | 0.95 | 1 | 1 | 0.95 | 0.85 |
| gauss_hierarchical | 0.9 | 0.9 | 1 | 0.85 | 0.9 | 0.75 |
| gauss_jacobian | 0.95 | 0.9 | 1 | 0.8 | 0.9 | 0.85 |
| gauss_jacobian_FD | 0.95 | 0.9 | 1 | 0.8 | 0.9 | 0.85 |
| langevin_fnpe | 0.95 | 0.9 | 0.95 | 0.75 | 0.7 | 0.35 |
| attn_hier:arruda_direct | 0.95 | 1 | 1 | 0.6 | 0.75 | 0.7 |
| attn_hier:arruda_gauss_direct | 0.95 | 0.95 | 1 | 0.75 | 0.85 | 0.8 |
| attn_hier:gauss_jacobian_FD | 0.95 | 0.95 | 1 | 0.7 | 0.85 | 0.8 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.95 | 0.95 | 1 | 0.75 | 0.8 | 0.8 |
| attn_full:gauss_hierarchical | 0.9 | 0.95 | 1 | 0.85 | 0.75 | 0.65 |
| attn_full:gauss_jacobian_FD | 0.95 | 0.9 | 1 | 0.85 | 0.7 | 0.65 |
| attn_full:arruda_schur | 0.95 | 1 | 1 | 1 | 1 | 0.95 |
| attn_full:arruda_laplace | 0.95 | 0.95 | 1 | 0.85 | 0.9 | 0.8 |
| bayesflow_arruda | 0.95 | 0.4 | 0.7 | 0.5 | 0.4 | 0.4 |
| bayesflow_arruda_diffprior | 0.95 | 0.95 | 1 | 0.7 | 0.8 | 0.7 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.95 | 0.9 | 1 | 0.9 | 0.8 | 0.8 |
| legacy_w1.5:gauss_jacobian_FD | 0.95 | 0.9 | 1 | 0.85 | 0.75 | 0.75 |
| legacy_w1.5:gauss_hierarchical | 0.9 | 0.9 | 1 | 0.9 | 0.9 | 0.75 |
| attn_hier_tsm:gauss_jacobian_FD | 0.95 | 0.95 | 1 | 0.65 | 0.65 | 0.55 |
| attn_hier_tsm:arruda_direct | 0.95 | 0.95 | 1 | 0.5 | 0.45 | 0.25 |

## Q3 population spread σ_g

**|mean err|/sd** (`sigma_mean_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0435 | 0.0299 | 0.041 | 0.0356 | 0.0323 | 0.0361 |
| uncorrected | 0.0354 | 0.517 | 0.644 | 3.59 | 0.635 | 1.79 |
| gauss | 0.0256 | 0.607 | 0.687 | 1.32 | 1.45 | 2.81 |
| gauss_hierarchical | 0.0408 | 0.454 | 0.683 | 1.51 | 0.966 | 1.51 |
| gauss_jacobian | 0.0354 | 0.484 | 0.724 | 2.46 | 1 | 2.76 |
| gauss_jacobian_FD | 0.0354 | 0.478 | 0.719 | 2.44 | 0.979 | 2.69 |
| langevin_fnpe | 0.0468 | 0.576 | 0.683 | 4.43 | 2.44 | 2.88 |
| attn_hier:arruda_direct | 0.0373 | 0.492 | 0.601 | 1.6 | 0.999 | 1.56 |
| attn_hier:arruda_gauss_direct | 0.0373 | 0.452 | 0.613 | 1.09 | 0.705 | 1.34 |
| attn_hier:gauss_jacobian_FD | 0.0373 | 0.451 | 0.614 | 1.11 | 0.728 | 1.22 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0367 | 0.347 | 0.587 | 1.1 | 0.655 | 1.22 |
| attn_full:gauss_hierarchical | 0.0381 | 0.443 | 0.716 | 2.09 | 1.2 | 1.79 |
| attn_full:gauss_jacobian_FD | 0.0405 | 0.479 | 0.758 | 2.16 | 1.43 | 2.7 |
| attn_full:arruda_schur | 0.159 | 0.34 | 0.3 | 4.66 | 3.44 | 2.14 |
| attn_full:arruda_laplace | 0.0432 | 0.394 | 0.417 | 2.74 | 3.18 | 3.97 |
| bayesflow_arruda | 0.0424 | 0.844 | 0.745 | 3.34 | 1.15 | 1.85 |
| bayesflow_arruda_diffprior | 0.0393 | 0.458 | 0.637 | 1.53 | 1.01 | 0.958 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0465 | 0.42 | 0.651 | 1.3 | 0.672 | 1.01 |
| legacy_w1.5:gauss_jacobian_FD | 0.0513 | 0.457 | 0.751 | 1.56 | 0.999 | 1.46 |
| legacy_w1.5:gauss_hierarchical | 0.0411 | 0.449 | 0.746 | 1.6 | 1.17 | 1.33 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0413 | 0.424 | 0.585 | 1.29 | 1.4 | 1.44 |
| attn_hier_tsm:arruda_direct | 0.0413 | 0.48 | 0.553 | 1.93 | 1.32 | 1.26 |

**width ratio** (`sigma_width`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 1 | 1 | 1.01 | 1.01 | 0.987 | 1 |
| uncorrected | 1 | 1.07 | 1.12 | 2.85 | 1.85 | 1.7 |
| gauss | 0.995 | 1.08 | 1.17 | 4.47 | 3.45 | 3.86 |
| gauss_hierarchical | 0.992 | 1.07 | 1.15 | 3.16 | 2.7 | 2.19 |
| gauss_jacobian | 1 | 1.04 | 1.1 | 2.98 | 2.4 | 2.92 |
| gauss_jacobian_FD | 1 | 1.04 | 1.1 | 2.94 | 2.36 | 2.91 |
| langevin_fnpe | 1 | 1 | 1.02 | 4.95 | 2.83 | 2.31 |
| attn_hier:arruda_direct | 1.01 | 1.08 | 1.16 | 3.38 | 1.89 | 2.54 |
| attn_hier:arruda_gauss_direct | 1.01 | 1.07 | 1.15 | 3.18 | 2.21 | 3.08 |
| attn_hier:gauss_jacobian_FD | 1.01 | 1.07 | 1.15 | 3.08 | 2.2 | 2.94 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 1 | 1.07 | 1.14 | 3.06 | 2.2 | 3.29 |
| attn_full:gauss_hierarchical | 0.995 | 1.06 | 1.14 | 2.5 | 1.74 | 1.74 |
| attn_full:gauss_jacobian_FD | 1.01 | 1.04 | 1.09 | 2.67 | 1.86 | 2.59 |
| attn_full:arruda_schur | 1.09 | 1.18 | 1.18 | 9.23 | 6.66 | 7.27 |
| attn_full:arruda_laplace | 1.01 | 1.14 | 1.21 | 4.05 | 4.13 | 5.71 |
| bayesflow_arruda | 0.994 | 0.466 | 0.748 | 3.11 | 1.67 | 2.74 |
| bayesflow_arruda_diffprior | 1.01 | 1.08 | 1.18 | 3.67 | 2.7 | 2.81 |
| attn_hier_w1.5:gauss_jacobian_FD | 1.01 | 1.13 | 1.18 | 2.97 | 2.49 | 1.83 |
| legacy_w1.5:gauss_jacobian_FD | 1.01 | 1.08 | 1.12 | 2.8 | 1.8 | 1.23 |
| legacy_w1.5:gauss_hierarchical | 0.994 | 1.09 | 1.2 | 4.1 | 3.01 | 2.95 |
| attn_hier_tsm:gauss_jacobian_FD | 1.01 | 1.09 | 1.17 | 2.36 | 1.68 | 1.41 |
| attn_hier_tsm:arruda_direct | 1.01 | 1.1 | 1.19 | 2.35 | 1.05 | 0.921 |

**W1/sd** (`sigma_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0523 | 0.0464 | 0.0505 | 0.0563 | 0.0593 | 0.0609 |
| uncorrected | 0.0576 | 0.52 | 0.645 | 3.59 | 0.872 | 2.06 |
| gauss | 0.0394 | 0.608 | 0.689 | 2.68 | 1.87 | 3.35 |
| gauss_hierarchical | 0.052 | 0.455 | 0.685 | 2.04 | 1.08 | 1.69 |
| gauss_jacobian | 0.0576 | 0.487 | 0.726 | 2.72 | 1.15 | 2.76 |
| gauss_jacobian_FD | 0.0576 | 0.482 | 0.721 | 2.7 | 1.12 | 2.71 |
| langevin_fnpe | 0.0528 | 0.578 | 0.693 | 4.5 | 2.42 | 2.88 |
| attn_hier:arruda_direct | 0.0573 | 0.493 | 0.621 | 2.58 | 1.5 | 1.88 |
| attn_hier:arruda_gauss_direct | 0.0573 | 0.457 | 0.614 | 1.78 | 1.26 | 1.9 |
| attn_hier:gauss_jacobian_FD | 0.0573 | 0.456 | 0.615 | 1.67 | 1.25 | 1.75 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0525 | 0.358 | 0.588 | 1.56 | 1.26 | 1.95 |
| attn_full:gauss_hierarchical | 0.0522 | 0.443 | 0.718 | 2.1 | 1.2 | 1.79 |
| attn_full:gauss_jacobian_FD | 0.0579 | 0.479 | 0.76 | 2.18 | 1.45 | 2.71 |
| attn_full:arruda_schur | 0.172 | 0.464 | 0.434 | 5.97 | 5.01 | 5.22 |
| attn_full:arruda_laplace | 0.0607 | 0.398 | 0.421 | 2.98 | 3.54 | 4.64 |
| bayesflow_arruda | 0.0588 | 0.968 | 0.823 | 3.32 | 1.31 | 2.45 |
| bayesflow_arruda_diffprior | 0.0522 | 0.461 | 0.638 | 1.97 | 1.37 | 1.34 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0621 | 0.421 | 0.653 | 1.4 | 1.13 | 1.33 |
| legacy_w1.5:gauss_jacobian_FD | 0.0608 | 0.458 | 0.753 | 1.55 | 0.997 | 1.47 |
| legacy_w1.5:gauss_hierarchical | 0.0553 | 0.45 | 0.748 | 1.62 | 1.23 | 1.38 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0563 | 0.428 | 0.586 | 1.46 | 1.41 | 1.49 |
| attn_hier_tsm:arruda_direct | 0.0563 | 0.482 | 0.564 | 2 | 1.32 | 1.35 |

**90% coverage** (`sigma_cover90`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.65 | 0.9 | 1 | 1 | 1 | 0.95 |
| uncorrected | 0.65 | 0.7 | 0.7 | 0.7 | 0.65 | 0.4 |
| gauss | 0.65 | 0.75 | 0.9 | 0.9 | 0.95 | 0.8 |
| gauss_hierarchical | 0.7 | 0.75 | 0.8 | 0.85 | 0.75 | 0.5 |
| gauss_jacobian | 0.65 | 0.7 | 0.65 | 0.55 | 0.75 | 0.55 |
| gauss_jacobian_FD | 0.65 | 0.7 | 0.65 | 0.55 | 0.75 | 0.55 |
| langevin_fnpe | 0.65 | 0.55 | 0.5 | 0.4 | 0.4 | 0.45 |
| attn_hier:arruda_direct | 0.65 | 0.75 | 0.85 | 0.9 | 0.8 | 0.7 |
| attn_hier:arruda_gauss_direct | 0.65 | 0.75 | 0.75 | 0.9 | 1 | 0.9 |
| attn_hier:gauss_jacobian_FD | 0.65 | 0.75 | 0.75 | 0.95 | 1 | 0.9 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.65 | 0.75 | 0.85 | 0.9 | 0.95 | 0.9 |
| attn_full:gauss_hierarchical | 0.65 | 0.7 | 0.65 | 0.6 | 0.5 | 0.45 |
| attn_full:gauss_jacobian_FD | 0.65 | 0.7 | 0.6 | 0.5 | 0.5 | 0.65 |
| attn_full:arruda_schur | 0.9 | 1 | 1 | 1 | 1 | 1 |
| attn_full:arruda_laplace | 0.65 | 0.85 | 1 | 0.9 | 0.95 | 0.95 |
| bayesflow_arruda | 0.65 | 0 | 0.25 | 0.35 | 0.5 | 0.7 |
| bayesflow_arruda_diffprior | 0.7 | 0.85 | 0.8 | 1 | 0.95 | 0.85 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.65 | 0.75 | 0.8 | 0.9 | 0.85 | 0.75 |
| legacy_w1.5:gauss_jacobian_FD | 0.65 | 0.7 | 0.7 | 0.8 | 0.75 | 0.6 |
| legacy_w1.5:gauss_hierarchical | 0.65 | 0.75 | 0.75 | 0.8 | 0.65 | 0.6 |
| attn_hier_tsm:gauss_jacobian_FD | 0.65 | 0.75 | 0.95 | 0.85 | 0.65 | 0.55 |
| attn_hier_tsm:arruda_direct | 0.65 | 0.8 | 0.95 | 0.85 | 0.55 | 0.4 |

**joint global C2ST** (`global_c2st`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.502 | 0.51 | 0.489 | 0.508 | 0.491 | 0.49 |
| uncorrected | 0.499 | 0.652 | 0.812 | 0.972 | 0.951 | 0.992 |
| gauss | 0.503 | 0.696 | 0.839 | 0.947 | 0.948 | 0.982 |
| gauss_hierarchical | 0.497 | 0.649 | 0.796 | 0.948 | 0.869 | 0.978 |
| gauss_jacobian | 0.499 | 0.643 | 0.824 | 0.935 | 0.871 | 0.964 |
| gauss_jacobian_FD | 0.499 | 0.641 | 0.822 | 0.936 | 0.865 | 0.956 |
| langevin_fnpe | 0.506 | 0.688 | 0.85 | 0.94 | 0.946 | 0.99 |
| attn_hier:arruda_direct | 0.502 | 0.667 | 0.804 | 0.979 | 0.987 | 0.997 |
| attn_hier:arruda_gauss_direct | 0.502 | 0.649 | 0.801 | 0.965 | 0.985 | 0.995 |
| attn_hier:gauss_jacobian_FD | 0.502 | 0.653 | 0.801 | 0.968 | 0.985 | 0.994 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.502 | 0.645 | 0.792 | 0.966 | 0.985 | 0.994 |
| attn_full:gauss_hierarchical | 0.485 | 0.624 | 0.818 | 0.927 | 0.935 | 0.988 |
| attn_full:gauss_jacobian_FD | 0.499 | 0.621 | 0.828 | 0.895 | 0.883 | 0.968 |
| attn_full:arruda_schur | 0.574 | 0.794 | 0.935 | 0.983 | 0.988 | 0.991 |
| attn_full:arruda_laplace | 0.5 | 0.604 | 0.778 | 0.913 | 0.945 | 0.978 |
| bayesflow_arruda | 0.503 | 0.94 | 0.927 | 0.96 | 0.911 | 0.957 |
| bayesflow_arruda_diffprior | 0.49 | 0.649 | 0.748 | 0.864 | 0.885 | 0.96 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.501 | 0.591 | 0.74 | 0.887 | 0.929 | 0.96 |
| legacy_w1.5:gauss_jacobian_FD | 0.5 | 0.603 | 0.828 | 0.836 | 0.869 | 0.945 |
| legacy_w1.5:gauss_hierarchical | 0.49 | 0.61 | 0.81 | 0.851 | 0.896 | 0.959 |
| attn_hier_tsm:gauss_jacobian_FD | 0.498 | 0.631 | 0.749 | 0.939 | 0.961 | 0.99 |
| attn_hier_tsm:arruda_direct | 0.498 | 0.639 | 0.731 | 0.937 | 0.988 | 0.994 |

## Q2+Q3 globals in mirror-adapted coordinates

**P(u>0) (exact 0.5)** (`mode_balance`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.496 | 0.495 | 0.493 | 0.498 | 0.485 | 0.496 |
| uncorrected | 0.505 | 0.505 | 0.514 | 0.521 | 0.632 | 0.639 |
| gauss | 0.505 | 0.502 | 0.515 | 0.544 | 0.576 | 0.642 |
| gauss_hierarchical | 0.5 | 0.491 | 0.524 | 0.522 | 0.521 | 0.526 |
| gauss_jacobian | 0.505 | 0.506 | 0.516 | 0.522 | 0.519 | 0.568 |
| gauss_jacobian_FD | 0.505 | 0.511 | 0.514 | 0.526 | 0.52 | 0.556 |
| langevin_fnpe | 0.498 | 0.494 | 0.517 | 0.52 | 0.511 | 0.554 |
| attn_hier:arruda_direct | 0.501 | 0.529 | 0.518 | 0.525 | 0.674 | 0.647 |
| attn_hier:arruda_gauss_direct | 0.501 | 0.531 | 0.525 | 0.557 | 0.615 | 0.691 |
| attn_hier:gauss_jacobian_FD | 0.501 | 0.532 | 0.526 | 0.546 | 0.599 | 0.716 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.51 | 0.531 | 0.515 | 0.565 | 0.65 | 0.718 |
| attn_full:gauss_hierarchical | 0.497 | 0.505 | 0.509 | 0.487 | 0.564 | 0.461 |
| attn_full:gauss_jacobian_FD | 0.504 | 0.5 | 0.506 | 0.493 | 0.517 | 0.457 |
| attn_full:arruda_schur | 0.532 | 0.515 | 0.515 | 0.527 | 0.535 | 0.515 |
| attn_full:arruda_laplace | 0.498 | 0.524 | 0.536 | 0.445 | 0.329 | 0.133 |
| bayesflow_arruda | 0.513 | 0.507 | 0.518 | 0.527 | 0.582 | 0.577 |
| bayesflow_arruda_diffprior | 0.506 | 0.479 | 0.503 | 0.329 | 0.34 | 0.224 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.502 | 0.486 | 0.496 | 0.492 | 0.496 | 0.559 |
| legacy_w1.5:gauss_jacobian_FD | 0.507 | 0.49 | 0.528 | 0.524 | 0.558 | 0.559 |
| legacy_w1.5:gauss_hierarchical | 0.494 | 0.489 | 0.516 | 0.476 | 0.488 | 0.503 |
| attn_hier_tsm:gauss_jacobian_FD | 0.493 | 0.503 | 0.505 | 0.545 | 0.55 | 0.521 |
| attn_hier_tsm:arruda_direct | 0.493 | 0.505 | 0.512 | 0.527 | 0.555 | 0.532 |

**|P(u>0) - ref|** (`mode_balance_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0343 | 0.0158 | 0.0168 | 0.024 | 0.021 | 0.018 |
| uncorrected | 0.0197 | 0.0285 | 0.023 | 0.0848 | 0.138 | 0.285 |
| gauss | 0.0278 | 0.0255 | 0.0175 | 0.109 | 0.069 | 0.143 |
| gauss_hierarchical | 0.0157 | 0.0277 | 0.0212 | 0.0833 | 0.049 | 0.037 |
| gauss_jacobian | 0.0197 | 0.0222 | 0.0225 | 0.0503 | 0.0373 | 0.0712 |
| gauss_jacobian_FD | 0.0197 | 0.0272 | 0.0255 | 0.0393 | 0.0397 | 0.0713 |
| langevin_fnpe | 0.009 | 0.0162 | 0.0255 | 0.037 | 0.0823 | 0.114 |
| attn_hier:arruda_direct | 0.0202 | 0.036 | 0.0363 | 0.0428 | 0.171 | 0.2 |
| attn_hier:arruda_gauss_direct | 0.0202 | 0.0425 | 0.0418 | 0.063 | 0.107 | 0.184 |
| attn_hier:gauss_jacobian_FD | 0.0202 | 0.0425 | 0.0448 | 0.0515 | 0.102 | 0.213 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0155 | 0.038 | 0.0398 | 0.0688 | 0.168 | 0.217 |
| attn_full:gauss_hierarchical | 0.0195 | 0.0215 | 0.0143 | 0.0675 | 0.102 | 0.053 |
| attn_full:gauss_jacobian_FD | 0.0207 | 0.0262 | 0.014 | 0.0313 | 0.0607 | 0.0847 |
| attn_full:arruda_schur | 0.0388 | 0.0253 | 0.0375 | 0.13 | 0.105 | 0.137 |
| attn_full:arruda_laplace | 0.0227 | 0.0198 | 0.0305 | 0.159 | 0.177 | 0.369 |
| bayesflow_arruda | 0.0238 | 0.0103 | 0.0222 | 0.0323 | 0.104 | 0.109 |
| bayesflow_arruda_diffprior | 0.0153 | 0.02 | 0.0225 | 0.16 | 0.159 | 0.279 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0185 | 0.032 | 0.0205 | 0.032 | 0.0182 | 0.075 |
| legacy_w1.5:gauss_jacobian_FD | 0.0195 | 0.023 | 0.0387 | 0.0283 | 0.0603 | 0.086 |
| legacy_w1.5:gauss_hierarchical | 0.0157 | 0.024 | 0.035 | 0.026 | 0.045 | 0.032 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0267 | 0.039 | 0.0165 | 0.104 | 0.0685 | 0.193 |
| attn_hier_tsm:arruda_direct | 0.0267 | 0.037 | 0.0125 | 0.158 | 0.232 | 0.465 |

**u=(μ0+μ1)/√2 W1/sd (bimodal)** (`sym_u_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.074 | 0.0659 | 0.057 | 0.0419 | 0.0538 | 0.0451 |
| uncorrected | 0.0629 | 0.167 | 0.204 | 0.473 | 0.591 | 0.782 |
| gauss | 0.0723 | 0.14 | 0.261 | 0.537 | 0.727 | 0.899 |
| gauss_hierarchical | 0.0612 | 0.161 | 0.255 | 0.418 | 0.395 | 0.532 |
| gauss_jacobian | 0.0629 | 0.193 | 0.276 | 0.414 | 0.398 | 0.507 |
| gauss_jacobian_FD | 0.0629 | 0.192 | 0.272 | 0.413 | 0.398 | 0.512 |
| langevin_fnpe | 0.0528 | 0.193 | 0.259 | 0.49 | 0.406 | 0.567 |
| attn_hier:arruda_direct | 0.0648 | 0.126 | 0.197 | 0.433 | 0.495 | 0.652 |
| attn_hier:arruda_gauss_direct | 0.0648 | 0.159 | 0.222 | 0.427 | 0.345 | 0.576 |
| attn_hier:gauss_jacobian_FD | 0.0648 | 0.159 | 0.223 | 0.436 | 0.345 | 0.587 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.068 | 0.148 | 0.219 | 0.507 | 0.456 | 0.664 |
| attn_full:gauss_hierarchical | 0.0596 | 0.152 | 0.232 | 0.436 | 0.439 | 0.449 |
| attn_full:gauss_jacobian_FD | 0.0633 | 0.158 | 0.237 | 0.463 | 0.31 | 0.494 |
| attn_full:arruda_schur | 0.202 | 0.388 | 0.667 | 0.974 | 1.81 | 2.12 |
| attn_full:arruda_laplace | 0.0807 | 0.173 | 0.342 | 0.518 | 0.95 | 1.06 |
| bayesflow_arruda | 0.0704 | 0.445 | 0.3 | 0.537 | 0.548 | 0.49 |
| bayesflow_arruda_diffprior | 0.0547 | 0.154 | 0.134 | 0.443 | 0.588 | 0.637 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0594 | 0.167 | 0.221 | 0.289 | 0.257 | 0.497 |
| legacy_w1.5:gauss_jacobian_FD | 0.0571 | 0.183 | 0.272 | 0.35 | 0.283 | 0.402 |
| legacy_w1.5:gauss_hierarchical | 0.0581 | 0.153 | 0.274 | 0.401 | 0.356 | 0.517 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0702 | 0.192 | 0.209 | 0.474 | 0.387 | 0.659 |
| attn_hier_tsm:arruda_direct | 0.0702 | 0.181 | 0.145 | 0.584 | 0.68 | 1.02 |

**w=σ0-σ1 W1/sd (bimodal)** (`sym_w_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0458 | 0.0469 | 0.049 | 0.0604 | 0.0666 | 0.0526 |
| uncorrected | 0.0536 | 0.137 | 0.338 | 1.68 | 0.766 | 1.04 |
| gauss | 0.0497 | 0.129 | 0.326 | 1.86 | 1.52 | 2.1 |
| gauss_hierarchical | 0.0522 | 0.11 | 0.358 | 1.23 | 0.996 | 0.808 |
| gauss_jacobian | 0.0536 | 0.148 | 0.333 | 1.26 | 0.767 | 0.971 |
| gauss_jacobian_FD | 0.0536 | 0.147 | 0.332 | 1.23 | 0.747 | 1 |
| langevin_fnpe | 0.0482 | 0.153 | 0.408 | 2.86 | 1.05 | 0.829 |
| attn_hier:arruda_direct | 0.0584 | 0.136 | 0.307 | 2.01 | 1.03 | 1.74 |
| attn_hier:arruda_gauss_direct | 0.0584 | 0.136 | 0.297 | 1.48 | 1.01 | 1.83 |
| attn_hier:gauss_jacobian_FD | 0.0584 | 0.137 | 0.298 | 1.39 | 0.96 | 1.69 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.056 | 0.153 | 0.299 | 1.39 | 1.15 | 1.91 |
| attn_full:gauss_hierarchical | 0.062 | 0.104 | 0.371 | 1.23 | 0.693 | 0.738 |
| attn_full:gauss_jacobian_FD | 0.058 | 0.141 | 0.35 | 1.48 | 1.38 | 1.87 |
| attn_full:arruda_schur | 0.0893 | 0.166 | 0.325 | 4.94 | 3.28 | 2.76 |
| attn_full:arruda_laplace | 0.066 | 0.132 | 0.271 | 3.11 | 3.8 | 3.61 |
| bayesflow_arruda | 0.0672 | 0.405 | 0.431 | 1.62 | 0.566 | 1.47 |
| bayesflow_arruda_diffprior | 0.0668 | 0.17 | 0.333 | 1.76 | 1.43 | 1.39 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0593 | 0.18 | 0.26 | 1.07 | 1.05 | 0.763 |
| legacy_w1.5:gauss_jacobian_FD | 0.0626 | 0.152 | 0.332 | 0.93 | 0.494 | 0.727 |
| legacy_w1.5:gauss_hierarchical | 0.0634 | 0.115 | 0.357 | 1.21 | 0.953 | 0.977 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0591 | 0.122 | 0.265 | 1.28 | 0.807 | 1.62 |
| attn_hier_tsm:arruda_direct | 0.0591 | 0.132 | 0.264 | 1.64 | 0.852 | 1.44 |

**v=(μ1-μ0)/√2 W1/sd** (`sym_v_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0576 | 0.0536 | 0.062 | 0.0498 | 0.0651 | 0.0711 |
| uncorrected | 0.0643 | 0.315 | 0.432 | 2.44 | 1.37 | 2.59 |
| gauss | 0.0664 | 0.274 | 0.571 | 2.55 | 2.34 | 4.97 |
| gauss_hierarchical | 0.0682 | 0.247 | 0.47 | 1.9 | 1.01 | 2.49 |
| gauss_jacobian | 0.0643 | 0.394 | 0.468 | 1.83 | 0.886 | 2.41 |
| gauss_jacobian_FD | 0.0643 | 0.395 | 0.463 | 1.82 | 0.878 | 2.31 |
| langevin_fnpe | 0.0659 | 0.481 | 0.439 | 1.52 | 1.09 | 1.22 |
| attn_hier:arruda_direct | 0.0796 | 0.248 | 0.586 | 2.14 | 0.945 | 1.89 |
| attn_hier:arruda_gauss_direct | 0.0796 | 0.366 | 0.531 | 1.82 | 0.834 | 1.86 |
| attn_hier:gauss_jacobian_FD | 0.0796 | 0.368 | 0.53 | 1.95 | 0.822 | 1.89 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0824 | 0.297 | 0.523 | 1.95 | 0.772 | 1.65 |
| attn_full:gauss_hierarchical | 0.0738 | 0.213 | 0.486 | 2.05 | 1.54 | 2.27 |
| attn_full:gauss_jacobian_FD | 0.0701 | 0.357 | 0.483 | 2.2 | 0.898 | 2.1 |
| attn_full:arruda_schur | 0.104 | 0.956 | 2.45 | 8.32 | 9.81 | 14.6 |
| attn_full:arruda_laplace | 0.0804 | 0.24 | 0.342 | 2.14 | 2.01 | 3.98 |
| bayesflow_arruda | 0.0621 | 0.869 | 0.578 | 1.14 | 0.871 | 1.49 |
| bayesflow_arruda_diffprior | 0.0511 | 0.25 | 0.433 | 1.36 | 0.901 | 1.98 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0693 | 0.373 | 0.462 | 1.28 | 0.889 | 2.02 |
| legacy_w1.5:gauss_jacobian_FD | 0.0635 | 0.406 | 0.496 | 1.41 | 0.88 | 2.04 |
| legacy_w1.5:gauss_hierarchical | 0.076 | 0.267 | 0.477 | 1.52 | 0.975 | 2.17 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0731 | 0.383 | 0.451 | 1.37 | 0.996 | 1.33 |
| attn_hier_tsm:arruda_direct | 0.0731 | 0.304 | 0.445 | 1.32 | 0.952 | 1.34 |

**s=σ0+σ1 W1/sd** (`sym_s_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0682 | 0.0532 | 0.0614 | 0.0514 | 0.0585 | 0.0658 |
| uncorrected | 0.0543 | 0.784 | 1.07 | 5.05 | 1.45 | 4.26 |
| gauss | 0.0567 | 0.955 | 1.22 | 3.63 | 3.88 | 7.25 |
| gauss_hierarchical | 0.0601 | 0.685 | 1.2 | 3.22 | 1.85 | 4.18 |
| gauss_jacobian | 0.0543 | 0.766 | 1.27 | 4.22 | 2.07 | 5.35 |
| gauss_jacobian_FD | 0.0543 | 0.757 | 1.25 | 4.18 | 2.02 | 5.29 |
| langevin_fnpe | 0.0649 | 0.866 | 1.21 | 7.2 | 5.02 | 7.71 |
| attn_hier:arruda_direct | 0.0509 | 0.735 | 1.01 | 2.75 | 2.23 | 3.71 |
| attn_hier:arruda_gauss_direct | 0.0509 | 0.677 | 1.1 | 2.19 | 1.59 | 3.31 |
| attn_hier:gauss_jacobian_FD | 0.0509 | 0.677 | 1.11 | 2.18 | 1.69 | 3.34 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0546 | 0.574 | 1.03 | 2.33 | 1.84 | 3.68 |
| attn_full:gauss_hierarchical | 0.0599 | 0.649 | 1.25 | 2.83 | 1.93 | 5.48 |
| attn_full:gauss_jacobian_FD | 0.0508 | 0.744 | 1.3 | 3.28 | 2.49 | 6.97 |
| attn_full:arruda_schur | 0.205 | 0.678 | 0.83 | 9.13 | 8.68 | 12.9 |
| attn_full:arruda_laplace | 0.0532 | 0.582 | 0.899 | 4.23 | 5.89 | 13.1 |
| bayesflow_arruda | 0.0608 | 1.25 | 1.33 | 5.46 | 2.21 | 6.12 |
| bayesflow_arruda_diffprior | 0.0374 | 0.737 | 1.13 | 2.38 | 1.78 | 2.41 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0604 | 0.633 | 1.32 | 2.56 | 2.06 | 2.89 |
| legacy_w1.5:gauss_jacobian_FD | 0.0561 | 0.694 | 1.42 | 2.85 | 1.78 | 3.64 |
| legacy_w1.5:gauss_hierarchical | 0.0577 | 0.706 | 1.46 | 2.97 | 2.09 | 3.47 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0493 | 0.66 | 1.04 | 2.38 | 2.18 | 3.49 |
| attn_hier_tsm:arruda_direct | 0.0493 | 0.702 | 0.927 | 2.56 | 2.58 | 2.27 |

**mirror-invariant W1/sd** (`sym_inv_w1`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0522 | 0.0504 | 0.0606 | 0.058 | 0.0644 | 0.0626 |
| uncorrected | 0.0596 | 0.403 | 0.607 | 3.29 | 1.42 | 2.72 |
| gauss | 0.0601 | 0.396 | 0.677 | 3.13 | 2.61 | 5.71 |
| gauss_hierarchical | 0.0647 | 0.347 | 0.698 | 2.2 | 1.09 | 2.52 |
| gauss_jacobian | 0.0596 | 0.43 | 0.703 | 2.47 | 1.09 | 2.99 |
| gauss_jacobian_FD | 0.0596 | 0.43 | 0.696 | 2.45 | 1.04 | 2.94 |
| langevin_fnpe | 0.0597 | 0.505 | 0.65 | 3.93 | 2.4 | 3.32 |
| attn_hier:arruda_direct | 0.0653 | 0.368 | 0.577 | 1.98 | 1.37 | 2.31 |
| attn_hier:arruda_gauss_direct | 0.0653 | 0.38 | 0.641 | 1.7 | 1.13 | 3.02 |
| attn_hier:gauss_jacobian_FD | 0.0653 | 0.381 | 0.644 | 1.73 | 1.09 | 2.84 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0618 | 0.317 | 0.614 | 1.79 | 1.14 | 3.05 |
| attn_full:gauss_hierarchical | 0.0622 | 0.315 | 0.713 | 2.23 | 1.29 | 3.08 |
| attn_full:gauss_jacobian_FD | 0.0642 | 0.398 | 0.705 | 2.29 | 1.33 | 4.46 |
| attn_full:arruda_schur | 0.195 | 0.633 | 1.22 | 8.07 | 7.39 | 11.3 |
| attn_full:arruda_laplace | 0.0687 | 0.343 | 0.538 | 2.94 | 3.86 | 7.64 |
| bayesflow_arruda | 0.0609 | 0.953 | 0.798 | 3.08 | 1.4 | 3.35 |
| bayesflow_arruda_diffprior | 0.0575 | 0.346 | 0.598 | 2.04 | 1.36 | 2.14 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0676 | 0.396 | 0.616 | 1.63 | 1.11 | 1.94 |
| legacy_w1.5:gauss_jacobian_FD | 0.0638 | 0.431 | 0.709 | 1.8 | 0.948 | 1.65 |
| legacy_w1.5:gauss_hierarchical | 0.0665 | 0.34 | 0.724 | 1.97 | 1.23 | 1.83 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0613 | 0.396 | 0.593 | 1.49 | 1.58 | 2.45 |
| attn_hier_tsm:arruda_direct | 0.0613 | 0.383 | 0.514 | 1.59 | 1.64 | 2.01 |

## Q4 global–local dependence

**|Δ corr(g, θ_i)|** (`dep_corr_err`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0344 | 0.0413 | 0.0354 | 0.0308 | 0.0299 | 0.027 |
| uncorrected | 0.0487 | 0.178 | 0.338 | 0.378 | 0.306 | 0.477 |
| gauss | 0.0524 | 0.219 | 0.368 | 0.449 | 0.334 | 0.51 |
| gauss_hierarchical | 0.0347 | 0.174 | 0.323 | 0.37 | 0.308 | 0.44 |
| gauss_jacobian | 0.0487 | 0.156 | 0.326 | 0.379 | 0.286 | 0.399 |
| gauss_jacobian_FD | 0.0487 | 0.153 | 0.325 | 0.373 | 0.277 | 0.396 |
| langevin_fnpe | 0.0359 | 0.185 | 0.34 | 0.446 | 0.325 | 0.487 |
| attn_hier:arruda_direct | 0.0486 | 0.134 | 0.302 | 0.353 | 0.317 | 0.494 |
| attn_hier:arruda_gauss_direct | 0.0486 | 0.139 | 0.303 | 0.355 | 0.301 | 0.466 |
| attn_hier:gauss_jacobian_FD | 0.0744 | 0.181 | 0.355 | 0.423 | 0.322 | 0.508 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.0885 | 0.185 | 0.361 | 0.43 | 0.32 | 0.498 |
| attn_full:gauss_hierarchical | 0.0345 | 0.175 | 0.328 | 0.393 | 0.307 | 0.456 |
| attn_full:gauss_jacobian_FD | 0.0484 | 0.143 | 0.331 | 0.412 | 0.288 | 0.403 |
| attn_full:arruda_schur | 0.0662 | 0.182 | 0.333 | 0.389 | 0.332 | 0.487 |
| attn_full:arruda_laplace | 0.0397 | 0.0951 | 0.274 | 0.29 | 0.26 | 0.367 |
| bayesflow_arruda | 0.0433 | 0.25 | 0.336 | 0.432 | 0.212 | 0.19 |
| bayesflow_arruda_diffprior | 0.047 | 0.0729 | 0.245 | 0.243 | 0.193 | 0.268 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.0772 | 0.123 | 0.325 | 0.316 | 0.312 | 0.42 |
| legacy_w1.5:gauss_jacobian_FD | 0.0511 | 0.109 | 0.328 | 0.323 | 0.266 | 0.377 |
| legacy_w1.5:gauss_hierarchical | 0.0376 | 0.131 | 0.325 | 0.309 | 0.286 | 0.395 |
| attn_hier_tsm:gauss_jacobian_FD | 0.0754 | 0.152 | 0.339 | 0.388 | 0.321 | 0.464 |
| attn_hier_tsm:arruda_direct | 0.0458 | 0.109 | 0.278 | 0.308 | 0.324 | 0.387 |

**ref |corr| (signal)** (`dep_corr_ref`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| uncorrected | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| gauss | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| gauss_hierarchical | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| gauss_jacobian | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| gauss_jacobian_FD | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| langevin_fnpe | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_hier:arruda_direct | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_hier:arruda_gauss_direct | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_hier:gauss_jacobian_FD | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_full:gauss_hierarchical | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_full:gauss_jacobian_FD | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_full:arruda_schur | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_full:arruda_laplace | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| bayesflow_arruda | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| bayesflow_arruda_diffprior | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| legacy_w1.5:gauss_jacobian_FD | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| legacy_w1.5:gauss_hierarchical | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_hier_tsm:gauss_jacobian_FD | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |
| attn_hier_tsm:arruda_direct | 0.089 | 0.254 | 0.397 | 0.489 | 0.346 | 0.52 |

**joint (g, θ_i) C2ST** (`joint_c2st`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.502 | 0.495 | 0.501 | 0.495 | 0.5 | 0.502 |
| uncorrected | 0.5 | 0.641 | 0.791 | 0.964 | 0.932 | 0.991 |
| gauss | 0.727 | 0.798 | 0.857 | 0.94 | 0.947 | 0.973 |
| gauss_hierarchical | 0.504 | 0.637 | 0.792 | 0.931 | 0.856 | 0.968 |
| gauss_jacobian | 0.5 | 0.622 | 0.803 | 0.926 | 0.845 | 0.956 |
| gauss_jacobian_FD | 0.5 | 0.625 | 0.8 | 0.927 | 0.843 | 0.951 |
| langevin_fnpe | 0.506 | 0.688 | 0.837 | 0.936 | 0.919 | 0.983 |
| attn_hier:arruda_direct | 0.502 | 0.649 | 0.791 | 0.966 | 0.972 | 0.992 |
| attn_hier:arruda_gauss_direct | 0.502 | 0.641 | 0.787 | 0.947 | 0.978 | 0.992 |
| attn_hier:gauss_jacobian_FD | 0.518 | 0.652 | 0.798 | 0.95 | 0.978 | 0.989 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.514 | 0.642 | 0.803 | 0.947 | 0.976 | 0.989 |
| attn_full:gauss_hierarchical | 0.498 | 0.608 | 0.805 | 0.919 | 0.924 | 0.983 |
| attn_full:gauss_jacobian_FD | 0.508 | 0.602 | 0.815 | 0.887 | 0.853 | 0.96 |
| attn_full:arruda_schur | 0.578 | 0.788 | 0.918 | 0.962 | 0.977 | 0.981 |
| attn_full:arruda_laplace | 0.502 | 0.595 | 0.758 | 0.899 | 0.929 | 0.965 |
| bayesflow_arruda | 0.502 | 0.926 | 0.915 | 0.959 | 0.896 | 0.944 |
| bayesflow_arruda_diffprior | 0.498 | 0.631 | 0.739 | 0.86 | 0.867 | 0.943 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.522 | 0.613 | 0.767 | 0.867 | 0.923 | 0.955 |
| legacy_w1.5:gauss_jacobian_FD | 0.512 | 0.597 | 0.81 | 0.836 | 0.836 | 0.932 |
| legacy_w1.5:gauss_hierarchical | 0.495 | 0.604 | 0.802 | 0.835 | 0.865 | 0.947 |
| attn_hier_tsm:gauss_jacobian_FD | 0.522 | 0.642 | 0.762 | 0.936 | 0.952 | 0.985 |
| attn_hier_tsm:arruda_direct | 0.509 | 0.642 | 0.738 | 0.932 | 0.983 | 0.994 |

**same, dependence shuffled out** (`joint_c2st_indep`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.526 | 0.576 | 0.638 | 0.652 | 0.598 | 0.624 |
| uncorrected | 0.536 | 0.666 | 0.807 | 0.967 | 0.936 | 0.992 |
| gauss | 0.743 | 0.816 | 0.863 | 0.944 | 0.949 | 0.975 |
| gauss_hierarchical | 0.524 | 0.662 | 0.814 | 0.932 | 0.859 | 0.969 |
| gauss_jacobian | 0.536 | 0.664 | 0.824 | 0.93 | 0.851 | 0.961 |
| gauss_jacobian_FD | 0.536 | 0.661 | 0.82 | 0.927 | 0.846 | 0.955 |
| langevin_fnpe | 0.545 | 0.72 | 0.849 | 0.937 | 0.919 | 0.984 |
| attn_hier:arruda_direct | 0.551 | 0.677 | 0.81 | 0.968 | 0.973 | 0.993 |
| attn_hier:arruda_gauss_direct | 0.551 | 0.666 | 0.806 | 0.948 | 0.978 | 0.99 |
| attn_hier:gauss_jacobian_FD | 0.528 | 0.669 | 0.806 | 0.951 | 0.979 | 0.989 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 0.53 | 0.658 | 0.812 | 0.953 | 0.977 | 0.99 |
| attn_full:gauss_hierarchical | 0.522 | 0.643 | 0.827 | 0.927 | 0.926 | 0.985 |
| attn_full:gauss_jacobian_FD | 0.534 | 0.649 | 0.831 | 0.9 | 0.858 | 0.96 |
| attn_full:arruda_schur | 0.603 | 0.804 | 0.915 | 0.962 | 0.975 | 0.981 |
| attn_full:arruda_laplace | 0.544 | 0.648 | 0.783 | 0.918 | 0.94 | 0.967 |
| bayesflow_arruda | 0.549 | 0.925 | 0.92 | 0.964 | 0.907 | 0.948 |
| bayesflow_arruda_diffprior | 0.527 | 0.676 | 0.786 | 0.888 | 0.88 | 0.95 |
| attn_hier_w1.5:gauss_jacobian_FD | 0.534 | 0.654 | 0.79 | 0.872 | 0.933 | 0.964 |
| legacy_w1.5:gauss_jacobian_FD | 0.532 | 0.647 | 0.828 | 0.857 | 0.843 | 0.94 |
| legacy_w1.5:gauss_hierarchical | 0.534 | 0.648 | 0.821 | 0.854 | 0.865 | 0.949 |
| attn_hier_tsm:gauss_jacobian_FD | 0.534 | 0.663 | 0.78 | 0.939 | 0.952 | 0.985 |
| attn_hier_tsm:arruda_direct | 0.538 | 0.671 | 0.771 | 0.941 | 0.983 | 0.994 |

## Cost

**s / dataset** (`seconds`)

| method | N=1 | N=5 | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|---|
| reference | 0.0882 | 0.157 | 0.267 | 0.606 | 1.46 | 2.96 |
| uncorrected | 6.1 | 11.7 | 13.6 | 31.1 | 63.6 | 111 |
| gauss | 19 | 26.6 | 30.8 | 63.3 | 130 | 223 |
| gauss_hierarchical | 34.7 | 49 | 62.6 | 105 | 147 | 152 |
| gauss_jacobian | 21.9 | 35.7 | 59.2 | 192 | 351 | 648 |
| gauss_jacobian_FD | 5.81 | 28 | 56.4 | 161 | 410 | 484 |
| langevin_fnpe | 14.7 | 37.6 | 72.6 | 175 | 331 | 617 |
| attn_hier:arruda_direct | 6.79 | 18.6 | 37.4 | 118 | 209 | 100 |
| attn_hier:arruda_gauss_direct | 10.4 | 63.6 | 132 | 336 | 609 | 1.2e+03 |
| attn_hier:gauss_jacobian_FD | 9.89 | 40.6 | 96.6 | 221 | 445 | 743 |
| attn_hier:gauss_jacobian_FD_r5_churn4 | 4.77 | 32.5 | 49.9 | 166 | 314 | 618 |
| attn_full:gauss_hierarchical | 26.1 | 53 | 70.9 | 111 | 182 | 171 |
| attn_full:gauss_jacobian_FD | 5.82 | 28 | 49.8 | 173 | 389 | 770 |
| attn_full:arruda_schur | 8.22 | 34.5 | 107 | 305 | 625 | 1.36e+03 |
| attn_full:arruda_laplace | 14.4 | 84.7 | 185 | 493 | 997 | 2.07e+03 |
| bayesflow_arruda | 10.7 | 29.9 | 109 | 103 | 136 | 126 |
| bayesflow_arruda_diffprior | 11.9 | 65.6 | 155 | 181 | 198 | 211 |
| attn_hier_w1.5:gauss_jacobian_FD | 19 | 50.6 | 105 | 383 | 601 | 639 |
| legacy_w1.5:gauss_jacobian_FD | 18.5 | 51.5 | 103 | 333 | 652 | 614 |
| legacy_w1.5:gauss_hierarchical | 30.5 | 58.6 | 94.9 | 143 | 222 | 386 |
| attn_hier_tsm:gauss_jacobian_FD | 57.3 | 50.6 | 69.1 | 176 | 326 | 175 |
| attn_hier_tsm:arruda_direct | 28 | 34.5 | 50.3 | 110 | 181 | 358 |

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
| attn_hier_w1.5:gauss_jacobian_FD | 0 | 0 | 0 | 0 | 0 | 0 |
| legacy_w1.5:gauss_jacobian_FD | 0 | 0 | 0 | 0 | 0 | 0 |
| legacy_w1.5:gauss_hierarchical | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_hier_tsm:gauss_jacobian_FD | 0 | 0 | 0 | 0 | 0 | 0 |
| attn_hier_tsm:arruda_direct | 0 | 0 | 0 | 0 | 0 | 0 |

## BayesFlow tuned (d1, shift)

| method | N | d1 | shift |
|---|---|---|---|
| bayesflow_arruda | 1 | 1 | 0 |
| bayesflow_arruda | 5 | 1 | 5 |
| bayesflow_arruda | 10 | 1 | 0 |
| bayesflow_arruda | 25 | 1 | 0 |
| bayesflow_arruda | 50 | 1 | 0 |
| bayesflow_arruda | 100 | 0.01 | 0 |
| bayesflow_arruda_diffprior | 1 | 1 | 0 |
| bayesflow_arruda_diffprior | 5 | 1 | 0 |
| bayesflow_arruda_diffprior | 10 | 1 | 2 |
| bayesflow_arruda_diffprior | 25 | 1 | 2 |
| bayesflow_arruda_diffprior | 50 | 1 | 0 |
| bayesflow_arruda_diffprior | 100 | 1 | 0 |
