# GM (exact posterior)

| net arm | method | configs | mean W1/sd | mean abs branch err | mean width | mean local W1 | mean s |
|---|---|---|---|---|---|---|---|
| base | GH (pilot) | 16 | 0.958 | 0.188 | 1.169 | 1.724 | 128 |
| base | JAC-FD r9 | 16 | 1.459 | 0.143 | 2.287 | 1.725 | 149 |
| cov | GH (pilot) | 16 | 0.903 | 0.432 | 0.287 | 1.724 | 103 |
| cov | JAC-FD r9 | 16 | 1.138 | 0.307 | 1.469 | 1.725 | 116 |
| cov | JAC-netcov | 16 | 0.931 | 0.265 | 1.611 | 1.594 | 180 |
| cov_det | GH (pilot) | 16 | 0.851 | 0.360 | 0.254 | 1.721 | 62 |
| cov_det | JAC-FD r9 | 16 | 0.975 | 0.255 | 1.619 | 1.723 | 67 |
| cov_det | JAC-netcov | 16 | 1.037 | 0.180 | 1.923 | 1.545 | 103 |

<details><summary>per config</summary>

| net | regime | n | method | W1/sd | branch err | width | local W1 | s |
|---|---|---|---|---|---|---|---|---|
| base_s7 | asymmetric | 8 | GH (pilot) | 0.413 | -0.114 | 0.789 | 1.338 | 102 |
| base_s7 | asymmetric | 8 | JAC-FD r9 | 0.561 | -0.071 | 1.378 | 1.352 | 110 |
| base_s7 | asymmetric | 16 | GH (pilot) | 0.973 | -0.687 | 0.569 | 1.664 | 136 |
| base_s7 | asymmetric | 16 | JAC-FD r9 | 1.247 | -0.331 | 1.858 | 1.655 | 135 |
| base_s7 | asymmetric | 32 | GH (pilot) | 1.036 | -0.509 | 0.143 | 1.902 | 173 |
| base_s7 | asymmetric | 32 | JAC-FD r9 | 1.775 | -0.376 | 2.396 | 1.904 | 211 |
| base_s7 | asymmetric | 64 | GH (pilot) | 0.931 | -0.300 | 0.278 | 2.067 | 293 |
| base_s7 | asymmetric | 64 | JAC-FD r9 | 2.022 | -0.373 | 2.732 | 2.068 | 436 |
| base_s7 | symmetric | 8 | GH (pilot) | 0.262 | -0.028 | 0.911 | 1.289 | 132 |
| base_s7 | symmetric | 8 | JAC-FD r9 | 0.377 | -0.005 | 1.325 | 1.290 | 126 |
| base_s7 | symmetric | 16 | GH (pilot) | 0.727 | -0.015 | 0.445 | 1.511 | 157 |
| base_s7 | symmetric | 16 | JAC-FD r9 | 0.738 | -0.062 | 1.641 | 1.514 | 154 |
| base_s7 | symmetric | 32 | GH (pilot) | 0.847 | -0.080 | 0.222 | 1.920 | 165 |
| base_s7 | symmetric | 32 | JAC-FD r9 | 1.298 | -0.155 | 1.961 | 1.921 | 178 |
| base_s7 | symmetric | 64 | GH (pilot) | 0.716 | -0.131 | 0.241 | 2.068 | 211 |
| base_s7 | symmetric | 64 | JAC-FD r9 | 1.563 | -0.183 | 2.457 | 2.066 | 272 |
| base_s8 | asymmetric | 8 | GH (pilot) | 0.443 | +0.151 | 0.642 | 1.352 | 42 |
| base_s8 | asymmetric | 8 | JAC-FD r9 | 0.541 | -0.017 | 1.321 | 1.364 | 37 |
| base_s8 | asymmetric | 16 | GH (pilot) | 0.535 | -0.198 | 1.218 | 1.671 | 63 |
| base_s8 | asymmetric | 16 | JAC-FD r9 | 1.119 | -0.178 | 1.959 | 1.661 | 63 |
| base_s8 | asymmetric | 32 | GH (pilot) | 1.263 | -0.234 | 2.013 | 1.917 | 88 |
| base_s8 | asymmetric | 32 | JAC-FD r9 | 2.052 | -0.182 | 2.934 | 1.919 | 101 |
| base_s8 | asymmetric | 64 | GH (pilot) | 3.236 | -0.271 | 3.983 | 2.078 | 139 |
| base_s8 | asymmetric | 64 | JAC-FD r9 | 3.877 | -0.206 | 4.758 | 2.080 | 176 |
| base_s8 | symmetric | 8 | GH (pilot) | 0.128 | -0.000 | 1.003 | 1.287 | 51 |
| base_s8 | symmetric | 8 | JAC-FD r9 | 0.408 | +0.063 | 1.315 | 1.288 | 44 |
| base_s8 | symmetric | 16 | GH (pilot) | 0.245 | -0.100 | 1.084 | 1.521 | 63 |
| base_s8 | symmetric | 16 | JAC-FD r9 | 0.776 | +0.031 | 1.741 | 1.523 | 63 |
| base_s8 | symmetric | 32 | GH (pilot) | 0.916 | -0.103 | 1.744 | 1.924 | 88 |
| base_s8 | symmetric | 32 | JAC-FD r9 | 1.648 | -0.019 | 2.591 | 1.925 | 101 |
| base_s8 | symmetric | 64 | GH (pilot) | 2.656 | -0.084 | 3.412 | 2.073 | 139 |
| base_s8 | symmetric | 64 | JAC-FD r9 | 3.349 | +0.041 | 4.220 | 2.071 | 173 |
| cov_det_s7 | asymmetric | 8 | GH (pilot) | 0.471 | -0.145 | 0.710 | 1.340 | 53 |
| cov_det_s7 | asymmetric | 8 | JAC-FD r9 | 0.423 | -0.069 | 1.293 | 1.354 | 70 |
| cov_det_s7 | asymmetric | 8 | JAC-netcov | 0.510 | -0.124 | 1.398 | 1.186 | 77 |
| cov_det_s7 | asymmetric | 16 | GH (pilot) | 0.956 | -0.719 | 0.404 | 1.667 | 79 |
| cov_det_s7 | asymmetric | 16 | JAC-FD r9 | 1.094 | -0.346 | 1.673 | 1.658 | 91 |
| cov_det_s7 | asymmetric | 16 | JAC-netcov | 0.970 | -0.259 | 1.763 | 1.479 | 112 |
| cov_det_s7 | asymmetric | 32 | GH (pilot) | 1.085 | -0.747 | 0.038 | 1.909 | 114 |
| cov_det_s7 | asymmetric | 32 | JAC-FD r9 | 1.573 | -0.465 | 2.203 | 1.910 | 125 |
| cov_det_s7 | asymmetric | 32 | JAC-netcov | 1.451 | -0.358 | 2.301 | 1.722 | 198 |
| cov_det_s7 | asymmetric | 64 | GH (pilot) | 1.133 | -0.746 | 0.116 | 2.071 | 166 |
| cov_det_s7 | asymmetric | 64 | JAC-FD r9 | 2.404 | -0.627 | 2.825 | 2.073 | 223 |
| cov_det_s7 | asymmetric | 64 | JAC-netcov | 1.914 | -0.444 | 3.010 | 1.861 | 380 |
| cov_det_s7 | symmetric | 8 | GH (pilot) | 0.400 | -0.098 | 0.751 | 1.286 | 53 |
| cov_det_s7 | symmetric | 8 | JAC-FD r9 | 0.282 | +0.027 | 1.226 | 1.288 | 57 |
| cov_det_s7 | symmetric | 8 | JAC-netcov | 0.288 | -0.037 | 1.244 | 1.105 | 91 |
| cov_det_s7 | symmetric | 16 | GH (pilot) | 0.787 | -0.178 | 0.338 | 1.516 | 73 |
| cov_det_s7 | symmetric | 16 | JAC-FD r9 | 0.585 | -0.078 | 1.522 | 1.518 | 91 |
| cov_det_s7 | symmetric | 16 | JAC-netcov | 0.574 | -0.054 | 1.540 | 1.352 | 112 |
| cov_det_s7 | symmetric | 32 | GH (pilot) | 0.904 | -0.265 | 0.097 | 1.923 | 127 |
| cov_det_s7 | symmetric | 32 | JAC-FD r9 | 1.155 | -0.238 | 1.811 | 1.925 | 109 |
| cov_det_s7 | symmetric | 32 | JAC-netcov | 0.982 | -0.115 | 1.945 | 1.718 | 195 |
| cov_det_s7 | symmetric | 64 | GH (pilot) | 0.874 | -0.470 | 0.063 | 2.073 | 98 |
| cov_det_s7 | symmetric | 64 | JAC-FD r9 | 1.861 | -0.369 | 2.483 | 2.071 | 110 |
| cov_det_s7 | symmetric | 64 | JAC-netcov | 1.565 | -0.164 | 2.702 | 1.860 | 184 |
| cov_det_s8 | asymmetric | 8 | GH (pilot) | 0.693 | -0.030 | 0.505 | 1.336 | 31 |
| cov_det_s8 | asymmetric | 8 | JAC-FD r9 | 0.309 | +0.003 | 1.121 | 1.351 | 24 |
| cov_det_s8 | asymmetric | 8 | JAC-netcov | 0.437 | -0.097 | 1.375 | 1.218 | 34 |
| cov_det_s8 | asymmetric | 16 | GH (pilot) | 1.064 | -0.299 | 0.126 | 1.666 | 27 |
| cov_det_s8 | asymmetric | 16 | JAC-FD r9 | 0.619 | -0.165 | 1.324 | 1.657 | 23 |
| cov_det_s8 | asymmetric | 16 | JAC-netcov | 0.746 | -0.144 | 1.689 | 1.502 | 26 |
| cov_det_s8 | asymmetric | 32 | GH (pilot) | 1.029 | -0.008 | 0.031 | 1.904 | 26 |
| cov_det_s8 | asymmetric | 32 | JAC-FD r9 | 0.916 | +0.074 | 1.436 | 1.908 | 21 |
| cov_det_s8 | asymmetric | 32 | JAC-netcov | 1.170 | -0.066 | 2.045 | 1.733 | 36 |
| cov_det_s8 | asymmetric | 64 | GH (pilot) | 0.863 | +0.246 | 0.118 | 2.070 | 49 |
| cov_det_s8 | asymmetric | 64 | JAC-FD r9 | 1.239 | +0.240 | 1.845 | 2.071 | 41 |
| cov_det_s8 | asymmetric | 64 | JAC-netcov | 2.007 | +0.046 | 2.714 | 1.875 | 70 |
| cov_det_s8 | symmetric | 8 | GH (pilot) | 0.624 | +0.306 | 0.571 | 1.281 | 11 |
| cov_det_s8 | symmetric | 8 | JAC-FD r9 | 0.365 | +0.176 | 1.106 | 1.283 | 6 |
| cov_det_s8 | symmetric | 8 | JAC-netcov | 0.292 | +0.050 | 1.245 | 1.147 | 10 |
| cov_det_s8 | symmetric | 16 | GH (pilot) | 0.936 | +0.494 | 0.079 | 1.509 | 15 |
| cov_det_s8 | symmetric | 16 | JAC-FD r9 | 0.560 | +0.290 | 1.182 | 1.511 | 11 |
| cov_det_s8 | symmetric | 16 | JAC-netcov | 0.646 | +0.166 | 1.484 | 1.360 | 19 |
| cov_det_s8 | symmetric | 32 | GH (pilot) | 0.915 | +0.500 | 0.028 | 1.921 | 26 |
| cov_det_s8 | symmetric | 32 | JAC-FD r9 | 0.868 | +0.427 | 1.242 | 1.922 | 21 |
| cov_det_s8 | symmetric | 32 | JAC-netcov | 1.131 | +0.316 | 1.846 | 1.739 | 36 |
| cov_det_s8 | symmetric | 64 | GH (pilot) | 0.886 | +0.500 | 0.088 | 2.068 | 49 |
| cov_det_s8 | symmetric | 64 | JAC-FD r9 | 1.351 | +0.489 | 1.619 | 2.068 | 41 |
| cov_det_s8 | symmetric | 64 | JAC-netcov | 1.910 | +0.431 | 2.459 | 1.871 | 70 |
| cov_s7 | asymmetric | 8 | GH (pilot) | 0.554 | -0.195 | 0.697 | 1.351 | 81 |
| cov_s7 | asymmetric | 8 | JAC-FD r9 | 0.455 | -0.048 | 1.230 | 1.363 | 67 |
| cov_s7 | asymmetric | 8 | JAC-netcov | 0.423 | -0.036 | 1.267 | 1.260 | 98 |
| cov_s7 | asymmetric | 16 | GH (pilot) | 1.036 | -0.727 | 0.571 | 1.667 | 100 |
| cov_s7 | asymmetric | 16 | JAC-FD r9 | 1.170 | -0.400 | 1.694 | 1.658 | 98 |
| cov_s7 | asymmetric | 16 | JAC-netcov | 0.992 | -0.319 | 1.708 | 1.542 | 150 |
| cov_s7 | asymmetric | 32 | GH (pilot) | 1.058 | -0.683 | 0.157 | 1.910 | 136 |
| cov_s7 | asymmetric | 32 | JAC-FD r9 | 1.899 | -0.528 | 2.150 | 1.910 | 160 |
| cov_s7 | asymmetric | 32 | JAC-netcov | 1.494 | -0.452 | 2.189 | 1.782 | 250 |
| cov_s7 | asymmetric | 64 | GH (pilot) | 1.025 | -0.479 | 0.179 | 2.069 | 213 |
| cov_s7 | asymmetric | 64 | JAC-FD r9 | 3.452 | -0.604 | 2.791 | 2.070 | 275 |
| cov_s7 | asymmetric | 64 | JAC-netcov | 2.295 | -0.473 | 2.965 | 1.925 | 455 |
| cov_s7 | symmetric | 8 | GH (pilot) | 0.487 | -0.096 | 0.655 | 1.287 | 82 |
| cov_s7 | symmetric | 8 | JAC-FD r9 | 0.267 | +0.005 | 1.205 | 1.287 | 69 |
| cov_s7 | symmetric | 8 | JAC-netcov | 0.263 | +0.040 | 1.215 | 1.176 | 97 |
| cov_s7 | symmetric | 16 | GH (pilot) | 0.772 | +0.024 | 0.343 | 1.513 | 102 |
| cov_s7 | symmetric | 16 | JAC-FD r9 | 0.648 | -0.070 | 1.445 | 1.515 | 99 |
| cov_s7 | symmetric | 16 | JAC-netcov | 0.581 | -0.008 | 1.483 | 1.405 | 151 |
| cov_s7 | symmetric | 32 | GH (pilot) | 0.871 | -0.087 | 0.488 | 1.914 | 135 |
| cov_s7 | symmetric | 32 | JAC-FD r9 | 1.548 | -0.277 | 1.703 | 1.915 | 167 |
| cov_s7 | symmetric | 32 | JAC-netcov | 1.142 | -0.142 | 1.825 | 1.778 | 235 |
| cov_s7 | symmetric | 64 | GH (pilot) | 0.830 | -0.344 | 0.246 | 2.066 | 162 |
| cov_s7 | symmetric | 64 | JAC-FD r9 | 2.686 | -0.399 | 2.261 | 2.064 | 224 |
| cov_s7 | symmetric | 64 | JAC-netcov | 1.836 | -0.342 | 2.522 | 1.921 | 383 |
| cov_s8 | asymmetric | 8 | GH (pilot) | 0.776 | -0.405 | 0.496 | 1.348 | 50 |
| cov_s8 | asymmetric | 8 | JAC-FD r9 | 0.331 | -0.024 | 1.035 | 1.362 | 43 |
| cov_s8 | asymmetric | 8 | JAC-netcov | 0.291 | -0.060 | 1.176 | 1.249 | 60 |
| cov_s8 | asymmetric | 16 | GH (pilot) | 1.122 | -0.749 | 0.047 | 1.671 | 62 |
| cov_s8 | asymmetric | 16 | JAC-FD r9 | 0.666 | -0.355 | 1.176 | 1.662 | 62 |
| cov_s8 | asymmetric | 16 | JAC-netcov | 0.632 | -0.345 | 1.332 | 1.537 | 93 |
| cov_s8 | asymmetric | 32 | GH (pilot) | 1.124 | -0.750 | 0.036 | 1.913 | 86 |
| cov_s8 | asymmetric | 32 | JAC-FD r9 | 0.982 | -0.550 | 1.234 | 1.916 | 98 |
| cov_s8 | asymmetric | 32 | JAC-netcov | 0.964 | -0.508 | 1.454 | 1.768 | 153 |
| cov_s8 | asymmetric | 64 | GH (pilot) | 1.311 | -0.750 | 0.073 | 2.076 | 132 |
| cov_s8 | asymmetric | 64 | JAC-FD r9 | 1.337 | -0.677 | 1.368 | 2.078 | 166 |
| cov_s8 | asymmetric | 64 | JAC-netcov | 1.365 | -0.650 | 1.675 | 1.907 | 278 |
| cov_s8 | symmetric | 8 | GH (pilot) | 0.725 | -0.217 | 0.473 | 1.286 | 48 |
| cov_s8 | symmetric | 8 | JAC-FD r9 | 0.441 | +0.101 | 1.031 | 1.288 | 41 |
| cov_s8 | symmetric | 8 | JAC-netcov | 0.318 | +0.043 | 1.134 | 1.180 | 58 |
| cov_s8 | symmetric | 16 | GH (pilot) | 0.946 | -0.442 | 0.028 | 1.517 | 60 |
| cov_s8 | symmetric | 16 | JAC-FD r9 | 0.592 | -0.119 | 1.055 | 1.518 | 60 |
| cov_s8 | symmetric | 16 | JAC-netcov | 0.547 | -0.113 | 1.202 | 1.396 | 90 |
| cov_s8 | symmetric | 32 | GH (pilot) | 0.912 | -0.488 | 0.031 | 1.924 | 83 |
| cov_s8 | symmetric | 32 | JAC-FD r9 | 0.824 | -0.339 | 0.990 | 1.924 | 95 |
| cov_s8 | symmetric | 32 | JAC-netcov | 0.807 | -0.304 | 1.211 | 1.775 | 152 |
| cov_s8 | symmetric | 64 | GH (pilot) | 0.899 | -0.483 | 0.064 | 2.073 | 113 |
| cov_s8 | symmetric | 64 | JAC-FD r9 | 0.912 | -0.419 | 1.143 | 2.072 | 130 |
| cov_s8 | symmetric | 64 | JAC-netcov | 0.946 | -0.403 | 1.414 | 1.905 | 176 |

</details>

# Partial pooling (30 datasets x 100 subjects)

| net arm | method | seeds | mean r | global RMSE | cover90 | z_rms | local RMSE | s/dataset |
|---|---|---|---|---|---|---|---|---|
| cov | GH (pilot) | 2 | 0.939 | 0.344 | 0.85 | 1.03 | 0.156 | 169 |
| cov | JAC-FD r9 | 2 | 0.945 | 0.333 | 0.81 | 1.21 | 0.153 | 126 |
| cov | JAC-netcov | 2 | 0.946 | 0.338 | 0.83 | 1.19 | 0.153 | 118 |
| cov_det | GH (pilot) | 2 | 0.938 | 0.339 | 0.88 | 0.98 | 0.156 | 229 |
| cov_det | JAC-FD r9 | 2 | 0.949 | 0.316 | 0.85 | 1.07 | 0.153 | 118 |
| cov_det | JAC-netcov | 2 | 0.947 | 0.355 | 0.87 | 1.05 | 0.156 | 114 |

| net | method | mean r | global RMSE | cover90 | z_rms | local RMSE | s/dataset |
|---|---|---|---|---|---|---|---|
| cov_det_s7 | GH (pilot) | 0.942 | 0.326 | 0.88 | 0.97 | 0.157 | 229 |
| cov_det_s7 | JAC-FD r9 | 0.951 | 0.298 | 0.86 | 1.03 | 0.152 | 127 |
| cov_det_s7 | JAC-netcov | 0.950 | 0.323 | 0.90 | 0.90 | 0.155 | 141 |
| cov_det_s8 | GH (pilot) | 0.934 | 0.352 | 0.88 | 0.99 | 0.156 | 230 |
| cov_det_s8 | JAC-FD r9 | 0.948 | 0.333 | 0.85 | 1.10 | 0.154 | 108 |
| cov_det_s8 | JAC-netcov | 0.944 | 0.387 | 0.84 | 1.21 | 0.157 | 87 |
| cov_s7 | GH (pilot) | 0.934 | 0.351 | 0.82 | 1.03 | 0.156 | 79 |
| cov_s7 | JAC-FD r9 | 0.938 | 0.339 | 0.79 | 1.16 | 0.153 | 137 |
| cov_s7 | JAC-netcov | 0.938 | 0.346 | 0.81 | 1.13 | 0.153 | 125 |
| cov_s8 | GH (pilot) | 0.944 | 0.337 | 0.88 | 1.02 | 0.156 | 260 |
| cov_s8 | JAC-FD r9 | 0.952 | 0.327 | 0.82 | 1.26 | 0.154 | 114 |
| cov_s8 | JAC-netcov | 0.954 | 0.330 | 0.85 | 1.25 | 0.153 | 111 |
