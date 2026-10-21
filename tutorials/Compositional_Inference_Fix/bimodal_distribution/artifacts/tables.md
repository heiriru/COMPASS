### Network fidelity (relative RMS of the learned row score against the closed form)

| benchmark | score source | config | parameters | lam 0.05 | lam 0.2 | lam 1 | mean |
|---|---|---|---|---|---|---|---|
| sign | learned | h32d3 | 552,352 | 0.440 | 0.152 | 0.069 | 0.166 |
| sign | learned_large | h128d6 | 5,202,688 | 0.132 | 0.045 | 0.028 | 0.058 |
| control | learned | h32d3 | 552,352 | 0.866 | 0.182 | 0.046 | 0.230 |
| control | learned_large | h128d6 | 5,202,688 | 0.053 | 0.020 | 0.012 | 0.023 |

### control: exact targets

modes = 2; P(g>0) = 0.643; gap draws = 0.24 of 3000; mode s.d. = 0.376; corr(g,l) = 0.094; sign coherence = 0.000

### control / analytic

| arm | W1(g)/sigma | W1(l)/sigma | modes | gap draws | P(g>0) | mode s.d. | corr(g,l) | seconds | score calls |
|---|---|---|---|---|---|---|---|---|---|
| DPM2 + churn 8 + gauss_hierarchical | 0.2662 | 0.0212 | 2 | 0 | 0.576 | 0.131 | 0.108 | 1.3 | 198 |
| DPM2 + churn 8 + gauss_jacobian | 0.2950 | 0.0239 | 2 | 0 | 0.554 | 0.145 | 0.110 | 4.9 | 594 |
| Langevin + F-NPSE | 0.2217 | 0.0308 | 2 | 0 | 0.602 | 0.145 | 0.104 | 3.2 | 1000 |
| DPM2 + gauss_hierarchical (10 correctors) | 0.2416 | 0.0260 | 2 | 0 | 0.589 | 0.144 | 0.107 | 6.5 | 1268 |
| DPM2 + gauss_jacobian (10 correctors) | 0.3050 | 0.0312 | 2 | 0 | 0.547 | 0.147 | 0.103 | 30.3 | 3804 |

_Seconds and score calls overstate the cost of any arm whose draws had to be chunked to fit in memory (the 5.2M-parameter `gauss_jacobian` arms); the sampled statistics are unaffected._

### control / learned_large

| arm | W1(g)/sigma | W1(l)/sigma | modes | gap draws | P(g>0) | mode s.d. | corr(g,l) | seconds | score calls |
|---|---|---|---|---|---|---|---|---|---|
| DPM2 + churn 8 + gauss_hierarchical | 0.2173 | 0.0283 | 2 | 0 | 0.608 | 0.134 | 0.113 | 98.0 | 198 |
| DPM2 + churn 8 + gauss_jacobian | 0.2962 | 0.0423 | 2 | 0 | 0.554 | 0.145 | 0.099 | 112.7 | 2376 |
| Langevin + F-NPSE | 0.2196 | 0.0289 | 2 | 0 | 0.604 | 0.143 | 0.097 | 139.9 | 1000 |
| DPM2 + gauss_hierarchical (10 correctors) | 0.1950 | 0.0348 | 2 | 0 | 0.623 | 0.124 | 0.101 | 252.4 | 1268 |
| DPM2 + gauss_jacobian (10 correctors) | 0.2766 | 0.0373 | 2 | 0 | 0.567 | 0.126 | 0.098 | 726.5 | 15216 |

_Seconds and score calls overstate the cost of any arm whose draws had to be chunked to fit in memory (the 5.2M-parameter `gauss_jacobian` arms); the sampled statistics are unaffected._

### control / learned

| arm | W1(g)/sigma | W1(l)/sigma | modes | gap draws | P(g>0) | mode s.d. | corr(g,l) | seconds | score calls |
|---|---|---|---|---|---|---|---|---|---|
| DPM2 + churn 8 + gauss_hierarchical | 0.2279 | 0.0333 | 2 | 0 | 0.601 | 0.106 | 0.117 | 43.9 | 198 |
| DPM2 + churn 8 + gauss_jacobian | 0.2556 | 0.0293 | 2 | 0 | 0.580 | 0.117 | 0.111 | 19.0 | 594 |
| Langevin + F-NPSE | 0.1944 | 0.1064 | 2 | 0 | 0.624 | 0.113 | 0.163 | 31.2 | 1000 |
| DPM2 + gauss_hierarchical (10 correctors) | 0.2178 | 0.1532 | 2 | 0 | 0.609 | 0.089 | 0.174 | 78.7 | 1268 |
| DPM2 + gauss_jacobian (10 correctors) | 0.3110 | 0.1458 | 2 | 0 | 0.546 | 0.096 | 0.178 | 134.7 | 3804 |

_Seconds and score calls overstate the cost of any arm whose draws had to be chunked to fit in memory (the 5.2M-parameter `gauss_jacobian` arms); the sampled statistics are unaffected._

### sign: exact targets

modes = 2; P(g>0) = 0.500; gap draws = 0.07 of 3000; mode s.d. = 0.199; corr(g,l) = 0.962; sign coherence = 0.965

### sign / analytic

| arm | W1(g)/sigma | W1(l)/sigma | modes | gap draws | P(g>0) | mode s.d. | corr(g,l) | seconds | score calls |
|---|---|---|---|---|---|---|---|---|---|
| DPM2 + churn 8 + gauss_hierarchical | 0.0749 | 0.0243 | 2 | 76 | 0.499 | 0.300 | 0.929 | 1.9 | 198 |
| DPM2 + churn 8 + gauss_jacobian | 0.0435 | 0.0293 | 2 | 34 | 0.516 | 0.260 | 0.943 | 8.1 | 594 |
| Langevin + F-NPSE | 0.0707 | 0.0346 | 2 | 68 | 0.512 | 0.298 | 0.930 | 2.9 | 1000 |
| DPM2 + gauss_hierarchical (10 correctors) | 0.0199 | 0.0223 | 2 | 2 | 0.510 | 0.211 | 0.959 | 6.2 | 1268 |
| DPM2 + gauss_jacobian (10 correctors) | 0.0129 | 0.0089 | 2 | 4 | 0.498 | 0.217 | 0.957 | 26.2 | 3804 |

_Seconds and score calls overstate the cost of any arm whose draws had to be chunked to fit in memory (the 5.2M-parameter `gauss_jacobian` arms); the sampled statistics are unaffected._

### sign / learned_large

| arm | W1(g)/sigma | W1(l)/sigma | modes | gap draws | P(g>0) | mode s.d. | corr(g,l) | seconds | score calls |
|---|---|---|---|---|---|---|---|---|---|
| DPM2 + churn 8 + gauss_hierarchical | 0.0808 | 0.0655 | 3 | 74 | 0.527 | 0.297 | 0.934 | 136.0 | 198 |
| DPM2 + churn 8 + gauss_jacobian | 0.0559 | 0.0432 | 2 | 40 | 0.516 | 0.256 | 0.946 | 119.9 | 2376 |
| Langevin + F-NPSE | 0.1467 | 0.1406 | 3 | 65 | 0.564 | 0.300 | 0.939 | 153.9 | 1000 |
| DPM2 + gauss_hierarchical (10 correctors) | 0.1899 | 0.1751 | 2 | 3 | 0.582 | 0.175 | 0.969 | 296.1 | 1268 |
| DPM2 + gauss_jacobian (10 correctors) | 0.1268 | 0.1157 | 2 | 6 | 0.549 | 0.177 | 0.970 | 1060.4 | 15216 |

_Seconds and score calls overstate the cost of any arm whose draws had to be chunked to fit in memory (the 5.2M-parameter `gauss_jacobian` arms); the sampled statistics are unaffected._

### sign / learned

| arm | W1(g)/sigma | W1(l)/sigma | modes | gap draws | P(g>0) | mode s.d. | corr(g,l) | seconds | score calls |
|---|---|---|---|---|---|---|---|---|---|
| DPM2 + churn 8 + gauss_hierarchical | 0.1116 | 0.0696 | 2 | 24 | 0.465 | 0.258 | 0.957 | 44.5 | 198 |
| DPM2 + churn 8 + gauss_jacobian | 0.0901 | 0.0457 | 2 | 6 | 0.521 | 0.226 | 0.963 | 22.0 | 594 |
| Langevin + F-NPSE | 0.1335 | 0.0989 | 2 | 9 | 0.457 | 0.218 | 0.973 | 30.9 | 1000 |
| DPM2 + gauss_hierarchical (10 correctors) | 0.1851 | 0.1453 | 2 | 0 | 0.437 | 0.163 | 0.980 | 80.2 | 1268 |
| DPM2 + gauss_jacobian (10 correctors) | 0.0932 | 0.0570 | 2 | 0 | 0.492 | 0.164 | 0.981 | 124.6 | 3804 |

_Seconds and score calls overstate the cost of any arm whose draws had to be chunked to fit in memory (the 5.2M-parameter `gauss_jacobian` arms); the sampled statistics are unaffected._
