# extra4a_nll_posterior: NLL of the true θ under the posterior (lower is better)

Mean ± s.e.m. over the evaluated observations/conditionals. Δ = method − Simformer, paired over the same observations; **bold** where |Δ| > 2 s.e.m.

| task | sims | Simformer (official) | COMPASS (eps bug fixed) | Δ COMPASS (eps bug fixed) |
|---|---|---|---|---|
| Gaussian Linear | 1,000 | 0.492 ± 0.047 | — | — |
| Gaussian Linear | 10,000 | -0.200 ± 0.042 | — | — |
| Gaussian Linear | 100,000 | -0.746 ± 0.032 | — | — |
| Gaussian Mixture | 1,000 | 1.505 ± 0.035 | 1.663 ± 0.040 | **+0.159 ± 0.053** |
| Gaussian Mixture | 10,000 | 1.245 ± 0.034 | 1.146 ± 0.034 | **-0.098 ± 0.048** |
| Gaussian Mixture | 100,000 | 1.181 ± 0.033 | 1.173 ± 0.037 | -0.008 ± 0.050 |
| Two Moons | 1,000 | -2.968 ± 0.022 | -3.076 ± 0.016 | **-0.108 ± 0.027** |
| Two Moons | 10,000 | -3.652 ± 0.014 | -3.691 ± 0.010 | **-0.040 ± 0.018** |
| Two Moons | 100,000 | -3.752 ± 0.012 | -3.738 ± 0.012 | +0.014 ± 0.017 |
| SLCP | 1,000 | 5.896 ± 0.051 | 9.803 ± 0.069 | **+3.906 ± 0.085** |
| SLCP | 10,000 | 3.056 ± 0.052 | 3.084 ± 0.047 | +0.028 ± 0.069 |
| SLCP | 100,000 | 2.346 ± 0.052 | 2.405 ± 0.045 | +0.059 ± 0.069 |
