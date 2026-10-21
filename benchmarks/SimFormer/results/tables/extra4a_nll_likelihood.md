# extra4a_nll_likelihood: NLL of the true x under the likelihood (lower is better)

Mean ± s.e.m. over the evaluated observations/conditionals. Δ = method − Simformer, paired over the same observations; **bold** where |Δ| > 2 s.e.m.

| task | sims | Simformer (official) | COMPASS (eps bug fixed) | Δ COMPASS (eps bug fixed) |
|---|---|---|---|---|
| Gaussian Linear | 1,000 | 3.664 ± 0.043 | — | — |
| Gaussian Linear | 10,000 | 3.033 ± 0.039 | — | — |
| Gaussian Linear | 100,000 | 2.713 ± 0.032 | — | — |
| Gaussian Mixture | 1,000 | 1.578 ± 0.036 | 1.732 ± 0.041 | **+0.154 ± 0.054** |
| Gaussian Mixture | 10,000 | 1.335 ± 0.035 | 1.215 ± 0.035 | **-0.120 ± 0.049** |
| Gaussian Mixture | 100,000 | 1.308 ± 0.035 | 1.184 ± 0.038 | **-0.124 ± 0.051** |
| Two Moons | 1,000 | -3.566 ± 0.021 | -3.725 ± 0.018 | **-0.159 ± 0.028** |
| Two Moons | 10,000 | -4.234 ± 0.013 | -4.279 ± 0.011 | **-0.045 ± 0.017** |
| Two Moons | 100,000 | -4.321 ± 0.010 | -4.282 ± 0.012 | **+0.039 ± 0.016** |
| SLCP | 1,000 | 14.430 ± 0.134 | 16.699 ± 0.108 | **+2.268 ± 0.175** |
| SLCP | 10,000 | 11.942 ± 0.152 | 11.551 ± 0.122 | **-0.390 ± 0.193** |
| SLCP | 100,000 | 11.229 ± 0.155 | 10.663 ± 0.127 | **-0.565 ± 0.202** |
