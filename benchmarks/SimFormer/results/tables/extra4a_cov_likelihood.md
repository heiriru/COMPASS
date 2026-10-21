# extra4a: cov_likelihood — mean(coverage − level); > 0 conservative, < 0 overconfident; |.| = mean absolute calibration error

| task | sims | Simformer (official) signed / abs | COMPASS (eps bug fixed) signed / abs |
|---|---|---|---|
| Gaussian Linear | 1,000 | +0.108 / 0.108 | — |
| Gaussian Linear | 10,000 | +0.052 / 0.052 | — |
| Gaussian Linear | 100,000 | -0.005 / 0.006 | — |
| Gaussian Mixture | 1,000 | +0.043 / 0.043 | +0.065 / 0.068 |
| Gaussian Mixture | 10,000 | -0.002 / 0.005 | +0.022 / 0.050 |
| Gaussian Mixture | 100,000 | -0.009 / 0.009 | +0.036 / 0.036 |
| Two Moons | 1,000 | +0.061 / 0.061 | +0.094 / 0.094 |
| Two Moons | 10,000 | +0.022 / 0.022 | +0.047 / 0.047 |
| Two Moons | 100,000 | +0.001 / 0.004 | +0.060 / 0.060 |
| SLCP | 1,000 | +0.020 / 0.020 | +0.179 / 0.179 |
| SLCP | 10,000 | +0.009 / 0.010 | +0.005 / 0.017 |
| SLCP | 100,000 | +0.002 / 0.003 | -0.003 / 0.012 |
