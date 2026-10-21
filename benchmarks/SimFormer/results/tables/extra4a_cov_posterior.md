# extra4a: cov_posterior — mean(coverage − level); > 0 conservative, < 0 overconfident; |.| = mean absolute calibration error

| task | sims | Simformer (official) signed / abs | COMPASS (eps bug fixed) signed / abs |
|---|---|---|---|
| Gaussian Linear | 1,000 | +0.161 / 0.161 | — |
| Gaussian Linear | 10,000 | +0.096 / 0.096 | — |
| Gaussian Linear | 100,000 | +0.004 / 0.005 | — |
| Gaussian Mixture | 1,000 | +0.040 / 0.040 | +0.080 / 0.080 |
| Gaussian Mixture | 10,000 | -0.002 / 0.005 | +0.011 / 0.045 |
| Gaussian Mixture | 100,000 | -0.005 / 0.006 | +0.032 / 0.032 |
| Two Moons | 1,000 | +0.064 / 0.064 | +0.074 / 0.074 |
| Two Moons | 10,000 | +0.016 / 0.016 | +0.026 / 0.026 |
| Two Moons | 100,000 | -0.005 / 0.006 | +0.041 / 0.041 |
| SLCP | 1,000 | +0.162 / 0.162 | +0.428 / 0.428 |
| SLCP | 10,000 | +0.034 / 0.034 | +0.067 / 0.067 |
| SLCP | 100,000 | -0.010 / 0.010 | -0.005 / 0.010 |
