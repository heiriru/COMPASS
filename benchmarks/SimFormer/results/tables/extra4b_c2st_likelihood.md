# extra4b_c2st_likelihood: C2ST of the likelihood p(x | θ) (lower is better)

Mean ± s.e.m. over the evaluated observations/conditionals. Δ = method − Simformer, paired over the same observations; **bold** where |Δ| > 2 s.e.m.

| task | sims | Simformer (official) | COMPASS (eps bug fixed) | Δ COMPASS (eps bug fixed) |
|---|---|---|---|---|
| Tree (all cond.) | 1,000 | 0.673 ± 0.006 | 0.709 ± 0.010 | **+0.036 ± 0.010** |
| Tree (all cond.) | 10,000 | 0.551 ± 0.003 | 0.526 ± 0.003 | **-0.025 ± 0.003** |
| Tree (all cond.) | 100,000 | 0.502 ± 0.002 | 0.501 ± 0.001 | -0.000 ± 0.002 |
| HMM (all cond.) | 1,000 | 0.660 ± 0.013 | 0.857 ± 0.010 | **+0.196 ± 0.009** |
| HMM (all cond.) | 10,000 | 0.556 ± 0.007 | 0.581 ± 0.008 | **+0.025 ± 0.004** |
| HMM (all cond.) | 100,000 | 0.504 ± 0.002 | 0.514 ± 0.002 | **+0.009 ± 0.002** |
| Two Moons (all cond.) | 1,000 | 0.613 ± 0.007 | 0.644 ± 0.007 | **+0.031 ± 0.007** |
| Two Moons (all cond.) | 10,000 | 0.518 ± 0.002 | 0.537 ± 0.002 | **+0.019 ± 0.002** |
| Two Moons (all cond.) | 100,000 | 0.506 ± 0.001 | 0.521 ± 0.002 | **+0.014 ± 0.002** |
| SLCP (all cond.) | 1,000 | 0.873 ± 0.017 | 0.901 ± 0.011 | **+0.028 ± 0.010** |
| SLCP (all cond.) | 10,000 | 0.817 ± 0.018 | 0.664 ± 0.019 | **-0.153 ± 0.020** |
| SLCP (all cond.) | 100,000 | 0.802 ± 0.017 | 0.625 ± 0.020 | **-0.177 ± 0.021** |
