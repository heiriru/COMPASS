# extra4b_c2st_joint: C2ST of the joint p(θ, x) (lower is better)

Mean ± s.e.m. over the evaluated observations/conditionals. Δ = method − Simformer, paired over the same observations; **bold** where |Δ| > 2 s.e.m.

| task | sims | Simformer (official) | COMPASS (eps bug fixed) | Δ COMPASS (eps bug fixed) |
|---|---|---|---|---|
| Tree (all cond.) | 1,000 | 0.577 ± 0.001 | 0.583 ± 0.001 | **+0.006 ± 0.001** |
| Tree (all cond.) | 10,000 | 0.510 ± 0.001 | 0.512 ± 0.001 | +0.001 ± 0.001 |
| Tree (all cond.) | 100,000 | 0.501 ± 0.001 | 0.502 ± 0.001 | +0.001 ± 0.001 |
| HMM (all cond.) | 1,000 | 0.551 ± 0.001 | 0.722 ± 0.001 | **+0.171 ± 0.001** |
| HMM (all cond.) | 10,000 | 0.508 ± 0.001 | 0.525 ± 0.001 | **+0.017 ± 0.002** |
| HMM (all cond.) | 100,000 | 0.502 ± 0.001 | 0.505 ± 0.001 | **+0.004 ± 0.001** |
| Two Moons (all cond.) | 1,000 | 0.529 ± 0.001 | 0.549 ± 0.001 | **+0.020 ± 0.001** |
| Two Moons (all cond.) | 10,000 | 0.504 ± 0.001 | 0.507 ± 0.001 | **+0.003 ± 0.001** |
| Two Moons (all cond.) | 100,000 | 0.501 ± 0.001 | 0.504 ± 0.001 | +0.003 ± 0.002 |
| SLCP (all cond.) | 1,000 | 0.603 ± 0.001 | 0.712 ± 0.001 | **+0.108 ± 0.001** |
| SLCP (all cond.) | 10,000 | 0.546 ± 0.001 | 0.527 ± 0.001 | **-0.019 ± 0.001** |
| SLCP (all cond.) | 100,000 | 0.539 ± 0.001 | 0.509 ± 0.001 | **-0.030 ± 0.001** |
