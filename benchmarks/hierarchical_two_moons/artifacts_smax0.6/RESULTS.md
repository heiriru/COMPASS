# Hierarchical Two Moons, small-σ regime (max σ ≤ 0.6) — findings

Test datasets: prior conditioned on max(σ_0, σ_1) ≤ 0.6 (`./run.sh run --sigma-max 0.6`); networks
unchanged (trained on the full prior σ ~ U(0.1, 3)). 10 datasets per N, 500 draws per method.
Reference: collapsed SMC-MCMC, mirror-symmetrized (exact), checked against an independent 4x larger
run (C2ST 0.48–0.54, `checks/check_reference_smax.py`). Full tables: `TABLES.md`.

## The global posterior is bimodal — by an exact symmetry

x is unchanged under θ_i → (−θ_i1, −θ_i0), so the posterior is exactly invariant under
g → (−μ_1, −μ_0, σ_1, σ_0). In u = (μ_0+μ_1)/√2 and w = σ_0−σ_1 the two modes sit at ±truth;
v = (μ_1−μ_0)/√2 and s = σ_0+σ_1 are invariant. With small σ the reference shows two well-separated
modes in u from N≈10 and in w from N≈50 (`figures/bimodality/reference_overview.png`). The 1-D
marginals of μ_0 or μ_1 mix the truth with the other mode's −μ_1 and hide it.

## Findings (medians over 10 datasets; reference = Monte-Carlo floor)

| metric | reference | best learned arm(s) | N where all learned arms fail |
|---|---|---|---|
| local C2ST | 0.50 | 0.52–0.55 (gauss_hierarchical, gauss_jacobian*, attn_*:gauss_*, bayesflow) | none — locals OK except `gauss` (0.95) and Arruda/Schur/Laplace on attention nets (0.61–0.66, crescent 2x too wide) |
| global C2ST | 0.49–0.51 | 0.62–0.65 at N=5, 0.75–0.83 at N=10 | ≥ 10: every arm ≥ 0.75, ≥ 0.86 at N ≥ 25 |
| mirror-invariant W1/sd | 0.06 | 0.3–0.4 at N=5, 0.6–0.7 at N=10 | ≥ 25: 1–11 |
| mode balance err | 0.02 | ≤ 0.05 up to N=25 for gauss_jacobian(_FD), attn_full:gauss_jacobian_FD | 0.05–0.37 at N=100 |
| dependence |Δcorr| | 0.03 | 0.07–0.10 at N=5 | ≥ 10: 0.25–0.5 |

What goes wrong at N ≥ 10 (`figures/bimodality/joint_N*_ds*.png`, `mirror_N*_ds*_*.png`):
* no learned arm resolves the two sharp modes; draws form a broad cloud spanning both modes or a bridge between them;
* σ is far too wide for the legacy-net rules and BayesFlow (mass up to σ = 1–3 when the reference
  has σ ≤ 0.6), while the attention-net Arruda / Jacobian arms pile up at the σ = 0.1 prior edge;
* `attn_full:arruda_schur` collapses onto the prior box edges.

Possible cause (not tested): max σ ≤ 0.6 is ~3% of the training prior's mass, and at small σ the
per-group scores must be composed very precisely to give a sharp product.

## Importance sampling on the globals (exact likelihood) — `checks/importance_sampling.py`

Proposal = Gaussian KDE (probit z) of each method's 500 global draws + 10% broad Gaussian + 5% prior,
optionally mirror-symmetrized; 20k proposals weighted with the exact p(x|g)p(g); 500 resampled.
Medians over 10 datasets (`importance_sampling.json`). Best proposals (gauss_hierarchical,
gauss_jacobian_FD, attn_full:gauss_hierarchical, bayesflow_arruda_diffprior):

| N | ESS / proposals | global C2ST after IS (raw method) | mirror-inv W1 after IS (raw) |
|---|---|---|---|
| 5 | 26–37% | 0.51–0.53 (0.62–0.65) | 0.05–0.07 (0.3–0.4) |
| 10 | 1.6–4.6% | 0.51–0.55 (0.75–0.83) | 0.07–0.09 (0.6–0.7) |
| 25 | 0.4–1.7% | 0.56–0.66 (0.86–0.95) | 0.10–0.14 (1.7–2.5) |
| 50 | 0.9–2.6% | 0.53–0.58 (0.87–0.94) | 0.09–0.14 (1.0–1.4) |
| 100 | 0.04–0.15% | 0.76–0.88 (0.96–0.99) | 0.19–0.45 (2.1–3.1) |

The prior as proposal: 1% (N=5) → 0.02% (N=100). Mirror symmetrization helps only slightly.
IS repairs the global posterior up to N≈50 but needs ~1/ESS-fraction more proposals (20–250x
from N=10 on); at N=100 the proposals are too wide (σ) for one-shot IS to work.

## Network or rule? Score check (`checks/score_check.py`) and two network fixes

Per-group global score of the attn_hier nets vs the exact Tweedie-IS oracle (agrees with autograd to
~10%), at reference posterior draws + noise, σ ≤ 0.6 datasets. rel_err = ||s_net − s*|| / ||likelihood part||,
median; independent of N. The summed error over N groups exceeds the exact composed signal for λ ≤ 0.2:
**the network, not the rule, is the limit.**

| λ | attn_hier | attn_hier_w1.5 (1.5x wider global training prior) | attn_hier_tsm (target score matching on globals) |
|---|---|---|---|
| 0.02 | 0.89 | 0.73–0.77 | 0.83–0.91 |
| 0.05 | 0.80–0.86 | 0.55–0.60 | 0.66–0.77 |
| 0.2 | 0.53–0.58 | 0.32–0.36 | 0.45–0.53 |
| 1.0 | 0.33–0.35 | 0.20–0.22 | 0.30–0.34 |

Benchmark (global C2ST at N = 5/10/25/50): attn_hier:gauss_jacobian_FD 0.65/0.80/0.97/0.99,
_tsm 0.63/0.75/0.94/0.96, _w1.5 0.59/0.74/0.89/0.93; legacy_w1.5 rules also improve at N=25/50.
Both fixes help; neither brings the globals close to the reference (0.49–0.51).
