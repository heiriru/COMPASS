# Shared-posterior figures for the exact marginal product

Produced by `inverse_kinematics/experiments/plot_marginal_product.py` from the
saved arrays in `artifacts/composition_attribution/*/` — no model and no
sampling, so these figures are of exactly the numbers in
`docs/nonsymmetric_composition.md` rather than of a fresh run that would differ
by Monte Carlo noise. Regenerate with:

    python inverse_kinematics/experiments/plot_marginal_product.py

| figure | what it shows |
|---|---|
| `01_global_posteriors.png` | `p(angle_1 \| x_1..x_n)` from each method over the ABC reference, one panel per dataset, W1/sigma in the legend |
| `02_product_construction.png` | how the product is built: eight individually near-uninformative single-observation posteriors, then their product |
| `03_diagnostics.png` | branch mass, width ratio, W1/sigma and the locals' endpoint error across all five datasets |

## Reading them

**Figure 01, panels 1-3 (the asymmetric cases).** The reference is lopsided —
0.24 of its mass on the positive branch in panel 1 — and both Gaussian rules
return a broad, near-symmetric density with a filled-in valley between the
modes. The marginal product tracks the reference's branch weights and its
valley. Panel 3 (n=16) shows the rules getting *worse* with more data, which is
the wrong direction.

**Figure 02.** The top panel is the point. Each single-observation posterior is
broad and nearly symmetric: on its own, no observation knows which branch is
right. The information about the branch lives in the *product* of eight slight
preferences, and a mean-plus-covariance summary of each observation is exactly
what discards it.

**Figure 01, panel 4 (`x1 = 0`, symmetric control) is where the new method
loses**, and it is in the folder for that reason. The network carries a
systematic +0.028 single-obs-sd bias toward the true branch; at n=8 the exact
product amplifies it to a branch mass of 0.77 against a true 0.50, while the
rules' symmetrising bias accidentally hides it. Panel 5 (sharp unimodal
posterior) is a wash.

**Figure 03, bottom panel** is the one result that holds everywhere: the locals'
endpoint error is 0.014-0.024 for the marginal product against 0.066-0.084 for
every composition rule, *including* the two datasets where it loses on the
shared coordinate. This metric is reference-free — the forward map is
deterministic, so a correct draw reproduces its own observation.
