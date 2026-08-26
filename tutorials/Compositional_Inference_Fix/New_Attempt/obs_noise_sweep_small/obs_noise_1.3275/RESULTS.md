# MAP comparison: gauss_jacobian + Newton vs. gauss_hierarchical + Tweedie vs. Langevin/F-NPSE KDE

Case: `obs_noise_sweep_small/obs_noise_1.3275/` (S0L=1, SXH=1.3275,
model_size='small', 30 real observations, checkpoint and
`reference.npz` reused unchanged from the existing obs_noise sweep). All three
methods are scored against the *analytic* joint posterior in `reference.npz`
(`exact_joint_mean` / `exact_joint_covariance`) -- no oracle moments are fed
into any sampler or MAP estimator.

## Methods

1. **New (pilot-free)**: DPM-Solver-2, 50 timesteps, `correction="gauss_jacobian"`
   posterior draws (deterministic, no Langevin correctors), plus
   `MultiObsSampler.newton_map_estimate(correction="gauss_jacobian", curvature="jacobian")`.
   Both the composition weights and the Newton curvature come from the
   network's own forward-mode Jacobian -- no pilot DDIM pass at all.
2. **Old**: reuses the existing `gauss_hierarchical_dpm50_deterministic.npz`
   posterior draws unchanged; MAP via the Tweedie-ascent
   `MultiObsSampler.map_estimate(correction="gauss_hierarchical", ...)`, which
   needs a pilot single-observation covariance. That covariance is not saved
   in the existing `.npz` (only diagnostics are), so it was rebuilt here with
   the network's own automatic pilot-estimation routine
   (`estimate_posterior_moments`, 4096 draws x
   100 steps -- the same automatic procedure
   `model.sample(correction="gauss_hierarchical", ...)` runs internally, and
   the same sample count used when the plotted posterior was produced).
3. **Langevin + F-NPSE**: reuses the existing `langevin_fnpe.npz` posterior
   draws unchanged. F-NPSE's bridging score has no reverse-diffusion MAP
   objective, so its "MAP" is the KDE mode of the posterior global samples,
   computed the same way as `Partial_Pooling/infer_partial_pooling.py::_fnpse_kde_map`
   (adapted for this toy problem's single global scalar).

## Results

| Method | Global error (sigma) | Local mean \|error\| (sigma) | Local max \|error\| (sigma) | MAP runtime (s) | Score-network calls |
|---|---|---|---|---|---|
| gauss_jacobian + Newton (new) | 1.026 | 0.098 | 0.342 | 63.92 | 3008 |
| gauss_hierarchical + Tweedie (old) | 0.812 | 0.060 | 0.241 | 42.72 | 7272 |
| Langevin + F-NPSE (KDE mode) | 0.138 | 0.667 | 2.102 | 0.23 | 0 |

Full table: `map_method_comparison.csv`. Bar-chart comparison:
`05_map_method_comparison.png`. New method's posterior dashboard:
`04_gauss_jacobian_dpm50_deterministic.png`
(sampling runtime 84.6 s for
3000 draws x 30 observations, 50 steps).

## Interpretation

On this real trained network and these 30 real observations
(no oracle moments anywhere), the new pilot-free `gauss_jacobian` + Newton
combination does not beat the old
`gauss_hierarchical` + Tweedie MAP on the global parameter
(1.026 vs 0.812
sigma) and does not beat it on the
local parameters (0.098 vs
0.060 sigma mean error), while using
3008 vs 7272 score-network
calls (old total includes the 4096-draw pilot pass Tweedie
needs and Newton does not). Against the Langevin/F-NPSE KDE-mode baseline, the
new method does not beat it on
the global parameter (1.026 vs
0.138 sigma) and
beats it on the locals
(0.098 vs
0.667 sigma). The KDE-mode
"MAP" is fundamentally a different object -- a discrete selection among
existing posterior draws rather than a continuous optimum -- so it is
expected to be coarser regardless of the underlying sampler's quality.
