# True compositional GAUSS for hierarchical (global + local) SBI

## 1. What was wrong

`compare_shared_local_composition_methods.py` produces
`02f_dpm2_gauss_global_local.png` and `02g_dpm2_gauss_moment.png` using
`correction="Gauss_global_local"` together with `global_posterior_mean`,
`global_posterior_covariance` and (for 02g) `posterior_mean`. These moments
come from `build_single_observation_covariance_bank`, which draws them from

```python
joint_precision = np.array([[5.0, 4.0], [4.0, 5.0]])
```

This is not an estimate — it is the **exact analytic** posterior precision of
`p(g, l_j | x_j)` for the toy model `g ~ N(0,1)`, `l_j ~ N(0,1)`,
`x_j = g + l_j + N(0, 0.5^2)` (prior precision `I` plus a rank-1 likelihood
update `4·[[1,1],[1,1]]` from `1/0.5^2`). The trained score network is barely
involved: `MultiObsSampler._marginal_global_scores` uses this externally
supplied mean/covariance to evaluate a *closed-form Gaussian score*
`-Σ_t⁻¹(θ_t - mean)` directly, **discarding the network's actual score
output** for the shared coordinate. 02g goes further and does the same
substitution for every coordinate via `_moment_projected_scores`. Both
figures are therefore showing an analytic Gaussian projection dressed up as a
diffusion sampler result, not compositional score modeling.

## 2. The paper's actual algorithm (GAUSS, Linhart et al. 2024)

For a *single*, fully shared parameter block `θ` (no per-observation locals),
the paper derives (Lemma 3.1 / 3.2) that the diffused tall-data posterior
score is well approximated, using a constant-in-`θ` Gaussian/Tweedie
covariance `Σ_{t,j}` per observation, by

```
s(θ_t) ≈ Λ(t)⁻¹ [ Σⱼ Σ_{t,j}⁻¹ sⱼ(θ_t) + (1-n) Σ_{t,λ}⁻¹ s_λ(θ_t) ]
Λ(t)  = Σⱼ Σ_{t,j}⁻¹ + (1-n) Σ_{t,λ}⁻¹
```

`sⱼ` is the network's **real** score at observation `j`; only the weighting
covariance `Σ_{t,j} = (Σ_{0,j}⁻¹ + (α_t/υ_t) I)⁻¹` is a Gaussian
approximation, and `Σ_{0,j}` is estimated by running the trained network's own
DDIM sampler on `x_j` and taking the empirical covariance of the draws
("GAUSS", Algorithm 2). COMPASS already implements this correctly and
without any oracle as `correction="gauss"`/`"full_gaussian"`.

## 3. Generalizing to a global + local hierarchy

Now `θ = (g, l_1, ..., l_n)`: `g` is shared, `l_j` is private to observation
`j`. The tall posterior factorizes as
`p(g,l_{1:n}|x_{1:n}) ∝ λ(g)^{1-n} ∏ⱼ p(g,l_j|x_j)`, and the trained network
gives the *joint* score `sⱼ = (sⱼ[g], sⱼ[l])` for each observation's `(g,l_j)`
block. The question is how to combine the global components across
observations without discarding the local information or the network's real
score.

**Naive idea (rejected):** treat `p(g|x_j) := ∫ p(g,l_j|x_j) dl_j` as "the"
single-observation posterior and apply GAUSS directly to it. This needs the
*marginal* score `∇_g log p_t(g_t|x_j)`, which is **not** what the network's
`sⱼ[g]` gives (that is the *conditional* derivative at the current `l_{j,t}`).
Substituting a marginal Gaussian score built from pilot moments (mean +
covariance) — exactly what `Gauss_global_local`'s oracle path does — is the
cheat identified above, whether the moments come from an oracle or from a
real DDIM estimate.

**Correct approach: don't marginalize — solve the full block system.**
Under GAUSS's own Tweedie-Gaussian approximation (`Σ_{t,j}` constant in `θ`,
which is what makes the paper's correction term `F` vanish), the *joint*
target `p_t(g,l_{1:n}|x_{1:n})` is itself approximately Gaussian with an
**arrow-shaped precision matrix**: `g` couples to every `l_j`, but the `l_j`
are mutually independent given `g` and untouched by the `(1-n)` prior
correction (each `l_j`'s own prior contributes to exactly one factor, so
there is no over-counting to correct). Writing `Pⱼ = Σ_{0,j}⁻¹ + (α_t/υ_t) I`
(the network-only, no-mean-needed GAUSS Tweedie precision of observation
`j`'s joint `(g,l_j)` block) and `Pλ` for the (diffused) prior precision on
`g`, the exact composed score solves the block-arrow linear system

```
Λ_gg = (1-n) Pλ + Σⱼ Pⱼ[g,g]          Λ_{g,lⱼ} = Pⱼ[g,l]         Λ_{lⱼ,lⱼ} = Pⱼ[l,l]
[Λ] · [composed score] = [ (1-n) Pλ s_λ + Σⱼ Pⱼ sⱼ  (embedded) ]
```

Eliminating the (block-diagonal, independent) local rows via a Schur
complement gives, after simplification (the algebra is in
`tests/test_gauss_hierarchical_composition.py::test_gauss_hierarchical_matches_dense_arrowhead_reference`):

```
Λⱼ(t)      = ( Pⱼ[g,g] - Pⱼ[g,l] Pⱼ[l,l]⁻¹ Pⱼ[l,g] )      # marginal precision of g_t | x_j
             = Var(g_t | x_j)⁻¹                            # = the (g,g) BLOCK of the covariance Σ_{t,j}, inverted

composed_g  = ( Σⱼ Λⱼ(t) + (1-n) Λλ(t) )⁻¹ [ Σⱼ Λⱼ(t) sⱼ[g] + (1-n) Λλ(t) s_λ ]   ← exactly GAUSS/Algorithm 2

composed_lⱼ = sⱼ[l] + ( Pⱼ[l,l]⁻¹ Pⱼ[l,g] ) · ( sⱼ[g] - composed_g )
```

The remarkable (and reassuring) simplification: **the `l`-dependent terms in
the numerator cancel exactly**, so the composed global score only needs the
network's real conditional score `sⱼ[g]` (not a marginal one) weighted by the
marginal precision `Λⱼ(t)`, and the local score is the network's real local
score plus a linear correction proportional to how much the shared
composition just moved `g` away from observation `j`'s own opinion. This is
*exactly* what COMPASS's existing `Gauss_schur_global` (global part) +
`Gauss_global_local`'s cross-term (local part) already compute, **as long as
they are never handed `global_posterior_mean`/`posterior_mean`** — i.e. as
long as they stay in the `global_scores = scores[:, :, hierarchy]` branch of
`_compositional_score`, which uses the network's real output.

Sanity checks (all covered by tests):
- `n=1`: the arrow system collapses to `Pⱼ`, so the composed score exactly
  equals the network's raw score — no correction, as required.
- No local dimensions (`m_l=0`): the system collapses to the ordinary GAUSS
  formula, matching `"full_gaussian"`/`"gauss"` exactly.
- Zero cross-covariance `Cov(g,l_j)=0`: the local correction vanishes, and the
  global part matches the single-block ("full_gaussian") reduction — already
  covered by `test_zero_cross_covariance_schur_matches_full_gaussian`.

## 4. What was implemented

`correction="gauss_hierarchical"` in `MultiObsSampler` (`src/compass/MultiObsSampler.py`):

- Reuses all existing Schur/marginal-covariance machinery (`_effective_global_factors`,
  `estimate_posterior_moments`) — the pilot `Σ_{0,j}` is estimated from the
  trained network's own DDIM draws, exactly as GAUSS prescribes. No pilot
  *mean* is ever estimated or used.
- Always takes the real-score branch of `_compositional_score`
  (`global_scores = scores[:, :, hierarchy]`), and always applies the local
  cross-correction (added `LOCAL_CROSS_CORRECTIONS`).
- The API refuses `posterior_mean` and `global_posterior_mean` outright
  (`NO_MOMENT_SUBSTITUTION_CORRECTIONS`), in both `sample()` and
  `map_estimate()`, so the "moment projection" cheat is structurally
  impossible to reintroduce for this correction by accident.
- `"Gauss_global_local"` itself is left untouched (including its
  oracle-mean shortcut) since it is used elsewhere as a deliberate ablation
  (what does an *exact* Gaussian projection look like); its docstring now
  says explicitly that this is not compositional score modeling.

See `tests/test_gauss_hierarchical_composition.py` for the arrowhead-matrix
cross-check, the n=1/no-locals reductions, and the API guards.

## 5. Experiment

`run_gauss_hierarchical_experiment.py` reuses the exact checkpoint (
`shared_local_mixture_full`), the 30 real observations, and the exact-joint
reference truth from `06b_shared_local/raw_plot_data.npz` — the same inputs
`02a`–`02g` were generated from — and runs `gauss_hierarchical` with
covariances estimated from the trained network's own draws
(`posterior_covariance=None`, `precision_est_samples=512`). Two settings are
run: dense Langevin correctors (comparable to 02a/02f) and a fully
deterministic zero-corrector DPM-2 sampler (comparable to 02g). `02g`'s own
`.npz` (missing from disk, only its plot survived) is regenerated unmodified
for a real number to compare against, and `02a`/`02c`/`02f`'s already-computed
results are reused unchanged. See `README.md` for the resulting numbers.
