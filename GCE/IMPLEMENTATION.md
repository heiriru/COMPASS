# Is the Milky Way's high-mass IMF universal? Implementation plan

Status 2026-10-29. This turns `PLAN.md` (the experiment design) into concrete work packages.
Each package has the code to write, the runs, a gate that must pass before the next package
starts, and a cost estimate.

Nothing scientific has been run yet. The code in `gce/` is wiring-tested at toy size only.

## The question, stated so that it can be answered

> Do the abundances and ages of Milky Way disc stars require a high-mass IMF slope that varies
> with the stars' birth conditions? If not, how large a variation do they exclude?

Three forms of variation, one model each (`gce/models.py`), all nested in the universal IMF:

| model | parameters | null | what it measures |
|---|---|---|---|
| `universal` | α, log N_Ia | — | the baseline: one slope for all stars |
| `pooled` | μ_α, ln σ_α, log N_Ia; per-star α_i | σ_α < 0.02 | any birth-environment dependence, whatever its driver |
| `time` | α₀, α_t, log N_Ia | α_t = 0 | top-heavy early (what density-dependent prescriptions such as Durrant et al. 2026 imply, since early star formation was denser) |
| `metallicity` | α₀, α_Z, log N_Ia | α_Z = 0 | top-heavy at low Z (IGIMF-type claims) |

**Primary result:** the posteriors of σ_α, α_t and α_Z, as measured values or 95% upper limits.

**Secondary result:** Bayes factors against the universal IMF (Savage–Dickey / interval null),
reported only together with
- their calibrated false-positive rate (WP5);
- their prior-width sensitivity.

**Negative controls:** `universal_kroupa`, `universal_salpeter`. Abundances should not
separate low-mass IMF shapes.

## Decision rule, fixed before any real data is seen

We claim "the IMF varies" only if all of the following hold:
1. the 95% credible interval of the variation parameter excludes the null, for all 3 training
   seeds;
2. the log Bayes factor exceeds the threshold at which the false-positive rate in WP5 is ≤ 5%;
3. the verdict survives:
   - 2 of the 3 yield sets (WP7);
   - a systematic floor σ_sys = 0.05 dex (WP6);
   - 3 random star subsamples of the survey.

Otherwise we report upper limits, and the same five checks decide how robust those limits are.

---

## WP0: Make the code base complete and current (≈ 1 day, CPU)

**What needs fixing:**
1. **Vendored Chempy is missing from the integration branch.** Only the main tree has it
   (`COMPASS/GCE/vendor/Chempy`, 14 MB, untracked). Copy it into the integration branch, with
   `vendor/CHEMPY_PATCHES.md` describing the port, and commit it. The repository `.gitignore`
   already drops the rebuildable `init_abun.npz`.
2. **Bring the network recipe up to the final method.** The `compass_setup.RECIPE`/`train()`
   of 2026-10-22 predates the benchmark (`benchmarks/paper/FINDINGS.md`):

   | setting | change | evidence |
   |---|---|---|
   | Rg residual on the globals | add it (`experiments_arruda_schur/precondition.py`, as `pp.py --precondition Rg`) | on PP over 3 seeds, without it results depend on the seed (one seed cov90 0.60), the error is higher (RMSE 0.41 vs 0.36) and log σ collapses appear |
   | loss weight | pass `time_loss_weight=None` explicitly | EDM hurts posterior recovery on PP: 0.957 without vs 0.945 with |
   | seeded validation, final weights | keep; check that `train()` does not override `validation_noise_seed` | seeded validation is the largest single training effect |
   | covariance output | leave off for now | optional; netcov is faster but worse on IK |
   | sampler | `SAMPLER` stays at gauss_jacobian_FD r9 | the library default |
   | prior | keep the 1.5× wide training prior | divided out by `set_training_prior` |

3. **Run the tests.**
   - `CUDA_VISIBLE_DEVICES=-1 pytest GCE/tests -q`: 17 tests.
   - A new test: the recipe sidecar records Rg and the loss weight.

**Gate:** all tests pass; `scripts/mini_test.py` runs end to end on the new recipe.

## WP1: Simulator gates (≈ ½ day CPU)

- **Failure map.** Draw 2,000 rows from each model's 1.5×-wide prior. Record the failure rate
  and where failures occur (which parameters). Failures truncate the prior, so the rate goes
  into every bank manifest.
- **Sensitivity check.** Change α_t, α_Z and σ_α one at a time, from 0 to the size of
  Durrant et al.'s variation (α −2.3 → −1.6). Plot Δ[X/Fe] and Δ[Fe/H] per element against
  0.05 dex. This shows which elements carry the IMF signal, mostly O and Mg relative to Fe
  through core-collapse vs SN Ia enrichment, and it fixes the minimum element set of a usable
  survey profile.

**Gate:** failure rate ≤ 5% everywhere in the bulk of the prior; at least one observable moves
by more than 0.05 dex for the Durrant-sized variation.

## WP2: Simulation banks (≈ 3 days CPU at 3 cores)

`python -m gce.generate --model <m> --n 200000` (resumable; train / validation / test splits).

| bank | model | yields | rows | time at 3 CPUs |
|---|---|---|---|---|
| B1 | universal | TNG | 200k | 6.5 h |
| B2 | pooled | TNG | 200k | 6.5 h |
| B3 | time | TNG | 200k | 6.5 h |
| B4 | metallicity | TNG | 200k | 6.5 h |
| B5, B6 | universal_kroupa, universal_salpeter | TNG | 100k each | 6.5 h together |
| B7–B10 | universal, time (plus whichever variation model WP4 finds most detectable) | 2 alternative sets (e.g. Nomoto2013 / Karakas / Seitenzahl) | 200k each | 26 h |

- Banks hold clean abundances. Noise and error nodes are added at training time per profile
  (`observation.make_observations`), so a new survey profile costs only training.
- Ask whether more than 3 CPUs may be used for banks: the work parallelises perfectly, and 12
  cores would turn 3 days into 18 h.

## WP3: Networks (≈ 3–4 GPU-days in total)

`scripts/train.py`, one network per (model, profile, seed): about 3 h each on a 2080 Ti.

| set | models | profiles | seeds | networks |
|---|---|---|---|---|
| N1 | universal, pooled, time, metallicity | `tng_mock` (known answer), `ideal8` | 3 | 24 |
| N2 | the 2 controls | `ideal8` | 1 | 2 |
| N3 | the 4 models | the survey profile (WP8) | 3 | 12 |
| N4 | universal + the most powerful variation model | `ideal8` on B7–B10 | 1 | 4 |

**Gate per network:** training runs the full 500 epochs (seeded validation), there is no
collapse of the globals, and its per-star (n = 1) posterior passes an SBC check on 200 test
rows.

## WP4: Known answer and the n-sweep (E1), the method gate (≈ 2 GPU-days)

1. **Chempy galaxies from `universal`:** 20 galaxies × n = 10, 50, 100, 200, 500, 1000 stars.
   - Composed posterior of (α, log N_Ia).
   - Bias in posterior sd, width, coverage over the 20 galaxies.
2. **TNG mock:** a different simulator, so this is the misspecified known answer. Compare with
   the NumPyro HMC reference for n = 1, 10, 200 (`paths.EXTERNAL["hmc_reference"]`:
   α = −2.281 ± 0.007, log N_Ia = −2.888 ± 0.008 at n = 200).
3. **Arms:**
   - gauss_jacobian_FD r9 (the method);
   - netcov (if the covariance output is trained);
   - gauss_hierarchical;
   - the plain Arruda sum, as the "naive" baseline.

**Why this gate matters:** the composition benchmark shows that network error summed over many
groups is the method's weak spot. AR(1) at 4,096 groups and the Gaussian toy at J = 1000
(1.35 sd mean error) both failed. Stars are many groups.

**Gate: the largest n at which bias stays below 0.5 sd and coverage stays at 0.85–0.95 sets
n_max for everything after this.** If n_max is below 200, the science runs use random
subsamples of size n_max, repeated, and report the spread. Before that, try more seeds and
netcov.

## WP5: Power and false positives (E2 + E3), the science gate (≈ 2–3 GPU-days)

1. **Power.** Galaxies from `time`, `metallicity` and `pooled` with true variation on a grid:
   - α_t ∈ {0, 0.1, 0.2, 0.4, 0.7} per 10 Gyr;
   - α_Z ∈ {0, 0.1, 0.2, 0.4} per dex;
   - σ_α ∈ {0.02, 0.05, 0.1, 0.2}.

   Run 10 galaxies per cell at n = 200 and n_max, with the `tng_mock`-like profile and ages.
   Output: detection probability and the 95% upper limit against the true variation.
2. **False positives.** 50 galaxies from `universal` through every variation network: the
   distribution of the log Bayes factor gives the threshold at 5% false positives. The two
   controls must come out indistinguishable from `universal`.
3. **Seed spread:** the same galaxies through 3 seeds. Seed scatter must be small compared with
   the effects we want to detect (validation C9).

**Gate: if a Durrant-sized variation (Δα ≈ 0.7 between early/dense and late star formation)
is not detectable at n_max, the answer is "abundances cannot test it".** That is publishable
as a limit paper, and we stop before WP8. Otherwise we continue.

## WP6: Misspecification nulls (E4) (≈ 1 GPU-day)

Universal-IMF data that Chempy did not generate, run through every variation network:
- the TNG mock;
- the Chempy alternative-yield mock;

each with σ_sys = 0, 0.02 and 0.05 dex (`ObsProfile.systematic`).

**Question:** does simulator or yield error produce a "variation" verdict?

**Gate:** with the σ_sys we adopt, false "variation" verdicts stay below 5%. The σ_sys that
achieves this is fixed for the real data.

## WP7: Yield robustness (E5) (≈ 1 GPU-day)

Repeat WP4 and WP5 on B7–B10 at reduced size (one seed, the strongest cells). Report the IMF
result for each yield set and model-averaged over yield sets. Philcox et al. 2018 found α
depends on the yield set, so this is expected to be the dominant systematic.

## WP8: Real data (E6) (≈ 1 week of data work, then ≈ 1 GPU-day)

1. **Nissen solar twins** (n = 71, ready): first contact with real data. Expect weak power,
   because [Fe/H] spans only ±0.3 dex and the stars are near-solar.
2. **A disc survey.** Build `data_sources.load_apogee` (DR17 allStar plus an age catalogue,
   astroNN or distmass) or `load_galah` (DR4, isochrone ages):
   - quality: clean flags, SNR > 100, σ_age/age < 30%;
   - population: a narrow Galactocentric radius and |z| cut, low-α (thin disc) only, since a
     one-zone model is a local model. The thick disc is a separate run;
   - elements: those WP1 found informative. Giants without C/N, or with a mixing correction;
   - zero points: checked against Chempy's Asplund09 solar scale;
   - profile: a new `ObsProfile`, with networks N3 trained on it;
   - three random subsamples at n = 200 and n_max.
3. **Run all four models** with 3 seeds each, then apply the decision rule above.

## WP9 (optional): density-based variation

Chempy's `evolve()` already has each step's gas mass and SFR (`cube.cube["gas"]`,
`basic_sfr.sfr`). A schedule `slope_schedule(log_z, t, sfr, gas)` could implement Durrant et
al.'s sigmoid α(n) = α_high + (α_low − α_high) / (1 + (n / n_pivot)^γ), with an SFR-intensity
proxy for n:
- either as a fixed hypothesis (their α_high = −1.6, γ = 0.65), compared with the universal
  IMF;
- or with α_high and n_pivot free.

**Limitation:** a one-zone model has no physical density, so the proxy normalisation is an
assumption the paper must state. Do this only after WP5 shows the time trend is detectable.
If it is not, the density version will not be detectable either.

## Paper outline (application paper)

1. The question; prior work:
   - Philcox & Rybizki 2018/2019 and SBI-Chempy (Güneş et al. 2025): fitted a universal α;
   - IGIMF;
   - Durrant et al. 2026: a density-dependent IMF in COLIBRE, tested only on bulk galaxy
     properties, not stellar archaeology.
2. Method: hierarchical COMPASS over stars, and the nested IMF models (WP0, WP3).
3. Validation: known answer, HMC agreement, n_max (WP4); power and false positives (WP5);
   misspecification and yields (WP6, WP7).
4. Result on real data (WP8): posteriors or upper limits of σ_α, α_t, α_Z; what they mean for
   density-dependent IMF prescriptions.
5. Limitations: one-zone model, yield tables, n_max.

## Order and time

| week | work |
|---|---|
| 1 | WP0, WP1, start banks (WP2) |
| 2 | finish banks; networks N1 and N2 (WP3) |
| 3 | **WP4** (method gate), **WP5** (science gate) |
| 4 | WP6, WP7; start survey selection (WP8) |
| 5–6 | networks N3, real-data runs, write-up |

**Compute:**
- CPU: about 70 h of Chempy runs at 3 cores, much less with more cores.
- GPU: about 10–12 GPU-days on 2080 Ti cards.

Inference at n = 1,000 costs roughly 20–30 min per galaxy on today's code. The planned sampler
speed-ups (batching across galaxies, 25 steps, netcov; see `benchmarks/paper/FINDINGS.md`)
would make WP5's grid much cheaper. They are worth doing before WP5.

**People:** yield-set and survey-selection choices need a GCE expert, for example Tobias Buck or
Berkay Güneş (SBI-Chempy).
