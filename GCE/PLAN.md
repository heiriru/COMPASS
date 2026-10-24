# GCE application: is the high-mass IMF universal? — plan

Status 2026-10-22: code prepared and wiring-tested at toy size (README.md). **No bank,
network or posterior of scientific size exists yet.** Nothing below is a result except the
simulator facts in section 2.

## 1. The question and why COMPASS fits it

Each star's abundances are one Chempy one-zone run (ChempyMulti picture, Philcox & Rybizki
2019): the star's birth environment has local ISM parameters (SFE, SFR peak, outflow) and a
birth time, while the stellar-physics parameters — high-mass IMF slope α and SN Ia
normalisation N_Ia — are shared by the whole galaxy. With n stars this is exactly the
global/local hierarchy that `gauss_jacobian_FD` composes (G = 2–3 globals, L = 4–5 locals per
star, n = 10²–10⁴ stars).

The SBI-Chempy work (arXiv:2503.02456) *assumed* a universal α and fitted it. The new question
is whether the data prefer a universal IMF over specific alternatives. Framed as **nested
hypotheses inside an encompassing model**, it becomes parameter estimation: one network per
alternative, and a Bayes factor against the universal IMF from that same composed posterior
(Savage–Dickey). No separate likelihood network and no PF-ODE evidence are needed. Non-nested
comparisons (IMF low-mass shape, yield sets) use COMPASS `compare()` with the counted
information criteria (validation A1/A2).

## 2. Simulator facts (measured)

- Vendored Chempy ported to numpy 2; agrees with the old-numpy outputs to 1.8e-15 on 40 random
  rows (`vendor/CHEMPY_PATCHES.md`). Note: under numpy ≥ 2 the unpatched code raises a
  `TypeError` in `shorten_sfr`, which the old data-generation scripts caught and recorded as a
  *failed simulation*. Any bank made with numpy 2 and the old copy is silently empty.
- Per-SSP IMF hook (`simulator.evolve`): a constant schedule is bit-identical to stock Chempy.
  α_Z = 0.5 shifts abundances by 0.01–0.18 dex for single stars, against 0.05 dex noise, so
  the effect is not invisible per star. Whether n stars can resolve it is experiment E2.
- 0.3–0.5 s per run. At the repository's 3-CPU cap that is about 30k runs/h, so 200k runs take
  about 6.5 h per (model, yield set).
- Failure rate about 2% at the 1.5×-wide training prior (6 negative-mass and 2
  SFR-too-low out of 400). Failures truncate the prior, so the rate is reported per bank.

## 3. Models (`gce/models.py`)

| name | globals | locals per star | per-SSP slope | null vs universal |
|---|---|---|---|---|
| `universal` | α, log N_Ia | log SFE, log SFR, x_out, u_birth | α | — |
| `pooled` | μ_α, ln σ_α, log N_Ia | **α_i**, log SFE, log SFR, x_out, u_birth | α_i | interval null σ_α < 0.02 |
| `metallicity` | α₀, α_Z, log N_Ia | as universal | α₀ + α_Z [Z] | point null α_Z = 0 |
| `time` | α₀, α_t, log N_Ia | as universal | α₀ + α_t (t − t_☉)/10 Gyr | point null α_t = 0 |
| `universal_kroupa`, `universal_salpeter` | as universal | as universal | α, different low-mass IMF | non-nested (negative control) |

- Priors: Chempy's for α, log N_Ia and the ISM parameters. Birth time is uniform on
  [1, 13.7] Gyr, stored as a probit coordinate so every local prior is exactly Gaussian. New
  priors (**yours to confirm**): α_Z ~ N(0, 0.3) per dex, α_t ~ N(0, 0.3) per 10 Gyr,
  ln σ_α ~ N(ln 0.1, 1). The Bayes factors depend on these widths: report a prior-width
  sensitivity (×0.5, ×2) next to every Bayes factor.
- Sign convention: α_Z < 0 or α_t < 0 is top-heavy at low Z or early times (Marks+2012 and
  IGIMF-type claims). Slopes are clamped to [−3.5, −1.5] and [Z] is floored at −2; both are
  part of the model definition.
- `pooled` is the drift-diffusion partial-pooling structure (μ, ln σ globals, locals drawn from
  them). Expect the same failure mode, a ln σ collapse to the prior floor, and use the same
  fixes: wide training prior, residual on the globals, and TSM if needed.
- A SFR-dependent IMF (IGIMF) is left out on purpose: Chempy normalises each zone's SFR, so
  the absolute SFR the IGIMF depends on is not modelled.
- Candidate extra model (not built): a **shared-ISM galaxy**, where SFE, SFR peak and x_out
  are global and only the birth time is local. It is physically closer to one galaxy, far
  more informative per star, and a structural alternative worth comparing to the
  ChempyMulti picture.

## 4. Experiments, in order

Every network: tokenized trunk with hierarchical latent attention, six-conditional menu,
`log_sigma` time sampling, 1.5× training prior divided out at inference, final-epoch weights,
patience 100, **≥ 3 training seeds** before any verdict. Every result gets a plot (measured ±
CI against the prediction or reference).

| # | Experiment | Answers | Needs | Cost (3 CPUs / 1 GPU) |
|---|---|---|---|---|
| G0 | Simulator gates: port, constant hook, failure map over the prior (where and how often Chempy fails) | is the forward model what we think | done except the failure map | 1 h CPU |
| E1 | **Universal model, known answer.** Composed posterior of (α, log N_Ia) vs n = 1, 10, 50, 200, 1000 on Chempy galaxies; SBC/coverage of the globals; then the TNG mock against the NumPyro HMC reference (vgiusepp: n = 200 gives α −2.281 ± 0.007, log N_Ia −2.888 ± 0.008). Arms: JAC_FD r9, gauss_hierarchical, Arruda plain sum | does hierarchical COMPASS reproduce the established result, and does it stay calibrated as n grows | bank 200k + 100 galaxies × 1000 stars (100k runs) | ~10 h CPU, 1 h train, ~2 h inference |
| E2 | **Power analysis for IMF variation.** Galaxies from `metallicity` / `time` / `pooled` on a grid of true α_Z, α_t, σ_α. Posterior of the variation parameter and the log BF vs n | smallest variation detectable at n = 200 / 1000; decides whether the science question is answerable before any real data | 3 banks (200k each) + galaxies | ~25 h CPU, 3 h train, ~6 h inference |
| E3 | **Calibration of the verdict.** Galaxies from `universal`: distribution of the log BF (false-positive rate at a fixed threshold). Negative control: Kroupa/Salpeter vs Chabrier must come out indistinguishable | does COMPASS cry wolf; is the seed spread (validation C9) small against the BF | E2's networks + 2 shape banks | ~15 h CPU |
| E4 | **Misspecification nulls.** TNG mock (universal truth, not Chempy) and the alternative-yield Chempy mock through every variable-IMF network. With and without a systematic floor σ_sys (0, 0.02, 0.05 dex, `ObsProfile.systematic`) | does one-zone or yield error masquerade as IMF variation; this is the main threat to a real-data claim | E2's networks, retrained per σ_sys | ~3 h train per σ_sys |
| E5 | **Yield robustness.** Repeat E1/E4 on 2–3 yield sets (e.g. Nomoto2013 / Karakas / Seitenzahl, as in the earlier model comparison); model-average the IMF verdict over yield sets | is the IMF answer stable over yields (Philcox+2018 found α depends on the yield set) | banks per yield set | ~7 h CPU per set |
| E6 | **Real data**, only after E2–E4 pass: Nissen solar twins first (n = 71, profile `nissen`), then an APOGEE or GALAH disc sample (n = 10³–10⁴) | the science result | section 5 | — |

Notes on the order:
- E2 comes before any real data because the solar twins span only about ±0.3 dex in [Fe/H].
  The metallicity lever arm may be far too short to constrain α_Z there, and E2 answers that
  for free.
- E1 doubles as the thesis's main composition benchmark on a real application: an n sweep
  with an external HMC reference.
- n ≥ 1000 with gauss_jacobian_FD needs draws × n ≤ about 64k rows per call on an 11 GB card.
  Either chunk the draws, or this is the place to test the observation-mini-batched JAC.

## 5. Observational data

### 5.1 Ready (files on this machine; loaders in `gce/data_sources.py`)
- **TNG mock** (1000 star particles, one IllustrisTNG galaxy): 8 elements at 0.05 dex, noisy
  birth times (σ 0.4–1.75 Gyr). True α = −2.3, N_Ia = 10^−2.89. It is not a Chempy run, so it
  is the *misspecified* known-answer test.
- **Chempy mocks** (TNG and alternative yields): yield misspecification.
- **Nissen solar twins**: 71 stars after dropping one negative age; C, Fe, Mg, O, Si at
  0.006–0.03 dex; ages ±0.5–1.3 Gyr. To check before science use: the preprocessed file is
  assumed to be [Fe/H], [X/Fe] differential to the Sun (compare with Nissen et al. 2020,
  Table 2). Also, the solar twins are only a near-solar slice: no He, N or Ne.

### 5.2 How a survey becomes `x` (`gce/observation.py`)
- `x` = profile elements in Chempy's convention ([Fe/H], [X/Fe]). With `error_nodes`, each
  star's own errors enter as observed log₁₀ σ nodes (trained over a log-uniform range), so one
  network serves heteroscedastic data. An observed birth time t_obs = 13.8 − age enters with
  its σ_t. Errors outside the training range are clipped and reported, never silently used.
- **One network per survey profile.** Elements a survey does not measure are left out of `x`
  rather than made latent. This is simplest and keeps latent attention's g/l/x roles clean.
  The banks are shared, so a new profile costs only training.

### 5.3 APOGEE DR17 (to build)
- Selection: ASPCAPFLAG/STARFLAG clean, SNR > 100, a narrow Galactocentric annulus and
  |z| cut (a one-zone model is a local model), and the low-α (thin-disc) sequence. A
  high-α-only sample is a separate run, not mixed in.
- Ages from distmass or astroNN, with σ_age/age < 30%.
- Giants: C and N are altered by dredge-up/mixing, so either restrict C/N to dwarfs and
  subgiants or apply a mixing correction. He and Ne are not measured (profile `apogee_like`).
- Check zero points against Chempy's Asplund09 solar scale (APOGEE's scale is
  solar-neighbourhood calibrated) and add a systematic floor (E4 decides its size).
- Random subsamples of n = 200 / 1000 / 5000, with ≥ 3 subsamples each to separate sample
  noise from a real trend.

### 5.4 GALAH DR4 (to build)
- `flag_sp == 0`, `flag_X_fe == 0` per element; isochrone ages; main-sequence turn-off and
  subgiant stars (best ages).
- Elements C, O, Mg, Si, Fe (N for few). More elements are possible later by extending
  `TRACED`, though yield quality varies by element.
- Watch atomic diffusion in old dwarfs (about 0.05 dex) and NLTE corrections for O.

### 5.5 Threats to an "IMF varies" claim
Every one of these can produce a non-universal verdict on its own; E4/E5 exist to measure them.
1. Yield-set error.
2. One-zone inadequacy: radial migration, and mixing thin and thick disc.
3. Metallicity-dependent *yields*, which Chempy already has, being partly degenerate with a
   metallicity-dependent IMF.
4. Survey zero points.
5. Network bias summed n times: the composition lesson from IK and partial pooling. Check by
   seed spread and by the E3 false-positive rate at the real n.

## 6. Decisions that are yours

1. Priors of α_Z, α_t, σ_α (section 3) and the Bayes-factor threshold for a claim.
2. Whether to add the shared-ISM galaxy model.
3. Whether to raise the 3-CPU cap for data generation: about 6.5 h per 200k-run bank at 3 CPUs.
4. Which survey after Nissen: APOGEE (more stars, but C/N mixing) or GALAH (better ages,
   fewer stars).
5. Whether to make the systematic floor σ_sys a fitted global parameter instead of a fixed
   training-time floor. The fitted version is more honest, but adds one global and one more
   thing to calibrate.

## 7. First commands when it is time to run
```bash
cd COMPASS/GCE
python -m gce.generate --model universal --n 200000                  # ~6.5 h at 3 CPUs
python -m gce.generate --model universal --n 20000 --split validation
python -m gce.generate --model universal --n 100 --galaxy-stars 1000  # E1 known-answer galaxies
python scripts/train.py --model universal --profile tng_mock --seed 7  # autocvd picks the GPU
python scripts/infer.py --network <dir> --name universal-tng_mock-s7 --source tng_mock --stars 200
```
