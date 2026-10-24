# GCE: hierarchical COMPASS on Chempy, and IMF model comparison

Many stars of one galaxy share the stellar physics (high-mass IMF slope α, SN Ia rate
N_Ia); each star has its own birth environment and birth time. COMPASS learns one star,
then composes n stars with `gauss_jacobian_FD`. Several IMF hypotheses (universal,
pooled across environments, metallicity- or time-dependent) are separate models, tested
against the universal IMF by Bayes factors from the composed posterior.

**What to test and in which order: [PLAN.md](PLAN.md).** Status: prepared, wiring-tested at
toy size only; no scientific run yet.

## Layout

```
gce/paths.py          data root (/export/data/rheinric/compass_gce), external inputs, CPU cap + autocvd
gce/simulator.py      one star = one Chempy run; per-SSP IMF hook (bit-identical to stock when constant)
gce/models.py         the IMF hypotheses as global/local models, priors, nested nulls
gce/observation.py    survey profiles: elements, noise / error nodes, observed birth time
gce/generate.py       resumable simulation banks (clean abundances) and known-answer galaxies
gce/data_sources.py   TNG mock, Chempy mocks, Nissen solar twins; APOGEE/GALAH stubs
gce/compass_setup.py  layout, normaliser, network recipe, training, composition, Bayes factors
scripts/mini_test.py  end-to-end toy check (CPU)
scripts/train.py      train one network (autocvd on cuda)
scripts/infer.py      composed inference on a sample or galaxy (+ Bayes factor)
tests/test_gce.py     fast checks (17 tests, ~6 s)
vendor/Chempy         numpy-2 port of Chempy, see vendor/CHEMPY_PATCHES.md
```

## Run

All scripts cap themselves to 3 CPUs (COMPASS/CLAUDE.md) and pick a GPU with autocvd when
`--device cuda` and `CUDA_VISIBLE_DEVICES` is unset.

```bash
cd COMPASS/GCE
CUDA_VISIBLE_DEVICES=-1 ../.COMPASS/bin/python -m pytest tests -q
CUDA_VISIBLE_DEVICES=-1 ../.COMPASS/bin/python scripts/mini_test.py --root <scratch dir>
../.COMPASS/bin/python -m gce.generate --model universal --n 200000        # a real bank (~6.5 h)
```

## Verified so far (2026-10-22)

- Chempy port: matches old-numpy outputs to 1.8e-15; the constant-slope hook equals stock
  Chempy exactly; α_Z = 0.5 moves abundances by up to 0.18 dex.
- Every registered model simulates. Runs take 0.3–0.5 s; about 2% fail at the 1.5×-wide
  training prior.
- Toy pipeline: 400-run bank → 2-epoch network (CPU) → gauss_jacobian_FD composition of 5
  TNG-mock stars gives finite, synchronised globals. Its numbers mean nothing.
- Loaders: TNG mock (1000×10), Chempy-alternative mock (1052×10), Nissen (71×12) in their
  profiles' layouts.

## Data

Banks, galaxies and networks go to `$COMPASS_GCE_DATA` (default
`/export/data/rheinric/compass_gce`), never the home directory. External inputs are read in
place (paths in `gce/paths.py::EXTERNAL`). Of the vendored files, the repository `.gitignore`
drops only `vendor/Chempy/input/yields/West17/init_abun.npz`, a cache Chempy rebuilds when
it is missing. Nothing here is committed yet.
