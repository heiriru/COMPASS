# Vendored Chempy: origin and patches

Source: `/export/home/bguenes/sbi_chemical_abundances/Chempy` (B. Güneş' copy used for the
SBI-Chempy paper, arXiv:2503.02456, and the earlier COMPASS GCE model comparison), copied
2026-10-22. Upstream: https://github.com/jan-rybizki/Chempy (MIT, `Chempy/LICENSE`).
Default configuration unchanged: Chabrier_TNG IMF, TNG yields (SN II `TNG_net`, AGB `TNG_net`,
SN Ia `TNG`), Maoz delay time, 28 time steps, elements C, Fe, H, He, Mg, N, Ne, O, Si.

All patches are numpy-2 / path fixes. **None changes a number**: on 40 random rows of
`chempy_TNG_val_data.npz` (generated with the old numpy) the patched code agrees to
1.8e-15 (`scripts/mini_test.py`, `tests/test_gce.py`).

| file | change | why |
|---|---|---|
| `cem_function.py::shorten_sfr` | birth step found with `int(np.argmin(...))` instead of a nested `np.where` tuple; `float()` on scalars | numpy >= 2 refuses `float()` / `int()` of a 1-element array. Old scripts caught exactly this `TypeError` and recorded the run as a *failed simulation*. `argmin` takes the first index on ties, as `np.where(...)[0][0]` did |
| `infall.py`, `making_abundances.py`, `data_to_test.py` | `[...][0]` before `int()` / `float()` of `np.where` look-ups; element masses taken as scalars | same numpy-2 rule |
| `weighted_yield.py` (3x) | `float(-np.ravel(self.sn1a_yields[0.02]['mass_in_remnants'])[0])` | same |
| all `*.py` | `np.core.records.fromarrays` -> `np.rec.fromarrays` | `np.core` is private in numpy 2 |
| `parameter.py` | proto-Sun file loaded relative to the package (`os.path.dirname(__file__)`) | was relative to the working directory |
| `yields.py::West17` | `init_abun.npz` read/written via `localpath` | same |

Environment: `h5py` was added to `.COMPASS` (TNG yields are HDF5).

The per-SSP variable IMF is **not** a patch to Chempy: `gce/simulator.py::evolve` is a copy of
`wrapper.Chempy_all_times` with one added block, so the vendored package stays a pure port.
