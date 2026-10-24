"""Locations used by the GCE application.

Simulation banks and checkpoints are large, and the home quota is tight, so
they live under ``/export/data`` (override with ``COMPASS_GCE_DATA``). The
vendored Chempy is put on ``sys.path`` by :func:`use_vendored_chempy`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

GCE_ROOT = Path(__file__).resolve().parent.parent
COMPASS_ROOT = GCE_ROOT.parent
VENDOR = GCE_ROOT / "vendor"
DATA_ROOT = Path(os.environ.get("COMPASS_GCE_DATA", "/export/data/rheinric/compass_gce"))

# Read-only external inputs found on this machine (2026-10-22). Nothing here is copied;
# the loaders in data_sources.py read them in place.
EXTERNAL = {
    # 1000 star particles of one IllustrisTNG galaxy, Chabrier IMF slope -2.3, N_Ia = 10^-2.89
    "tng_mock": Path("/export/home/rfeliu/ChempyMulti/tutorial_data/Mock_Data_TNG.npz"),
    # Chempy mocks from ChempyMulti (TNG yields and an alternative yield set)
    "chempy_tng_mock": Path("/export/home/rfeliu/ChempyMulti/tutorial_data/Mock_Data_Chempy+TNG.npz"),
    "chempy_alt_mock": Path("/export/home/rfeliu/ChempyMulti/tutorial_data/Mock_Data_Chempy+Alternative.npz"),
    # Nissen et al. solar twins as preprocessed by B. Guenes: rows age, C, Fe, Mg, O, Si
    "nissen": Path("/export/home/tbuck/COMPASS_paper/GCE_compass/data/MA_data/nissen_gce_data/nissen_solar.npy"),
    "nissen_std": Path("/export/home/tbuck/COMPASS_paper/GCE_compass/data/MA_data/nissen_gce_data/nissen_solar_std.npy"),
    # Old-numpy Chempy outputs (TNG yields) used to check the vendored port bit for bit
    "chempy_tng_val": Path("/export/home/tbuck/COMPASS_paper/GCE_compass/data/MA_data/Chempy_data/chempy_TNG_val_data.npz"),
    # NumPyro HMC reference posteriors on the TNG mock (ChempyMulti model, NN emulator)
    "hmc_reference": Path("/export/data/vgiusepp/chempy_hmc/hmc_numpyro/TNG_set/paper_net/posteriors"),
}


def configure(cpus=3, gpu=False):
    """The repository's runtime rule (COMPASS/CLAUDE.md): cap this process to ``cpus`` logical
    CPUs (3 unless deliberately overridden) before numeric imports, and with ``gpu`` wait for
    one free GPU via autocvd unless CUDA_VISIBLE_DEVICES was chosen explicitly."""
    if str(COMPASS_ROOT) not in sys.path:
        sys.path.insert(0, str(COMPASS_ROOT))
    from Partial_Pooling.runtime import configure_cpu_limit, configure_runtime
    if gpu and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return configure_runtime(max_threads=cpus)
    return configure_cpu_limit(max_threads=cpus)


def use_vendored_chempy():
    """Make ``import Chempy`` resolve to GCE/vendor/Chempy (the numpy-2 port)."""
    path = str(VENDOR)
    if path not in sys.path:
        sys.path.insert(0, path)
    import Chempy
    if Path(Chempy.__file__).resolve().parent.parent != VENDOR.resolve():
        raise ImportError(f"Chempy resolved to {Chempy.__file__}, not the vendored copy in {VENDOR}; "
                          "import gce.paths before anything that imports Chempy.")
    return Chempy
