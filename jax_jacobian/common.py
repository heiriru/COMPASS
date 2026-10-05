"""Shared constants for the JAX-vs-PyTorch score-Jacobian prototype.

Imports neither torch nor jax, so both environments can use it.
"""

import importlib.util
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
WEIGHTS = ARTIFACTS / "weights.npz"
REFERENCE = ARTIFACTS / "reference.npz"

# The trial-token H64/d2 model the gauss_jacobian ladder runs were made with.
CHECKPOINT = Path(
    "/export/home/rheinric/COMPASS/Partial_Pooling/artifacts/checkpoints/small/"
    "sde_joint-tok-trial-h64d2x8m4-sub100000-train-400000/"
    "sde_joint-tok-trial-h64d2x8m4-sub100000-train-400000-checkpoint.pt"
)

THETA_DIM = 10            # 7 global + 3 local; the latent block of every row
NODES = 100               # theta + 30 trials x 3
TRIALS, ITEM_DIM = 30, 3
SIGMA = 25.0              # VESDE

# Batch shapes (observation rows, draws per row) to time: the overhead-bound
# draw batch of 6 used in the churn benchmark, up to the 256-draw ladder
# protocol at n=20. Larger batches are chunked by the library anyway
# (FD_JACOBIAN_MAX_ROWS = 16384), and 10 stacked tangents of 20 x 256 rows
# already approach the 11 GB of an RTX 2080 Ti.
BATCHES = [(20, 6), (20, 64), (20, 256)]

# Noise levels at which accuracy is checked; 0.003 is where FD was worst.
LAMBDAS = [0.003, 0.03, 0.3, 3.0]

FD_STEP = 1e-2            # library default jacobian_fd_step (h = FD_STEP * lambda)


def vesde_std(t):
    return math.sqrt((SIGMA ** (2 * t) - 1.0) / (2 * math.log(SIGMA)))


def vesde_time_of_lambda(lam):
    return math.log1p(2 * math.log(SIGMA) * lam ** 2) / (2 * math.log(SIGMA))


def configure(gpu):
    """Cap this process at 3 CPU threads and, if ``gpu``, wait for one free GPU
    via autocvd. Must run before torch, numpy or jax is imported.

    Loads ``Partial_Pooling/runtime.py`` by path: importing the package would
    import its config module, and with it numpy, before the cap is set.
    """
    path = HERE.parent / "Partial_Pooling" / "runtime.py"
    spec = importlib.util.spec_from_file_location("pp_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    if gpu:
        return runtime.configure_runtime(max_threads=3)
    return runtime.configure_cpu_limit(max_threads=3)
