"""Recovery test sets, one named generator stream per (kind, layout, n).

``single``: ``count`` independent prior-predictive rows -- all four parameters
local, one observation each. It is the same problem for both layouts, so both
networks are run on the same rows.

``hier``: ``count`` tall datasets. The globals of the layout are drawn once per
dataset from the prior, the locals once per observation, exactly the generative
model the composition rules assume. No selection is applied.
"""
from pathlib import Path

import numpy as np

import hierarchical as H
import simulator as sim
from layouts import LAYOUTS, PRIOR_STD

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SEED = 20260929


def folder(kind, layout=None, n=1):
    return ARTIFACTS / ("single" if kind == "single" else f"{layout}_n{n}")


def make(kind, layout=None, n=1, count=200, seed=SEED):
    """``{"theta": (count, n, 4), "x": (count, n, 2)}`` in node order."""
    if kind == "single":
        rng = sim.generator("recovery-single", seed)
        theta = rng.normal(0.0, PRIOR_STD, (count, 1, 4))
    else:
        g = LAYOUTS[layout].n_global
        rng = sim.generator(f"recovery-{layout}-n{n}", seed)
        theta = rng.normal(0.0, PRIOR_STD, (count, n, 4))
        theta[:, :, :g] = rng.normal(0.0, PRIOR_STD[:g], (count, 1, g))
    x = H.forward(theta.reshape(-1, 4)).reshape(count, theta.shape[1], 2)
    return {"theta": theta, "x": x}


def load_or_make(kind, layout=None, n=1, count=200, seed=SEED):
    path = folder(kind, layout, n) / "data.npz"
    if path.exists():
        data = dict(np.load(path))
        if len(data["x"]) != count:
            raise ValueError(f"{path} holds {len(data['x'])} datasets, asked for {count}")
        return data
    data = make(kind, layout, n, count, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data
