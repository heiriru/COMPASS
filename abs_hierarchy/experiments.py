"""Registry of the toy hierarchies sharing this pipeline (train_abs.py, run.py, plot.py).

Every model exposes the same interface: ``simulate``, ``simulate_dataset``,
``global_density``, ``gibbs``, ``plot_half_width``, ``label`` and the Gaussian
prior moments the sampler API takes. ``abs`` keeps its original artifact root so
its cached networks and runs stay valid.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def get(name):
    if name == "abs":
        from abs_model import DEFAULT
        return {"model": DEFAULT, "g_true": 1.0, "root": HERE / "artifacts"}
    if name == "square":
        from square_model import DEFAULT
        return {"model": DEFAULT, "g_true": 0.7, "root": HERE / "artifacts" / "square"}
    if name == "lean":
        from lean_model import DEFAULT
        return {"model": DEFAULT, "g_true": 1.5, "root": HERE / "artifacts" / "lean"}
    if name == "curved":
        from curved_model import DEFAULT
        return {"model": DEFAULT, "g_true": 1.0, "root": HERE / "artifacts" / "curved"}
    raise ValueError(f"unknown model {name!r}; choose abs, square, lean or curved")


NAMES = ("abs", "square", "lean", "curved")
