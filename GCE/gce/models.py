"""The IMF hypotheses as hierarchical (global / local) models.

Every model has the same per-star structure: one Chempy one-zone run per star
(simulator.py). What differs is which parameters are *global* (shared by all
stars of the galaxy, composed across stars by COMPASS) and how the high-mass IMF
slope enters the run.

All parameters are kept in *inference coordinates*, which are what the network
sees (after a per-coordinate affine normalisation, compass_setup.py):

* the birth time t in [1, 13.7] Gyr is stored as ``u_birth = Phi^-1((t - t0) / (t1 - t0))``,
  so its uniform prior is exactly N(0, 1). Every local prior is then Gaussian, which
  is what MultiObsSampler's ``local_prior=(mean, std)`` and the Gaussian rules assume;
* the pooled-IMF scatter is stored as ``ln_sigma_alpha``.

Priors follow Chempy/ChempyMulti (alpha N(-2.3, 0.3), log10 N_Ia N(-2.89, 0.3),
log10 SFE N(-0.3, 0.3), log10 SFR scale N(0.55, 0.1), x_out N(0.5, 0.1)). The new
parameters have priors chosen here and listed in PLAN.md; revisit them before any
real-data run.

Training data are drawn with the globals' prior sd widened by ``global_scale``
(1.5 by default, as ``experiments_arruda_schur/pp.py --train-prior-scale 1.5``, which removed
the log_sigma collapses on partial pooling);
``MultiObsSampler.set_training_prior`` divides that wider prior back out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.special import ndtr, ndtri

from .simulator import BIRTH_TIME_RANGE, END, StarParameters

T_PIVOT = END - 4.6        # the Sun's birth time; slope "at the pivot" = slope of the solar SSP
TIME_UNIT = 10.0           # alpha_t is the slope change per 10 Gyr


@dataclass(frozen=True)
class Gaussian:
    mean: float
    sd: float


# Local ISM priors (Chempy parameter.py) and the birth-time coordinate.
ISM_PRIORS = {
    "log10_sfe": Gaussian(-0.3, 0.3),
    "log10_sfr_scale": Gaussian(0.55, 0.1),
    "outflow": Gaussian(0.5, 0.1),
    "u_birth": Gaussian(0.0, 1.0),
}
ALPHA = Gaussian(-2.3, 0.3)
LOG10_N_IA = Gaussian(-2.89, 0.3)


def birth_time_from_u(u):
    lo, hi = BIRTH_TIME_RANGE
    return lo + (hi - lo) * ndtr(np.asarray(u, dtype=float))


def u_from_birth_time(t):
    lo, hi = BIRTH_TIME_RANGE
    p = np.clip((np.asarray(t, dtype=float) - lo) / (hi - lo), 1e-9, 1 - 1e-9)
    return ndtri(p)


@dataclass(frozen=True)
class HierarchicalModel:
    name: str
    description: str
    globals: Tuple[Tuple[str, Gaussian], ...]
    pooled_alpha: bool = False                 # alpha_i is a local drawn from N(mu_alpha, sigma_alpha)
    schedule: str = "constant"                 # "constant", "metallicity" or "time"
    imf_type: str = "Chabrier_TNG"
    # For Bayes factors against the universal IMF (compass_setup.savage_dickey /
    # interval_null_bayes_factor): {parameter: point} or {parameter: (low, high)}.
    nested_null: Dict[str, object] = field(default_factory=dict)

    # ---- layout -----------------------------------------------------------------
    @property
    def global_names(self):
        return tuple(name for name, _ in self.globals)

    @property
    def local_names(self):
        return (("alpha_i",) if self.pooled_alpha else ()) + tuple(ISM_PRIORS)

    @property
    def theta_names(self):
        return self.global_names + self.local_names

    def global_prior(self, scale=1.0):
        mean = np.array([p.mean for _, p in self.globals])
        sd = np.array([p.sd for _, p in self.globals]) * scale
        return mean, sd

    # ---- prior draws ----------------------------------------------------------------
    def sample_globals(self, rng, n, scale=1.0):
        mean, sd = self.global_prior(scale)
        return rng.normal(mean, sd, size=(n, len(mean)))

    def sample_locals(self, rng, globals_):
        """One star's locals per row of ``globals_`` (the model's p(l | g); it does not
        depend on the training-prior widening)."""
        n = len(globals_)
        columns = []
        if self.pooled_alpha:
            g = dict(zip(self.global_names, globals_.T))
            columns.append(rng.normal(g["mu_alpha"], np.exp(g["ln_sigma_alpha"])))
        for p in ISM_PRIORS.values():
            columns.append(rng.normal(p.mean, p.sd, size=n))
        return np.stack(columns, axis=1)

    # ---- mapping to one Chempy run --------------------------------------------------------
    def star(self, g_row, l_row) -> Tuple[StarParameters, Optional[Callable[[float, float], float]]]:
        g = dict(zip(self.global_names, np.asarray(g_row, dtype=float)))
        loc = dict(zip(self.local_names, np.asarray(l_row, dtype=float)))
        if self.pooled_alpha:
            alpha = loc["alpha_i"]
        elif self.schedule == "constant":
            alpha = g["alpha"]
        else:
            alpha = g["alpha_0"]
        schedule = None
        if self.schedule == "metallicity":
            alpha_0, alpha_z = g["alpha_0"], g["alpha_z"]
            schedule = lambda log_z, t: alpha_0 + alpha_z * log_z  # noqa: E731
        elif self.schedule == "time":
            alpha_0, alpha_t = g["alpha_0"], g["alpha_t"]
            schedule = lambda log_z, t: alpha_0 + alpha_t * (t - T_PIVOT) / TIME_UNIT  # noqa: E731
        star = StarParameters(alpha=alpha, log10_n_ia=g["log10_n_ia"], log10_sfe=loc["log10_sfe"],
                              log10_sfr_scale=loc["log10_sfr_scale"], outflow=loc["outflow"],
                              birth_time=float(birth_time_from_u(loc["u_birth"])))
        return star, schedule


MODELS: Dict[str, HierarchicalModel] = {}


def register(model: HierarchicalModel):
    MODELS[model.name] = model
    return model


register(HierarchicalModel(
    "universal",
    "One high-mass slope for every SSP of every star (the ChempyMulti / SBI-Chempy model).",
    (("alpha", ALPHA), ("log10_n_ia", LOG10_N_IA)),
))
register(HierarchicalModel(
    "pooled",
    "Each star's birth environment has its own slope alpha_i ~ N(mu_alpha, sigma_alpha): "
    "partial pooling of the IMF across environments (the drift-diffusion PP structure).",
    (("mu_alpha", ALPHA), ("ln_sigma_alpha", Gaussian(float(np.log(0.1)), 1.0)), ("log10_n_ia", LOG10_N_IA)),
    pooled_alpha=True,
    nested_null={"ln_sigma_alpha": (-np.inf, float(np.log(0.02)))},
))
register(HierarchicalModel(
    "metallicity",
    "alpha(Z) = alpha_0 + alpha_z [Z] for the SSP born at gas metallicity [Z] (traced metals, "
    "floored at -2); alpha_z < 0 is top-heavy at low Z. Slopes clamped to [-3.5, -1.5].",
    (("alpha_0", ALPHA), ("alpha_z", Gaussian(0.0, 0.3)), ("log10_n_ia", LOG10_N_IA)),
    schedule="metallicity",
    nested_null={"alpha_z": 0.0},
))
register(HierarchicalModel(
    "time",
    "alpha(t) = alpha_0 + alpha_t (t - t_sun) / 10 Gyr for the SSP born at time t; "
    "alpha_t < 0 is top-heavy early. Slopes clamped to [-3.5, -1.5].",
    (("alpha_0", ALPHA), ("alpha_t", Gaussian(0.0, 0.3)), ("log10_n_ia", LOG10_N_IA)),
    schedule="time",
    nested_null={"alpha_t": 0.0},
))
# Low-mass-shape controls: same parameters as "universal", different IMF below 1 Msun.
# Abundances barely see the low-mass IMF, so model comparison should NOT separate these
# from "universal"; a confident verdict here is a calibration failure (PLAN.md, E3).
register(HierarchicalModel(
    "universal_kroupa",
    "Universal slope, Kroupa (2001) low-mass part (-1.3 below 0.5 Msun) instead of Chabrier.",
    (("alpha", ALPHA), ("log10_n_ia", LOG10_N_IA)),
    imf_type="normed_3slope",
))
register(HierarchicalModel(
    "universal_salpeter",
    "Universal single power law from 0.1 to 100 Msun (no low-mass turnover).",
    (("alpha", ALPHA), ("log10_n_ia", LOG10_N_IA)),
    imf_type="salpeter",
))


def get(name) -> HierarchicalModel:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name]
