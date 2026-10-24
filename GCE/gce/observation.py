"""From clean Chempy abundances to what a survey measures.

An :class:`ObsProfile` fixes the observed node vector ``x`` of one network:

    x = [abundances of `elements`] + [log10 sigma per element]?  + [t_obs, log10 sigma_t]?

* Abundances are Chempy's convention: [Fe/H] for Fe, [X/Fe] for everything else (dex).
* ``error_nodes``: heteroscedastic errors. Training draws a per-star, per-element
  sigma log-uniformly from ``error_range`` and passes log10 sigma as an *observed*
  node, so one network serves any error a star actually has. Without it the noise
  is a fixed ``abundance_error``.
* ``time_node``: an observed birth time t_obs = t + N(0, sigma_t^2), with log10 sigma_t
  observed alongside (ages are what ChempyMulti and the TNG mock use to pin each
  star's local birth time). A survey's age is turned into t_obs = END - age.
* ``systematic``: a floor added in quadrature to every abundance error in training, a
  stand-in for yield-table / one-zone model error (Philcox & Rybizki 2019 fit one).

The same profile builds ``x`` for real data (:func:`observed_vector`), so training and
deployment cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .simulator import END, TRACED


@dataclass(frozen=True)
class ObsProfile:
    name: str
    elements: Tuple[str, ...]
    abundance_error: Optional[float] = 0.05          # fixed sigma (dex); None if error_nodes
    error_nodes: bool = False
    error_range: Tuple[float, float] = (0.01, 0.1)    # log-uniform training range, dex
    time_node: bool = False
    time_error_range: Tuple[float, float] = (0.3, 2.0)  # log-uniform training range, Gyr
    systematic: float = 0.0

    @property
    def x_names(self):
        names = [f"[{e}/H]" if e == "Fe" else f"[{e}/Fe]" for e in self.elements]
        if self.error_nodes:
            names += [f"log10_err_{e}" for e in self.elements]
        if self.time_node:
            names += ["t_obs", "log10_err_t"]
        return tuple(names)

    @property
    def x_size(self):
        return len(self.x_names)


PROFILES = {
    # The setup of the existing GCE model-comparison work: 8 elements, 5% (0.05 dex) noise,
    # no ages. Chempy mocks / model-level calibration (PLAN.md E1-E3).
    "ideal8": ObsProfile("ideal8", ("C", "Fe", "He", "Mg", "N", "Ne", "O", "Si"), 0.05),
    # IllustrisTNG mock of ChempyMulti: 8 elements at 0.05 dex, noisy birth times (0.4-1.75 Gyr).
    "tng_mock": ObsProfile("tng_mock", ("C", "Fe", "He", "Mg", "N", "Ne", "O", "Si"), 0.05,
                           time_node=True, time_error_range=(0.2, 3.0)),
    # Nissen et al. solar twins: C, Fe, Mg, O, Si at 0.006-0.03 dex, isochrone ages 0.4-1.3 Gyr.
    "nissen": ObsProfile("nissen", ("C", "Fe", "Mg", "O", "Si"), None, error_nodes=True,
                         error_range=(0.004, 0.05), time_node=True, time_error_range=(0.2, 2.5)),
    # APOGEE-like (placeholder until the selection is fixed, PLAN.md section 5): no He or Ne;
    # C and N only for dwarfs or after a mixing correction.
    "apogee_like": ObsProfile("apogee_like", ("C", "Fe", "Mg", "N", "O", "Si"), None, error_nodes=True,
                              error_range=(0.01, 0.1), time_node=True, time_error_range=(0.5, 4.0)),
}


def _log_uniform(rng, low, high, size):
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def make_observations(abundances, birth_time, profile: ObsProfile, rng):
    """Noisy ``x`` rows for simulated stars. ``abundances``: [n, len(TRACED)] clean Chempy
    output; ``birth_time``: [n] Gyr."""
    abundances = np.asarray(abundances, dtype=float)
    columns = [TRACED.index(e) for e in profile.elements]
    clean = abundances[:, columns]
    n, k = clean.shape
    if profile.error_nodes:
        sigma = _log_uniform(rng, *profile.error_range, size=(n, k))
    else:
        sigma = np.full((n, k), float(profile.abundance_error))
    total = np.sqrt(sigma ** 2 + profile.systematic ** 2)
    parts = [clean + total * rng.standard_normal((n, k))]
    if profile.error_nodes:
        parts.append(np.log10(sigma))
    if profile.time_node:
        sigma_t = _log_uniform(rng, *profile.time_error_range, size=n)
        t_obs = np.asarray(birth_time, dtype=float) + sigma_t * rng.standard_normal(n)
        parts.append(np.stack([t_obs, np.log10(sigma_t)], axis=1))
    return np.concatenate(parts, axis=1)


def observed_vector(values, errors, profile: ObsProfile, age=None, age_error=None):
    """``x`` rows for real stars in the profile's layout. ``values``/``errors``: [n, k] in the
    order of ``profile.elements``; ``age``/``age_error`` in Gyr (converted to birth time).
    Errors outside the training range are clipped to it and reported, never silently used."""
    values = np.asarray(values, dtype=float)
    parts = [values]
    notes = []
    if profile.error_nodes:
        errors = np.asarray(errors, dtype=float)
        low, high = profile.error_range
        outside = (errors < low) | (errors > high)
        if outside.any():
            notes.append(f"{int(outside.sum())} abundance errors clipped to the training range {profile.error_range}")
        parts.append(np.log10(np.clip(errors, low, high)))
    if profile.time_node:
        if age is None:
            raise ValueError(f"profile {profile.name!r} needs ages")
        t_obs = END - np.asarray(age, dtype=float)
        sigma_t = np.asarray(age_error, dtype=float)
        low, high = profile.time_error_range
        if ((sigma_t < low) | (sigma_t > high)).any():
            notes.append(f"{int(((sigma_t < low) | (sigma_t > high)).sum())} age errors clipped to {profile.time_error_range}")
        parts.append(np.stack([t_obs, np.log10(np.clip(sigma_t, low, high))], axis=1))
    return np.concatenate(parts, axis=1), notes
