"""Observed (or mock-observed) star samples, in the layout of an ObsProfile.

Ready now (files already on this machine, read in place):

* ``tng_mock``        1000 star particles of one IllustrisTNG galaxy (ChempyMulti). Known
                      answer: Chabrier IMF with slope -2.3 and N_Ia = 10^-2.89 everywhere. Not a
                      Chempy run, so it also carries one-zone model error: the misspecified
                      known-answer test.
* ``chempy_tng_mock`` / ``chempy_alt_mock``  ChempyMulti's Chempy mocks with TNG / alternative
                      yields (yield misspecification).
* ``nissen``          72 solar twins (Nissen et al.), C, Fe, Mg, O, Si + isochrone ages, as
                      preprocessed for the earlier yield-set comparison.

Not built yet (PLAN.md section 5): APOGEE DR17, GALAH DR4. Their loaders raise with the
selection that is planned for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .observation import ObsProfile, observed_vector
from .paths import EXTERNAL
from .simulator import END


@dataclass
class ObservedSample:
    name: str
    elements: tuple
    values: np.ndarray                 # [n, k] dex, Chempy convention ([Fe/H], [X/Fe])
    errors: np.ndarray                 # [n, k] dex
    age: Optional[np.ndarray] = None   # [n] Gyr
    age_error: Optional[np.ndarray] = None
    truth: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def __len__(self):
        return len(self.values)

    def subset(self, index):
        index = np.asarray(index)
        pick = lambda a: None if a is None else a[index]  # noqa: E731
        return ObservedSample(self.name, self.elements, self.values[index], self.errors[index],
                              pick(self.age), pick(self.age_error), dict(self.truth), list(self.notes))

    def x(self, profile: ObsProfile):
        """Rows in the profile's x layout (elements reordered to the profile)."""
        missing = [e for e in profile.elements if e not in self.elements]
        if missing:
            raise ValueError(f"{self.name} has no {missing} for profile {profile.name!r}")
        columns = [self.elements.index(e) for e in profile.elements]
        x, notes = observed_vector(self.values[:, columns], self.errors[:, columns], profile,
                                   self.age, self.age_error)
        return x, notes


def _chempymulti_mock(path, name, truth):
    z = np.load(path)
    elements = tuple(str(e) for e in z["elements"])
    # obs_time is a noisy *birth time* (Gyr after the start), as in ChempyMulti's run_pymc3.py
    age = END - np.asarray(z["obs_time"], dtype=float)
    sample = ObservedSample(name, elements, np.asarray(z["abundances"], float), np.asarray(z["abundance_errs"], float),
                            age, np.asarray(z["obs_time_err"], float), truth)
    sample.truth["true_birth_time"] = np.asarray(z["true_time"], float)
    sample.truth["true_abundances"] = np.asarray(z["true_abuns"], float)
    return sample


def load_tng_mock():
    return _chempymulti_mock(EXTERNAL["tng_mock"], "tng_mock",
                             {"alpha": -2.3, "log10_n_ia": -2.89,
                              "note": "TNG uses a Chabrier IMF and N_Ia = 1.3e-3 per Msun; "
                                      "no Chempy one-zone history, so a perfect fit is not expected"})


def load_chempy_mock(which="tng"):
    key = {"tng": "chempy_tng_mock", "alt": "chempy_alt_mock"}[which]
    return _chempymulti_mock(EXTERNAL[key], key, {"alpha": -2.3, "log10_n_ia": -2.89})


def load_nissen(drop_nonpositive_age=True):
    """Rows of the preprocessed file: age, C, Fe, Mg, O, Si (assumed [Fe/H] and [X/Fe],
    differential to the Sun -- check against Nissen et al. 2020, Table 2, before using it
    for science)."""
    data = np.load(EXTERNAL["nissen"]).astype(float)
    std = np.load(EXTERNAL["nissen_std"]).astype(float)
    sample = ObservedSample("nissen", ("C", "Fe", "Mg", "O", "Si"), data[1:].T, std[1:].T,
                            data[0], std[0])
    if drop_nonpositive_age:
        bad = sample.age <= 0
        if bad.any():
            sample = sample.subset(np.flatnonzero(~bad))
            sample.notes.append(f"dropped {int(bad.sum())} star(s) with age <= 0")
    return sample


def load_apogee(*_, **__):
    raise NotImplementedError(
        "APOGEE DR17 is not wired up yet. Planned selection (PLAN.md 5.3): allStar with "
        "ASPCAPFLAG/STARFLAG clean, SNR > 100, solar-neighbourhood low-alpha thin disc in a narrow "
        "Galactocentric annulus, ages from distmass/astroNN with sigma_age < 30%, dwarfs and subgiants for "
        "C/N or a mixing correction for giants; He and Ne are not measured (profile 'apogee_like').")


def load_galah(*_, **__):
    raise NotImplementedError(
        "GALAH DR4 is not wired up yet (PLAN.md 5.4): flag_sp == 0 and flag_X_fe == 0 per element, "
        "isochrone ages, main-sequence turn-off stars; Chempy elements C, O, Mg, Si, Fe (+ N for few).")


SOURCES = {"tng_mock": load_tng_mock, "chempy_tng_mock": lambda: load_chempy_mock("tng"),
           "chempy_alt_mock": lambda: load_chempy_mock("alt"), "nissen": load_nissen,
           "apogee": load_apogee, "galah": load_galah}
