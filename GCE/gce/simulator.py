"""One star = one Chempy one-zone run, with an optional IMF that changes per time step.

This follows the ChempyMulti picture (Philcox & Rybizki 2019): every star has
its own one-zone birth environment (local ISM parameters and birth time), while
the stellar-physics (SSP) parameters -- the high-mass IMF slope and the SN Ia
normalisation -- are shared by every star of a galaxy.

Stock Chempy builds the IMF once per run (``SSP_wrap.__init__``) and reuses it
for every simple stellar population. A variable-IMF model needs the slope of the
SSP born at step ``i`` to depend on that step's gas metallicity or time, so
:func:`evolve` is a copy of ``Chempy.wrapper.Chempy_all_times`` whose only
change is that it rebuilds ``basic_ssp.imf`` before each SSP when a schedule is
given. With no schedule it is the stock loop; with a constant schedule it must
reproduce the stock output bit for bit (``tests/test_gce.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .paths import use_vendored_chempy

use_vendored_chempy()

from Chempy.cem_function import shorten_sfr  # noqa: E402
from Chempy.imf import IMF  # noqa: E402
from Chempy.parameter import ModelParameters  # noqa: E402
from Chempy.solar_abundance import solar_abundances  # noqa: E402
from Chempy.wrapper import SSP_wrap, initialise_stuff  # noqa: E402

# Chempy's default traced elements. Outputs are [Fe/H] for Fe and [X/Fe] for every
# other element (so the H entry is -[Fe/H] and carries no information).
TRACED = ("C", "Fe", "H", "He", "Mg", "N", "Ne", "O", "Si")
OBSERVABLE = tuple(e for e in TRACED if e != "H")

END = 13.797617          # Chempy's simulation end (Planck 2015 age), Gyr
BIRTH_TIME_RANGE = (1.0, 13.7)   # birth time t (Gyr after the start); age = END - t
SLOPE_RANGE = (-3.5, -1.5)       # per-SSP slope clamp for the variable-IMF schedules
LOG_Z_FLOOR = -2.0               # [Z] floor for the metallicity schedule (step 0 has Z = 0)


@dataclass(frozen=True)
class YieldSet:
    """Chempy yield tables (names as in Chempy.parameter / Chempy.yields)."""
    sn2: str = "TNG_net"
    agb: str = "TNG_net"
    sn1a: str = "TNG"
    hn: str = "Nomoto2013"

    @property
    def tag(self):
        return f"{self.agb}_{self.sn2}_{self.sn1a}"


TNG_YIELDS = YieldSet()


@dataclass(frozen=True)
class StarParameters:
    """Physical inputs of one Chempy run. ``alpha`` is the slope above 1 Msun at solar
    metallicity / the pivot time; a schedule (if any) varies it per SSP."""
    alpha: float
    log10_n_ia: float
    log10_sfe: float
    log10_sfr_scale: float
    outflow: float
    birth_time: float


@dataclass
class SimulationResult:
    abundances: Optional[np.ndarray]          # [len(TRACED)] or None on failure
    failure: str = ""                         # "", "sfr_too_low", "negative_mass", "exception: ..."
    slopes: list = field(default_factory=list)  # per-SSP slopes actually used (schedule runs only)


def chempy_parameters(yields: YieldSet = TNG_YIELDS, imf_type: str = "Chabrier_TNG"):
    a = ModelParameters()
    a.yield_table_name_sn2 = yields.sn2
    a.yield_table_name_agb = yields.agb
    a.yield_table_name_1a = yields.sn1a
    a.yield_table_name_hn = yields.hn
    a.imf_type_name = imf_type
    if tuple(a.elements_to_trace) != TRACED:
        raise RuntimeError(f"vendored Chempy traces {a.elements_to_trace}, expected {TRACED}")
    return a


def imf_parameter(a, slope):
    """The ``imf_parameter`` tuple for ``a.imf_type_name`` with high-mass slope ``slope``
    (negative, e.g. -2.3). Mirrors Chempy.cem_function.extract_parameters_and_priors, plus
    the two shapes it does not handle (Salpeter takes a positive exponent; a Kroupa-like
    two-slope law via normed_3slope)."""
    kind = a.imf_type_name
    if kind == "Chabrier_TNG":
        return (a.chabrier_para1, a.chabrier_para2, a.chabrier_para3, a.chabrier_para4, slope)
    if kind == "Chabrier_1":
        return (a.chabrier_para1, a.chabrier_para2, slope)
    if kind == "Chabrier_2":
        return (a.chabrier_para1, a.chabrier_para2, a.chabrier_para3, slope)
    if kind == "salpeter":
        return -slope                       # Chempy's salpeter() uses x**(-alpha)
    if kind == "normed_3slope":             # Kroupa (2001): -1.3 below 0.5 Msun, `slope` above
        return (-1.3, slope, slope, 0.5, 1.0)
    raise ValueError(f"no slope mapping for IMF type {kind!r}")


def _set_inputs(a, star: StarParameters):
    """What extract_parameters_and_priors does for the five SSP/ISM parameters."""
    a.high_mass_slope = float(star.alpha)
    a.log10_N_0 = float(star.log10_n_ia)
    a.N_0 = 10.0 ** a.log10_N_0
    a.log10_starformation_efficiency = float(star.log10_sfe)
    a.starformation_efficiency = 10.0 ** a.log10_starformation_efficiency
    a.log10_sfr_scale = float(star.log10_sfr_scale)
    a.sfr_scale = 10.0 ** a.log10_sfr_scale
    a.outflow_feedback_fraction = float(star.outflow)
    a.imf_parameter = imf_parameter(a, a.high_mass_slope)
    if a.time_delay_functional_form != "maoz":
        raise RuntimeError("only the Maoz SN Ia delay-time distribution is wired up")
    a.sn1a_parameter = [a.N_0, a.sn1a_time_delay, a.sn1a_exponent, a.dummy]
    return a


def traced_solar_metallicity(a):
    """Solar mass fraction of the traced metals: Chempy's cube['Z'] sums only the
    traced elements other than H and He, so [Z] must be normalised by the same sum."""
    sun = solar_abundances()
    getattr(sun, a.solar_abundance_name)()
    names = list(np.hstack(sun.all_elements))
    fractions = np.asarray(sun.fractions, dtype=float)
    return float(sum(fractions[names.index(e)] for e in TRACED if e not in ("H", "He")))


def evolve(a, slope_schedule: Optional[Callable[[float, float], float]] = None, record=None):
    """Copy of Chempy.wrapper.Chempy_all_times. ``slope_schedule(log_z, t)`` returns the
    high-mass slope for the SSP born at gas metallicity [Z] = ``log_z`` and time ``t``.
    Returns (cube, abundances) like the original, or (-inf, [0]) on a negative mass."""
    from numpy.lib.recfunctions import append_fields
    from Chempy.infall import PRIMORDIAL_INFALL
    from Chempy.making_abundances import mass_fraction_to_abundances
    from Chempy.time_integration import ABUNDANCE_MATRIX

    basic_solar, basic_sfr, basic_infall = initialise_stuff(a)
    elements_to_trace = a.elements_to_trace
    basic_primordial = PRIMORDIAL_INFALL(list(elements_to_trace), np.copy(basic_solar.table))
    basic_primordial.primordial()
    gas_reservoir_mass_factor = a.gas_reservoir_mass_factor / a.shortened_sfr_rescaling
    gas_at_start = a.gas_at_start / a.shortened_sfr_rescaling

    cube = ABUNDANCE_MATRIX(np.copy(basic_sfr.t), np.copy(basic_sfr.sfr), np.copy(basic_infall.infall),
                            list(elements_to_trace), list(basic_primordial.symbols),
                            list(basic_primordial.fractions), float(gas_at_start),
                            list(basic_primordial.symbols), list(basic_primordial.fractions),
                            float(gas_reservoir_mass_factor), float(a.outflow_feedback_fraction),
                            bool(a.check_processes), float(a.starformation_efficiency), float(a.gas_power),
                            float(a.sfr_factor_for_cosmic_accretion), list(basic_primordial.symbols),
                            list(basic_primordial.fractions))

    basic_ssp = SSP_wrap(a)
    z_sun = traced_solar_metallicity(a) if slope_schedule is not None else None

    for i in range(len(basic_sfr.t) - 1):
        j = len(basic_sfr.t) - i
        element_fractions = []
        for item in elements_to_trace:
            element_fractions.append(float(np.copy(cube.cube[item][max(i - 1, 0)] / cube.cube['gas'][max(i - 1, 0)])))
        metallicity = float(cube.cube['Z'][i])
        if slope_schedule is not None:
            log_z = max(np.log10(max(metallicity, 1e-30) / z_sun), LOG_Z_FLOOR)
            slope = float(np.clip(slope_schedule(log_z, float(basic_sfr.t[i])), *SLOPE_RANGE))
            imf = IMF(a.mmin, a.mmax, a.mass_steps)
            getattr(imf, a.imf_type_name)(imf_parameter(a, slope))
            basic_ssp.imf = imf
            if record is not None:
                record.append(slope)
        time_steps = np.copy(basic_sfr.t[:j])
        basic_ssp.calculate_feedback(float(metallicity), list(elements_to_trace), list(element_fractions),
                                     np.copy(time_steps))
        cube.advance_one_step(i + 1, np.copy(basic_ssp.table), np.copy(basic_ssp.sn2_table),
                              np.copy(basic_ssp.agb_table), np.copy(basic_ssp.sn1a_table),
                              np.copy(basic_ssp.bh_table))
        for item in elements_to_trace:
            if cube.cube[item][i] < 0:
                return -np.inf, [0]
        if cube.cube['gas'][i] < 0 or cube.gas_reservoir['gas'][i] < 0:
            return -np.inf, [0]

    abundances, elements, _ = mass_fraction_to_abundances(np.copy(cube.cube), np.copy(basic_solar.table))
    abundances = append_fields(abundances, 'weights', cube.cube['sfr'])
    abundances = append_fields(abundances, 'time', cube.cube['time'])
    abundances = np.array(abundances)
    for element in elements:
        if element != 'Fe':
            bad = np.where(np.logical_not(np.isfinite(abundances['Fe'])))
            abundances[element][bad] = np.inf
            good = np.where(np.isfinite(abundances['Fe']))
            abundances[element][good] -= abundances['Fe'][good]
    return cube, abundances


def simulate_star(star: StarParameters, yields: YieldSet = TNG_YIELDS, imf_type: str = "Chabrier_TNG",
                  slope_schedule: Optional[Callable[[float, float], float]] = None,
                  record_slopes: bool = False) -> SimulationResult:
    """Abundances of a star born at ``star.birth_time`` (Chempy's single_timestep_chempy path)."""
    a = chempy_parameters(yields, imf_type)
    _set_inputs(a, star)
    try:
        a = shorten_sfr(a, a.end - float(star.birth_time))
        if isinstance(a, float):            # np.inf: the SFR at birth is < 5% of the mean SFR
            return SimulationResult(None, "sfr_too_low")
        slopes = [] if record_slopes else None
        cube, abundances = evolve(a, slope_schedule, slopes)
        if isinstance(cube, float):
            return SimulationResult(None, "negative_mass")
        values = np.array([abundances[e][-1] for e in a.elements_to_trace], dtype=float)
        if not np.all(np.isfinite(values)) or np.all(values == 0):
            return SimulationResult(None, "non_finite")
        return SimulationResult(values, "", slopes or [])
    except Exception as error:              # Chempy raises on some corners of parameter space
        return SimulationResult(None, f"exception: {type(error).__name__}: {error}"[:200])


def simulate_star_stock(theta6, yields: YieldSet = TNG_YIELDS):
    """The unmodified Chempy entry point used by the existing GCE data sets
    (``single_timestep_chempy`` on [alpha, log10_N_Ia, log10_SFE, log10_SFR_scale, x_out, t]).
    Kept as the reference that :func:`simulate_star` is tested against."""
    from Chempy.cem_function import single_timestep_chempy
    a = chempy_parameters(yields)
    out = single_timestep_chempy((np.asarray(theta6, dtype=float), a))
    if isinstance(out, float):
        return None
    return np.asarray(out[0], dtype=float)
