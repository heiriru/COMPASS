"""Compatibility shim: churn now lives in ``compass.churn``.

This file used to hold a copy of the churned DPM-Solver-2 predictor. The
implementation has been lifted into the library, where it is the sampler's
**default** (eta = 2.0) and reachable as ``churn_eta=`` on ``SBIm.sample`` /
``Sampler.sample`` / ``MultiObsSampler.sample`` -- including on the
single-observation route, which had no churn path at all while this helper was a
monkeypatch over ``MultiObsSampler``.

``tests/test_churn.py`` pins the library version bit-identical to the patch that
produced every published churn number in this repository, so the tables that
cite this file remain reproducible.

The names below are kept so existing scripts import unchanged. Prefer passing
``churn_eta`` directly in new code; the monkeypatch route is unnecessary now and
cannot churn the ``n = 1`` sampler.
"""

import contextlib

from compass.churn import GAMMA_CAP, DEFAULT_ETA, apply_churn, churn_sigma
from compass.MultiObsSampler import MultiObsSampler

# Captured at import, before anything can patch it: the wrapper below runs
# *as* MultiObsSampler._dpm_sampler, so it must call the library body by an
# alias rather than by attribute lookup, or it recurses into itself.
_LIBRARY_DPM_SAMPLER = MultiObsSampler._dpm_sampler


def churned_dpm_sampler(eta, s_noise=1.0, sigma_min=0.0, sigma_max=float("inf")):
    """Deprecated. Pass ``churn_eta=eta`` to ``sample`` instead.

    A thin wrapper over the library predictor, so scripts that patch the sampler
    keep working and keep reproducing their published numbers.
    """
    def sampler(self, data, condition_mask, idx, order=2, snr=0.1,
                corrector_steps_interval=5, corrector_steps=5,
                final_corrector_steps=3, terminal_corrector_steps=0):
        if corrector_steps or final_corrector_steps or terminal_corrector_steps:
            raise ValueError(
                "churned_dpm_sampler is a corrector-free predictor; got "
                "corrector_steps=" + repr(corrector_steps)
                + ", final_corrector_steps=" + repr(final_corrector_steps)
                + ", terminal_corrector_steps=" + repr(terminal_corrector_steps)
                + "."
            )
        saved = (getattr(self, "churn_eta", DEFAULT_ETA),
                 getattr(self, "churn_s_noise", 1.0),
                 getattr(self, "churn_sigma_min", 0.0),
                 getattr(self, "churn_sigma_max", float("inf")))
        self.churn_eta = float(eta)
        self.churn_s_noise = float(s_noise)
        self.churn_sigma_min = float(sigma_min)
        self.churn_sigma_max = float(sigma_max)
        try:
            return _LIBRARY_DPM_SAMPLER(
                self, data, condition_mask, idx, order=order, snr=snr,
                corrector_steps_interval=corrector_steps_interval,
                corrector_steps=0, final_corrector_steps=0,
                terminal_corrector_steps=0,
            )
        finally:
            (self.churn_eta, self.churn_s_noise,
             self.churn_sigma_min, self.churn_sigma_max) = saved

    return sampler


@contextlib.contextmanager
def patched_dpm_sampler(builder):
    """Swap ``MultiObsSampler._dpm_sampler`` for one run, then restore it."""
    original = MultiObsSampler._dpm_sampler
    MultiObsSampler._dpm_sampler = builder
    try:
        yield
    finally:
        MultiObsSampler._dpm_sampler = original
