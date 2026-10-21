"""Shared plumbing for the latent-attention / arruda_schur experiments.

Nothing here edits a benchmark's files. Each driver imports the benchmark's own
modules and patches them in-process:

- ``ConfiguredSampler`` is a ``MarginalSchurSampler`` whose settings come from
  ``ACTIVE``, so a benchmark that constructs ``MultiObsSampler(network)``
  itself can be pointed at it by rebinding ``compass.MultiObsSampler.MultiObsSampler``.
  Production corrections pass straight through to the parent, unchanged.
- ``NEW_METHODS`` maps method names to ``(sampler settings, sampler kwargs)``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPASS = ROOT.parent
sys.path.insert(0, str(COMPASS / "src"))

ACTIVE = {}

DPM = {"method": "dpm", "order": 2, "corrector_steps": 0,
       "final_corrector_steps": 0, "terminal_corrector_steps": 0}

# name -> (MarginalSchurSampler settings, sampler.sample kwargs)
NEW_METHODS = {
    "gauss_jacobian_FD": ({}, {**DPM, "correction": "gauss_jacobian_FD"}),
    "arruda_direct": ({"estimator": "direct"}, {**DPM, "correction": "arruda_schur"}),
    "arruda_schur": ({"estimator": "schur"}, {**DPM, "correction": "arruda_schur"}),
    "arruda_laplace": ({"estimator": "laplace"}, {**DPM, "correction": "arruda_schur"}),
    "arruda_gauss_direct": ({"estimator": "direct"},
                            {**DPM, "correction": "arruda_schur_gauss"}),
    "arruda_gauss_laplace": ({"estimator": "laplace"},
                             {**DPM, "correction": "arruda_schur_gauss"}),
}


def configured_sampler_class():
    from compass.experimental.marginal_schur import MarginalSchurSampler

    class ConfiguredSampler(MarginalSchurSampler):
        def __init__(self, SBIm):
            super().__init__(SBIm, **ACTIVE)

    return ConfiguredSampler


def install_configured_sampler():
    """Make every ``MultiObsSampler(...)`` constructed afterwards a ``ConfiguredSampler``."""
    import compass

    # ``compass/__init__`` star-imports the class over the submodule's name, so
    # ``import compass.MultiObsSampler as m`` returns the *class*. The module
    # that ``from compass.MultiObsSampler import MultiObsSampler`` reads is the
    # one in sys.modules.
    module = sys.modules["compass.MultiObsSampler"]
    cls = configured_sampler_class()
    module.MultiObsSampler = cls
    compass.MultiObsSampler = cls
    return cls


def load_with_latent_attention(loader):
    """Wrap a benchmark's checkpoint loader so a sidecar'd mode is re-applied."""
    from compass.experimental.latent_attention import load_latent_attention

    def load(checkpoint, *args, **kwargs):
        network = loader(checkpoint, *args, **kwargs)
        load_latent_attention(network, Path(checkpoint).parent, required=False)
        return network

    return load


def build_with_latent_attention(builder, mode, roles_of):
    """Wrap a benchmark's network builder so the new network gets ``mode``."""
    from compass.experimental.latent_attention import enable_latent_attention

    def build(*args, **kwargs):
        network = builder(*args, **kwargs)
        if mode != "none":
            enable_latent_attention(network, mode, roles_of(network))
        return network

    return build
