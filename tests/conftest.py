"""Reserve one free GPU before any test module imports torch.

This mirrors the pattern already used by every tutorial script in this repo:
``autocvd`` must run *before* torch is imported so it can restrict
``CUDA_VISIBLE_DEVICES`` to a single, currently-idle GPU. Without this,
``torch.device("cuda")`` (used by test_multiobs_analytic.py) silently defaults
to physical GPU 0, which may already be running another user's job on this
shared host.

An explicit ``CUDA_VISIBLE_DEVICES`` is a deliberate choice of device and wins
over the search: ``autocvd`` blocks until a GPU is *idle*, which on this shared
host can be never, and a test session that hangs at collection looks identical
to one that is merely slow.
"""
import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    try:
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
    except ImportError:
        pass


import pytest


@pytest.fixture(autouse=True)
def _restore_torch_default_dtype():
    """Undo a test's change of torch's global default dtype.

    ``pd_study/clamp_study.py`` calls ``torch.set_default_dtype(torch.float64)`` at
    import, and ``test_conservative_clamp.py`` imports it inside a test; every test
    after it then ran in float64 and ~75 of them failed with Double/Float mismatches
    in a full-suite run while passing in isolation.
    """
    import torch

    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)
