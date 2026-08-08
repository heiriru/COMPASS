"""Reserve one free GPU before any test module imports torch.

This mirrors the pattern already used by every tutorial script in this repo:
``autocvd`` must run *before* torch is imported so it can restrict
``CUDA_VISIBLE_DEVICES`` to a single, currently-idle GPU. Without this,
``torch.device("cuda")`` (used by test_multiobs_analytic.py) silently defaults
to physical GPU 0, which may already be running another user's job on this
shared host.
"""
try:
    from autocvd import autocvd
    autocvd(num_gpus=1, interval=1)
except ImportError:
    pass
