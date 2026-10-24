"""Framework-agnostic helpers: wall-clock timing and relative error."""

import time

import numpy as np

WARMUP, REPEATS = 5, 30


def time_call(call, warmup=WARMUP, repeats=REPEATS):
    """Median and spread, in milliseconds, of ``call()`` after warm-up.

    ``call`` must block until the device is done (``block_until_ready`` in
    JAX, ``torch.cuda.synchronize`` in PyTorch). The first warm-up call also
    absorbs compilation, which is reported separately.
    """
    start = time.perf_counter()
    call()
    first = time.perf_counter() - start
    for _ in range(warmup - 1):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    samples = 1e3 * np.asarray(samples)
    return {"median_ms": float(np.median(samples)),
            "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90)),
            "first_call_ms": 1e3 * first}


def relative_error(estimate, exact):
    """Frobenius-norm relative error, pooled over the batch."""
    estimate = np.asarray(estimate, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    return float(np.linalg.norm(estimate - exact) / np.linalg.norm(exact))
