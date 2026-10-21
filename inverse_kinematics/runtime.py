"""Runtime guard applied before numerical libraries are imported.

Copied in behaviour from ``Partial_Pooling/runtime.py`` rather than imported:
this package stands alone and must not take a dependency on the partial-pooling
benchmark. Per the repository ``CLAUDE.md`` rule, the cap is an absolute thread
count, not a percentage of the host CPU count.

No GPU variant is provided. Nothing in this package touches a GPU: the simulator
is closed-form NumPy and the figures are matplotlib.
"""

import os

CPU_THREAD_LIMIT = 3


def configure_cpu_limit(max_threads=CPU_THREAD_LIMIT):
    """Restrict this process to at most ``max_threads`` logical CPUs.

    Must be called before NumPy or matplotlib are imported: the environment
    variables below are read once, when the native thread pools initialize.
    """
    total = os.cpu_count()
    if total is None or total < 1:
        raise RuntimeError("Cannot determine the host CPU count.")
    permitted = min(int(max_threads), total)
    if permitted < 1:
        raise RuntimeError(f"A cap of {max_threads} threads permits no CPUs.")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("This package requires Linux CPU-affinity support.")
    available = sorted(os.sched_getaffinity(0))
    active = available[: min(permitted, len(available))]
    if not active:
        raise RuntimeError("The current affinity mask exposes no CPUs.")
    os.sched_setaffinity(0, active)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(len(active))
    return {
        "host_cpus": total,
        "active_cpus": len(active),
        "percent": 100.0 * len(active) / total,
    }
