"""Runtime guard applied before numerical libraries are imported.

Copied in behaviour from ``Partial_Pooling/runtime.py`` rather than imported:
this benchmark stands alone. Per the repository ``CLAUDE.md`` rule the cap is an
absolute thread count, not a percentage of the host CPU count, and per
``AGENTS.md`` anything that may touch a GPU selects one through ``autocvd``
first, so a training or sampling run cannot land on a card another user holds.
"""

import logging
import os

CPU_THREAD_LIMIT = 3


def configure_cpu_limit(max_threads=CPU_THREAD_LIMIT):
    """Restrict this process to at most ``max_threads`` logical CPUs.

    Must be called before NumPy/PyTorch are imported: the environment variables
    below are read once, when the native thread pools initialize.
    """
    total = os.cpu_count()
    if total is None or total < 1:
        raise RuntimeError("Cannot determine the host CPU count.")
    permitted = min(int(max_threads), total)
    if permitted < 1:
        raise RuntimeError(f"A cap of {max_threads} threads permits no CPUs.")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("This benchmark requires Linux CPU-affinity support.")
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("The current affinity mask exposes no CPUs.")
    permitted = min(permitted, len(available))
    # PID-offset block, as in Partial_Pooling: taking ``available[:permitted]``
    # pins every concurrent script to the same three logical CPUs.
    start = (os.getpid() * permitted) % len(available)
    active = sorted({available[(start + offset) % len(available)]
                     for offset in range(permitted)})
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
        "cpu_ids": active,
    }


def configure_runtime(max_threads=CPU_THREAD_LIMIT, gpu=True):
    """CPU cap plus, when ``gpu``, exactly one free GPU chosen by ``autocvd``."""
    info = configure_cpu_limit(max_threads)
    if not gpu:
        return info
    try:
        from autocvd import autocvd
    except ImportError as error:  # pragma: no cover - environment problem
        raise RuntimeError(
            "GPU runs require autocvd so a busy card is never taken."
        ) from error
    selected = autocvd(num_gpus=1, interval=1)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info("Selected physical GPU %s via autocvd.", list(selected))
    info["gpu_ids"] = list(selected)
    return info
