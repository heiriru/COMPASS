"""Runtime guard: call before any numeric library is imported.

Mirrors ``Partial_Pooling/runtime.py`` -- see the repository ``CLAUDE.md``: this
host has a large logical CPU count and a small cgroup quota, so every standalone
script pins itself to at most three logical CPUs before torch/numpy read the
host core count.
"""

import logging
import os

CPU_THREAD_LIMIT = 3


def configure_cpu_limit(max_threads=CPU_THREAD_LIMIT):
    """Restrict this process to at most ``max_threads`` logical CPUs."""
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
    # Offset by PID so concurrent scripts do not all land on CPUs 0-2.
    start = (os.getpid() * permitted) % len(available)
    active = [available[(start + offset) % len(available)]
              for offset in range(permitted)]
    os.sched_setaffinity(0, sorted(set(active)))
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(len(active))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info("CPU affinity: %d/%d CPUs", len(active), total)
    return {"host_cpus": total, "active_cpus": len(active), "cpu_ids": active}


def configure_runtime(max_threads=CPU_THREAD_LIMIT):
    """CPU cap plus ``autocvd`` selection of exactly one free GPU.

    This host is shared. Never pick a GPU by hand in these scripts; autocvd is
    what guarantees the device is actually idle.
    """
    cpu_info = configure_cpu_limit(max_threads)
    try:
        from autocvd import autocvd
    except ImportError as error:
        raise RuntimeError(
            "GPU runs require autocvd for safe GPU selection on this shared host."
        ) from error
    gpu_ids = autocvd(num_gpus=1, interval=1)
    if len(gpu_ids) != 1:
        raise RuntimeError(f"autocvd selected {len(gpu_ids)} GPUs; expected exactly one.")
    logging.info("Selected physical GPU %s via autocvd.", gpu_ids[0])
    return cpu_info
