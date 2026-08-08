"""Runtime guard used before numerical libraries are imported by CLI scripts."""

import logging
import os


def configure_runtime(max_threads=3):
    """Limit CPU use and select exactly one free GPU before numeric imports."""
    cpu_info = configure_cpu_limit(max_threads)
    try:
        from autocvd import autocvd
    except ImportError as error:
        raise RuntimeError(
            "The partial-pooling benchmark requires autocvd for safe GPU selection."
        ) from error
    gpu_ids = autocvd(num_gpus=1, interval=1)
    if len(gpu_ids) != 1:
        raise RuntimeError(f"autocvd selected {len(gpu_ids)} GPUs; expected exactly one.")
    logging.info("Selected physical GPU %s via autocvd.", gpu_ids[0])
    return cpu_info


def configure_cpu_limit(max_threads=3):
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
    active = available[: min(permitted, len(available))]
    if not active:
        raise RuntimeError("The current affinity mask exposes no CPUs.")
    os.sched_setaffinity(0, active)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(len(active))
    info = {
        "host_cpus": total,
        "active_cpus": len(active),
        "active_fraction": len(active) / total,
        "cpu_ids": active,
    }
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info(
        "CPU affinity: %d/%d CPUs (%.3f%%)",
        len(active), total, 100 * info["active_fraction"],
    )
    if len(active) > max_threads:
        raise RuntimeError(f"The active CPU affinity exceeds the requested {max_threads}-thread cap.")
    return info
