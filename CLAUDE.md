# CPU usage cap for standalone scripts

This host has a large logical CPU count but a small cgroup CPU quota. PyTorch/NumPy/BLAS
auto-detect all logical CPUs and spawn that many intra-op threads, which then get throttled
by the cgroup — causing contention/context-switch overhead that's slower than just using a
handful of threads.

**Rule:** every standalone executable script (tutorials, benchmarks, CLI entry points) must
cap itself to **at most 3 threads/logical CPUs**, set *before* importing numeric libraries
(torch/numpy). Do this via `os.sched_setaffinity` plus the `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`
env vars.

- Canonical implementation: `Partial_Pooling/runtime.py` — `configure_cpu_limit(max_threads=3)`
  (and `configure_runtime(max_threads=3)` if GPU selection via `autocvd` is also needed).
- New scripts under `Partial_Pooling/` should import and call this helper rather than
  reimplementing the cap inline.
- Scripts elsewhere in the repo (`tutorials/`, `tests/`) that can't import `Partial_Pooling.runtime`
  should follow the same pattern: hardcode a `CPU_THREAD_LIMIT = 3` (not a percentage —
  the percentage varies by host and previously used a stale 6% figure) and reproduce the
  affinity + env-var logic above.
- Do not express the cap as a percentage of `os.cpu_count()`; express it as an absolute
  thread count (3), since the whole point is a small fixed number of threads regardless of
  host core count.
