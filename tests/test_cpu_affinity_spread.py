"""The per-process CPU cap must not pin every process to the same cores.

CLAUDE.md caps each standalone script at three logical CPUs. That cap is per
process; taking ``available[:3]`` satisfies it while stacking every concurrent
job onto CPUs 0-2, which is what happened in practice -- three cores at 100%,
the other 69 at ~90% idle, and a run queue above 50.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "Partial_Pooling"

# Several benchmark modules call configure_cpu_limit at import time, so a full
# pytest session can narrow its *own* mask to three CPUs before this test runs.
# Children inherit that mask, which would make every child see the same three
# CPUs no matter what the helper does. The child therefore widens its mask back
# to the whole machine first, and the test skips if it is not allowed to.
SNIPPET = """
import os, sys
try:
    os.sched_setaffinity(0, range(os.cpu_count()))
except OSError:
    print("DENIED")
    raise SystemExit(0)
sys.path.insert(0, {root!r})
from runtime import configure_cpu_limit
configure_cpu_limit(3)
print(",".join(str(c) for c in sorted(os.sched_getaffinity(0))))
"""


def affinity_of_a_fresh_process():
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET.format(root=str(RUNTIME))],
        capture_output=True, text=True, check=True,
    )
    output = result.stdout.strip()
    if output == "DENIED":
        pytest.skip("not permitted to widen this process's CPU affinity")
    return tuple(int(c) for c in output.split(","))


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"),
                    reason="requires Linux CPU affinity")
def test_cap_is_respected():
    assert len(affinity_of_a_fresh_process()) <= 3


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"),
                    reason="requires Linux CPU affinity")
def test_concurrent_processes_do_not_all_land_on_the_same_cpus():
    # Checked at call time, not collection time: an earlier test in the session
    # may already have narrowed this process's mask.
    if (os.cpu_count() or 0) < 12:
        pytest.skip("needs a machine with room to spread")
    masks = {affinity_of_a_fresh_process() for _ in range(8)}
    assert len(masks) > 1, f"every process was pinned to the same CPUs: {masks}"


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"),
                    reason="requires Linux CPU affinity")
def test_every_chosen_cpu_was_actually_available():
    assert set(affinity_of_a_fresh_process()) <= set(range(os.cpu_count()))
