"""Runtime setup for GPU validation scripts: 3-thread CPU cap + one free GPU via autocvd.

Import this first (instead of ``validation.common.runtime``) in any script that trains
or evaluates on a GPU. ``autocvd`` waits until a GPU is actually free, so other users'
jobs are never disturbed (see AGENTS.md).
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (ROOT, os.path.join(ROOT, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from Partial_Pooling.runtime import configure_runtime  # noqa: E402

CPU_INFO = configure_runtime(max_threads=3)
