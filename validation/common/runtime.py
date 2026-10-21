"""Runtime setup for validation scripts: 3-thread CPU cap before numeric imports.

Import this module first in every script (see the repository CLAUDE.md).
GPU runs must additionally go through ``Partial_Pooling.runtime.configure_runtime``
(autocvd). Everything in A1/A2 runs on CPU.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (ROOT, os.path.join(ROOT, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from Partial_Pooling.runtime import configure_cpu_limit  # noqa: E402

CPU_INFO = configure_cpu_limit(max_threads=3)
