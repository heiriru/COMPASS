"""Shared machinery for the Simformer comparison.

Import order matters: every executable script must call
``common.runtime.configure_cpu_limit()`` *before* torch is imported, so this
package deliberately does not import torch at module level.
"""

from . import runtime  # noqa: F401  (safe: no numeric imports)

__all__ = ["runtime", "paper", "tasks", "masks", "metrics", "reference",
           "model", "paths"]
