import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)
