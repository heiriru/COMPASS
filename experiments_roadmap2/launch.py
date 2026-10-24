"""Run a main-tree experiment driver against a chosen COMPASS ``src`` with extra train kwargs.

    COMPASS_SRC=/path/to/src TRAIN_KW='{"time_loss_weight": "edm"}' \
        python launch.py /export/home/rheinric/COMPASS/experiments_arruda_schur/pp.py train ...

The drivers insert the main tree's ``src`` on sys.path themselves; this puts ``COMPASS_SRC``
in front of it, and wraps ``ScoreBasedInferenceModel.train`` so every call gets ``TRAIN_KW``
(explicit keyword arguments of the driver win only if TRAIN_KW does not name them).
"""

import json
import os
import runpy
import sys
from pathlib import Path

CPU_THREAD_LIMIT = 3
if len(os.sched_getaffinity(0)) > CPU_THREAD_LIMIT:
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT])
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(CPU_THREAD_LIMIT)

script = Path(sys.argv[1]).resolve()
src = os.environ["COMPASS_SRC"]
sys.path.insert(0, str(script.parent))
import common  # noqa: E402,F401  (inserts the main tree's src; ours goes in front)
sys.path.insert(0, src)

import compass  # noqa: E402
assert Path(compass.__file__).resolve().is_relative_to(Path(src).resolve()), compass.__file__
print(f"[launch] compass from {compass.__file__}; {len(os.sched_getaffinity(0))} CPUs", flush=True)

extra = json.loads(os.environ.get("TRAIN_KW", "{}"))
if extra:
    from compass import ScoreBasedInferenceModel
    original = ScoreBasedInferenceModel.train

    def train(self, *args, **kwargs):
        kwargs.update(extra)
        print(f"[launch] SBIm.train extra kwargs {extra}", flush=True)
        return original(self, *args, **kwargs)

    ScoreBasedInferenceModel.train = train

sys.argv = [str(script)] + sys.argv[2:]
os.chdir(script.parent)
runpy.run_path(str(script), run_name="__main__")
