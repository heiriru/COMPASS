#!/usr/bin/env python3
"""Run the main tree's partial-pooling driver (experiments_arruda_schur/pp.py) against this
branch's ``src`` with the network covariance output switched on. No main-tree file is edited;
outputs land in the driver's own ``artifacts/pp/<...>_<variant>`` folders.

    COVOUT_TRAIN=joint|detached   attach the covariance output and train it (second-order DSM)
    COVOUT_NETCOV=1               at inference, use the network's covariance output in place of
                                  the score Jacobian (needs a ``gauss_jacobian*`` method)
    COVOUT_RANK=8, COVOUT_LATENT=10 (latent nodes are 0..COVOUT_LATENT-1, the rest observed)
    TRAIN_KW='{...}'              extra SBIm.train kwargs, as experiments_roadmap2/launch.py

    python launch_pp.py /export/home/rheinric/COMPASS/experiments_arruda_schur/pp.py train ...
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

SRC = str(Path(__file__).resolve().parents[1] / "src")
script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parent))
import common  # noqa: E402,F401  (inserts the main tree's src; ours goes in front)
sys.path.insert(0, SRC)
import compass  # noqa: E402
assert Path(compass.__file__).resolve().is_relative_to(Path(SRC).resolve()), compass.__file__
from compass import MultiObsSampler, ScoreBasedInferenceModel  # noqa: E402
from compass.covariance_output import (  # noqa: E402
    CovarianceObjective, NetworkCovarianceHead, enable_covariance_output,
)
print(f"[covout] compass from {compass.__file__}", flush=True)

RANK = int(os.environ.get("COVOUT_RANK", "8"))
LATENT = int(os.environ.get("COVOUT_LATENT", "10"))
mode = os.environ.get("COVOUT_TRAIN", "")
extra = json.loads(os.environ.get("TRAIN_KW", "{}"))
original_train = ScoreBasedInferenceModel.train


def train(self, *args, **kwargs):
    kwargs.update(extra)
    if mode:
        head = enable_covariance_output(self, rank=RANK)
        kwargs["covariance_objective"] = CovarianceObjective(head, weight=1.0,
                                                             detach_trunk=(mode == "detached"))
        print(f"[covout] training with covariance output ({mode}, rank {RANK})", flush=True)
    return original_train(self, *args, **kwargs)


ScoreBasedInferenceModel.train = train

if os.environ.get("COVOUT_NETCOV") == "1":
    # attach at sample time: SBIm builds its sampler before the checkpoint is loaded
    original_sample = MultiObsSampler.sample

    def sample(self, *args, **kwargs):
        if getattr(self, "covariance_head", None) is None:
            SBIm = self.SBIm
            if getattr(SBIm, "covariance_output", None) is None:
                raise RuntimeError("COVOUT_NETCOV=1 but the checkpoint has no covariance output")
            self.set_covariance_head(NetworkCovarianceHead(
                SBIm, list(range(LATENT)), list(range(LATENT, SBIm.covariance_output.nodes))))
            print("[covout] sampler uses the network covariance output", flush=True)
        return original_sample(self, *args, **kwargs)

    MultiObsSampler.sample = sample

sys.argv = [str(script)] + sys.argv[2:]
os.chdir(script.parent)
runpy.run_path(str(script), run_name="__main__")
