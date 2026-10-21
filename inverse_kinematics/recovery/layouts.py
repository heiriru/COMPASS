"""The two global/local partitions of the arm studied here.

Both use the same node order ``[angle_1, height_arm, angle_2, angle_3 | x1, x2]``
(see ``hierarchical.py``); they differ only in how many leading latents are
shared across observations.

* ``angle``        -- angle_1 global; (height, angle_2, angle_3) local. The
                      layout every earlier IK experiment used.
* ``angle_height`` -- (angle_1, height) global; (angle_2, angle_3) local.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simulator as sim  # noqa: E402

IK = Path(__file__).resolve().parents[1]
MODELS = IK / "artifacts" / "models"
NODE_NAMES = ("angle_1", "height_arm", "angle_2", "angle_3")
PRIOR_STD = np.array([0.5, sim.PRIOR_SCALES[0], 0.5, 0.5])


@dataclass(frozen=True)
class Layout:
    name: str
    n_global: int
    checkpoint: Path

    @property
    def hierarchy(self):
        return tuple(range(self.n_global))

    @property
    def local_indices(self):
        return tuple(range(self.n_global, 4))

    @property
    def global_names(self):
        return NODE_NAMES[:self.n_global]

    @property
    def local_names(self):
        return NODE_NAMES[self.n_global:]


LAYOUTS = {
    "angle": Layout("angle", 1, MODELS / "legacy_h16d2_n200000_six_weighted_log_sigma" / "Model_checkpoint.pt"),
    "angle_height": Layout("angle_height", 2, MODELS / "legacy_h16d2_n200000_six_weighted_log_sigma_globalheight"
                           / "Model_checkpoint.pt"),
}
