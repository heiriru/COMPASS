#!/usr/bin/env python3
"""Train the IK network for the layout with TWO globals: (angle_1, height_arm).

The node order is unchanged -- ``[angle_1, height_arm, angle_2, angle_3, x1, x2]``
-- so the training rows are byte-identical to the angle_1-only checkpoint
``legacy_h16d2_n200000_six_weighted_log_sigma`` (same seed, same generator
names). The only thing that changes is which conditionals the six-conditional
mask menu draws: with ``hierarchical_roles(schema, 2, 2)``, ``p(g|l,x)`` and
``p(l|g,x)`` now split the latents 2+2 instead of 1+3. That is what the
compositional sampler asks for when ``hierarchy=[0, 1]``.

Usage (from COMPASS/):
    .COMPASS/bin/python inverse_kinematics/recovery/train_height_global.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IK = HERE.parent
sys.path[:0] = [str(HERE), str(IK)]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd  # noqa: E402

    autocvd(num_gpus=1, interval=1)

import json  # noqa: E402

import hierarchical as H  # noqa: E402
import layouts  # noqa: E402

layout = layouts.LAYOUTS["angle_height"]
H.GLOBAL_NAMES = list(layout.global_names)
H.LOCAL_NAMES = list(layout.local_names)

import train_model as trainer  # noqa: E402

SUFFIX = "_globalheight"
ARGS = ["--arch", "legacy", "--config", "h16d2", "--simulations", "200000",
        "--mask-menu", "six_weighted", "--time-sampling", "log_sigma",
        "--seed", "7", "--output-suffix", SUFFIX]

if __name__ == "__main__":
    status = trainer.main(ARGS + sys.argv[1:])
    directory = layout.checkpoint.parent
    (directory / "partition.json").write_text(json.dumps({
        "global_names": H.GLOBAL_NAMES, "local_names": H.LOCAL_NAMES,
        "hierarchy": list(layout.hierarchy),
        "note": "six_weighted roles (2 global, 2 local); rows identical to the angle_1-only checkpoint",
    }, indent=2))
    raise SystemExit(status)
