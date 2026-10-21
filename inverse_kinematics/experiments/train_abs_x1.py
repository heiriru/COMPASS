#!/usr/bin/env python3
"""Train the IK network for the |x1| variant: the arm observes (|x1|, x2) instead of (x1, x2).

The flip (angle_1, height, angle_2, angle_3) -> -(...) maps x1 -> -x1 and leaves x2 alone, and every
prior is a zero-mean Gaussian, so with |x1| observed the likelihood and prior are both invariant under
it: the angle_1 posterior is exactly symmetric at every n, with no selection of the data. That makes it
a bimodal benchmark whose truth, exact reference and training distribution agree.

Everything else matches the stock angle_1-global checkpoint
``legacy_h16d2_n200000_six_weighted_log_sigma`` (same seed, generators, arch, menu, time sampling);
only ``hierarchical.forward`` is wrapped, so the training rows are the stock rows with x1 -> |x1|.

Usage (from COMPASS/):
    .COMPASS/bin/python inverse_kinematics/experiments/train_abs_x1.py
"""
import os
import sys
from pathlib import Path

IK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(IK)]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd  # noqa: E402

    autocvd(num_gpus=1, interval=1)

import json  # noqa: E402

import numpy as np  # noqa: E402

import hierarchical as H  # noqa: E402

_forward = H.forward


def forward_abs_x1(theta_nodes):
    x = np.array(_forward(theta_nodes), copy=True)
    x[..., 0] = np.abs(x[..., 0])
    return x


H.forward = forward_abs_x1          # H.simulate looks forward up at call time

import train_model as trainer  # noqa: E402

SUFFIX = "_absx1"
ARGS = ["--arch", "legacy", "--config", "h16d2", "--simulations", "200000",
        "--mask-menu", "six_weighted", "--time-sampling", "log_sigma",
        "--seed", "7", "--output-suffix", SUFFIX]
CHECKPOINT = IK / "artifacts" / "models" / f"legacy_h16d2_n200000_six_weighted_log_sigma{SUFFIX}" / "Model_checkpoint.pt"

if __name__ == "__main__":
    status = trainer.main(ARGS + sys.argv[1:])
    (CHECKPOINT.parent / "variant.json").write_text(json.dumps({
        "observation": ["abs(x1)", "x2"],
        "note": "stock angle_1-global rows with x1 -> |x1|; angle_1 posterior exactly symmetric",
    }, indent=2) + "\n")
    raise SystemExit(status)
