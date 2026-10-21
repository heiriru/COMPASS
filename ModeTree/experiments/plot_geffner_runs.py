#!/usr/bin/env python
"""Plot the completed original-Geffner F-NPSE Gaussian-mixture and IK runs."""

import os


def _configure_cpu_limit():
    host = os.cpu_count() or 1
    allowed = sorted(os.sched_getaffinity(0))
    limit = int(host * 0.06)
    if limit < 1:
        raise RuntimeError("6% CPU limit leaves fewer than one logical CPU")
    active = min(limit, len(allowed))
    os.sched_setaffinity(0, set(allowed[:active]))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(active)
    print(f"CPU affinity: {active}/{host} CPUs", flush=True)


_configure_cpu_limit()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FIGURES = ARTIFACTS / "figures"


def gaussian_mixture():
    """Plot measured diagnostics from the original F-NPSE step ladder."""
    payload = json.loads((ARTIFACTS / "gm_oracle" /
                          "geffner_ladder_mu7.5.json").read_text())["128"]
    entries = []
    for steps in (10, 50, 200):
        row = payload[f"fnpe/T=100,L={steps},snr=0.05"]
        entries.append((steps, row["w1_over_sd"], row["width_ratio"],
                        row["weight_tv"]))
    values = np.asarray(entries, dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(values[:, 0], values[:, 1], "o-", color="#d55e00",
                 label="F-NPSE Langevin")
    axes[0].set(xscale="log", xlabel="inner Langevin steps per level (L)",
                ylabel="W1 / posterior sd", title="Gaussian mixture, n=128")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(values[:, 0], values[:, 2], "o-", color="#0072b2",
                 label="sampled / exact width")
    axes[1].axhline(1.0, color="black", lw=1, ls="--", label="exact width")
    axes[1].set(xscale="log", xlabel="inner Langevin steps per level (L)",
                ylabel="width ratio", title="Original 1/n step scaling")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    path = FIGURES / "geffner_original_gm_n128_ladder.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def inverse_kinematics():
    """Overlay the actual ABC reference and actual sampler density."""
    payload = json.loads((ARTIFACTS /
                          "ik_geffner_original_L100_n128_gpu6_panels.json").read_text())
    panel = payload["bimodal|128"]
    grid = np.asarray(panel["grid"])
    exact = np.asarray(panel["exact"])
    sampled = np.asarray(panel["curves"]["langevin_fnpse"])

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(grid, exact, color="#0072b2", lw=2.3, label="ABC reference")
    axis.plot(grid, sampled, color="#d55e00", lw=2.0,
              label="original F-NPSE Langevin")
    axis.set(xlim=(-1.0, 1.0), xlabel=r"shared angle $a_1$", ylabel="density",
             title="Inverse kinematics, bimodal data, n=128")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    axis.text(0.02, 0.96, "100 levels × 100 inner steps; 2,000 walkers",
              transform=axis.transAxes, va="top")
    path = FIGURES / "geffner_original_ik_n128.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (gaussian_mixture(), inverse_kinematics()):
        print(path)


if __name__ == "__main__":
    main()
