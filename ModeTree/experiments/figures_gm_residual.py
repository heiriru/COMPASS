#!/usr/bin/env python
"""Figures for Experiment 1: where the GM composed field loses the data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "gm_residual"
FLOOR = 0.03167


def cancellation_panel(axis):
    """|s_post| / |Delta s| on the (g, lambda) plane, at one observation."""
    data = json.load(open(ARTIFACTS / "single_fields.json"))
    grid = np.array(data["grid"])
    lambdas = sorted({float(key.split("|")[1]) for key in data["fields"]})
    image = []
    for lam in lambdas:
        field = data["fields"][f"{1.50:+.2f}|{lam:.4g}"]
        residual = np.abs(np.array(field["s_exact"]) - np.array(field["s_prior"]))
        ratio = np.abs(np.array(field["s_exact"])) / np.maximum(residual, 1e-6)
        image.append(ratio)
    image = np.log10(np.clip(np.array(image), 0.3, 300))
    mesh = axis.pcolormesh(grid, np.arange(len(lambdas)), image,
                           cmap="magma", shading="nearest")
    axis.set_yticks(np.arange(len(lambdas)))
    axis.set_yticklabels([f"{lam:g}" for lam in lambdas], fontsize=7)
    axis.axhline(np.searchsorted(lambdas, FLOOR) - 0.5, color="cyan", lw=1.2,
                 ls="--")
    axis.set_xlabel("g")
    axis.set_ylabel("lambda")
    axis.set_title(r"$\log_{10}\,|s_{post}|/|\Delta s|$  (x = +1.5)", fontsize=9)
    axis.set_xlim(-5, 5)
    plt.colorbar(mesh, ax=axis, fraction=0.046)


def snr_panel(axis):
    rows = json.load(open(ARTIFACTS / "indistribution_rows.json"))
    for regime, style in (("training", "o-"), ("posterior", "s--")):
        selected = [r for r in rows if r["regime"] == regime]
        selected.sort(key=lambda r: r["lambda"])
        axis.plot([r["lambda"] for r in selected],
                  [r["snr_residual"] for r in selected], style, label=regime)
    axis.axhline(1.0, color="0.5", lw=0.8)
    axis.axvline(FLOOR, color="crimson", lw=1.0, ls=":")
    axis.text(FLOOR * 1.05, 0.06, "trained floor", color="crimson", fontsize=7,
              rotation=90, va="bottom")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("lambda")
    axis.set_ylabel(r"$|\Delta s^*| \,/\, |\Delta s_\phi - \Delta s^*|$")
    axis.set_title("likelihood-residual signal-to-error", fontsize=9)
    axis.legend(fontsize=7)


def scaling_panel(axis):
    rows = json.load(open(ARTIFACTS / "ladder_rows.json"))
    for lam, colour in zip((0.5, 0.1, 0.05), ("C0", "C1", "C2")):
        selected = sorted([r for r in rows if r["lambda"] == lam],
                          key=lambda r: r["n"])
        n = np.array([r["n"] for r in selected])
        axis.plot(n, [r["rms_network_error"] for r in selected], "o-",
                  color=colour, label=f"network error, lam={lam:g}")
        axis.plot(n, [r["rms_composed_signal"] for r in selected], "^:",
                  color=colour, alpha=0.6, label=f"signal, lam={lam:g}")
    n = np.array([1, 128])
    axis.plot(n, 1.0 * n, "k--", lw=0.8)
    axis.text(20, 14, r"$\propto n$", fontsize=8)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("n")
    axis.set_ylabel("RMS")
    axis.set_title(r"composed error grows as $n^{1.00}$: coherent bias", fontsize=9)
    axis.legend(fontsize=6)


def pinning_panel(axis):
    data = json.load(open(ARTIFACTS / "ladder_fields.json"))
    grid = np.array(data["grid"])
    # each curve is scaled to unit peak: the exact posterior narrows by a
    # factor of 13 across the ladder and the learned one by 40, so on a shared
    # density axis every curve but the narrowest is invisible.
    for n, colour in zip((1, 8, 32, 128), ("C0", "C1", "C2", "C3")):
        field = data["fields"][f"{n}|0.05"]
        exact = np.array(field["clean"])
        learned = np.array(field["density_learned"])
        axis.plot(grid, exact / exact.max(), "-", color=colour, lw=1.4,
                  label=f"exact n={n}")
        axis.plot(grid, learned / learned.max(), "--", color=colour, lw=1.0)
    for centre in (-2.5, 2.5):
        axis.axvline(centre, color="0.6", lw=0.8, ls=":")
    axis.text(-2.45, 1.02, "prior modes", color="0.4", fontsize=7)
    axis.set_xlim(-3.5, 3.5)
    axis.set_xlabel("g")
    axis.set_ylabel("density / peak")
    axis.set_title("solid exact, dashed learned (lam=0.05):\n"
                   "the learned modes never leave the prior", fontsize=9)
    axis.legend(fontsize=6)


def main():
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    cancellation_panel(axes[0, 0])
    snr_panel(axes[0, 1])
    scaling_panel(axes[1, 0])
    pinning_panel(axes[1, 1])
    figure.suptitle("Gaussian mixture: the composed first-order field, "
                    "against the exact diffused score", fontsize=11)
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "gm_residual.png", dpi=150)
    print(f"wrote {ARTIFACTS / 'gm_residual.png'}")


if __name__ == "__main__":
    main()
