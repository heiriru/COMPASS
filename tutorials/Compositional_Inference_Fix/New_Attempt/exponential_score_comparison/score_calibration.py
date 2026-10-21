#!/usr/bin/env python3
"""Is the learned score *miscalibrated*, and if so, how?

The composed posteriors in ``inverse_kinematics`` are systematically too narrow
(widths 0.72-0.85 of reference, 0.68 at n=16) and a Fisher-identity transplant
of the marginal score collapsed to width 0.156.  Both are consistent with one
hypothesis: the network's score has the **right shape but the wrong scale** --
a multiplicative error that is invisible at n=1 and amplified by composition,
which subtracts ``(n-1)`` prior scores from a sum of ``n`` row scores.

There the hypothesis could only be probed through a KDE.  Here it cannot: the
exponential hierarchy admits the exact diffused score at every noise level
(:func:`hierarchy.diffused_score`), so the learned field can be regressed
directly on the truth with no density estimate anywhere.

For each noise level ``lam`` and each observation ``x_j`` we draw states from
the *exact* diffused single-observation posterior ``p_lam(g_t, l_t | x_j)`` --
the states the sampler actually visits -- and fit

    learned = slope * exact + intercept

by ordinary least squares.  ``slope = 1`` is a calibrated score.  ``slope < 1``
is systematic under-shooting, and it is the quantity that composition
amplifies: an ``n``-fold sum against an ``(n-1)``-fold subtraction turns a
scale error of ``1 - slope`` into a composed-precision error of roughly
``n * (1 - slope)``.

The regression is reported per noise level and never pooled across them: the
score magnitude runs as ``1/lam``, so a pooled fit would measure the ladder
rather than the network.

Usage:

    python score_calibration.py --models h128d6_n200000
    python score_calibration.py --models h8d1_n200000 h8d1_n200000_log_sigma
    python score_calibration.py --replot
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from compass.Sampler import Sampler  # noqa: E402

import hierarchy  # noqa: E402
import train as recipe  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "artifacts" / "models"
OUTPUT_ROOT = ROOT / "artifacts" / "score_calibration"

# Log-spaced from the terminal noise the sampler denoises at up to the prior
# scale.  The bottom two rungs are the regime the composed posterior's width is
# decided in and the regime the t=0 product identity lives in.
LAMBDAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)

COMPONENTS = {"shared": hierarchy.GLOBAL_INDEX, "local": hierarchy.LOCAL_INDEX}


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def exact_and_learned(sampler, x, lam, count, seed, device):
    """Exact and learned row scores at states drawn from ``p_lam(.|x_j)``.

    Each observation gets its *own* diffused shared state -- these are
    uncomposed single-observation posteriors, not a synchronized trajectory,
    because the object under test is what the network supplies to a composition
    rule, before any rule touches it.

    Returns ``(exact, learned)``, both ``(N, count, 2)`` in ``(g, l)`` order.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    state = torch.zeros(n, count, hierarchy.NODES, dtype=torch.float64)
    exact = torch.empty(n, count, 2, dtype=torch.float64)

    for j, value in enumerate(x):
        shared, local, _, _ = hierarchy.sample_diffused(
            [value], lam, count, seed + 1000 * j
        )
        shared = torch.as_tensor(shared, dtype=torch.float64)
        local = torch.as_tensor(local, dtype=torch.float64)
        grid = hierarchy.quadrature_grid([value], points=4001)
        score_g, score_l = hierarchy.diffused_score(
            shared, local, [value], lam, grid
        )
        exact[j, :, 0] = score_g
        exact[j, :, 1] = score_l[:, 0]
        state[j, :, hierarchy.GLOBAL_INDEX] = shared
        state[j, :, hierarchy.LOCAL_INDEX] = local[:, 0]
        state[j, :, hierarchy.OBSERVED_INDEX] = float(value)

    mask = hierarchy.CONDITION_MASK.reshape(1, 1, -1).repeat(n, count, 1)
    t = sampler.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1)
    learned = sampler._get_score(
        state.to(dtype=torch.float32, device=device),
        t.to(device), mask.to(device), None,
    ).to(torch.float64).cpu()
    columns = [hierarchy.GLOBAL_INDEX, hierarchy.LOCAL_INDEX]
    return exact, learned[:, :, columns]


def calibration(exact, learned):
    """OLS of ``learned`` on ``exact`` plus the scale-free error summaries."""
    exact = np.asarray(exact, dtype=np.float64).reshape(-1)
    learned = np.asarray(learned, dtype=np.float64).reshape(-1)
    exact_centred = exact - exact.mean()
    learned_centred = learned - learned.mean()
    variance = float(np.mean(exact_centred**2))
    slope = float(np.mean(exact_centred * learned_centred) / max(variance, 1e-300))
    intercept = float(learned.mean() - slope * exact.mean())
    correlation = float(
        np.mean(exact_centred * learned_centred)
        / max(np.sqrt(variance * float(np.mean(learned_centred**2))), 1e-300)
    )
    residual = learned - exact
    scale = max(float(np.sqrt(np.mean(exact**2))), 1e-300)
    # What a post-hoc affine recalibration ``s -> (s - intercept) / slope`` could
    # remove: everything except the part of the error uncorrelated with the
    # truth. If this is close to ``relative_rmse`` the miscalibration is not
    # worth correcting -- the error is noise, not scale.
    corrected = (learned - intercept) / slope if abs(slope) > 1e-12 else learned
    return {
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
        "relative_rmse": float(np.sqrt(np.mean(residual**2)) / scale),
        "relative_rmse_after_affine": float(
            np.sqrt(np.mean((corrected - exact) ** 2)) / scale
        ),
        # In units of the exact score's own spread, so it is comparable across
        # noise levels even though the score magnitude runs as 1/lam.
        "bias_in_score_sd": float(residual.mean() / max(np.sqrt(variance), 1e-300)),
        "exact_score_sd": float(np.sqrt(variance)),
    }


def measure(model_name, arguments, device):
    directory = MODEL_ROOT / model_name
    checkpoint = directory / "Model_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    model = recipe.load_to(checkpoint, device)
    sampler = Sampler(model)
    # ``_get_score`` reads two attributes that ``sample()`` would have set; this
    # calls the score directly, so disable the attention capture explicitly.
    sampler.device = torch.device(device)
    sampler.model = model.model.to(device).eval()
    sampler.score_network_calls = 0
    sampler.evaluated_subject_rows = 0
    sampler.return_attn_weights = False
    sampler.attn_weights_time = torch.tensor(float("nan"))
    _, _, x = hierarchy.observations(arguments.observations, arguments.seed)

    rows = []
    for lam in arguments.lambdas:
        exact, learned = exact_and_learned(
            sampler, x, lam, arguments.states, arguments.seed, device
        )
        for name, index in COMPONENTS.items():
            column = 0 if index == hierarchy.GLOBAL_INDEX else 1
            summary = calibration(exact[:, :, column], learned[:, :, column])
            rows.append({"model": model_name, "lambda": lam,
                         "component": name, **summary})
            print(f"  lam={lam:<6g} {name:<7s} slope={summary['slope']:.3f} "
                  f"r={summary['correlation']:.4f} "
                  f"rel_rmse={summary['relative_rmse']:.3f} "
                  f"bias={summary['bias_in_score_sd']:+.3f}")
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_rows(rows, path):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows, path):
    models = sorted({row["model"] for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), layout="constrained")
    palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.75, max(len(models), 2)))
    for axis, (key, label, reference) in zip(axes, (
            ("slope", "OLS slope  (learned on exact)", 1.0),
            ("correlation", "correlation", 1.0),
            ("bias_in_score_sd", "mean error / score sd", 0.0))):
        for colour, model in zip(palette, models):
            for component, style in (("shared", "-o"), ("local", "--s")):
                picked = [r for r in rows
                          if r["model"] == model and r["component"] == component]
                picked.sort(key=lambda r: r["lambda"])
                if not picked:
                    continue
                axis.plot([r["lambda"] for r in picked],
                          [r[key] for r in picked], style, color=colour,
                          markersize=3.4, linewidth=1.4,
                          label=f"{model} · {component}")
        axis.axhline(reference, color="0.55", linewidth=0.9, zorder=0)
        axis.set_xscale("log")
        axis.set_xlabel(r"noise level $\lambda$")
        axis.set_title(label, fontsize=10)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.6, loc="lower right")
    figure.suptitle(
        "Learned score against the exact diffused score, exponential hierarchy",
        fontsize=11,
    )
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["h128d6_n200000"])
    parser.add_argument("--observations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--replot", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit(
            "refusing to touch a shared GPU without an explicit "
            "CUDA_VISIBLE_DEVICES; this host has other users on it"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    table = arguments.output / "calibration.csv"

    if arguments.replot:
        with table.open() as handle:
            rows = [{k: (v if k in ("model", "component") else float(v))
                     for k, v in row.items()}
                    for row in csv.DictReader(handle)]
    else:
        rows = []
        for model_name in arguments.models:
            print(f"{model_name}:")
            rows.extend(measure(model_name, arguments, arguments.device))
        write_rows(rows, table)
        (arguments.output / "config.json").write_text(
            json.dumps(vars(arguments), indent=2, default=str)
        )

    print(f"Wrote {table}")
    print(f"Wrote {plot(rows, arguments.output / '01_score_calibration.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
