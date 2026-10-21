#!/usr/bin/env python3
"""How much of the diffusion-time embedding a trained network actually uses.

``time_embedding_size`` is the width of the sinusoidal embedding of the scalar
diffusion time, and it also sets the input width of every adaLN modulation, so
it dominates the parameter budget at small ``hidden_size``: 131,584 of the
463,364 parameters of the trial-h64/d2 arm, 28.4% of the model, spent on a
function of one variable.

Whether that width buys anything is answerable directly. Everything downstream
of the embedding sees only the modulation vectors it produces, so sweep t across
its training range, collect those vectors, and take their singular values. The
number of directions needed to reproduce the sweep is the *effective* width; any
excess is capacity the network cannot use no matter how it is trained.

The same sweep also compares two trained arms: if a 64-wide embedding tracks the
256-wide one's modulation curve, the width was redundant rather than informative.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit(3)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.model_pipeline import load_model  # noqa: E402
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402

NAME = "sde_joint"


def time_grid(count=512, eps=1e-3):
    """The diffusion times training actually visits, log-spaced in sigma."""
    return torch.linspace(eps, 1.0, count).unsqueeze(1)


def modulation_sweep(model, grid):
    """``{block: (times, 6*hidden)}`` modulation vectors along the time grid."""
    network = model.model
    out = {}
    with torch.no_grad():
        embedded = network.t_embedder(grid)
        out["t_embedder"] = embedded.cpu().numpy()
        for index, block in enumerate(network.blocks):
            out[f"block{index}"] = block.adaLN_modulation(embedded).cpu().numpy()
        out["final"] = network.final_layer.adaLN_modulation(embedded).cpu().numpy()
    return out


def spectrum(matrix):
    """Singular values of the centred sweep, and the rank that holds 99%/99.9%."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    values = np.linalg.svd(centred, compute_uv=False)
    energy = np.cumsum(values**2) / max(np.sum(values**2), 1e-30)
    return {
        "singular_values": values[:16].tolist(),
        "rank_99": int(np.searchsorted(energy, 0.99) + 1),
        "rank_999": int(np.searchsorted(energy, 0.999) + 1),
        "width": int(matrix.shape[1]),
    }


def parse_arm(text):
    label, _, overrides = text.partition(":")
    kwargs = {}
    for item in filter(None, overrides.split(",")):
        key, _, value = item.partition("=")
        if value in ("True", "False"):
            value = value == "True"
        elif value.replace(".", "", 1).isdigit():
            value = float(value) if "." in value else int(value)
        kwargs[key] = value
    return label, kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="LABEL:key=value,... overrides for get_config")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--times", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    paths = BenchmarkPaths.default().ensure()
    grid = time_grid(args.times)
    records = []
    for entry in args.arm:
        label, kwargs = parse_arm(entry)
        config = get_config(args.preset, **kwargs)
        model = load_model(config, NAME, paths, device="cpu")
        model.model.eval()
        sweep = modulation_sweep(model, grid)
        records.append({
            "label": label,
            "time_embedding_size": model.time_embedding_size,
            "parameters": sum(p.numel() for p in model.model.parameters()),
            "spectra": {key: spectrum(value) for key, value in sweep.items()},
        })

    print(f"{'arm':14s} {'t_emb':>6s} {'params':>9s} {'component':12s} "
          f"{'width':>6s} {'rank99':>7s} {'rank99.9':>9s}")
    for record in records:
        for component, data in record["spectra"].items():
            print(f"{record['label'][:14]:14s} {record['time_embedding_size']:6d} "
                  f"{record['parameters']:9,d} {component:12s} "
                  f"{data['width']:6d} {data['rank_99']:7d} {data['rank_999']:9d}")

    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
