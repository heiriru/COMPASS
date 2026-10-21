#!/usr/bin/env python3
"""Where does the shared-coordinate bias come from? Sweep the observation count.

At ``n = 1`` the arrow system collapses to the network's own score -- no
composition happens at all -- so any error there is the *network's*. As ``n``
grows, composition averages error that is independent across observations down
by roughly ``sqrt(n)`` while leaving error that is coherent across them
untouched. That gives two distinguishable predictions for the shared posterior:

* **coherent network bias**: the absolute shift stays roughly constant while the
  posterior narrows, so ``bias / sigma`` *grows* with ``n``
* **composition-rule error**: the shift is absent at ``n = 1`` (where the rule is
  a no-op) and appears only once observations are combined

Both are measured against :func:`hierarchical.reference_global`, which passed
simulation-based calibration (``sbc_reference.py``), so it can referee this.

Usage:
    python inverse_kinematics/observation_sweep.py --counts 1 2 4 8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

CPU_INFO = configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchical as H  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402

ROOT = Path(__file__).resolve().parent


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    result.add_argument("--config", default="h16d2")
    result.add_argument("--arch", default="legacy")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--draws", type=int, default=2_000)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--pilot-draws", type=int, default=2_000)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--reference-draws", type=int, default=100_000)
    result.add_argument("--reference-bandwidth", type=float, default=0.05)
    result.add_argument("--seed", type=int, default=3)
    result.add_argument("--device", default="cuda")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    directory = trainer.model_directory(
        arguments.artifacts, arguments.config, arguments.arch,
        arguments.simulations,
    )
    model = trainer.load_to(directory / "Model_checkpoint.pt", arguments.device)
    print(f"Loaded {arguments.arch}/{arguments.config}")

    # One nested sequence of datasets: the n-observation dataset is the first n
    # rows of the largest. Sharing the prefix means a change across the sweep is
    # the observation count and not a fresh draw of the data.
    largest = max(arguments.counts)
    dataset_rng = sim.generator("dataset", arguments.seed)
    truth, local_truth, x_all = H.simulate_dataset(largest, dataset_rng)
    print(f"shared angle_1 = {truth:.4f}\n")

    header = (f"{'n':>4}{'rule':>24}{'ref mean':>10}{'ref sd':>9}"
              f"{'post mean':>11}{'post sd':>9}{'bias/sig':>10}{'W1/sig':>9}")
    print(header)
    rows = []
    for count in arguments.counts:
        x = x_all[:count]
        grid, density = H.reference_global(
            x, draws=arguments.reference_draws,
            bandwidth=arguments.reference_bandwidth, seed=arguments.seed,
        )
        reference_mean, reference_std, _ = H.grid_moments(grid, density)
        data = torch.as_tensor(x, dtype=torch.float32)

        for method in inference.METHODS:
            samples, sampler, runtime = inference.draw(
                model, data, method, arguments, arguments.device
            )
            shared = samples[0, :, H.HIERARCHY[0]].astype(np.float64)
            endpoint = np.concatenate([
                np.linalg.norm(H.forward(samples[j, :, :4]) - x[j], axis=1)
                for j in range(count)
            ])
            row = {
                "observations": int(count), "method": method,
                "reference_mean": reference_mean, "reference_std": reference_std,
                "posterior_mean": float(shared.mean()),
                "posterior_std": float(shared.std()),
                "bias_sigma": float((shared.mean() - reference_mean) / reference_std),
                "bias_absolute": float(shared.mean() - reference_mean),
                "width_ratio": float(shared.std() / reference_std),
                "w1_over_sigma": inference.wasserstein_to_grid(
                    shared, grid, density) / reference_std,
                "endpoint_error": float(endpoint.mean()),
                "truth_error": float(shared.mean() - truth),
                "seconds": runtime,
            }
            rows.append(row)
            print(f"{count:>4}{method:>24}{reference_mean:>10.4f}"
                  f"{reference_std:>9.4f}{row['posterior_mean']:>11.4f}"
                  f"{row['posterior_std']:>9.4f}{row['bias_sigma']:>10.3f}"
                  f"{row['w1_over_sigma']:>9.3f}", flush=True)

    output = arguments.artifacts / "inference" / "observation_sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "arch": arguments.arch, "config": arguments.config,
        "truth": truth, "counts": arguments.counts,
        "churn_eta": arguments.churn_eta, "draws": arguments.draws,
        "reference_draws": arguments.reference_draws,
        "reference_bandwidth": arguments.reference_bandwidth,
        "rows": rows,
    }, indent=2, sort_keys=True))
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
