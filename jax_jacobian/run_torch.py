"""Accuracy and timing of the PyTorch Jacobian variants, on one free GPU.

    ../../COMPASS/.COMPASS/bin/python run_torch.py [--compile]

``--compile`` wraps the network call in ``torch.compile`` (all variants that
can be compiled; failures are recorded, not fatal). Results go to
``artifacts/torch_results[_compiled].json``.
"""

import argparse
import json
import sys

import common

common.configure(gpu=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import (  # noqa: E402
    ARTIFACTS, BATCHES, CHECKPOINT, REFERENCE, THETA_DIM, vesde_time_of_lambda,
)
from timing import relative_error, time_call  # noqa: E402
from torch_variants import VARIANTS, Score, load_network, make_inputs  # noqa: E402

TIMING_LAMBDA = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    compile_mode = "default" if args.compile else None
    device = torch.device("cuda")
    network = load_network(CHECKPOINT, device=device)
    reference = np.load(REFERENCE)
    results = {"framework": "torch", "torch": torch.__version__,
               "device": torch.cuda.get_device_name(), "compile": compile_mode,
               "accuracy": [], "timing": [], "failures": {}}

    c = torch.as_tensor(reference["c"], dtype=torch.float32, device=device)
    for index, lam in enumerate(reference["lambdas"]):
        t = float(reference[f"t_{index}"])
        f = Score(network, t, c, compile_mode)
        z = torch.as_tensor(reference[f"z_{index}"], dtype=torch.float32, device=device)
        with torch.no_grad():
            row = {"lambda": float(lam), "score_rel_err": relative_error(
                f(z).cpu().numpy(), reference[f"scores_{index}"])}
        for name, variant in VARIANTS.items():
            try:
                jacobian = variant(f, z, float(lam))
                row[f"{name}_rel_err"] = relative_error(
                    jacobian.detach().cpu().numpy(), reference[f"jacobian_{index}"])
            except Exception as error:                 # recorded, not fatal
                results["failures"][name] = f"{type(error).__name__}: {error}"[:500]
        results["accuracy"].append(row)
        print(row, file=sys.stderr)

    results["timing"] = timing(network, reference, compile_mode, device, results["failures"])
    suffix = "_compiled" if args.compile else ""
    path = ARTIFACTS / f"torch_results{suffix}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


def timing(network, reference, compile_mode, device, failures):
    generator = torch.Generator().manual_seed(1)
    t = vesde_time_of_lambda(TIMING_LAMBDA)
    rows_out = []
    for rows, draws in BATCHES:
        observations = {
            "theta": torch.as_tensor(reference["theta_all"][:rows]),
            "obs": torch.as_tensor(reference["obs_all"][:rows]),
        }
        z, c = make_inputs(observations, draws, TIMING_LAMBDA, generator)
        z, c = z.to(device), c.to(device)
        f = Score(network, t, c, compile_mode)
        row = {"rows": rows, "draws": draws, "batch": rows * draws}

        def forward():
            with torch.no_grad():
                f(z)
            torch.cuda.synchronize()
        row["forward"] = time_call(forward)
        for name, variant in VARIANTS.items():
            if name in failures:
                continue

            def call(variant=variant):
                variant(f, z, TIMING_LAMBDA)
                torch.cuda.synchronize()
            try:
                row[name] = time_call(call)
            except Exception as error:                 # e.g. OOM at the big batch
                failures[f"{name}@{rows}x{draws}"] = f"{type(error).__name__}: {error}"[:500]
                torch.cuda.empty_cache()
        rows_out.append(row)
        print({k: (v["median_ms"] if isinstance(v, dict) else v) for k, v in row.items()},
              file=sys.stderr)
    return rows_out


if __name__ == "__main__":
    main()
