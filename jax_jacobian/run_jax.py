"""Accuracy and timing of the JAX Jacobian variants.

    /export/data/rheinric/jax_jacobian_env/bin/python run_jax.py          # float32, GPU
    /export/data/rheinric/jax_jacobian_env/bin/python run_jax.py --x64    # port check

``--x64`` runs the port in float64 against the float64 PyTorch reference: that
isolates translation errors from float32 rounding. Results go to
``artifacts/jax_results[_x64].json``.
"""

import argparse
import json
import sys

import common

common.configure(gpu="--cpu" not in sys.argv)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from common import (  # noqa: E402
    ARTIFACTS, BATCHES, NODES, REFERENCE, THETA_DIM, WEIGHTS, vesde_std,
    vesde_time_of_lambda,
)
import jax_model  # noqa: E402
from timing import relative_error, time_call  # noqa: E402

TIMING_LAMBDA = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--cpu", action="store_true",
                        help="run on CPU (set JAX_PLATFORMS=cpu too); skips autocvd")
    args = parser.parse_args()
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if args.x64 else jnp.float32
    print(f"jax {jax.__version__} on {jax.devices()}", file=sys.stderr)

    params = jax_model.load_params(WEIGHTS, dtype)
    reference = np.load(REFERENCE)
    results = {"framework": "jax", "jax": jax.__version__, "dtype": str(np.dtype(dtype)),
               "device": str(jax.devices()[0]), "accuracy": [], "timing": []}

    c = jnp.asarray(reference["c"], dtype)
    for index, lam in enumerate(reference["lambdas"]):
        t = float(reference[f"t_{index}"])
        score = jax_model.make_score(params, jnp.asarray(t, dtype), vesde_std(t), c)
        z = jnp.asarray(reference[f"z_{index}"], dtype)
        row = {"lambda": float(lam),
               "score_rel_err": relative_error(np.asarray(score(z)),
                                               reference[f"scores_{index}"])}
        exact = reference[f"jacobian_{index}"]
        for name, variant in jax_model.VARIANTS.items():
            jacobian = np.asarray(jax.jit(lambda z: variant(score, z))(z))
            row[f"{name}_rel_err"] = relative_error(jacobian, exact)
        results["accuracy"].append(row)
        print(row, file=sys.stderr)

    if not args.x64:
        results["timing"], results["failures"] = timing(params, reference, dtype)

    name = "jax_results_x64.json" if args.x64 else "jax_results.json"
    (ARTIFACTS / name).write_text(json.dumps(results, indent=2))
    print(f"wrote {ARTIFACTS / name}")


def timing(params, reference, dtype):
    rng = np.random.default_rng(1)
    t = vesde_time_of_lambda(TIMING_LAMBDA)
    rows_out = []
    failures = {}
    for rows, draws in BATCHES:
        theta = reference["theta_all"][:rows]
        obs = reference["obs_all"][:rows]
        z_theta = theta[:, None] + TIMING_LAMBDA * rng.standard_normal((rows, draws, THETA_DIM))
        z = np.concatenate([z_theta, np.broadcast_to(obs[:, None], (rows, draws, obs.shape[1]))], -1)
        z = jnp.asarray(z.reshape(rows * draws, NODES), dtype)
        c = jnp.zeros_like(z).at[:, THETA_DIM:].set(1.0)
        score = jax_model.make_score(params, jnp.asarray(t, dtype), vesde_std(t), c)
        calls = {"forward": jax.jit(score)}
        for name, variant in jax_model.VARIANTS.items():
            calls[name] = jax.jit(lambda z, variant=variant: variant(score, z))
        row = {"rows": rows, "draws": draws, "batch": rows * draws}
        for name, call in calls.items():
            try:
                row[name] = time_call(lambda: call(z).block_until_ready())
            except jax.errors.JaxRuntimeError as error:   # OOM at the big batch
                failures[f"{name}@{rows}x{draws}"] = str(error).splitlines()[0][:500]
        rows_out.append(row)
        print(row, file=sys.stderr)
    return rows_out, failures


if __name__ == "__main__":
    main()
