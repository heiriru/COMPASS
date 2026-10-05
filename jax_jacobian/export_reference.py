"""Dump the checkpoint's weights and a float64 reference for the JAX port.

Runs in the COMPASS (torch) environment on CPU:

    ../../COMPASS/.COMPASS/bin/python export_reference.py

Writes
- artifacts/weights.npz: every parameter of the trial-token H64/d2 network.
- artifacts/reference.npz: realistic network inputs (test-set observations,
  noised normalized parameters) at each lambda in ``common.LAMBDAS``, with the
  float64 scores and the exact float64 10x10 latent Jacobian per row, taken by
  10 forward-mode JVPs exactly as ``MultiObsSampler._row_score_jacobian`` does.
"""

import sys

import common

common.configure(gpu=False)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import (  # noqa: E402
    ARTIFACTS, CHECKPOINT, LAMBDAS, REFERENCE, THETA_DIM, WEIGHTS,
    vesde_std, vesde_time_of_lambda,
)
from torch_variants import (  # noqa: E402
    load_network, make_inputs, math_attention, score_fn,
)

TEST_DATA = (
    "/export/home/rheinric/COMPASS/Partial_Pooling/artifacts/data/"
    "partial_pooling-train-400000-test-100.pt"
)
NORMALIZATION = (
    "/export/home/rheinric/COMPASS/Partial_Pooling/artifacts/data/"
    "partial_pooling-train-400000-normalization.pt"
)
ROWS, DRAWS = 20, 6


def main():
    torch.set_num_threads(1)
    ARTIFACTS.mkdir(exist_ok=True)
    network = load_network(CHECKPOINT, device="cpu")
    np.savez(WEIGHTS, **{
        name: tensor.detach().cpu().numpy()
        for name, tensor in network.state_dict().items()
    })

    observations = load_observations(ROWS)
    network = network.double()
    reference = {"lambdas": np.asarray(LAMBDAS)}
    generator = torch.Generator().manual_seed(0)
    for index, lam in enumerate(LAMBDAS):
        t = vesde_time_of_lambda(lam)
        assert abs(vesde_std(t) - lam) < 1e-9 * max(1.0, lam)
        z, c = make_inputs(observations, DRAWS, lam, generator)
        z, c = z.double(), c.double()
        f = score_fn(network, t, c)
        with torch.no_grad():
            scores = f(z)
        with math_attention():
            jacobian = exact_jacobian(f, z)
        reference[f"z_{index}"] = z.numpy()
        reference[f"t_{index}"] = np.asarray(t)
        reference[f"scores_{index}"] = scores.numpy()
        reference[f"jacobian_{index}"] = jacobian.numpy()
        print(f"lambda {lam:g}: |J| {jacobian.abs().mean():.3e}", file=sys.stderr)
    reference["c"] = c.numpy()
    # Inputs for the timing runs, which go up to 100 observation rows.
    timing = load_observations(100)
    reference["theta_all"] = timing["theta"].numpy()
    reference["obs_all"] = timing["obs"].numpy()
    np.savez(REFERENCE, **reference)
    print(f"wrote {WEIGHTS} and {REFERENCE}")


def exact_jacobian(f, z):
    """(batch, 10, 10) d score_theta / d z_theta, one JVP per input coordinate."""
    from torch.func import jvp
    columns = []
    for k in range(THETA_DIM):
        tangent = torch.zeros_like(z)
        tangent[:, k] = 1.0
        _, column = jvp(f, (z,), (tangent,))
        columns.append(column[:, :THETA_DIM].detach())
    return torch.stack(columns, dim=-1)


def load_observations(rows):
    """Normalized trial data of the first ``rows`` subjects of test dataset 0,
    together with that dataset's normalized true parameters."""
    payload = torch.load(TEST_DATA, map_location="cpu", weights_only=False)
    source = payload["payload"]["sde"]
    normalizers = torch.load(NORMALIZATION, map_location="cpu",
                             weights_only=False)["normalizers"]
    theta_n, obs_n = normalizers["sde_joint_theta"], normalizers["sde_observations"]
    theta = torch.cat((
        source["globals"][0].expand(rows, -1), source["locals"][0, :rows],
    ), dim=-1)
    theta = (theta - theta_n["mean"]) / theta_n["scale"]
    obs = (source["observations"][0, :rows] - obs_n["mean"]) / obs_n["scale"]
    return {"theta": theta.float(), "obs": obs.float()}


if __name__ == "__main__":
    main()
