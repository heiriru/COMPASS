#!/usr/bin/env python3
"""Where along the trajectory does the cached Jacobian information change?

Runs r1 (refresh every call) on one toy dataset and records, per composed-score
call: lambda, median min/max eigenvalue of the lambda-free information K_j, and
the relative change of K_j since the previous call, ||K_now - K_prev|| / ||K_now||.
A lagged refresh can only hurt where that change is large.

    python trace_info.py --model lean --mode oracle_joint --n 64
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_toy  # noqa: E402  (sets sys.path)
from runtime import configure_runtime  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lean")
    parser.add_argument("--mode", default="oracle_joint")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    import torch
    import run
    from compass.MultiObsSampler import MultiObsSampler
    run.select(arguments.model)
    configure_runtime(gpu=arguments.device.startswith("cuda")
                      and not os.environ.get("CUDA_VISIBLE_DEVICES"))
    network = bench_toy.load_network(run, arguments.mode, arguments.device)
    network.model.eval()
    trace = []
    original = MultiObsSampler._effective_global_factors

    def traced(self, var_t, *args, **kwargs):
        before = getattr(self, "_jacobian_cached_information", None)
        before = None if before is None else before.clone()
        result = original(self, var_t, *args, **kwargs)
        info = self._jacobian_cached_information
        eig = torch.linalg.eigvalsh(0.5 * (info + info.mT).cpu()).clamp_min(0)
        change = float("nan")
        if before is not None and before.shape == info.shape:
            change = float((torch.linalg.matrix_norm((info - before).cpu())
                            / torch.linalg.matrix_norm(info.cpu()).clamp_min(1e-30)).median())
        lam2 = float(torch.as_tensor(var_t).reshape(()))
        trace.append({"lambda": lam2 ** 0.5, "k_min": float(eig[..., 0].median()),
                      "k_max": float(eig[..., -1].median()), "change": change})
        return result

    MultiObsSampler._effective_global_factors = traced
    _, x = run.dataset(arguments.n)
    bench_toy.draw(network, x, "r1", 0, arguments.draws, arguments.device, run)
    out = HERE / "artifacts" / "trace" / f"{arguments.model}_{arguments.mode}_n{arguments.n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace))
    for row in trace[::4]:
        r_max = row["lambda"] ** 2 * row["k_max"]
        r_min = row["lambda"] ** 2 * row["k_min"]
        print(f"lambda {row['lambda']:9.4f}  r_min {r_min:10.3g}  r_max {r_max:10.3g}  "
              f"rel. change since last call {row['change']:.3g}")


if __name__ == "__main__":
    main()
