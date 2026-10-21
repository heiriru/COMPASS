#!/usr/bin/env python3
"""Network posteriors for the recovery datasets.

``--kind single``: all four parameters local, one observation, the ordinary
``SBIm.sample`` posterior -- no composition anywhere. Run for each layout's
network on the same rows.

``--kind hier``: one ``MultiObsSampler`` call per tall dataset with the layout's
``hierarchy``, composing the single-observation scores. Settings follow the
earlier IK comparisons (``Arruda/compare_compass_ik.py``): churned DPM2 at
eta = 2, 100 timesteps, denoise clamp 5, eps = 1e-3.

Usage (from COMPASS/):
    .COMPASS/bin/python inverse_kinematics/recovery/run_network.py --kind single --layout angle
    .COMPASS/bin/python inverse_kinematics/recovery/run_network.py --kind hier --layout angle --n 8
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1] / "src")]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd  # noqa: E402

    autocvd(num_gpus=1, interval=1)

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import datasets  # noqa: E402
import hierarchical as H  # noqa: E402
import train_model as trainer  # noqa: E402
from layouts import LAYOUTS, PRIOR_STD  # noqa: E402

TIMESTEPS = 100
METHODS = {
    "gauss_hierarchical": {"correction": "gauss_hierarchical", "precision_est_samples": 1_000,
                           "precision_est_timesteps": TIMESTEPS},
    "gauss_jacobian_FD": {"correction": "gauss_jacobian_FD", "jacobian_refresh": 1,
                          "jacobian_refresh_low_noise": 20},
}


def single(model, x, draws, seed, batch=16):
    """``(count, draws, 4)`` from the ordinary one-observation posterior."""
    out = []
    for start in range(0, len(x), batch):
        torch.manual_seed(seed + start)
        s = model.sample(x=torch.as_tensor(x[start:start + batch], dtype=torch.float32),
                         num_samples=draws, timesteps=TIMESTEPS, method="dpm", order=2, churn_eta=2.0,
                         corrector_steps=0, final_corrector_steps=0, device="cuda", verbose=False,
                         terminal_denoise=True)
        out.append(s.detach().cpu().numpy())
    return np.concatenate(out)


def composed(model, layout, x, method, draws, seed, pair_budget=32_000, eps=1e-3):
    """``(draws, n, 4)`` from one compositional call (chunked to fit the card)."""
    from compass.MultiObsSampler import MultiObsSampler

    g = layout.n_global
    per_call = max(1, min(draws, pair_budget // len(x)))
    parts = []
    for k, start in enumerate(range(0, draws, per_call)):
        torch.manual_seed(seed + k)
        sampler = MultiObsSampler(model)
        s = sampler.sample(
            world_size=1, data=torch.as_tensor(x, dtype=torch.float32), condition_mask=H.CONDITION_MASK,
            num_samples=min(per_call, draws - start), timesteps=TIMESTEPS, eps=eps, hierarchy=list(layout.hierarchy),
            prior=(torch.zeros(g), torch.as_tensor(PRIOR_STD[:g], dtype=torch.float32)),
            local_prior=(torch.zeros(4 - g), torch.as_tensor(PRIOR_STD[g:], dtype=torch.float32)),
            denoise_clamp=5.0, device="cuda", verbose=False, method="dpm", order=2, churn_eta=2.0,
            corrector_steps=0, final_corrector_steps=0, terminal_corrector_steps=0, **METHODS[method])
        parts.append(s.detach().cpu().numpy()[:, :, :4].transpose(1, 0, 2))      # (draws, n, 4)
        torch.cuda.empty_cache()
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("single", "hier"), required=True)
    parser.add_argument("--layout", choices=tuple(LAYOUTS), required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="final diffusion time; the draws keep noise of sd lambda(eps) (0.032 at 1e-3)")
    a = parser.parse_args()
    layout = LAYOUTS[a.layout]
    data = datasets.load_or_make(a.kind, a.layout, a.n, a.count)
    folder = datasets.folder(a.kind, a.layout, a.n)
    model = trainer.load_to(layout.checkpoint, "cuda")
    sigma_end = float(model.sde.lambda_t(torch.tensor([1e-3])).item()) if hasattr(model, "sde") else None
    print(f"{layout.checkpoint.parent.name}; sigma at eps=1e-3: {sigma_end}", flush=True)

    if a.kind == "single":
        started = time.perf_counter()
        theta = single(model, data["x"][:, 0], a.draws, a.seed)[:, :, None, :]       # (count, draws, 1, 4)
        seconds = time.perf_counter() - started
        np.savez_compressed(folder / f"network_{a.layout}.npz", theta=theta.astype(np.float32))
        timing = {"seconds": seconds}
        print(f"single / {a.layout}: {seconds:.0f}s")
    else:
        timing = {}
        for method in a.methods:
            tag = "" if a.eps == 1e-3 else f"_eps{a.eps:g}"
            path = folder / f"network_{method}{tag}.npz"
            if path.exists():
                print(f"skip {path.name} (exists)")
                continue
            started = time.perf_counter()
            # One file per finished dataset, so an interrupted run resumes; seeds are per dataset,
            # so a resumed run is identical to an uninterrupted one.
            partial = folder / f"network_partial_{method}{tag}"
            partial.mkdir(exist_ok=True)
            for i in range(a.count):
                done = partial / f"{i:04d}.npy"
                if not done.exists():
                    np.save(partial / f"{i:04d}.tmp.npy",
                            composed(model, layout, data["x"][i], method, a.draws, a.seed + 1000 * i, eps=a.eps)
                            .astype(np.float32))
                    (partial / f"{i:04d}.tmp.npy").rename(done)
                if i % 10 == 9:
                    print(f"  {method}{tag}: {i + 1}/{a.count} datasets", flush=True)
            theta = np.stack([np.load(partial / f"{i:04d}.npy") for i in range(a.count)])   # (count, draws, n, 4)
            timing[method + tag] = time.perf_counter() - started
            np.savez_compressed(path, theta=theta.astype(np.float32))
            print(f"{folder.name} / {method}{tag}: {timing[method + tag]:.0f}s", flush=True)
    timing_path = folder / "network_timing.json"
    previous = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    previous.update({f"{a.layout}_{k}" if a.kind == "single" else k: v for k, v in timing.items()},
                    sigma_at_eps=sigma_end)
    timing_path.write_text(json.dumps(previous, indent=1))


if __name__ == "__main__":
    main()
