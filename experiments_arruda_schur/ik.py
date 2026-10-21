"""Inverse kinematics (recovery harness): latent-attention networks and the arruda_schur rules.

Reuses ``inverse_kinematics/recovery`` read-only: its tall datasets
(``<layout>_n<n>/data.npz``), its exact MCMC references (``reference.npz``) and
its grading (``versus_reference``, copied below verbatim). Networks are
tokenized trunks trained with the recovery recipe (200k rows, six_weighted,
log_sigma, seed 7, the layout's six-conditional roles), and the attention mode
is the only change between them. Outputs go under
``experiments_arruda_schur/artifacts/ik/``.

    python ik.py train --layout angle_height --mode full
    python ik.py run --layout angle_height --mode full --n 8 --methods gauss_jacobian_FD arruda_laplace
    python ik.py grade
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

IK = common.COMPASS / "inverse_kinematics"
RECOVERY = IK / "recovery"
sys.path[:0] = [str(RECOVERY), str(IK)]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd  # noqa: E402
    autocvd(num_gpus=1, interval=1)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

import hierarchical as H  # noqa: E402
import layouts  # noqa: E402

MODES = ("none", "global", "hierarchical", "full")
CONFIG = "h32d3wide"
TIMESTEPS = 100
DPM = dict(method="dpm", order=2, churn_eta=2.0, corrector_steps=0,
           final_corrector_steps=0, terminal_corrector_steps=0)
FD_SCHEDULE = {"jacobian_refresh": 1, "jacobian_refresh_low_noise": 20}
METHODS = {
    "gauss_hierarchical": ({}, {"correction": "gauss_hierarchical", "precision_est_samples": 1000,
                                "precision_est_timesteps": TIMESTEPS}),
    "gauss_jacobian_FD": ({}, {"correction": "gauss_jacobian_FD", **FD_SCHEDULE}),
    "arruda_direct": ({"estimator": "direct"}, {"correction": "arruda_schur"}),
    "arruda_laplace": ({"estimator": "laplace"}, {"correction": "arruda_schur", **FD_SCHEDULE}),
    "arruda_gauss_direct": ({"estimator": "direct"}, {"correction": "arruda_schur_gauss", **FD_SCHEDULE}),
    "arruda_gauss_laplace": ({"estimator": "laplace"}, {"correction": "arruda_schur_gauss", **FD_SCHEDULE}),
}


def set_layout(name):
    layout = layouts.LAYOUTS[name]
    H.GLOBAL_NAMES = list(layout.global_names)
    H.LOCAL_NAMES = list(layout.local_names)
    return layout


def model_dir(layout, mode):
    return HERE / "artifacts" / "ik" / "models" / f"tokenized_{CONFIG}_{layout}_{mode}"


def out_dir(layout, mode, n):
    return HERE / "artifacts" / "ik" / f"{layout}_n{n}" / mode


def train(arguments):
    layout = set_layout(arguments.layout)
    import train_model as trainer
    from compass import hierarchical_roles
    from compass.experimental.latent_attention import enable_latent_attention, save_latent_attention

    directory = model_dir(arguments.layout, arguments.mode)
    directory.mkdir(parents=True, exist_ok=True)
    original = trainer.build

    def build(config, arch, device="cpu"):
        model = original(config, arch, device=device)
        roles = hierarchical_roles(model.schema, layout.n_global, 4 - layout.n_global)
        enable_latent_attention(model, arguments.mode, roles)
        save_latent_attention(model, directory)
        return model

    trainer.build = build
    trainer.load_to = common.load_with_latent_attention(trainer.load_to)
    started = time.time()
    model, _, parameters = trainer.train_or_load(
        directory, CONFIG, "tokenized", 200_000, "cuda", seed=7, menu="six_weighted",
        time_sampling="log_sigma",
    )
    (directory / "run.json").write_text(json.dumps({
        "layout": arguments.layout, "mode": arguments.mode, "config": CONFIG,
        "parameters": parameters, "minutes": (time.time() - started) / 60,
        "latent_attention": getattr(model.model, "latent_attention", "none"),
    }, indent=2))


def composed(model, layout, x, method, draws, seed, pair_budget=32_000):
    """``(draws, n, 4)``, chunked exactly like ``recovery/run_network.composed``."""
    from compass.experimental.marginal_schur import MarginalSchurSampler

    settings, kwargs = METHODS[method]
    g = layout.n_global
    per_call = max(1, min(draws, pair_budget // len(x)))
    parts = []
    for k, start in enumerate(range(0, draws, per_call)):
        torch.manual_seed(seed + k)
        sampler = MarginalSchurSampler(model, **settings)
        s = sampler.sample(
            world_size=1, data=torch.as_tensor(x, dtype=torch.float32),
            condition_mask=H.CONDITION_MASK, num_samples=min(per_call, draws - start),
            timesteps=TIMESTEPS, eps=1e-3, hierarchy=list(layout.hierarchy),
            prior=(torch.zeros(g), torch.as_tensor(layouts.PRIOR_STD[:g], dtype=torch.float32)),
            local_prior=(torch.zeros(4 - g), torch.as_tensor(layouts.PRIOR_STD[g:], dtype=torch.float32)),
            denoise_clamp=5.0, device="cuda", verbose=False, **DPM, **kwargs)
        parts.append(s.detach().cpu().numpy()[:, :, :4].transpose(1, 0, 2))
        torch.cuda.empty_cache()
    return np.concatenate(parts)


def load_network(layout_name, mode):
    import train_model as trainer
    from compass.experimental.latent_attention import load_latent_attention
    checkpoint = model_dir(layout_name, mode) / "Model_checkpoint.pt"
    model = trainer.load_to(checkpoint, "cuda")
    load_latent_attention(model, checkpoint.parent, required=False)
    model.model.eval()
    return model


def run(arguments):
    layout = set_layout(arguments.layout)
    data = dict(np.load(RECOVERY / "artifacts" / f"{arguments.layout}_n{arguments.n}" / "data.npz"))
    model = load_network(arguments.layout, arguments.mode)
    folder = out_dir(arguments.layout, arguments.mode, arguments.n)
    folder.mkdir(parents=True, exist_ok=True)
    count = min(arguments.count, len(data["x"]))
    for method in arguments.methods:
        path = folder / f"{method}.npz"
        if path.exists():
            print(f"skip {path}")
            continue
        partial = folder / f"partial_{method}"
        partial.mkdir(exist_ok=True)
        started = time.perf_counter()
        for i in range(count):
            done = partial / f"{i:04d}.npy"
            if not done.exists():
                theta = composed(model, layout, data["x"][i], method, arguments.draws, 2026 + 1000 * i)
                np.save(partial / f"{i:04d}.tmp.npy", theta.astype(np.float32))
                (partial / f"{i:04d}.tmp.npy").rename(done)
            if i % 10 == 9:
                print(f"  {arguments.layout} n={arguments.n} {arguments.mode}/{method}: {i + 1}/{count}", flush=True)
        theta = np.stack([np.load(partial / f"{i:04d}.npy") for i in range(count)])
        np.savez_compressed(path, theta=theta, seconds=time.perf_counter() - started)
        print(f"{arguments.layout} n={arguments.n} {arguments.mode}/{method}: "
              f"{time.perf_counter() - started:.0f}s", flush=True)


def versus_reference(draws, reference):
    """Verbatim from ``inverse_kinematics/recovery/plot_recovery.py``."""
    half = reference.shape[1] // 2
    ref_a, ref_b = reference[:, :half], reference[:, half:]
    ref_sd = np.maximum(reference.std(1), 1e-12)
    rng = np.random.default_rng(0)
    sub = draws[:, rng.choice(draws.shape[1], size=min(half, draws.shape[1]), replace=False)]
    w1 = np.array([wasserstein_distance(d, r) for d, r in zip(sub, ref_a)]) / ref_sd
    floor = np.array([wasserstein_distance(b, a) for b, a in zip(ref_b, ref_a)]) / ref_sd
    return {"median_w1_over_ref_sd": float(np.median(w1)), "mean_w1_over_ref_sd": float(w1.mean()),
            "p90_w1_over_ref_sd": float(np.quantile(w1, 0.9)),
            "floor_median_w1_over_ref_sd": float(np.median(floor)),
            "median_abs_mean_error_over_ref_sd": float(np.median(np.abs(draws.mean(1) - reference.mean(1)) / ref_sd)),
            "median_sd_ratio": float(np.median(draws.std(1) / ref_sd))}


def grade(arguments):
    rows = []
    root = HERE / "artifacts" / "ik"
    for folder in sorted(root.glob("*_n*/*")):
        layout_name, n = folder.parent.name.rsplit("_n", 1)
        layout = layouts.LAYOUTS[layout_name]
        reference = np.load(RECOVERY / "artifacts" / folder.parent.name / "reference.npz")["theta"]
        for result in sorted(folder.glob("*.npz")):
            theta = np.load(result)["theta"]
            count = theta.shape[0]
            row = {"layout": layout_name, "n": int(n), "mode": folder.name, "method": result.stem}
            for p in range(4):
                scope = "global" if p < layout.n_global else "local"
                if scope == "global":
                    stats = versus_reference(theta[:, :, 0, p], reference[:count, :, 0, p])
                else:
                    stats = versus_reference(theta[:, :, :, p].transpose(0, 2, 1).reshape(-1, theta.shape[1]),
                                             reference[:count, :, :, p].transpose(0, 2, 1).reshape(-1, reference.shape[1]))
                row[layouts.NODE_NAMES[p]] = {k: stats[k] for k in (
                    "median_w1_over_ref_sd", "floor_median_w1_over_ref_sd", "median_sd_ratio",
                    "median_abs_mean_error_over_ref_sd")}
            rows.append(row)
            g = [layouts.NODE_NAMES[p] for p in range(layout.n_global)]
            print(f"{layout_name:13s} n={n:>3} {folder.name:12s} {result.stem:22s} " + "  ".join(
                f"{name}: W1/sd {row[name]['median_w1_over_ref_sd']:.3f} sd-ratio {row[name]['median_sd_ratio']:.2f}"
                for name in g))
    (root / "grades.json").write_text(json.dumps(rows, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "run", "grade"))
    parser.add_argument("--layout", choices=tuple(layouts.LAYOUTS), default="angle_height")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--methods", nargs="+", default=["gauss_jacobian_FD", "arruda_laplace"])
    arguments = parser.parse_args()
    {"train": train, "run": run, "grade": grade}[arguments.action](arguments)


if __name__ == "__main__":
    main()
