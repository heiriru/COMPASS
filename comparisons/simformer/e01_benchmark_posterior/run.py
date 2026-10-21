#!/usr/bin/env python3
"""E01 -- posterior C2ST on the four sbibm benchmark tasks (paper Fig. 4a / A5a).

Stages, in order. Each is a separate invocation on purpose: ``train`` is the
only GPU-bound one and the only expensive one.

    # 0. once, from comparisons/simformer/
    python prepare_data.py --task gaussian_linear --task gaussian_mixture \
                           --task two_moons --task slcp

    # 1. fit one arm at one budget  (GPU; selects a free device via autocvd)
    python e01_benchmark_posterior/run.py train --task two_moons \
        --arm paper_matched --budget 10000 --device cuda

    # 2. draw 10k posterior samples for each of the 10 observations
    python e01_benchmark_posterior/run.py sample --task two_moons \
        --arm paper_matched --budget 10000

    # 3. C2ST against the reference posteriors, one row per observation
    python e01_benchmark_posterior/run.py score --task two_moons \
        --arm paper_matched --budget 10000

    # 4. the figure: C2ST vs simulation budget, our arms beside their curves
    python e01_benchmark_posterior/run.py plot

Read ``README.md`` in this directory before interpreting any number: the
architecture is matched only up to the four mismatches recorded in
``common/paper.py::MISMATCHES``, and the sampler step counts differ between arms
by design.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.runtime import configure_cpu_limit, configure_runtime  # noqa: E402

EXPERIMENT = HERE.name

import argparse  # noqa: E402
import json  # noqa: E402

TASKS = ("gaussian_linear", "gaussian_mixture", "two_moons", "slcp")
DEFAULT_DRAWS = 10_000


def _stage_train(args):
    configure_runtime() if args.device == "cuda" else configure_cpu_limit()
    from common.pipeline import train_arm

    out = train_arm(args.task, args.arm, args.budget, sde=args.sde,
                    seed=args.seed, arch=args.arch, device=args.device,
                    overwrite=args.overwrite)
    print(f"checkpoint: {out}")


def _stage_sample(args):
    configure_runtime() if args.device == "cuda" else configure_cpu_limit()
    import torch
    from common import paths, tasks as task_module
    from common.masks import posterior_mask
    from common.pipeline import load_arm, observation_path, sample_conditional

    task = task_module.get_task(args.task)
    obs = torch.load(observation_path(args.task, 1000), weights_only=False)
    model = load_arm(args.task, args.arm, args.budget, sde=args.sde,
                     seed=args.seed, arch=args.arch, device=args.device)
    mask = posterior_mask(task.nodes_size, task.theta_dim)

    out = paths.artifacts(EXPERIMENT, "samples",
                          f"{args.task}__{args.arm}__{args.arch}__{args.sde}"
                          f"__n{args.budget}__seed{args.seed}.pt")
    paths.ensure(out.parent)
    drawn = []
    for index in range(obs["x_observed"].shape[0]):
        z = torch.cat([obs["theta_true"][index], obs["x_observed"][index]], -1)
        samples = sample_conditional(
            model, z, mask, num_samples=args.draws, arm=args.arm,
            timesteps=args.timesteps, device=args.device, verbose=args.verbose)
        drawn.append(samples[..., : task.theta_dim].reshape(args.draws, -1))
        print(f"  observation {index}: {tuple(drawn[-1].shape)}")
    torch.save({"samples": torch.stack(drawn), "task": args.task,
                "arm": args.arm, "budget": args.budget}, out)
    paths.write_manifest(out.with_suffix(".manifest.json"), {
        "stage": "sample", "experiment": EXPERIMENT, "task": args.task,
        "arm": args.arm, "budget": args.budget, "seed": args.seed,
        "sde": args.sde, "arch": args.arch, "draws": args.draws,
        "timesteps": args.timesteps, "sha256": paths.file_digest(out),
    })
    print(f"samples: {out}")


def _stage_score(args):
    configure_cpu_limit()
    import torch
    from common import paths
    from common.metrics import c2st
    from common.pipeline import write_scores

    sample_file = paths.artifacts(
        EXPERIMENT, "samples",
        f"{args.task}__{args.arm}__{args.arch}__{args.sde}"
        f"__n{args.budget}__seed{args.seed}.pt")
    if not sample_file.exists():
        raise FileNotFoundError(f"{sample_file} is missing; run the sample stage.")
    drawn = torch.load(sample_file, weights_only=False)["samples"]

    rows = []
    for index in range(drawn.shape[0]):
        ref_file = paths.data_path(args.task, "reference",
                                   f"posterior_obs{index}.pt")
        if not ref_file.exists():
            raise FileNotFoundError(
                f"{ref_file} is missing; run prepare_data.py without "
                "--skip-reference.")
        reference = torch.load(ref_file, weights_only=False)["samples"]
        n = min(len(reference), drawn.shape[1])
        result = c2st(drawn[index][:n], reference[:n], seed=args.seed)
        result.update({"observation": index, "task": args.task,
                       "arm": args.arm, "budget": args.budget,
                       "sde": args.sde, "arch": args.arch, "seed": args.seed})
        rows.append(result)
        print(f"  observation {index}: C2ST = {result['c2st']:.4f}")

    mean = sum(r["c2st"] for r in rows) / len(rows)
    print(f"mean C2ST = {mean:.4f}  (paper's metric; 0.5 is perfect)")
    write_scores(EXPERIMENT,
                 f"c2st__{args.task}__{args.arm}__{args.arch}__{args.sde}"
                 f"__n{args.budget}__seed{args.seed}",
                 rows, extra={"mean_c2st": mean})


def _stage_plot(args):
    configure_cpu_limit()
    from common import paths

    scored = sorted(paths.artifacts(EXPERIMENT).glob("c2st__*.json"))
    if not scored:
        raise FileNotFoundError(
            "No scored tables yet; run the score stage for at least one arm.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {}
    for path in scored:
        with open(path) as handle:
            rows = json.load(handle)
        key = (rows[0]["task"], rows[0]["arm"], rows[0]["arch"], rows[0]["sde"])
        mean = sum(r["c2st"] for r in rows) / len(rows)
        curves.setdefault(key, []).append((rows[0]["budget"], mean))

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4 * len(TASKS), 3.2),
                             sharey=True)
    for axis, task in zip(axes, TASKS):
        for (name, arm, arch, sde), points in sorted(curves.items()):
            if name != task:
                continue
            points.sort()
            axis.plot([p[0] for p in points], [p[1] for p in points],
                      marker="o", label=f"{arm}/{arch}/{sde}")
        axis.set_xscale("log")
        axis.set_ylim(0.45, 1.0)
        axis.axhline(0.5, color="grey", lw=0.8, ls=":")
        axis.set_title(task)
        axis.set_xlabel("simulations")
    axes[0].set_ylabel("C2ST")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    out = paths.artifacts(EXPERIMENT, "figures", "c2st_vs_budget.png")
    paths.ensure(out.parent)
    fig.savefig(out, dpi=200)
    print(f"figure: {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common_args(p, with_device=True):
        p.add_argument("--task", choices=TASKS, required=True)
        p.add_argument("--arm", default="paper_matched",
                       choices=("paper_matched", "compass_native",
                                "posterior_only"))
        p.add_argument("--budget", type=int, required=True)
        p.add_argument("--sde", default="vesde", choices=("vesde", "vpsde"))
        p.add_argument("--arch", default="tokenized",
                       choices=("tokenized", "legacy"))
        p.add_argument("--seed", type=int, default=0)
        if with_device:
            p.add_argument("--device", default="cpu")

    train = sub.add_parser("train", help="fit one arm (GPU)")
    common_args(train)
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--verbose", action="store_true", default=True)
    train.set_defaults(func=_stage_train)

    sample = sub.add_parser("sample", help="draw posterior samples")
    common_args(sample)
    sample.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    sample.add_argument("--timesteps", type=int, default=None,
                        help="Override the arm's step count (see e05).")
    sample.add_argument("--verbose", action="store_true")
    sample.set_defaults(func=_stage_sample)

    score = sub.add_parser("score", help="C2ST against the reference posteriors")
    common_args(score, with_device=False)
    score.set_defaults(func=_stage_score)

    plot = sub.add_parser("plot", help="C2ST vs simulation budget")
    plot.set_defaults(func=_stage_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
