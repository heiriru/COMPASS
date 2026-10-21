#!/usr/bin/env python3
"""E05 -- C2ST as a function of reverse-SDE evaluation steps (paper Fig. A7).

They sweep 5, 10, 50, 100, 250, 500, 750, 1000 Euler-Maruyama steps and report a
sharp transition: above ~50 steps C2ST is at its best for every task except Two
Moons under the VPSDE.

The COMPASS-side question is different and is the cheap win in this directory:
their 500-step default is Euler-Maruyama, while COMPASS ships DPM-2 with churn
(``docs/unified_method.md`` measures eta=2.0 as the better predictor). If the
``compass_native`` arm reaches the same C2ST at 20-50 function evaluations, that
is a straight compute result on their own benchmark and it costs no retraining:
this experiment reuses e01's checkpoints and only re-samples.

    python e05_nfe_ablation/run.py sweep --task two_moons --arm paper_matched \
        --budget 100000 --device cuda
    python e05_nfe_ablation/run.py plot
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.runtime import configure_cpu_limit, configure_runtime  # noqa: E402

EXPERIMENT = HERE.name

import argparse  # noqa: E402
import json  # noqa: E402

STEPS = (5, 10, 20, 50, 100, 250, 500, 750, 1000)
TASKS = ("gaussian_linear", "gaussian_mixture", "two_moons", "slcp")


def _stage_sweep(args):
    configure_runtime() if args.device == "cuda" else configure_cpu_limit()
    import torch
    from common import paths, tasks as task_module
    from common.masks import posterior_mask
    from common.metrics import c2st
    from common.pipeline import (load_arm, observation_path, sample_conditional,
                                 write_scores)

    task = task_module.get_task(args.task)
    obs = torch.load(observation_path(args.task, 1000), weights_only=False)
    model = load_arm(args.task, args.arm, args.budget, sde=args.sde,
                     seed=args.seed, arch=args.arch, device=args.device)
    mask = posterior_mask(task.nodes_size, task.theta_dim)

    rows = []
    for steps in args.steps:
        per_observation = []
        for index in range(obs["x_observed"].shape[0]):
            z = torch.cat([obs["theta_true"][index], obs["x_observed"][index]], -1)
            samples = sample_conditional(
                model, z, mask, num_samples=args.draws, arm=args.arm,
                timesteps=steps, device=args.device)[..., : task.theta_dim]
            ref = torch.load(
                paths.data_path(args.task, "reference", f"posterior_obs{index}.pt"),
                weights_only=False)["samples"]
            n = min(len(ref), args.draws)
            per_observation.append(
                c2st(samples.reshape(args.draws, -1)[:n], ref[:n],
                     seed=args.seed)["c2st"])
        mean = sum(per_observation) / len(per_observation)
        rows.append({"steps": steps, "c2st": mean, "per_observation":
                     per_observation, "task": args.task, "arm": args.arm,
                     "budget": args.budget, "sde": args.sde})
        print(f"  steps={steps:5d}  mean C2ST={mean:.4f}")

    write_scores(EXPERIMENT,
                 f"nfe__{args.task}__{args.arm}__{args.arch}__{args.sde}"
                 f"__n{args.budget}__seed{args.seed}", rows)


def _stage_plot(args):
    configure_cpu_limit()
    from common import paths
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = sorted(paths.artifacts(EXPERIMENT).glob("nfe__*.json"))
    if not scored:
        raise FileNotFoundError("No sweeps scored yet.")
    fig, axes = plt.subplots(1, len(TASKS), figsize=(4 * len(TASKS), 3.2),
                             sharey=True)
    for path in scored:
        with open(path) as handle:
            rows = json.load(handle)
        task = rows[0]["task"]
        axis = axes[TASKS.index(task)]
        axis.plot([r["steps"] for r in rows], [r["c2st"] for r in rows],
                  marker="o", label=f"{rows[0]['arm']}/{rows[0]['sde']}")
        axis.set_xscale("log")
        axis.set_title(task)
        axis.set_xlabel("evaluation steps")
        axis.axhline(0.5, color="grey", lw=0.8, ls=":")
    axes[0].set_ylabel("C2ST")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    out = paths.artifacts(EXPERIMENT, "figures", "c2st_vs_steps.png")
    paths.ensure(out.parent)
    fig.savefig(out, dpi=200)
    print(f"figure: {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--task", choices=TASKS, required=True)
    sweep.add_argument("--arm", default="paper_matched",
                       choices=("paper_matched", "compass_native",
                                "posterior_only"))
    sweep.add_argument("--budget", type=int, required=True)
    sweep.add_argument("--sde", default="vesde", choices=("vesde", "vpsde"))
    sweep.add_argument("--arch", default="tokenized",
                       choices=("tokenized", "legacy"))
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--draws", type=int, default=10_000)
    sweep.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    sweep.add_argument("--device", default="cpu")
    sweep.set_defaults(func=_stage_sweep)

    plot = sub.add_parser("plot")
    plot.set_defaults(func=_stage_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
