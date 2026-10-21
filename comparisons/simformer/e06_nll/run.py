#!/usr/bin/env python3
"""E06 -- average negative log-likelihood of the true value (paper Fig. A8).

They evaluate on 5000 draws from the joint, for the posterior *and* the
likelihood, and compare to NPE/NLE -- which are trained to minimise exactly this
quantity, so they say so and give them the advantage. Simformer's density comes
from the probability-flow ODE, not the SDE it samples with; the paper names that
discrepancy as the reason NLL and C2ST sometimes disagree.

COMPASS is well placed here: ``ScoreBasedInferenceModel.log_prob`` already
solves the PF-ODE with an exact divergence (``divergence="exact"``), so this
experiment needs no new machinery -- only e01's checkpoints.

    python e06_nll/run.py evaluate --task slcp --arm paper_matched \
        --budget 100000 --target posterior --device cuda
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.runtime import configure_cpu_limit, configure_runtime  # noqa: E402

EXPERIMENT = HERE.name

import argparse  # noqa: E402

TASKS = ("gaussian_linear", "gaussian_mixture", "two_moons", "slcp")


def _stage_evaluate(args):
    configure_runtime() if args.device == "cuda" else configure_cpu_limit()
    import torch
    from common import paper, tasks as task_module
    from common.masks import likelihood_mask, posterior_mask
    from common.metrics import average_nll
    from common.pipeline import load_arm, write_scores

    task = task_module.get_task(args.task)
    model = load_arm(args.task, args.arm, args.budget, sde=args.sde,
                     seed=args.seed, arch=args.arch, device=args.device)

    generator = torch.Generator().manual_seed(args.seed + 9_000)
    theta, x = task.sample_joint(args.draws, generator=generator)
    z = torch.cat([theta, x], dim=-1)
    mask = (posterior_mask if args.target == "posterior" else likelihood_mask)(
        task.nodes_size, task.theta_dim)

    log_probs = model.log_prob(
        z, mask.expand(z.shape[0], -1), timesteps=args.timesteps,
        eps=paper.TIME_INTERVAL[0], divergence="exact", device=args.device,
        verbose=True)
    result = average_nll(log_probs)
    result.update({"task": args.task, "arm": args.arm, "budget": args.budget,
                   "target": args.target, "sde": args.sde,
                   "timesteps": args.timesteps, "seed": args.seed})
    print(f"average NLL ({args.target}) = {result['nll']:.4f} "
          f"over {result['n']} draws")
    write_scores(EXPERIMENT,
                 f"nll__{args.task}__{args.arm}__{args.arch}__{args.sde}"
                 f"__n{args.budget}__{args.target}__seed{args.seed}", [result])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--task", choices=TASKS, required=True)
    evaluate.add_argument("--arm", default="paper_matched",
                          choices=("paper_matched", "compass_native",
                                   "posterior_only"))
    evaluate.add_argument("--budget", type=int, required=True)
    evaluate.add_argument("--target", default="posterior",
                          choices=("posterior", "likelihood"))
    evaluate.add_argument("--sde", default="vesde", choices=("vesde", "vpsde"))
    evaluate.add_argument("--arch", default="tokenized",
                          choices=("tokenized", "legacy"))
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--draws", type=int, default=5_000)
    evaluate.add_argument("--timesteps", type=int, default=100)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(func=_stage_evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
