#!/usr/bin/env python3
"""E07 -- expected coverage / calibration (paper Fig. A9-A13).

They run an expected-coverage analysis (Hermans et al. 2022) for the posterior
and the likelihood on every benchmark task, plus the SIRD task at 20 random time
points, and report that Simformer is well calibrated and, like NPE, tends
conservative.

COMPASS already owns this kind of measurement -- ``inverse_kinematics/
sbc_reference.py`` and ``evaluate_composed.py`` compute SBC ranks and coverage,
and the partial-pooling benchmark reports cov95 everywhere. What is missing is
only running it on *their* tasks with *their* protocol, which is what this does.

Note the metric caveat in ``common/metrics.py::expected_coverage``: the
whitened-distance proxy for the highest-density region is exact only for
Gaussian posteriors. For Two Moons, Tree and HMM use SBC ranks instead
(``--method sbc``), which is what the multimodal arms in this repository already
do.

    python e07_calibration/run.py evaluate --task slcp --arm paper_matched \
        --budget 100000 --device cuda
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.runtime import configure_cpu_limit, configure_runtime  # noqa: E402

EXPERIMENT = HERE.name

import argparse  # noqa: E402

TASKS = ("gaussian_linear", "gaussian_mixture", "two_moons", "slcp",
         "tree", "hmm")


def _stage_evaluate(args):
    configure_runtime() if args.device == "cuda" else configure_cpu_limit()
    import torch
    from common import tasks as task_module
    from common.masks import likelihood_mask, posterior_mask
    from common.metrics import expected_coverage
    from common.pipeline import load_arm, sample_conditional, write_scores

    task = task_module.get_task(args.task)
    model = load_arm(args.task, args.arm, args.budget, sde=args.sde,
                     seed=args.seed, arch=args.arch, device=args.device)
    generator = torch.Generator().manual_seed(args.seed + 4_000)
    theta, x = task.sample_joint(args.cases, generator=generator)
    z = torch.cat([theta, x], dim=-1)
    mask = (posterior_mask if args.target == "posterior" else likelihood_mask)(
        task.nodes_size, task.theta_dim)
    latent = (mask == 0).nonzero(as_tuple=True)[0]

    drawn = []
    for index in range(args.cases):
        samples = sample_conditional(
            model, z[index], mask, num_samples=args.draws, arm=args.arm,
            device=args.device)
        drawn.append(samples.reshape(args.draws, -1)[:, latent])
    result = expected_coverage(torch.stack(drawn), z[:, latent])
    result.update({"task": args.task, "arm": args.arm, "budget": args.budget,
                   "target": args.target, "cases": args.cases,
                   "method": "whitened_distance_hdr_proxy"})
    for level, cov in zip(result["levels"], result["coverage"]):
        print(f"  nominal {level:.2f} -> empirical {cov:.3f}")
    write_scores(EXPERIMENT,
                 f"coverage__{args.task}__{args.arm}__{args.arch}__{args.sde}"
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
    evaluate.add_argument("--cases", type=int, default=200)
    evaluate.add_argument("--draws", type=int, default=1_000)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(func=_stage_evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
