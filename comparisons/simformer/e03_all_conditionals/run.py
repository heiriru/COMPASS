#!/usr/bin/env python3
"""E03 -- C2ST over arbitrary conditionals (paper Fig. 4b / A5b).

Tasks: Tree, HMM, Two Moons, SLCP. For each, 100 randomly drawn condition masks
are scored against MCMC references produced by the paper's own recipes.

This is the experiment that tests COMPASS's key-padding convention. Simformer
attends densely over all tokens; COMPASS sets ``key_padding_mask = 1 - c``, so a
*latent* token is never a key and latent-latent dependence has to travel through
the residual stream and MLP alone. HMM is the discriminating task -- its ten
latent theta nodes form a Markov chain and the posterior couples them all.

    python e03_all_conditionals/run.py masks --task hmm
    python e03_all_conditionals/run.py reference --task hmm
    python e03_all_conditionals/run.py sample --task hmm --arm paper_matched \
        --budget 10000
    python e03_all_conditionals/run.py score --task hmm --arm paper_matched \
        --budget 10000

``masks`` writes the frozen mask set and must be run once per task: every arm is
scored on the *same* masks, or the comparison is meaningless. ``reference`` is
the expensive stage (100 conditionals x 10,000 chains).

Training is shared with e01 -- train the arm there and this reads the checkpoint.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.runtime import configure_cpu_limit, configure_runtime  # noqa: E402

EXPERIMENT = HERE.name

import argparse  # noqa: E402

TASKS = ("tree", "hmm", "two_moons", "slcp")
DEFAULT_DRAWS = 10_000


def _mask_file(task):
    from common import paths
    return paths.artifacts(EXPERIMENT, "masks", f"{task}.pt")


def _stage_masks(args):
    configure_cpu_limit()
    import torch
    from common import paper, paths, tasks as task_module
    from common.masks import random_conditional_masks

    task = task_module.get_task(args.task)
    out = _mask_file(args.task)
    if out.exists() and not args.overwrite:
        print(f"{out} already exists; pass --overwrite to redraw. "
              "Redrawing invalidates every reference and score built on it.")
        return
    generator = torch.Generator().manual_seed(args.seed)
    masks = random_conditional_masks(task.nodes_size, paper.N_CONDITIONALS,
                                     generator=generator)
    generator_obs = torch.Generator().manual_seed(args.seed + 1)
    theta, x = task.sample_joint(paper.N_CONDITIONALS, generator=generator_obs)
    values = torch.cat([theta, x], dim=-1)
    paths.ensure(out.parent)
    torch.save({"masks": masks, "values": values, "task": args.task}, out)
    paths.write_manifest(out.with_suffix(".manifest.json"), {
        "stage": "masks", "experiment": EXPERIMENT, "task": args.task,
        "n_conditionals": paper.N_CONDITIONALS, "seed": args.seed,
        "sha256": paths.file_digest(out),
    })
    print(f"masks: {out}  ({paper.N_CONDITIONALS} conditionals, frozen)")


def _stage_reference(args):
    configure_cpu_limit()
    import time
    import torch
    from common import paths, tasks as task_module
    from common.reference import reference_conditional_samples

    task = task_module.get_task(args.task)
    payload = torch.load(_mask_file(args.task), weights_only=False)
    out_dir = paths.ensure(paths.artifacts(EXPERIMENT, "reference", args.task))
    for index in range(payload["masks"].shape[0]):
        out = out_dir / f"conditional_{index:03d}.pt"
        if out.exists() and not args.overwrite:
            continue
        started = time.time()
        samples, diagnostics = reference_conditional_samples(
            task, payload["masks"][index], payload["values"][index],
            args.draws, seed=args.seed + index)
        torch.save({"samples": samples, "mask": payload["masks"][index],
                    "conditioned": payload["values"][index],
                    "diagnostics": diagnostics}, out)
        print(f"  conditional {index}: {time.time() - started:.1f}s "
              f"accept={diagnostics.get('hmc_accept_rate', diagnostics.get('mh_accept_rate')):.3f}")


def _stage_sample(args):
    configure_runtime() if args.device == "cuda" else configure_cpu_limit()
    import torch
    from common import paths
    from common.pipeline import load_arm, sample_conditional

    payload = torch.load(_mask_file(args.task), weights_only=False)
    model = load_arm(args.task, args.arm, args.budget, sde=args.sde,
                     seed=args.seed, arch=args.arch, device=args.device)
    out = paths.artifacts(
        EXPERIMENT, "samples",
        f"{args.task}__{args.arm}__{args.arch}__{args.sde}"
        f"__n{args.budget}__seed{args.seed}.pt")
    paths.ensure(out.parent)

    drawn = []
    for index in range(payload["masks"].shape[0]):
        mask = payload["masks"][index]
        samples = sample_conditional(
            model, payload["values"][index], mask, num_samples=args.draws,
            arm=args.arm, device=args.device, verbose=False)
        latent = (mask == 0).nonzero(as_tuple=True)[0]
        drawn.append(samples.reshape(args.draws, -1)[:, latent])
    torch.save({"samples": drawn, "task": args.task, "arm": args.arm,
                "budget": args.budget}, out)
    paths.write_manifest(out.with_suffix(".manifest.json"), {
        "stage": "sample", "experiment": EXPERIMENT, "task": args.task,
        "arm": args.arm, "budget": args.budget, "seed": args.seed,
        "draws": args.draws, "sha256": paths.file_digest(out),
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
    drawn = torch.load(sample_file, weights_only=False)["samples"]
    ref_dir = paths.artifacts(EXPERIMENT, "reference", args.task)

    rows = []
    for index, samples in enumerate(drawn):
        ref_file = ref_dir / f"conditional_{index:03d}.pt"
        if not ref_file.exists():
            raise FileNotFoundError(f"{ref_file} is missing; run the reference stage.")
        reference = torch.load(ref_file, weights_only=False)["samples"]
        n = min(len(reference), len(samples))
        result = c2st(samples[:n], reference[:n], seed=args.seed)
        result.update({"conditional": index, "task": args.task,
                       "arm": args.arm, "budget": args.budget,
                       "n_latent": int(samples.shape[1])})
        rows.append(result)

    mean = sum(r["c2st"] for r in rows) / len(rows)
    print(f"mean C2ST over {len(rows)} conditionals = {mean:.4f}")
    write_scores(EXPERIMENT,
                 f"c2st__{args.task}__{args.arm}__{args.arch}__{args.sde}"
                 f"__n{args.budget}__seed{args.seed}",
                 rows, extra={"mean_c2st": mean})


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    masks = sub.add_parser("masks", help="freeze the 100 condition masks")
    masks.add_argument("--task", choices=TASKS, required=True)
    masks.add_argument("--seed", type=int, default=0)
    masks.add_argument("--overwrite", action="store_true")
    masks.set_defaults(func=_stage_masks)

    reference = sub.add_parser("reference", help="MCMC references (expensive)")
    reference.add_argument("--task", choices=TASKS, required=True)
    reference.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    reference.add_argument("--seed", type=int, default=0)
    reference.add_argument("--overwrite", action="store_true")
    reference.set_defaults(func=_stage_reference)

    for name, func in (("sample", _stage_sample), ("score", _stage_score)):
        p = sub.add_parser(name)
        p.add_argument("--task", choices=TASKS, required=True)
        p.add_argument("--arm", default="paper_matched",
                       choices=("paper_matched", "compass_native",
                                "posterior_only"))
        p.add_argument("--budget", type=int, required=True)
        p.add_argument("--sde", default="vesde", choices=("vesde", "vpsde"))
        p.add_argument("--arch", default="tokenized",
                       choices=("tokenized", "legacy"))
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
        if name == "sample":
            p.add_argument("--device", default="cpu")
        p.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
