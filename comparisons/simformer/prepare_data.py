#!/usr/bin/env python3
"""Stage 0 for every experiment: simulations, observations, reference posteriors.

CPU-only and shared. Run this once per task before any arm is trained; every
experiment in this directory reads the files it writes, which is what makes
their numbers comparable to each other.

    python comparisons/simformer/prepare_data.py --task two_moons
    python comparisons/simformer/prepare_data.py --all --skip-reference
    python comparisons/simformer/prepare_data.py --task tree --reference-only

The reference stage is the expensive one: it runs the paper's own MCMC recipes
(A2.2) to get ground-truth conditional samples, 10,000 chains per conditional.
Budget it accordingly, and validate it (``tests/test_reference.py``) before
scoring anything against it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runtime import configure_cpu_limit  # noqa: E402

CPU_INFO = configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from common import paper, paths, tasks as task_module  # noqa: E402
from common.masks import posterior_mask  # noqa: E402
from common.pipeline import generate_observations, generate_simulations  # noqa: E402
from common.reference import reference_conditional_samples  # noqa: E402

REFERENCE_DRAWS = 10_000     # sbibm's reference posterior size, and C2ST's


def reference_path(task_name, observation_index, tag="posterior"):
    return paths.data_path(
        task_name, "reference", f"{tag}_obs{observation_index}.pt")


def build_reference_posteriors(task_name, n_observations, draws, seed,
                               overwrite=False):
    """Ground-truth posterior samples for each held-out observation."""
    task = task_module.get_task(task_name)
    obs_file = generate_observations(task_name, n_observations, overwrite=False)
    payload = torch.load(obs_file, weights_only=False)
    mask = posterior_mask(task.nodes_size, task.theta_dim)

    written = []
    for index in range(payload["x_observed"].shape[0]):
        out = reference_path(task_name, index)
        if out.exists() and not overwrite:
            written.append(out)
            continue
        z = torch.cat([payload["theta_true"][index],
                       payload["x_observed"][index]], dim=-1)
        started = time.time()
        if task_name == "gaussian_linear":
            # Exact: no MCMC needed, and it doubles as a check on the sampler.
            mean, std = task.analytic_posterior(payload["x_observed"][index])
            generator = torch.Generator().manual_seed(seed + index)
            samples = mean + std * torch.randn(draws, task.theta_dim,
                                               generator=generator)
            diagnostics = {"method": "analytic"}
        else:
            samples, diagnostics = reference_conditional_samples(
                task, mask, z, draws, seed=seed + index)
            diagnostics["method"] = "mcmc"
        paths.ensure(out.parent)
        torch.save({"samples": samples, "mask": mask, "conditioned": z,
                    "task": task_name, "observation_index": index,
                    "diagnostics": diagnostics}, out)
        paths.write_manifest(out.with_suffix(".manifest.json"), {
            "stage": "reference", "task": task_name,
            "observation_index": index, "draws": int(draws),
            "seconds": time.time() - started, "diagnostics": diagnostics,
            "recipe": paper.REFERENCE_MCMC.get(task_name),
            "sha256": paths.file_digest(out),
        })
        written.append(out)
        print(f"  reference obs {index}: {time.time() - started:.1f}s "
              f"({diagnostics.get('method')})")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", choices=sorted(task_module.TASKS),
                        help="Repeatable. Default: all six tasks.")
    parser.add_argument("--all", action="store_true", help="All six tasks.")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=list(paper.SIMULATION_BUDGETS))
    parser.add_argument("--observations", type=int, default=paper.N_OBSERVATIONS)
    parser.add_argument("--reference-draws", type=int, default=REFERENCE_DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-reference", action="store_true",
                        help="Simulations and observations only (fast).")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = args.task or (sorted(task_module.TASKS) if args.all else None)
    if not selected:
        parser.error("Pass --task NAME (repeatable) or --all.")

    summary = {}
    for name in selected:
        print(f"[{name}]")
        entry = {}
        if not args.reference_only:
            for budget in args.budgets:
                path = generate_simulations(name, budget, seed=args.seed,
                                            overwrite=args.overwrite)
                print(f"  simulations n={budget}: {path.name}")
                entry.setdefault("simulations", []).append(str(path))
            obs = generate_observations(name, args.observations,
                                        overwrite=args.overwrite)
            print(f"  observations: {obs.name}")
            entry["observations"] = str(obs)
        if not args.skip_reference:
            entry["reference"] = [
                str(p) for p in build_reference_posteriors(
                    name, args.observations, args.reference_draws,
                    seed=args.seed + 500, overwrite=args.overwrite)
            ]
        summary[name] = entry

    out = paths.data_path("prepare_summary.json")
    paths.ensure(out.parent)
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
