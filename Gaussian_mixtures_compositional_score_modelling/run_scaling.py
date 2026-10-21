#!/usr/bin/env python3
"""The observation sweep: every rule, every regime, n = 1, 2, 4, 8, 16, 32.

One network, trained once on single observations, is composed by three rules --
Langevin + F-NPSE, and DPM2 + churn 4 with gauss_hierarchical and with
gauss_jacobian -- on four kinds of tall dataset and graded against the closed-form posterior. The
sweep also runs an ``exact_mc`` arm -- the same number of draws taken *from the
true posterior* and graded by the identical code -- so every reported error can
be read against the Monte-Carlo floor of the measurement itself rather than
against zero.

Outputs, under ``artifacts/scaling/<tag>/``:

    metrics.csv     one row per (regime, n, replicate, method)
    summary.json    median and IQR over replicates, per (regime, n, method)
    draws.npz       replicate 0's shared draws, for the density figure
    datasets.json   the datasets used, so a figure never re-draws them

Usage:
    python run_scaling.py                       # the full sweep
    python run_scaling.py --observations 1 8 --regimes symmetric asymmetric
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_runtime  # noqa: E402  (before numeric imports)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_OBSERVATIONS = (1, 2, 4, 8, 16, 32)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="h32d3wide")
    result.add_argument("--arch", default="tokenized")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--mask-menu", default="six_weighted")
    result.add_argument("--time-sampling", default="log_sigma")
    result.add_argument("--observations", type=int, nargs="+",
                        default=list(DEFAULT_OBSERVATIONS))
    result.add_argument("--regimes", nargs="+", default=None,
                        help="Subset of datasets.REGIMES; all four by default.")
    result.add_argument("--methods", nargs="+", default=None,
                        help="Composition rules to run. Defaults to the three "
                             "deployed arms in methods.ORDER; "
                             "methods.EXTRA holds the two diagnostic rules "
                             "(gauss_jacobian_mixture_prior, "
                             "marginal_product), which must be asked for.")
    result.add_argument("--replicates", type=int, default=3)
    result.add_argument("--draws", type=int, default=4_000)
    result.add_argument("--single-draws", type=int, default=20_000,
                        help="Per-observation draws for marginal_product's "
                             "stage 1; the KDE it feeds is that rule's only "
                             "approximation, so this is not a free knob.")
    result.add_argument("--timesteps", type=int, default=100,
                        help="Diffusion steps for every arm. The grid is "
                             "geometric in the noise scale (log-sigma) inside "
                             "the sampler, so this only changes its density.")
    result.add_argument("--stage1-churn", type=float, default=None,
                        help="Churn for the stage-1 single-observation sampler, "
                             "separately from --churn-eta. Defaults to it.")
    result.add_argument("--local-draws", type=int, default=0,
                        help="Cap on the draws stage 3 evaluates the locals at, "
                             "independent of --draws. 0 means equal to --draws.")
    result.add_argument("--stage1-chunk", type=int, default=16,
                        help="Observations per stage-1 sampler call. The "
                             "single-observation posteriors are independent, so "
                             "this only caps peak GPU memory; 0 means one call.")
    result.add_argument("--grid-points", type=int, default=1601,
                        help="Resolution of the shared grid the exact "
                             "factorisation and the tilt arms compose on.")
    result.add_argument("--kde-bandwidth", type=float, default=None)
    result.add_argument("--bag", type=int, default=64,
                        help="Bootstrap replicates over observations for the "
                             "bagged arm.")
    result.add_argument("--support-quantile", type=float, default=1e-3,
                        help="Two-sided quantile defining each observation's "
                             "empirical support; the composed density is "
                             "restricted to their intersection.")
    result.add_argument("--evidence-clip", type=float, default=8.0,
                        help="Bound on |log p(g|x_i) - log p(g)| for the "
                             "evidence-clipped arm. Composed log-error is the "
                             "sum of per-observation log-errors, so this bounds "
                             "the composed damage by n times this value.")
    result.add_argument("--branch-mass", type=float, default=0.75,
                        help="Exact posterior mass on the positive branch in "
                             "the 'asymmetric' regime, at every n.")
    result.add_argument("--spread", type=float, default=1.0)
    result.add_argument("--pilot-draws", type=int, default=1_000)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--churn-eta", type=float, default=4.0,
                        help="Karras churn on the DPM2 predictor for both "
                             "Gaussian rules. The Langevin arm never churns.")
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="cuda")
    result.add_argument("--tag", default="main")
    result.add_argument("--oracle", action="store_true",
                        help="Replace the trained network with the exact "
                             "diffused score (oracle.py). Whatever the sampler "
                             "then returns is the composition rule's own error, "
                             "with no network error in it at all.")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def load_network(arguments, train):
    """The trained checkpoint named by the recipe flags, or a clear failure."""
    directory = train.model_directory(
        arguments.artifacts, arguments.config, arguments.arch,
        arguments.simulations, arguments.mask_menu, arguments.time_sampling,
    )
    checkpoint = directory / "Model_checkpoint.pt"
    if not checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Train one first:\n"
            f"    python train.py --config {arguments.config} "
            f"--simulations {arguments.simulations}"
        )
    network = train.load_to(checkpoint, arguments.device)
    parameters = sum(p.numel() for p in network.model.parameters())
    print(f"{arguments.arch}/{arguments.config} {parameters:,} parameters "
          f"({arguments.mask_menu}, {arguments.time_sampling})", flush=True)
    return network, parameters, checkpoint


def main(argv=None):
    arguments = parser().parse_args(argv)
    info = configure_runtime(gpu=arguments.device.startswith("cuda"))
    print(f"CPU affinity: {info['active_cpus']}/{info['host_cpus']} CPUs "
          f"({info['percent']:.3f}%)", flush=True)

    import numpy as np

    import datasets
    import methods
    import metrics
    import model as M
    import train

    methods.COMMON["timesteps"] = int(arguments.timesteps)

    regimes = list(arguments.regimes or datasets.REGIMES)
    method_names = list(arguments.methods or methods.ORDER)
    unknown = set(method_names) - set(methods.METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")

    if arguments.oracle:
        import oracle

        network = oracle.OracleModel(M.DEFAULT,
                                     sigma=train.SDE_KWARGS["sigma"],
                                     device=arguments.device)
        parameters, checkpoint = 0, "oracle (exact diffused score)"
        print("Score source: the EXACT diffused score, not a network.",
              flush=True)
    else:
        network, parameters, checkpoint = load_network(arguments, train)


    mixture = M.DEFAULT
    output = Path(arguments.artifacts) / "scaling" / arguments.tag
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    stored, dataset_records = {}, []
    started = time.perf_counter()

    for regime in regimes:
        for n in arguments.observations:
            for replicate in range(int(arguments.replicates)):
                # ``hash`` on a str is salted per process, so the regime is
                # keyed by its position in a fixed tuple instead: a re-run has
                # to reproduce the same datasets.
                rng = np.random.default_rng(
                    [arguments.seed, datasets.REGIMES.index(regime), n,
                     replicate]
                )
                # The sampler seed moves with the replicate. At n = 1 the
                # constructed regimes are deterministic -- the single
                # observation *is* the prescribed total -- so replicates that
                # shared a sampler seed would report an identical draw twice and
                # an interquartile range of exactly zero.
                trial = argparse.Namespace(**vars(arguments))
                trial.seed = int(arguments.seed) + 1_000 * replicate
                data = datasets.make_dataset(
                    regime, n, rng, mixture=mixture,
                    branch_mass=arguments.branch_mass, spread=arguments.spread,
                )
                x = data["x"]
                if replicate == 0:
                    print(datasets.describe(data, mixture), flush=True)
                    dataset_records.append(
                        {"regime": regime, "observations": int(n),
                         "x": [float(v) for v in x],
                         "branch_mass": float(mixture.branch_weights(x)[1])}
                    )

                # The Monte-Carlo floor: the exact posterior, same draw count.
                exact = metrics.sample_exact(
                    mixture, x, arguments.draws,
                    np.random.default_rng([trial.seed, replicate, n]))
                exact_samples = np.zeros((n, arguments.draws, M.NODES))
                exact_samples[:, :, M.HIERARCHY[0]] = exact[:, 0][None, :]
                exact_samples[:, :, M.LOCAL_INDICES[0]] = exact[:, 1:].T
                exact_samples[:, :, M.OBSERVED_INDICES[0]] = x[:, None]
                scored = metrics.evaluate(exact_samples, mixture, x, 0.0)
                rows.append({"regime": regime, "observations": int(n),
                             "replicate": replicate, "method": "exact_mc",
                             **scored})
                if replicate == 0:
                    stored[f"{regime}_n{n}_exact_mc"] = exact[:, 0]

                for name in method_names:
                    try:
                        samples, seconds, _ = methods.draw(
                            network, x, name, trial, trial.device, mixture,
                        )
                    except Exception as error:  # noqa: BLE001
                        # One rule blowing up at one n must not cost the sweep
                        # the other 100 cells; the failure is recorded as such.
                        print(f"  !! {name} failed at {regime} n={n} "
                              f"replicate {replicate}: {error}", flush=True)
                        rows.append({"regime": regime, "observations": int(n),
                                     "replicate": replicate, "method": name,
                                     "failed": str(error)})
                        continue
                    scored = metrics.evaluate(samples, mixture, x, seconds)
                    rows.append({"regime": regime, "observations": int(n),
                                 "replicate": replicate, "method": name,
                                 **scored})
                    if replicate == 0:
                        stored[f"{regime}_n{n}_{name}"] = \
                            samples[0, :, M.HIERARCHY[0]]
                    print(f"  {regime:<14} n={n:<3} rep={replicate} "
                          f"{name:<30} W1/sd={scored['w1_over_sigma']:.4f} "
                          f"mass={scored['branch_mass']:.3f} "
                          f"(exact {scored['reference_branch_mass']:.3f}) "
                          f"{seconds:.1f}s", flush=True)

    fields = sorted({key for row in rows for key in row})
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for regime in regimes:
        for n in arguments.observations:
            for name in ["exact_mc"] + method_names:
                selected = [r for r in rows
                            if r["regime"] == regime
                            and r["observations"] == n
                            and r["method"] == name and "failed" not in r]
                if not selected:
                    continue
                summary[f"{regime}|{n}|{name}"] = {
                    key: metrics.summarise([r[key] for r in selected])
                    for key in ("w1_over_sigma", "branch_mass_error",
                                "width_ratio", "mean_error_sigma",
                                "local_w1_over_sigma", "seconds")
                }
    (output / "summary.json").write_text(json.dumps(summary, indent=2,
                                                    sort_keys=True))
    (output / "datasets.json").write_text(json.dumps(dataset_records, indent=2))
    np.savez(output / "draws.npz", **stored)
    (output / "settings.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "parameters": int(parameters),
        "score_source": "oracle" if arguments.oracle else "network",
        **{k: (str(v) if isinstance(v, Path) else v)
           for k, v in vars(arguments).items()},
    }, indent=2, sort_keys=True))
    print(f"\n{len(rows)} rows in {(time.perf_counter() - started) / 60:.1f} "
          f"minutes -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
