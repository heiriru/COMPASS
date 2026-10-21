#!/usr/bin/env python3
"""Sample the folded-hierarchy tall posterior with every composition rule, plus MCMC.

One dataset per n (shared g_true, nested: each n takes the first n observations of
one sequence). Per (network mode, n, method) the draws are cached as .npz under
artifacts/runs/; pass --force to redo.

    python run.py --mode none --methods langevin_fnpse gauss_hierarchical gauss_jacobian_FD
    python run.py --mode hierarchical --methods arruda_direct gauss_jacobian_FD
    python run.py --mode oracle_joint --methods langevin_fnpse gauss_hierarchical gauss_jacobian_FD
    python run.py --mode oracle_hierarchical --methods arruda_direct gauss_jacobian_FD
    python run.py --model square --mode none --methods ...   (step A, artifacts/square/)
    python run.py --model lean --mode oracle_joint --methods mixture_product
        (product arms: per-observation draws -> fitted mixtures -> grid product)
    python run.py --mcmc-only
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPASS = HERE.parent
GM = COMPASS / "Gaussian_mixtures_compositional_score_modelling"
sys.path[:0] = [str(HERE), str(GM), str(COMPASS / "experiments_arruda_schur")]

from runtime import configure_runtime  # noqa: E402

import argparse  # noqa: E402
import os  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

DATA_SEED = 11
MAX_N = 64        # first block: the original n <= 64 datasets, unchanged
EXTRA_N = 64      # appended from a second stream, so n = 128 keeps the first 64 as they were
RUNS = None      # set from --model in main()
EXPERIMENT = None


def select(model):
    """Point the module at one experiment (model, g_true, artifact root)."""
    import experiments
    global EXPERIMENT, RUNS
    EXPERIMENT = experiments.get(model)
    RUNS = EXPERIMENT["root"] / "runs"
    return EXPERIMENT


def dataset(n):
    import numpy as np
    if EXPERIMENT is None:
        select("abs")
    model, g_true = EXPERIMENT["model"], EXPERIMENT["g_true"]
    locals_, x = model.simulate_dataset(MAX_N, np.random.default_rng(DATA_SEED), g_true)
    if n > MAX_N:
        more_l, more_x = model.simulate_dataset(EXTRA_N, np.random.default_rng([DATA_SEED, 1]), g_true)
        locals_, x = np.concatenate([locals_, more_l]), np.concatenate([x, more_x])
    if n > len(x):
        raise ValueError(f"at most {len(x)} observations are defined")
    return locals_[:n], x[:n]


def run_mcmc(observations, draws):
    import numpy as np
    model, g_true = EXPERIMENT["model"], EXPERIMENT["g_true"]
    for n in observations:
        path = RUNS / f"mcmc_n{n}.npz"
        if path.exists():
            continue
        locals_, x = dataset(n)
        started = time.perf_counter()
        g, l = model.gibbs(x, draws, np.random.default_rng([DATA_SEED, n]))
        np.savez(path, g=g, l=l, x=x, l_true=locals_, g_true=g_true,
                 seconds=time.perf_counter() - started)
        print(f"mcmc n={n}: P(g>0)={np.mean(g > 0):.3f}  E|g|={np.abs(g).mean():.3f}", flush=True)


def register_methods():
    import common
    import methods

    for name, (settings, kwargs) in common.NEW_METHODS.items():
        methods.METHODS[name] = {"label": name, "churn": True, "mixture_prior": False,
                                 "kwargs": dict(kwargs)}
    methods.METHODS["gauss_jacobian_FD"]["churn"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="abs", help="abs, square or lean (experiments.py)")
    parser.add_argument("--mode", default="none")
    parser.add_argument("--methods", nargs="+", default=[])
    parser.add_argument("--observations", type=int, nargs="+", default=[4, 16, 64, 128])
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--mcmc-draws", type=int, default=20_000)
    parser.add_argument("--mcmc-only", action="store_true")
    parser.add_argument("--pilot-draws", type=int, default=1_000)
    parser.add_argument("--jacobian-refresh", type=int, default=5)
    parser.add_argument("--churn-eta", type=float, default=4.0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--single-draws", type=int, default=20_000,
                        help="per-observation draws for the product arms")
    parser.add_argument("--suffix", default="", help="appended to the mode in run file names")
    parser.add_argument("--sample-eps", type=float, default=None,
                        help="sampler end time (default: the network's training floor, else 1e-3)")
    parser.add_argument("--grid-points", type=int, default=4001)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    select(arguments.model)
    RUNS.mkdir(parents=True, exist_ok=True)

    # An explicitly chosen CUDA_VISIBLE_DEVICES skips the autocvd wait (as gm.py does).
    configure_runtime(gpu=not arguments.mcmc_only and arguments.device.startswith("cuda")
                      and not os.environ.get("CUDA_VISIBLE_DEVICES"))
    run_mcmc(arguments.observations, arguments.mcmc_draws)
    if arguments.mcmc_only:
        return

    import numpy as np
    import common
    import methods
    import train as gm_train
    DEFAULT = EXPERIMENT["model"]

    common.install_configured_sampler()
    register_methods()
    if arguments.mode.startswith("oracle_"):
        if arguments.model not in ("abs", "lean"):
            raise SystemExit("the exact-score oracle exists for --model abs and lean only")
        # Exact diffused row score instead of a network (oracle_abs.py).
        import oracle_abs
        network = oracle_abs.oracle_model(arguments.mode.removeprefix("oracle_"), arguments.device,
                                          model=DEFAULT)
    else:
        load = common.load_with_latent_attention(gm_train.load_to)
        directory = gm_train.model_directory(EXPERIMENT["root"] / arguments.mode, "h32d3wide",
                                             "tokenized", 200_000, "six_weighted", "log_sigma")
        network = load(directory / "Model_checkpoint.pt", arguments.device)
        import precondition
        config = precondition.load(network, directory, DEFAULT)
        if config is not None:
            # sample down to the floor the network was trained at
            methods.COMMON["eps"] = config["eps"]
    if arguments.sample_eps:
        methods.COMMON["eps"] = arguments.sample_eps
    if "eps" in methods.COMMON:
        print(f"sampling eps {methods.COMMON['eps']}", flush=True)
    for parameter in network.model.parameters():
        parameter.requires_grad_(False)
    network.model.eval()

    for n in arguments.observations:
        _, x = dataset(n)
        for name in arguments.methods:
            path = RUNS / f"{arguments.mode}{arguments.suffix}_{name}_n{n}.npz"
            if path.exists() and not arguments.force:
                continue
            common.ACTIVE.clear()
            common.ACTIVE.update(common.NEW_METHODS.get(name, ({}, {}))[0])
            try:
                if name in methods.SCORE_METHODS or name in common.NEW_METHODS:
                    samples, seconds = methods.draw_score_rule(network, x, name, arguments,
                                                               arguments.device, DEFAULT)
                else:  # product arms (mixture_product, ...): no composed score
                    samples, seconds, detail = methods.draw(network, x, name, arguments,
                                                            arguments.device, DEFAULT)
                    if detail.get("chosen_k") is not None:
                        print(f"   chosen K per observation: {np.bincount(detail['chosen_k']).tolist()}",
                              flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"!! {arguments.mode}/{name} n={n} failed: {error!r}", flush=True)
                continue
            g = samples[0, :, 0]
            l = samples[:, :, 1].T
            np.savez(path, g=g, l=l, x=x, seconds=seconds)
            print(f"{arguments.mode + arguments.suffix:<13} {name:<20} n={n:<3} P(g>0)={np.mean(g > 0):.3f} "
                  f"E|g|={np.abs(g).mean():.3f} sd={g.std():.3f}  {seconds:.0f}s", flush=True)
    (RUNS / f"settings_{arguments.mode}{arguments.suffix}.json").write_text(json.dumps(vars(arguments), indent=2))


if __name__ == "__main__":
    main()
