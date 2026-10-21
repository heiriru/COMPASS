#!/usr/bin/env python3
"""Train the single-observation (g, l, x) network for the folded hierarchy.

Reuses the GM benchmark's recipe unchanged (h32d3wide tokenized trunk, 200k
simulations, six_weighted mask menu, log_sigma time sampling, VESDE sigma 15);
only the simulator differs. ``--mode`` picks the latent-attention pattern from
``compass.experimental.latent_attention`` ("none" is the production trunk).

    python train_abs.py --mode none
    python train_abs.py --mode hierarchical
    python train_abs.py --model square --mode none      (step A)
    python train_abs.py --model lean --mode hierarchical --tag T --time
        (low-noise ablations, precondition.py; trained into <mode>_<tag>/)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPASS = HERE.parent
GM = COMPASS / "Gaussian_mixtures_compositional_score_modelling"
sys.path[:0] = [str(HERE), str(GM), str(COMPASS / "experiments_arruda_schur")]

from runtime import configure_runtime  # noqa: E402  (GM's: CPU cap + autocvd, before torch)

import argparse  # noqa: E402
import os  # noqa: E402
import json  # noqa: E402

RECIPE = dict(config="h32d3wide", arch="tokenized", simulations=200_000,
              mask_menu="six_weighted", time_sampling="log_sigma")


def directory(mode, model="abs", tag=None):
    import experiments
    import train as gm_train
    folder = f"{mode}_{tag}" if tag else mode
    return gm_train.model_directory(experiments.get(model)["root"] / folder, RECIPE["config"], RECIPE["arch"],
                                    RECIPE["simulations"], RECIPE["mask_menu"],
                                    RECIPE["time_sampling"])


def roles(network):
    from compass import hierarchical_roles
    return hierarchical_roles(network.schema, 1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="abs", help="abs, square or lean (experiments.py)")
    parser.add_argument("--mode", choices=("none", "global", "hierarchical", "full"), default="none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default=None, help="ablation name; trains into <mode>_<tag>/")
    parser.add_argument("--time", action="store_true", help="log-lambda time input")
    parser.add_argument("--eps", type=float, default=1e-3, help="training floor (stock 1e-3)")
    parser.add_argument("--residual", action="store_true", help="prior score + learned residual")
    arguments = parser.parse_args()
    if (arguments.time or arguments.residual or arguments.eps != 1e-3) and not arguments.tag:
        parser.error("an ablation needs --tag, so it never overwrites the stock network")
    # An explicitly chosen CUDA_VISIBLE_DEVICES skips the autocvd wait (as gm.py does).
    configure_runtime(gpu=arguments.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"))

    import common
    import train as gm_train
    import experiments
    DEFAULT = experiments.get(arguments.model)["model"]
    from compass.experimental.latent_attention import save_latent_attention

    import precondition
    target = directory(arguments.mode, arguments.model, arguments.tag)
    target.mkdir(parents=True, exist_ok=True)
    original_build = gm_train.build

    def build(*args, **kwargs):
        network = common.build_with_latent_attention(original_build, arguments.mode, roles)(*args, **kwargs)
        save_latent_attention(network, target)
        config = precondition.attach(network, DEFAULT, time=arguments.time, eps=arguments.eps,
                                     residual=arguments.residual)
        precondition.save(config, target)
        return network

    gm_train.build = build
    load_attention = common.load_with_latent_attention(gm_train.load_to)

    def load_to(checkpoint, device):
        network = load_attention(checkpoint, device)
        precondition.load(network, Path(checkpoint).parent, DEFAULT)
        return network

    gm_train.load_to = load_to
    network, seconds, parameters = gm_train.train_or_load(
        target, RECIPE["config"], RECIPE["arch"], RECIPE["simulations"], arguments.device,
        seed=arguments.seed, mask_menu=RECIPE["mask_menu"],
        time_sampling=RECIPE["time_sampling"], mixture=DEFAULT)
    if parameters is not None:
        (target / "training.json").write_text(json.dumps({
            **RECIPE, "mode": arguments.mode, "tag": arguments.tag,
            "precondition": {"time": arguments.time, "eps": arguments.eps,
                             "residual": arguments.residual}, "model_name": arguments.model, "parameters": int(parameters),
            "train_seconds": seconds, "seed": arguments.seed,
            "model": {"sigma_g": DEFAULT.sigma_g, "sigma_l": DEFAULT.sigma_l,
                      "sigma_x": DEFAULT.sigma_x},
        }, indent=2))
    print(f"trained {arguments.mode}: {parameters} parameters in {seconds / 60:.1f} min -> {target}")


if __name__ == "__main__":
    main()
