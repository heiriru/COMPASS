#!/usr/bin/env python3
"""Train the single-observation score network for the mixture hierarchy.

Three nodes -- ``(g, l, x)`` -- and one training row per ``(g, l, x)`` triple
drawn from the prior predictive, each with its own ``g``. Nothing about the tall
problem enters training: every cross-observation statement made later is the
composition rule's, which is the split this whole directory exists to measure.

Recipe, and why each piece is not the library default
-----------------------------------------------------
``time_sampling="log_sigma"``
    COMPASS ships ``"mixture"``, which spends half its budget uniform in ``t``
    and therefore puts almost nothing near small lambda -- where the sampler
    needs the score and where essentially all of the composed error lives.
    Measured elsewhere in this repository at ~3x better row scores for free.

``mask_menu="six_weighted"``
    The six conditionals with weights ``[7, 7, 7, 1, 1, 1]``. Under a uniform
    six-menu the three conditionals that *generate* ``x`` carry ~87% of the
    loss, so a uniform draw spends most of the gradient learning the forward
    model. Two of the three posterior-side conditionals are used directly by
    the samplers here: ``p(g, l | x)`` by every rule, and ``p(l | g, x)`` by
    ``marginal_product``'s third stage.

``sigma = 15`` in the VESDE
    The prior modes sit at ``+-2.5`` and ``x`` reaches ``+-6``, so the noise
    ceiling has to exceed the largest distance the reverse process must travel
    between the two branches. The stock 25 wastes schedule; 8 (the arm's value)
    is too small for a target this wide.

Usage:
    python train.py                       # 200k simulations, h32d3wide
    python train.py --config h16d2 --simulations 50000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_runtime  # noqa: E402  (before numeric imports)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

ROOT = Path(__file__).resolve().parent

SDE_KWARGS = {"sde_type": "vesde", "sigma": 15.0}

CONFIGS = {
    "h64d4wide": {"hidden_size": 64, "depth": 4, "num_heads": 8, "mlp_ratio": 8},
    "h32d3wide": {"hidden_size": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 4},
    "h32d3": {"hidden_size": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 2},
    "h16d2": {"hidden_size": 16, "depth": 2, "num_heads": 2, "mlp_ratio": 2},
    "h8d2": {"hidden_size": 8, "depth": 2, "num_heads": 2, "mlp_ratio": 2},
}

# A gradient-step budget rather than an epoch count, so a small data budget is
# not also handed less optimization and the difference reported as a data effect.
REFERENCE_SAMPLES = 200_000
MAX_EPOCHS = 200
PATIENCE = 40
BATCH_SIZE = 1024
LR = 5e-4
VALIDATION_FRACTION = 0.1
DEFAULT_TIME_SAMPLING = "log_sigma"
MASK_MENUS = ("six_weighted", "six", "bernoulli")


def schedule(train_samples):
    factor = REFERENCE_SAMPLES / float(train_samples)
    return (max(5, int(round(MAX_EPOCHS * factor))),
            max(5, int(round(PATIENCE * factor))))


def build(config, arch="tokenized", device="cpu"):
    import model as M
    from compass import ScoreBasedInferenceModel as SBIm

    return SBIm(nodes_size=M.NODES, arch=arch, device=device,
                **SDE_KWARGS, **CONFIGS[config])


def six_weighted_menu(network, posterior_weight=7.0):
    import model as M
    from compass import SIX_CONDITIONALS, ConditionalMenu, hierarchical_roles

    return ConditionalMenu(
        schema=network.schema,
        roles=hierarchical_roles(network.schema, len(M.GLOBAL_NAMES),
                                 len(M.LOCAL_NAMES)),
        patterns=SIX_CONDITIONALS,
        weights=[posterior_weight] * 3 + [1.0] * 3,
        bernoulli_fraction=0.0,
    )


def six_menu(network):
    import model as M
    from compass import SIX_CONDITIONALS, ConditionalMenu, hierarchical_roles

    return ConditionalMenu(
        schema=network.schema,
        roles=hierarchical_roles(network.schema, len(M.GLOBAL_NAMES),
                                 len(M.LOCAL_NAMES)),
        patterns=SIX_CONDITIONALS, weights=None, bernoulli_fraction=0.0,
    )


def build_mask_sampler(network, mask_menu):
    if mask_menu == "bernoulli":
        return None
    if mask_menu == "six":
        return six_menu(network)
    if mask_menu == "six_weighted":
        return six_weighted_menu(network)
    raise ValueError(f"Unknown mask_menu {mask_menu!r}; choose from {MASK_MENUS}.")


def model_directory(root, config, arch, train_samples, mask_menu,
                    time_sampling=DEFAULT_TIME_SAMPLING):
    """Every recipe is tagged into the path, so two can never overwrite."""
    return (Path(root) / "models"
            / f"{arch}_{config}_n{train_samples}_{mask_menu}_{time_sampling}")


def load_to(checkpoint, device):
    """``SBIm.load`` builds on the CPU regardless of its ``device`` argument."""
    from compass import ScoreBasedInferenceModel as SBIm

    network = SBIm.load(str(checkpoint), device=device)
    network.model.to(device)
    return network


def train_or_load(directory, config, arch, train_samples, device, seed=7,
                  force=False, verbose=True, mask_menu="six_weighted",
                  time_sampling=DEFAULT_TIME_SAMPLING, mixture=None,
                  max_epochs=None):
    import numpy as np
    import torch

    import model as M

    mixture = mixture or M.DEFAULT
    directory = Path(directory)
    checkpoint = directory / "Model_checkpoint.pt"
    if checkpoint.exists() and not force:
        print(f"Loading checkpoint: {checkpoint}")
        return load_to(checkpoint, device), 0.0, None

    validation_samples = max(2_000, int(VALIDATION_FRACTION * train_samples))
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_rows = mixture.simulate(train_samples,
                                  np.random.default_rng([seed, 1]))
    validation_rows = mixture.simulate(validation_samples,
                                       np.random.default_rng([seed, 2]))

    def split(rows):
        theta = torch.as_tensor(rows[:, M.LATENT_INDICES], dtype=torch.float32)
        x = torch.as_tensor(rows[:, M.OBSERVED_INDICES], dtype=torch.float32)
        return theta, x

    theta_train, x_train = split(train_rows)
    theta_validation, x_validation = split(validation_rows)

    network = build(config, arch, device=device)
    parameters = sum(p.numel() for p in network.model.parameters())
    mask_sampler = build_mask_sampler(network, mask_menu)
    epochs, patience = schedule(train_samples)
    if max_epochs is not None:
        epochs = int(max_epochs)
    print(
        f"Training {arch}/{config} ({parameters:,} parameters) on "
        f"{train_samples:,} simulations, batch {BATCH_SIZE}, up to {epochs} "
        f"epochs (patience {patience}), menu {mask_menu}, "
        f"time sampling {time_sampling}",
        flush=True,
    )
    directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    network.train(
        theta=theta_train, x=x_train, theta_val=theta_validation,
        x_val=x_validation, batch_size=BATCH_SIZE, max_epochs=epochs,
        early_stopping_patience=patience, lr=LR, time_sampling=time_sampling,
        device=device, verbose=verbose, path=str(directory),
        mask_sampler=mask_sampler,
    )
    seconds = time.perf_counter() - started
    return load_to(checkpoint, device), seconds, parameters


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", choices=sorted(CONFIGS), default="h32d3wide")
    result.add_argument("--arch", choices=("tokenized", "legacy"),
                        default="tokenized")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--mask-menu", choices=MASK_MENUS, default="six_weighted")
    result.add_argument("--time-sampling",
                        choices=("log_sigma", "mixture", "uniform"),
                        default=DEFAULT_TIME_SAMPLING)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--device", default="cuda")
    result.add_argument("--force", action="store_true")
    result.add_argument("--max-epochs", type=int, default=None,
                        help="Override the schedule's epoch cap. The schedule "
                             "equalizes gradient steps across data budgets, so "
                             "use this only for smoke tests or for an arm that "
                             "hit the cap while still improving.")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    info = configure_runtime(gpu=arguments.device.startswith("cuda"))
    print(f"CPU affinity: {info['active_cpus']}/{info['host_cpus']} CPUs "
          f"({info['percent']:.3f}%)")

    directory = model_directory(
        arguments.artifacts, arguments.config, arguments.arch,
        arguments.simulations, arguments.mask_menu, arguments.time_sampling,
    )
    _, seconds, parameters = train_or_load(
        directory, arguments.config, arguments.arch, arguments.simulations,
        arguments.device, seed=arguments.seed, force=arguments.force,
        mask_menu=arguments.mask_menu, time_sampling=arguments.time_sampling,
        max_epochs=arguments.max_epochs,
    )
    if parameters is not None:
        import model as M

        (directory / "training.json").write_text(json.dumps({
            "arch": arguments.arch, "config": arguments.config,
            "settings": CONFIGS[arguments.config], "sde": SDE_KWARGS,
            "nodes": M.NODES, "parameters": int(parameters),
            "mask_menu": arguments.mask_menu,
            "time_sampling": arguments.time_sampling,
            "simulations": int(arguments.simulations),
            "seed": int(arguments.seed), "train_seconds": seconds,
            "max_epochs_override": arguments.max_epochs,
            "mixture": {"mu_0": M.DEFAULT.mu_0, "tau_g": M.DEFAULT.tau_g,
                        "tau_l": M.DEFAULT.tau_l, "sigma_x": M.DEFAULT.sigma_x,
                        "weights": list(M.DEFAULT.weights)},
        }, indent=2, sort_keys=True))
        print(f"Trained in {seconds / 60:.1f} minutes -> {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
