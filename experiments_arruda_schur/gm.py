"""Gaussian-mixture benchmark: latent-attention networks and the arruda_schur rules.

Reuses ``Gaussian_mixtures_compositional_score_modelling`` unmodified: its
datasets, exact references, grader and ``run_scaling`` loop. Outputs go under
``experiments_arruda_schur/artifacts/gm/<mode>/``.

    python gm.py train --mode full
    python gm.py run --mode full --tag main -- --observations 2 8 32 128
    python gm.py run --mode oracle --tag oracle -- --observations 2 8 32 128

Everything after ``--`` goes to ``run_scaling`` unchanged. ``--methods`` there may
name any GM method or any key of ``common.NEW_METHODS``.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

GM = common.COMPASS / "Gaussian_mixtures_compositional_score_modelling"
sys.path.insert(0, str(GM))

MODES = ("none", "global", "hierarchical", "full", "oracle")
DEFAULT_RECIPE = dict(config="h32d3wide", arch="tokenized", simulations=200_000,
                      mask_menu="six_weighted", time_sampling="log_sigma")
DEFAULT_METHODS = ["gauss_jacobian_mixture_prior", "gauss_jacobian_FD", "arruda_direct",
                   "arruda_schur", "arruda_laplace", "arruda_gauss_laplace"]


def artifacts(mode):
    return HERE / "artifacts" / "gm" / mode


def roles(network):
    import model as M
    from compass import hierarchical_roles
    return hierarchical_roles(network.schema, len(M.GLOBAL_NAMES), len(M.LOCAL_NAMES))


def register_methods():
    import methods

    for name, (settings, kwargs) in common.NEW_METHODS.items():
        methods.METHODS[name] = {"label": name, "colour": "#444444", "churn": True,
                                 "mixture_prior": True, "kwargs": dict(kwargs)}
    original = methods.draw

    def draw(network, x, method, arguments, device, mixture=None):
        common.ACTIVE.clear()
        common.ACTIVE.update(common.NEW_METHODS.get(method, ({}, {}))[0])
        if mixture is None:
            return original(network, x, method, arguments, device)
        return original(network, x, method, arguments, device, mixture)

    methods.draw = draw


def link_production_checkpoint(recipe):
    """``none`` reuses the production checkpoint via a symlink, never a copy."""
    import train
    source = train.model_directory(GM / "artifacts", recipe["config"], recipe["arch"],
                                   recipe["simulations"], recipe["mask_menu"],
                                   recipe["time_sampling"])
    target = train.model_directory(artifacts("none"), recipe["config"], recipe["arch"],
                                   recipe["simulations"], recipe["mask_menu"],
                                   recipe["time_sampling"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.symlink(source, target)


def configure(arguments):
    """CPU cap, and autocvd unless CUDA_VISIBLE_DEVICES was set deliberately."""
    from runtime import configure_runtime
    explicit = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return configure_runtime(gpu=arguments.device.startswith("cuda") and not explicit)


def train_mode(arguments):
    configure(arguments)
    import train
    from compass.experimental.latent_attention import save_latent_attention

    recipe = DEFAULT_RECIPE
    directory = train.model_directory(artifacts(arguments.mode), recipe["config"],
                                      recipe["arch"], recipe["simulations"],
                                      recipe["mask_menu"], recipe["time_sampling"])
    directory.mkdir(parents=True, exist_ok=True)
    built = {}
    original_build = train.build

    def build(*args, **kwargs):
        network = common.build_with_latent_attention(original_build, arguments.mode,
                                                     roles)(*args, **kwargs)
        built["network"] = network
        save_latent_attention(network, directory)
        return network

    train.build = build
    train.load_to = common.load_with_latent_attention(train.load_to)
    network, seconds, parameters = train.train_or_load(
        directory, recipe["config"], recipe["arch"], recipe["simulations"],
        arguments.device, seed=arguments.seed, mask_menu=recipe["mask_menu"],
        time_sampling=recipe["time_sampling"],
    )
    print(f"trained {arguments.mode}: {parameters} parameters in {seconds / 60:.1f} min; "
          f"latent_attention={getattr(network.model, 'latent_attention', 'none')}")


def run_mode(arguments, passthrough):
    import run_scaling
    import train

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        # A deliberately chosen device: keep the CPU cap, skip the autocvd search.
        from runtime import configure_runtime
        run_scaling.configure_runtime = lambda max_threads=3, gpu=True: configure_runtime(
            max_threads, gpu=False)

    common.install_configured_sampler()
    register_methods()
    train.load_to = common.load_with_latent_attention(train.load_to)
    recipe = DEFAULT_RECIPE
    if arguments.mode == "none":
        link_production_checkpoint(recipe)
    argv = ["--artifacts", str(artifacts(arguments.mode)), "--tag", arguments.tag,
            "--config", recipe["config"], "--arch", recipe["arch"],
            "--simulations", str(recipe["simulations"]), "--mask-menu", recipe["mask_menu"],
            "--time-sampling", recipe["time_sampling"]]
    if arguments.mode == "oracle":
        argv.append("--oracle")
    if "--methods" not in passthrough:
        argv += ["--methods", *DEFAULT_METHODS]
    return run_scaling.main(argv + list(passthrough))


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "run"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args(argv)
    if arguments.action == "train":
        if arguments.mode in ("none", "oracle"):
            raise SystemExit("'none' reuses the production checkpoint; 'oracle' has no network.")
        return train_mode(arguments)
    return run_mode(arguments, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
