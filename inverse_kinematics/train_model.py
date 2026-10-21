#!/usr/bin/env python3
"""Train the joint score network for the hierarchical arm.

The architecture choice carries over the one measured result from the
partial-pooling capacity ladder (``docs/ladder_progress.md``): at
``nodes_size = N`` the legacy backbone spends
``6 * N * hidden_size`` parameters per block on adaLN modulation that is a
function of the diffusion time alone, replicated per node. Replacing it with the
tokenized trunk -- one ``Linear(256, 6H)`` shared by all tokens, ``LayerNorm(H)``
instead of ``LayerNorm((N, H))``, and a token-wise output head -- removes that
replication. Measured here at ``N = 6``:

    legacy    h16 d2  =  367,136 parameters
    tokenized h16 d2  =   78,833 parameters      4.7x smaller

so ``--arch tokenized`` is the default and ``--arch legacy`` is kept only as the
control that shows what the reduction cost, if anything.

Two knobs the ladder says *not* to bother with here. ``tokenization="trial"``
(the IID block) has nothing to group: each observation is a single 2-vector, not
a bag of exchangeable trials, so there is no trial axis to share an embedder
over. And ``time_embedding_size`` is fixed at 256 inside the network, which puts
a hard floor of ~33K parameters under any configuration -- at these budgets that
floor is most of the model, which is why the sweep below bottoms out where it
does rather than continuing to shrink.

Usage:
    python inverse_kinematics/train_model.py --config h16d2
    python inverse_kinematics/train_model.py --config h8d2 --arch legacy
    python inverse_kinematics/train_model.py --budgets      # parameter table only
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

CPU_INFO = configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402

ROOT = Path(__file__).resolve().parent

# The arm's reachable set spans about +-2.6 in each observable and the priors are
# far tighter, so the stock sigma=25 VESDE ceiling would spend most of its
# schedule far outside the data. 8.0 is the value the exponential bench settled
# on for a problem of the same scale.
# ``IK_VESDE_SIGMA`` raises the ceiling for the wide-height study (x1 spans +-30 there).
SDE_KWARGS = {"sde_type": "vesde", "sigma": float(os.environ.get("IK_VESDE_SIGMA") or 8.0)}

# Data-dependent capacity, measured at nodes_size=6 (see README Part 4). The
# legacy adaLN modulation is a function of diffusion time alone and scales as
# 6*N*H per block, while attention+MLP scale as H^2*(4 + 2*mlp_ratio). At
# h16d2/mlp2 that leaves only 5,056 of 367,136 parameters -- 1.4% -- able to
# respond to the observation at all. Raising H and mlp_ratio is what rebalances
# it; the ``wide`` configs below exist for that.
CONFIGS = {
    "h64d4wide": {"hidden_size": 64, "depth": 4, "num_heads": 8, "mlp_ratio": 8},
    "h32d3wide": {"hidden_size": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 4},
    "h32d3": {"hidden_size": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 2},
    "h16d2": {"hidden_size": 16, "depth": 2, "num_heads": 2, "mlp_ratio": 2},
    "h16d1": {"hidden_size": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 1},
    "h8d2": {"hidden_size": 8, "depth": 2, "num_heads": 2, "mlp_ratio": 2},
    "h8d1": {"hidden_size": 8, "depth": 1, "num_heads": 2, "mlp_ratio": 1},
    "h4d1": {"hidden_size": 4, "depth": 1, "num_heads": 1, "mlp_ratio": 1},
}

# A gradient-step budget, not an epoch count: an epoch on 50K rows is a fifth of
# an epoch on 250K, so holding epochs fixed across budgets would hand the large
# ones five times the optimization and report it as a data effect.
REFERENCE_SAMPLES = 200_000
MAX_EPOCHS = 300
# Long, deliberately. The denoising-score-matching validation loss averages a
# random diffusion time per row, so it plateaus in visible steps rather than
# descending smoothly; short patience stops on plateau noise and inverts the
# capacity ordering.
PATIENCE = 60
BATCH_SIZE = 1024
LR = 5e-4
VALIDATION_FRACTION = 0.1


def schedule(train_samples):
    factor = REFERENCE_SAMPLES / float(train_samples)
    return (max(5, int(round(MAX_EPOCHS * factor))),
            max(5, int(round(PATIENCE * factor))))


def build(config, arch, device="cpu"):
    from compass import ScoreBasedInferenceModel as SBIm

    return SBIm(
        nodes_size=H.NODES, arch=arch, device=device,
        **SDE_KWARGS, **CONFIGS[config],
    )


def posterior_menu(model, bernoulli_fraction=0.2, posterior_weight=6.0):
    """A ``ConditionalMenu`` concentrated on the mask inference actually uses.

    Under the historical Bernoulli(0.33) draw the deployed posterior layout --
    all four parameters latent, both observables conditioned -- has probability
    ``0.67^4 * 0.33^2 = 2.19%``. The network therefore spends ~98% of its
    gradient signal on conditionals no sampler here ever evaluates, which is the
    single largest difference from a dedicated amortized posterior.

    The menu keeps the likelihood and a mixed pattern so the general conditional
    capability is not thrown away, and keeps a Bernoulli fraction for the same
    reason; it just puts the mass where it is used.
    """
    from compass.schema import ConditionalMenu

    return ConditionalMenu(
        schema=model.schema,
        roles={"theta": H.LATENT_INDICES, "obs": H.OBSERVED_INDICES},
        patterns=[("posterior", ["obs"]), ("likelihood", ["theta"])],
        weights=[posterior_weight, 1.0],
        bernoulli_fraction=bernoulli_fraction,
        bernoulli_probability=0.33,
    )


def six_weighted_menu(model, bernoulli_fraction=0.0, posterior_weight=7.0):
    """The six conditionals, reweighted so the *gradient* splits evenly.

    ``six_menu`` equalizes how often each conditional is *drawn*; it does not
    equalize what each contributes. From the measured per-conditional validation
    losses (``docs/followup_results.md``), the three conditionals with the
    observations latent carry 87.2% of the total loss under uniform draws, so a
    uniform menu spends roughly seven eighths of its gradient learning to
    generate ``x`` -- which no sampler in this directory ever asks it to do.

    ``SIX_CONDITIONALS`` is ordered posterior-side first (``p(g,l|x)``,
    ``p(g|l,x)``, ``p(l|g,x)``) then generative (``p(x|g,l)``, ``p(x,l|g)``,
    ``p(x,g|l)``), so weights ``[7, 7, 7, 1, 1, 1]`` are what split the gradient
    about evenly. This is the same correction ``Partial_Pooling`` applies via
    ``six_mixed``, where uniform ``six`` measured *worse* than the Bernoulli
    control on ``p(g,l|x)`` (97,375 vs 95,263) while the weighted variants beat
    it (94,443 at pw8, 94,056 at ``six_mixed``).
    """
    from compass import SIX_CONDITIONALS, ConditionalMenu, hierarchical_roles

    return ConditionalMenu(
        schema=model.schema,
        roles=hierarchical_roles(model.schema, len(H.GLOBAL_NAMES),
                                 len(H.LOCAL_NAMES)),
        patterns=SIX_CONDITIONALS,
        weights=[posterior_weight] * 3 + [1.0] * 3,
        bernoulli_fraction=bernoulli_fraction,
    )


def six_menu(model, bernoulli_fraction=0.0):
    """The uniform six-conditional menu, via the COMPASS API.

    The node layout here is exactly the ``[g, l, x]`` partition the six
    conditionals are defined over: ``angle_1`` global, ``(height_arm, angle_2,
    angle_3)`` local, ``(x1, x2)`` observed. Each of the six is drawn with
    probability 1/6 = 16.7%, so the deployed posterior ``p(g,l|x)`` goes from
    2.19% under Bernoulli(0.33) to 16.7%.

    Note what uniform costs, because it is measured rather than hypothetical:
    the three conditionals with the observations *latent* carry several times
    the loss of the three with them observed, so a uniform draw spends most of
    its gradient on generating ``x``. ``posterior_menu`` is the alternative that
    weights instead of equalizing; this directory runs both.
    """
    from compass import SIX_CONDITIONALS, ConditionalMenu, hierarchical_roles

    return ConditionalMenu(
        schema=model.schema,
        roles=hierarchical_roles(model.schema, len(H.GLOBAL_NAMES),
                                 len(H.LOCAL_NAMES)),
        patterns=SIX_CONDITIONALS,
        weights=None,                       # uniform: 1/6 each
        bernoulli_fraction=bernoulli_fraction,
    )


MASK_MENUS = ("bernoulli", "six", "six_weighted", "posterior")


def build_mask_sampler(model, mask_menu):
    if mask_menu == "bernoulli":
        return None
    if mask_menu == "six":
        return six_menu(model)
    if mask_menu == "six_weighted":
        return six_weighted_menu(model)
    if mask_menu == "posterior":
        return posterior_menu(model)
    raise ValueError(f"Unknown mask_menu {mask_menu!r}; choose from {MASK_MENUS}.")


def parameter_count(config, arch):
    model = build(config, arch)
    return sum(p.numel() for p in model.model.parameters())


# COMPASS ships time_sampling="mixture", which spends half its budget uniformly
# in t and therefore puts almost nothing near small lambda -- exactly where the
# sampler needs the score and where essentially all of the error lives. Measured
# on ../tutorials/Compositional_Inference_Fix/bimodal_distribution against a
# closed-form row score, "log_sigma" took the mean relative row-score error from
# 0.166 to 0.054 and the error at lambda=0.05 from 0.440 to 0.088 -- i.e. the
# small network trained this way beat a 9x larger one trained the other way, at
# no inference cost. That is the diagnosed defect here too: this directory's
# network is 6.7x behind BayesFlow on endpoint error and its single-observation
# posterior is unimodal where BayesFlow's is bimodal, both symptoms of a less
# accurate row score.
DEFAULT_TIME_SAMPLING = "log_sigma"

# The recipe the existing checkpoints and every number in README.md were trained
# under. Its directories keep their original untagged names so no checkpoint can
# ever be silently re-pointed at weights trained under the other recipe.
LEGACY_TIME_SAMPLING = "mixture"


def model_directory(root, config, arch, train_samples, menu=False,
                    time_sampling=DEFAULT_TIME_SAMPLING):
    """``menu`` is the mask recipe: False/True keep the published spellings.

    ``menu=False`` -> Bernoulli(0.33), ``menu=True`` -> ``posterior_menu`` (both
    already on disk under those names), and the string ``"six"`` -> the uniform
    six-conditional menu. Every non-legacy recipe is tagged into the directory
    name so two recipes can never overwrite each other and then be compared.
    """
    # "posterior" and True are the same recipe under two spellings, and the
    # checkpoints already on disk use the published "_menu" one. Mapping the
    # string to "_posterior" instead would silently point every legacy --menu
    # invocation at a directory that does not exist.
    if menu is True or menu == "posterior":
        suffix = "_menu"
    elif menu in (False, None, "bernoulli"):
        suffix = ""
    else:
        suffix = f"_{menu}"
    if time_sampling != LEGACY_TIME_SAMPLING:
        suffix = f"{suffix}_{time_sampling}"
    return Path(root) / "models" / f"{arch}_{config}_n{train_samples}{suffix}"


def menu_name(menu):
    """Normalize the ``menu`` argument to a ``MASK_MENUS`` entry."""
    if menu is True:
        return "posterior"
    if menu in (False, None):
        return "bernoulli"
    return str(menu)


def load_to(checkpoint, device):
    """``SBIm.load``, but with the network actually on ``device``.

    ``SBIm.load`` builds the module on the CPU regardless of its ``device``
    argument, so anything touching ``model.model(...)`` directly -- the Jacobian
    JVPs, for one -- would hit a device mismatch. Move it here so the result does
    not depend on which call happens first.
    """
    from compass import ScoreBasedInferenceModel as SBIm

    model = SBIm.load(str(checkpoint), device=device)
    model.model.to(device)
    return model


def train_or_load(directory, config, arch, train_samples, device, seed=7,
                  force=False, verbose=True, menu=False,
                  time_sampling=DEFAULT_TIME_SAMPLING,
                  max_epochs=None, resume_from=None):
    """``max_epochs`` overrides :func:`schedule`; ``resume_from`` warm-starts.

    ``resume_from`` loads an existing checkpoint's **weights** before training,
    so raising the epoch cap on a run that hit it does not repeat the epochs
    already spent. The optimizer state is *not* restored -- ``SBIm.train`` builds
    a fresh one -- so this is a warm start, not an exact continuation, and the
    first epochs re-adapt the moment estimates. Train into a new directory: the
    trainer only writes on validation improvement, so the source checkpoint is
    left intact either way, but a separate directory keeps the two comparable.
    """
    directory = Path(directory)
    checkpoint = directory / "Model_checkpoint.pt"
    if checkpoint.exists() and not force:
        print(f"Loading checkpoint: {checkpoint}")
        return load_to(checkpoint, device), 0.0, None

    validation_samples = max(2_000, int(VALIDATION_FRACTION * train_samples))
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_rng = sim.generator("train", seed)
    validation_rng = sim.generator("validation", seed)

    theta_train, x_train = H.simulate(train_samples, train_rng)
    theta_validation, x_validation = H.simulate(validation_samples, validation_rng)

    if resume_from is not None:
        model = load_to(Path(resume_from), device)
        print(f"Warm start from {resume_from} (weights only; fresh optimizer)")
    else:
        model = build(config, arch, device=device)
    parameters = sum(p.numel() for p in model.model.parameters())
    mask_sampler = build_mask_sampler(model, menu_name(menu))
    epochs, patience = schedule(train_samples)
    if max_epochs is not None:
        epochs = int(max_epochs)
    steps = epochs * max(1, train_samples // BATCH_SIZE)
    print(
        f"Training {arch}/{config} ({parameters:,} parameters) on "
        f"{train_samples:,} simulations, batch {BATCH_SIZE}, up to {epochs} "
        f"epochs (patience {patience}) -- ~{steps:,} gradient steps"
    )
    directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model.train(
        theta=theta_train, x=x_train, theta_val=theta_validation,
        x_val=x_validation, batch_size=BATCH_SIZE, max_epochs=epochs,
        early_stopping_patience=patience, lr=LR, time_sampling=time_sampling,
        device=device, verbose=verbose, path=str(directory),
        mask_sampler=mask_sampler,
    )
    seconds = time.perf_counter() - started
    return load_to(checkpoint, device), seconds, parameters


def budget_table():
    rows = []
    for arch in ("legacy", "tokenized"):
        for config in CONFIGS:
            rows.append((arch, config, parameter_count(config, arch)))
    width = max(len(f"{count:,}") for _, _, count in rows)
    print(f"{'arch':11}{'config':8}{'parameters':>{width + 2}}")
    for arch, config, count in rows:
        print(f"{arch:11}{config:8}{count:>{width + 2},}")
    return rows


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", choices=sorted(CONFIGS), default="h16d2")
    result.add_argument("--arch", choices=("tokenized", "legacy"),
                        default="tokenized")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--device", default="cuda")
    result.add_argument("--mask-menu", choices=MASK_MENUS, default="bernoulli",
                        help="Training condition-mask distribution: "
                             "'bernoulli' (independent per node, the historical "
                             "default), 'six' (the six conditionals drawn "
                             "uniformly, 1/6 each) or 'posterior' (the bespoke "
                             "posterior-weighted menu).")
    result.add_argument("--menu", dest="mask_menu", action="store_const",
                        const="posterior",
                        help="Deprecated alias for --mask-menu posterior.")
    result.add_argument("--time-sampling",
                        choices=("log_sigma", "mixture", "uniform"),
                        default=DEFAULT_TIME_SAMPLING,
                        help="Diffusion-time sampling during training. "
                             "'log_sigma' is the default; 'mixture' reproduces "
                             "the checkpoints README.md's tables were run on.")
    result.add_argument("--force", action="store_true")
    result.add_argument("--max-epochs", type=int, default=None,
                        help="Override the schedule's epoch cap. The schedule "
                             "equalizes gradient steps across data budgets; "
                             "raise this only for an arm that HIT the cap while "
                             "still improving, and say so in the run's notes.")
    result.add_argument("--resume-from", type=Path, default=None,
                        help="Checkpoint whose weights warm-start this run "
                             "(fresh optimizer). Use with --max-epochs to "
                             "extend a truncated run without repeating it.")
    result.add_argument("--output-suffix", default="",
                        help="Appended to the model directory name, so an "
                             "extended run does not overwrite the one it "
                             "resumed from.")
    result.add_argument("--budgets", action="store_true",
                        help="Print the parameter table and exit; trains nothing.")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.budgets:
        budget_table()
        return 0

    print(
        f"CPU affinity: {CPU_INFO['active_cpus']}/{CPU_INFO['host_cpus']} CPUs "
        f"({CPU_INFO['percent']:.3f}%)"
    )
    directory = model_directory(
        arguments.artifacts, arguments.config, arguments.arch,
        arguments.simulations, menu=arguments.mask_menu,
        time_sampling=arguments.time_sampling,
    )
    if arguments.output_suffix:
        directory = directory.with_name(directory.name + arguments.output_suffix)
    model, seconds, parameters = train_or_load(
        directory, arguments.config, arguments.arch, arguments.simulations,
        arguments.device, seed=arguments.seed, force=arguments.force,
        menu=arguments.mask_menu, time_sampling=arguments.time_sampling,
        max_epochs=arguments.max_epochs, resume_from=arguments.resume_from,
    )
    if parameters is not None:
        (directory / "training.json").write_text(json.dumps({
            "arch": arguments.arch,
            "config": arguments.config,
            "settings": CONFIGS[arguments.config],
            "sde": SDE_KWARGS,
            "nodes": H.NODES,
            "parameters": int(parameters),
            "mask_menu": menu_name(arguments.mask_menu),
            "time_sampling": arguments.time_sampling,
            "simulations": int(arguments.simulations),
            "seed": int(arguments.seed),
            "train_seconds": seconds,
            "max_epochs_override": arguments.max_epochs,
            "resumed_from": (str(arguments.resume_from)
                             if arguments.resume_from else None),
        }, indent=2, sort_keys=True))
        print(f"Trained in {seconds / 60:.1f} minutes -> {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
