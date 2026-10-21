"""Shared command-line implementation; wrappers enforce CPU affinity first."""

import argparse
from pathlib import Path
import random
import numpy as np
import torch

from .config import PRESETS, SDE_TYPES, get_config
from .data_pipeline import generate_test_data, generate_training_data
from .model_pipeline import train_models
from .paths import BenchmarkPaths


def parser(stage):
    result = argparse.ArgumentParser(description=f"Partial-pooling benchmark: {stage}")
    result.add_argument("--preset", choices=PRESETS, default="smoke")
    result.add_argument("--sde-type", choices=SDE_TYPES, default="vesde")
    result.add_argument("--beta-min", type=float, default=0.1)
    result.add_argument("--beta-max", type=float, default=20.0)
    result.add_argument("--seed", type=int)
    result.add_argument("--root", type=Path)
    result.add_argument("--force", action="store_true")
    result.add_argument("--device", default="auto")
    result.add_argument(
        "--train-subset", type=int,
        help=(
            "Train on only the first N rows of the preset's training split "
            "(and the matching fraction of its validation split). The "
            "simulations, normalization constants and test datasets are the "
            "preset's own, so a subset run is directly comparable with the "
            "full-size run."
        ),
    )
    result.add_argument(
        "--arch", choices=("legacy", "tokenized"), default="legacy",
        help=(
            "legacy: one token per scalar with per-node adaLN modulation and a "
            "flattened output matmul (the historical network). tokenized: the "
            "schema-driven network with a token-count-agnostic trunk."
        ),
    )
    result.add_argument(
        "--tokenization", choices=("scalar", "trial", "group"), default="scalar",
        help=(
            "arch=tokenized only. scalar: 100 scalar tokens (isolates the trunk "
            "change). trial: 10 theta tokens plus 30 trial tokens through one "
            "shared (choice, log RT, censored) embedder -- lossless, the "
            "diffusion state stays the 90 raw scalars. group: --train-subjects "
            "subjects in one row, each with its own locals and its own trial "
            "block, read by the two-level GroupedTransformer so the group-level "
            "parameters see several subjects at once."
        ),
    )
    result.add_argument(
        "--time-sampling", choices=("mixture", "uniform", "log_sigma"),
        help=(
            "How diffusion times are drawn in training. 'log_sigma' is "
            "log-uniform in the noise scale; 'mixture' (the default every arm "
            "so far used) is 50/50 with uniform-in-t."
        ),
    )
    result.add_argument(
        "--init-seed", type=int,
        help=(
            "Seed for weight initialisation and the training shuffle only. "
            "Unlike --seed (which is root_seed, a data axis and therefore a "
            "different training split), this reruns the *same* experiment, "
            "which is what a seed-variance check needs."
        ),
    )
    result.add_argument(
        "--train-subjects", type=int,
        help=(
            "tokenization=group only. Subjects per training row, and therefore "
            "per composed factor at inference: 100 test subjects become "
            "100/train_subjects composed units. Unlike every other architecture "
            "flag this one also changes the *data*, so the arm generates its own "
            "training split; it still reuses the preset's normalization "
            "constants and test datasets."
        ),
    )
    result.add_argument(
        "--train-size", type=int,
        help=(
            "Rows to simulate for this arm's training split. Only meaningful "
            "with --train-subjects, which is the one setting that needs a split "
            "of its own: 100000 gives the globals-matched arm (the same number "
            "of group-parameter draws as the flat control) and 100000/K the "
            "subject-budget-matched one."
        ),
    )
    result.add_argument("--validation-size", type=int,
                        help="Validation rows for a --train-size split.")
    result.add_argument(
        "--depth-readout", type=int,
        help=(
            "tokenization=group only. Blocks after each subject vector is "
            "broadcast back onto its own tokens. 0 (default) is the recommended "
            "2+2+0 at 761,797 parameters; 1 costs 148,544 more and is the only "
            "extra block the 1,000,000-parameter cap allows."
        ),
    )
    result.add_argument(
        "--group-pool", choices=("mean", "pma"),
        help="tokenization=group only: how a repeat's tokens become one vector.",
    )
    result.add_argument(
        "--group-slot", choices=("learned", "sinusoidal", "none"),
        help=(
            "tokenization=group only. 'learned' (default) gives each subject "
            "slot its own embedding, so no subject-permutation symmetry is "
            "asserted; 'none' is the ablation that leaves the across-subject "
            "trunk equivariant."
        ),
    )
    result.add_argument(
        "--trunk", choices=("shared", "position"), default="shared",
        help="arch=tokenized only; 'position' reproduces the legacy trunk.",
    )
    result.add_argument(
        "--token-offset", action="store_true",
        help="arch=tokenized ablation: per-token offset on the shared modulation.",
    )
    result.add_argument(
        "--time-embedding-size", type=int,
        help=(
            "arch=tokenized only. Width of the sinusoidal diffusion-time "
            "embedding. It is a function of t alone, but it also sets the input "
            "width of every adaLN modulation, so it dominates the budget at "
            "small hidden_size: for trial/h64/d2 it is 131,584 of 463,364 "
            "parameters at 256, and dropping it to 64 takes the whole network "
            "to 192,836."
        ),
    )
    result.add_argument(
        "--per-position-head", action="store_true",
        help=(
            "arch=tokenized only. Give every scalar of an atomic block its own "
            "output readout instead of one shared direction per block. IID "
            "blocks stay shared, so trial exchangeability is preserved."
        ),
    )
    result.add_argument(
        "--mask-menu", choices=("bernoulli", "six", "six_mixed"),
        default="bernoulli",
        help=(
            "Which conditionals the training condition mask is drawn from. "
            "bernoulli: the historical per-token Bernoulli(0.33). six: uniform "
            "over p(g,l|x), p(g|l,x), p(l|g,x), p(x|g,l), p(x,l|g), p(x,g|l). "
            "six_mixed: those six mixed with the Bernoulli draw at "
            "--mask-bernoulli-fraction."
        ),
    )
    result.add_argument(
        "--mask-bernoulli-fraction", type=float,
        help="mask-menu=six_mixed only: probability a draw is Bernoulli.",
    )
    result.add_argument(
        "--mask-posterior-weight", type=float,
        help=(
            "mask-menu six/six_mixed only. Relative weight of the three "
            "observation-conditioned entries p(g,l|x), p(g|l,x), p(l|g,x) "
            "against the three that generate x. Uniform (1.0) spends most of "
            "the gradient on the harder x-latent conditionals."
        ),
    )
    result.add_argument("--hidden-size", type=int)
    result.add_argument("--depth", type=int)
    result.add_argument("--num-heads", type=int)
    result.add_argument("--mlp-ratio", type=int)
    result.add_argument("--train-epochs", type=int)
    result.add_argument("--patience", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--model", action="append", dest="models")
    return result


def run_stage(stage, cpu_limit):
    args = parser(stage).parse_args()
    config = get_config(
        args.preset, args.seed, args.sde_type,
        args.beta_min, args.beta_max,
        train_subset=args.train_subset,
        arch=args.arch, tokenization=args.tokenization, trunk=args.trunk,
        token_offset=args.token_offset or None,
        time_embedding_size=args.time_embedding_size,
        per_position_head=args.per_position_head or None,
        mask_menu=args.mask_menu,
        mask_bernoulli_fraction=args.mask_bernoulli_fraction,
        init_seed=args.init_seed,
        time_sampling=args.time_sampling,
        train_subjects=args.train_subjects,
        train_size=args.train_size,
        validation_size=args.validation_size,
        depth_readout=args.depth_readout,
        group_pool=args.group_pool,
        group_slot=args.group_slot,
        mask_posterior_weight=args.mask_posterior_weight,
        hidden_size=args.hidden_size, depth=args.depth,
        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        train_epochs=args.train_epochs, patience=args.patience,
        batch_size=args.batch_size,
    )
    random.seed(config.root_seed)
    np.random.seed(config.root_seed)
    torch.manual_seed(config.root_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.root_seed)
    paths = BenchmarkPaths(args.root) if args.root else BenchmarkPaths.default()
    paths.ensure()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        device = "cuda:0"
    if stage == "create_training_data":
        return generate_training_data(config, paths, args.force, cpu_limit)
    if stage == "create_test_data":
        return generate_test_data(config, paths, args.force, cpu_limit)
    if stage == "train_models":
        return train_models(config, args.models, paths, args.force, device)
    raise ValueError(stage)
