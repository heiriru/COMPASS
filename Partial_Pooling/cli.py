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
    result.add_argument("--model", action="append", dest="models")
    return result


def run_stage(stage, cpu_limit):
    args = parser(stage).parse_args()
    config = get_config(
        args.preset, args.seed, args.sde_type,
        args.beta_min, args.beta_max,
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
