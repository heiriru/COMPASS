#!/usr/bin/env python3
"""Train COMPASS on the frozen 16-D trial-set summary.

This is intentionally a separate experiment entry point.  The raw benchmark
models ``p(theta, X)`` over 100 nodes; this model instead learns the distinct
summary-space joint ``p(theta, z)`` over 26 scalar nodes.  Keeping the two
artifact namespaces separate prevents a raw-observation checkpoint or recovery
run from ever being mistaken for an encoder result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Partial_Pooling.runtime import configure_runtime  # noqa: E402

# Caps this process at three CPUs and claims exactly one genuinely free GPU.
# Must run before torch is imported: the thread caps are read at import time,
# and the host is shared with other users.
configure_runtime(max_threads=3)

import torch

from compass import Atomic, ScoreBasedInferenceModel, six_conditional_menu
from compass.SetEncoder import SetEncoder
from Partial_Pooling.config import get_config
from Partial_Pooling.data_pipeline import load_split
from Partial_Pooling.model_pipeline import load_normalizers
from Partial_Pooling.normalization import Normalizer
from Partial_Pooling.paths import BenchmarkPaths


TAG = "setenc-d16-w64x2s4"   # the first frozen encoder; the default --encoder-tag


def load_encoder(path: Path, device: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    encoder = SetEncoder(**config).to(device)
    encoder.load_state_dict(state["encoder_state_dict"])
    encoder.eval()
    return encoder, Normalizer.from_state_dict(state["item_normalizer"]), \
        Normalizer.from_state_dict(state["z_normalizer"])


def encode(source, encoder, item_normalizer, z_normalizer, device, batch_size):
    """Encode ``(rows, subjects, 90)`` observations into normalized z vectors."""
    raw = source["observations"]
    rows, subjects = raw.shape[:2]
    items = raw.reshape(rows * subjects, -1, 3)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            x = item_normalizer.transform(items[start:start + batch_size]).to(device)
            chunks.append(encoder(x).cpu())
    z = torch.cat(chunks)
    return z_normalizer.transform(z).reshape(rows, subjects, -1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-subset", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--time-embedding-size", type=int, default=256)
    parser.add_argument(
        "--encoder-tag", default=TAG,
        help=(
            "Stem of the frozen encoder under artifacts/encoders. The summary "
            "width is read off that encoder's own config rather than assumed, "
            "so a wider encoder needs no change here."
        ),
    )
    parser.add_argument(
        "--mask-recipe", choices=("six_w6", "mask6mix"), default="six_w6",
        help=(
            "six_w6: the six conditionals weighted 6:6:6:1:1:1 toward the "
            "posterior patterns, no Bernoulli component (what the first arm "
            "used). mask6mix: uniform over the six plus a 25%% Bernoulli "
            "fraction -- the recipe the raw-trial ladder arms are trained on, "
            "so it is the one that makes this arm comparable to them."
        ),
    )
    parser.add_argument("--time-sampling", choices=("mixture", "log_sigma"),
                        default="log_sigma")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = args.device if torch.cuda.is_available() else "cpu"
    paths = BenchmarkPaths.default().ensure()
    encoder_path = paths.data.parent / "encoders" / f"{args.encoder_tag}.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing frozen encoder: {encoder_path}")
    encoder, item_normalizer, z_normalizer = load_encoder(encoder_path, device)
    summary_dim = int(encoder.summary_dim)

    if args.mask_recipe == "six_w6":
        posterior_weight, bernoulli_fraction = 6.0, 0.0
    else:
        posterior_weight, bernoulli_fraction = 1.0, 0.25
    menu_weights = [posterior_weight] * 3 + [1.0] * 3

    # Reuse the exact small-preset simulations and theta normalization.  The
    # subset and validation split are scaled together so the two stay in the
    # same ratio as the ladder comparison whatever --train-subset is.
    config = get_config("small", train_subset=args.train_subset, arch="tokenized",
                        tokenization="scalar", hidden_size=args.hidden_size,
                        depth=args.depth, num_heads=args.heads, mlp_ratio=4,
                        mask_menu="six_mixed",
                        mask_bernoulli_fraction=bernoulli_fraction,
                        mask_posterior_weight=posterior_weight,
                        time_sampling=args.time_sampling)
    train = load_split(config, "train", paths)["sde"]
    validation = load_split(config, "validation", paths)["sde"]
    theta_normalizer = load_normalizers(config, paths)["sde_joint_theta"]
    z_train = encode(train, encoder, item_normalizer, z_normalizer, device, 1024)[:, 0]
    z_validation = encode(validation, encoder, item_normalizer, z_normalizer, device, 1024)[:, 0]
    theta_train = theta_normalizer.transform(torch.cat((train["globals"], train["locals"][:, 0]), 1))
    theta_validation = theta_normalizer.transform(torch.cat((validation["globals"], validation["locals"][:, 0]), 1))
    theta_train, z_train = theta_train[:args.train_subset], z_train[:args.train_subset]
    validation_rows = max(1, round(config.validation_size * args.train_subset / config.train_size))
    theta_validation, z_validation = theta_validation[:validation_rows], z_validation[:validation_rows]

    model = ScoreBasedInferenceModel(
        theta_size=10, x_spec=[Atomic("summary", summary_dim)], arch="tokenized",
        trunk="shared", time_embedding_size=args.time_embedding_size,
        sde_type=config.sde_type,
        sigma=config.sigma, hidden_size=args.hidden_size, depth=args.depth,
        num_heads=args.heads, mlp_ratio=4, device=device,
    )
    mask_sampler = six_conditional_menu(
        model.schema, global_size=7, local_size=3, weights=menu_weights,
        bernoulli_fraction=bernoulli_fraction,
    )
    out = paths.data.parent / "encoder_summary" / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    # The first arm's name is kept byte-identical so its checkpoint and its
    # recovery runs stay addressable; every new axis appends only when it
    # departs from that arm.
    name = f"{args.encoder_tag}-h{args.hidden_size}d{args.depth}-{args.time_sampling}"
    if args.time_embedding_size != 256:
        name += f"-te{args.time_embedding_size}"
    if args.mask_recipe != "six_w6":
        name += f"-{args.mask_recipe}"
    if args.train_subset != 100000:
        name += f"-n{args.train_subset // 1000}k"
    print(f"summary-space model: nodes={model.nodes_size}, tokens={model.token_count}, "
          f"parameters={sum(p.numel() for p in model.model.parameters()):,}", flush=True)
    model.train(theta_train, z_train, theta_validation, z_validation,
                batch_size=args.batch_size, max_epochs=args.epochs,
                early_stopping_patience=args.patience, device="cuda" if device.startswith("cuda") else "cpu",
                verbose=True, path=str(out), name=name, mask_sampler=mask_sampler,
                time_sampling=args.time_sampling)
    checkpoint = out / f"{name}_checkpoint.pt"
    manifest = out / f"{name}.json"
    manifest.write_text(json.dumps({
        "representation": "frozen_predictive_set_summary", "likelihood": "p(z|theta)",
        "encoder": str(encoder_path), "checkpoint": str(checkpoint),
        "theta_dim": 10, "summary_dim": summary_dim,
        "nodes": 10 + summary_dim, "encoder_tag": args.encoder_tag,
        "train_subset": args.train_subset, "validation_rows": validation_rows,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "time_sampling": args.time_sampling, "mask_recipe": args.mask_recipe,
        "mask_menu": (
            f"six_mixed(posterior_weight={posterior_weight:g}, "
            f"bernoulli_fraction={bernoulli_fraction:g})"
        ),
        "time_embedding_size": args.time_embedding_size,
        "hidden_size": args.hidden_size, "depth": args.depth, "heads": args.heads,
        "train_loss": model.trainer.train_loss, "validation_loss": model.trainer.val_loss,
    }, indent=2) + "\n")
    print(f"Wrote {checkpoint} and {manifest}", flush=True)


if __name__ == "__main__":
    main()
