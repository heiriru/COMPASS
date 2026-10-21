#!/usr/bin/env python3
"""Fine-tune an existing arm model without changing its Bernoulli mask task.

The hard endpoint region is oversampled by rejection on x1 only. Because the
selection probability depends only on x, it changes p(x) but not p(theta | x);
the conditional target remains valid. Trainer still draws GroupedBernoulli
(Bernoulli for this all-atomic legacy/schema model) masks with probability 0.33,
so every likelihood/posterior/marginal combination remains in the task.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit

CPU_INFO = configure_cpu_limit()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402

ROOT = Path(__file__).resolve().parent


def focused_rows(count: int, threshold: float, rng):
    """Prior-predictive rows conditional on |x1| < threshold."""
    theta_parts, x_parts = [], []
    accepted = 0
    while accepted < count:
        theta, x = H.simulate(max(20_000, 3 * (count - accepted)), rng)
        keep = x[:, 0].abs() < threshold
        theta_parts.append(theta[keep])
        x_parts.append(x[keep])
        accepted += int(keep.sum())
    return (torch.cat(theta_parts)[:count],
            torch.cat(x_parts)[:count])


def mixed_rows(count: int, focus_fraction: float, threshold: float, rng):
    focused = int(round(count * focus_fraction))
    ordinary = count - focused
    theta_o, x_o = H.simulate(ordinary, rng)
    if focused == 0:
        return theta_o, x_o
    if ordinary == 0:
        return focused_rows(focused, threshold, rng)
    theta_f, x_f = focused_rows(focused, threshold, rng)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(1234))
    return (torch.cat([theta_o, theta_f])[order],
            torch.cat([x_o, x_f])[order])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--base",
        type=Path,
        default=ROOT / "artifacts/models/tokenized_h32d3_n200000/Model_checkpoint.pt",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/models/bernoulli_focus_tokenized_h32d3_n200000",
    )
    result.add_argument("--samples", type=int, default=200_000)
    result.add_argument("--validation-samples", type=int, default=20_000)
    result.add_argument("--focus-fraction", type=float, default=0.5)
    result.add_argument("--focus-threshold", type=float, default=0.2)
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--patience", type=int, default=15)
    result.add_argument("--batch-size", type=int, default=1024)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--time-sampling",
                        choices=("uniform", "log_sigma", "mixture"),
                        default="mixture")
    result.add_argument("--posterior-mask-weight", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=71)
    result.add_argument("--device", default="cuda")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0.0 <= args.focus_fraction <= 1.0:
        raise ValueError("focus_fraction must lie in [0, 1].")
    if args.focus_threshold <= 0.0:
        raise ValueError("focus_threshold must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = mixed_rows(
        args.samples, args.focus_fraction, args.focus_threshold,
        sim.generator("bernoulli-focus-train", args.seed),
    )
    validation = mixed_rows(
        args.validation_samples, args.focus_fraction, args.focus_threshold,
        sim.generator("bernoulli-focus-validation", args.seed),
    )
    model = trainer.load_to(args.base, args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model.train(
        theta=train[0], x=train[1],
        theta_val=validation[0], x_val=validation[1],
        batch_size=args.batch_size, max_epochs=args.epochs,
        early_stopping_patience=args.patience, lr=args.lr,
        time_sampling=args.time_sampling, device=args.device,
        mask_loss_weights=[(H.CONDITION_MASK, args.posterior_mask_weight)],
        verbose=True, path=str(args.output),
    )
    seconds = time.perf_counter() - started
    metadata = {
        "base_checkpoint": str(args.base),
        "output_checkpoint": str(args.output / "Model_checkpoint.pt"),
        "samples": args.samples,
        "validation_samples": args.validation_samples,
        "focus_fraction": args.focus_fraction,
        "focus_selection": f"abs(x1) < {args.focus_threshold}",
        "selection_depends_only_on_observation": True,
        "mask_sampler": "GroupedBernoulli(schema, 0.33)",
        "posterior_mask_fixed": False,
        "posterior_mask_loss_weight": args.posterior_mask_weight,
        "time_sampling": args.time_sampling,
        "epochs_requested": args.epochs,
        "seed": args.seed,
        "seconds": seconds,
        "cpu_info": CPU_INFO,
    }
    (args.output / "training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(f"Wrote {args.output} in {seconds / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
