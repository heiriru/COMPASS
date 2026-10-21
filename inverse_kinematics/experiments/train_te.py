#!/usr/bin/env python3
"""Train one legacy arm at a chosen time-embedding width, without editing src.

Everything except the trunk rewiring is ``train_model.py``'s own code path --
the same simulator seeds, the same gradient-step schedule, the same mask menu
and time sampling -- so a ``--time-embedding 256`` arm is a control for the
sweep rather than a differently-built model. ``test_te_rewire.py`` pins that
the te=256 rewiring is the stock model bit-for-bit.

    python experiments/train_te.py --time-embedding 64 --device cuda:5
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from runtime import configure_cpu_limit  # noqa: E402

CPU_INFO = configure_cpu_limit()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402
import te_rewire as TE  # noqa: E402
import train_model as T  # noqa: E402

ROOT = HERE.parent


def build(te, device="cpu", config="h16d2"):
    model = T.build(config, "legacy", device=device)
    TE.rewire(model, te)
    model.model.to(device)
    return model


def directory(config, train_samples, menu, time_sampling, te, tag=""):
    name = (f"legacy_{config}_n{train_samples}_{menu}_{time_sampling}"
            f"_te{te}{tag}")
    return ROOT / "artifacts" / "models" / name


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time-embedding", type=int, required=True)
    p.add_argument("--config", default="h16d2")
    p.add_argument("--simulations", type=int, default=10_000)
    p.add_argument("--mask-menu", default="six_weighted", choices=T.MASK_MENUS)
    p.add_argument("--time-sampling", default="log_sigma")
    p.add_argument("--max-epochs", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    te = args.time_embedding
    out = directory(args.config, args.simulations, args.mask_menu,
                    args.time_sampling, te, args.tag)
    checkpoint = out / "Model_checkpoint.pt"
    if checkpoint.exists() and not args.force:
        print(f"Already trained: {checkpoint}")
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    theta_train, x_train = H.simulate(args.simulations, sim.generator("train", args.seed))
    n_validation = max(2_000, int(T.VALIDATION_FRACTION * args.simulations))
    theta_val, x_val = H.simulate(n_validation, sim.generator("validation", args.seed))

    model = build(te, args.device, args.config)
    report = TE.parameter_report(model.model)
    epochs, patience = T.schedule(args.simulations)
    if args.max_epochs is not None:
        epochs = int(args.max_epochs)
    mask_sampler = T.build_mask_sampler(model, args.mask_menu)

    print(f"te={te}  {report['total']:,} parameters  "
          f"data-dependent {report['data_dependent']:,} "
          f"({100 * report['data_dependent_share']:.1f}%)  "
          f"-> {out.name}, up to {epochs} epochs (patience {patience})")
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model.train(
        theta=theta_train, x=x_train, theta_val=theta_val, x_val=x_val,
        batch_size=T.BATCH_SIZE, max_epochs=epochs,
        early_stopping_patience=patience, lr=T.LR,
        time_sampling=args.time_sampling, device=args.device, verbose=True,
        path=str(out), mask_sampler=mask_sampler,
    )
    seconds = time.perf_counter() - started

    (out / "training.json").write_text(json.dumps({
        "time_embedding_size": te, "config": args.config,
        "arch": "legacy", "simulations": args.simulations,
        "mask_menu": args.mask_menu, "time_sampling": args.time_sampling,
        "max_epochs": epochs, "patience": patience, "seed": args.seed,
        "train_seconds": seconds, "device": args.device,
        "rewired_out_of_src": True, **report,
    }, indent=2))
    print(f"done in {seconds / 60:.1f} min -> {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
