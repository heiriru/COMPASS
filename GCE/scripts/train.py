#!/usr/bin/env python3
"""Train one single-star network: (model, observation profile, yield set) -> checkpoint.

Needs a train and a validation bank (gce.generate). On cuda it waits for a free GPU
via autocvd, as every GPU job here must.

    python scripts/train.py --model universal --profile tng_mock
    python scripts/train.py --model metallicity --profile ideal8 --seed 8
    CUDA_VISIBLE_DEVICES=-1 python scripts/train.py --model universal --profile tng_mock \
        --recipe smoke --device cpu --train-split smoke --val-split smoke     (wiring only)
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gce.paths import configure  # noqa: E402


def network_dir(root, model, yields_tag, profile, scale):
    return Path(root) / "networks" / model / yields_tag / f"{profile}-wide{scale:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--recipe", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--max-rows", type=int, default=0, help="> 0: use only this many training runs")
    parser.add_argument("--root", default=None)
    arguments = parser.parse_args()
    configure(cpus=3, gpu=arguments.device.startswith("cuda"))

    from gce import compass_setup, generate, models
    from gce.observation import PROFILES
    from gce.paths import DATA_ROOT
    from gce.simulator import TNG_YIELDS

    root = arguments.root or DATA_ROOT
    layout = compass_setup.Layout(models.get(arguments.model), PROFILES[arguments.profile])
    train_bank = generate.load_bank(generate.bank_dir(arguments.model, TNG_YIELDS, arguments.train_split,
                                                      arguments.scale, root))
    val_bank = generate.load_bank(generate.bank_dir(arguments.model, TNG_YIELDS, arguments.val_split,
                                                    arguments.scale, root))
    if arguments.max_rows:
        for key in ("globals", "locals", "abundances"):
            train_bank[key] = train_bank[key][:arguments.max_rows]
    recipe = compass_setup.RECIPE if arguments.recipe == "full" else compass_setup.SMOKE_RECIPE
    out = network_dir(root, arguments.model, TNG_YIELDS.tag, arguments.profile, arguments.scale)
    name = f"{arguments.model}-{arguments.profile}-s{arguments.seed}" + ("-smoke" if arguments.recipe == "smoke" else "")
    print(f"training {name}: {layout.describe()}; {len(train_bank['globals'])} runs "
          f"(bank failure rate {train_bank['failure_rate']:.2%})", flush=True)
    compass_setup.train(layout, train_bank, val_bank, out, name, recipe, arguments.device, arguments.scale,
                        arguments.seed)
    print(f"-> {out}/{name}.pt (+ {name}.gce.json)")


if __name__ == "__main__":
    main()
