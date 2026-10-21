"""Train one COMPASS network on one closed-form-likelihood family (B5/B6).

    .COMPASS/bin/python -m validation.b5_b6_pfode_accuracy.train \
        --family gauss --nu 0.5 --budget 16384 --seed 0

Simulation budget = training simulations; a separate 10% (min 1024) validation set
is drawn from a different stream. The seed controls the training data *and* the
initialisation. The final-epoch model is the one evaluated (best-validation
selection picks lucky draws on shared validation noise); the best checkpoint is
kept for C9.

Outputs: ``MODEL_ROOT/<family>_nu<nu>/n<budget>_s<seed>/{final.pt, Model_checkpoint.pt,
meta.json}``.
"""

import validation.common.runtime_gpu  # noqa: F401  (CPU cap + autocvd, before torch)

import argparse
import json
import os
import time

import torch

from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm
from validation.b5_b6_pfode_accuracy.families import Family

MODEL_ROOT = "/export/data/rheinric/compass_validation/b5b6/models"
ARCH = dict(sde_type="vesde", sigma=25.0, hidden_size=128, depth=4, num_heads=8, mlp_ratio=4)
TRAIN = dict(batch_size=256, max_epochs=400, lr=1e-3, early_stopping_patience=40,
             time_sampling="log_sigma")


def model_dir(family, nu, budget, seed, variant=""):
    """``variant`` tags a non-default training recipe (e.g. "loglam"), so two recipes
    can never overwrite or be confused with each other."""
    suffix = f"_{variant}" if variant else ""
    return os.path.join(MODEL_ROOT, Family(family, nu).tag, f"n{budget}_s{seed}{suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-embedding", default="raw", choices=("raw", "log_lambda"))
    parser.add_argument("--mask-menu", default="bernoulli",
                        choices=("bernoulli", "inference", "likelihood"),
                        help="'inference': 1/3 posterior mask, 1/3 likelihood mask, 1/3 "
                             "Bernoulli(0.33) (the directions compare() uses)")
    parser.add_argument("--loss-weight", default="none", choices=("none", "edm"))
    parser.add_argument("--hidden-size", type=int, default=ARCH["hidden_size"])
    parser.add_argument("--depth", type=int, default=ARCH["depth"])
    parser.add_argument("--patience", type=int, default=TRAIN["early_stopping_patience"])
    parser.add_argument("--max-epochs", type=int, default=TRAIN["max_epochs"])
    args = parser.parse_args()
    variant = "_".join(v for v in ({"raw": "", "log_lambda": "loglam"}[args.time_embedding],
                                   {"bernoulli": "", "inference": "menu",
                                    "likelihood": "likonly"}[args.mask_menu],
                                   {"none": "", "edm": "edm"}[args.loss_weight],
                                   "" if args.hidden_size == ARCH["hidden_size"] else f"h{args.hidden_size}",
                                   "" if args.depth == ARCH["depth"] else f"d{args.depth}",
                                   "" if (args.patience, args.max_epochs) == (
                                       TRAIN["early_stopping_patience"], TRAIN["max_epochs"])
                                   else f"p{args.patience}e{args.max_epochs}") if v)

    out = model_dir(args.family, args.nu, args.budget, args.seed, variant)
    if os.path.exists(os.path.join(out, "final.pt")):
        print(f"exists: {out}")
        return
    os.makedirs(out, exist_ok=True)
    fam = Family(args.family, args.nu)
    data_gen = torch.Generator().manual_seed(1000 * args.seed + 17)
    val_gen = torch.Generator().manual_seed(1000 * args.seed + 9001)
    theta, x = fam.simulate(args.budget, data_gen)
    theta_val, x_val = fam.simulate(max(1024, args.budget // 10), val_gen)

    torch.manual_seed(args.seed)
    arch = dict(ARCH, hidden_size=args.hidden_size, depth=args.depth)
    train_cfg = dict(TRAIN, early_stopping_patience=args.patience, max_epochs=args.max_epochs)
    model = SBIm(nodes_size=5, device="cuda", time_embedding=args.time_embedding, **arch)
    mask_sampler = None
    if args.mask_menu == "likelihood":
        # A dedicated likelihood network: only the mask theta -> x is ever trained.
        from compass.schema import ConditionalMenu
        mask_sampler = ConditionalMenu(model.schema, roles={"t": [0, 1], "x": [2, 3, 4]},
                                       patterns=[("p(x|theta)", "t")], bernoulli_fraction=0.0)
    if args.mask_menu == "inference":
        from compass.schema import ConditionalMenu
        mask_sampler = ConditionalMenu(
            model.schema, roles={"t": [0, 1], "x": [2, 3, 4]},
            patterns=[("p(theta|x)", "x"), ("p(x|theta)", "t")],
            bernoulli_fraction=1 / 3)
    t0 = time.time()
    model.train(theta, x, theta_val=theta_val, x_val=x_val, device="cuda", verbose=False,
                path=out, name="Model", mask_sampler=mask_sampler,
                time_loss_weight=None if args.loss_weight == "none" else args.loss_weight,
                **train_cfg)
    seconds = time.time() - t0
    model.save(out, name="final")
    meta = dict(family=args.family, nu=args.nu, budget=args.budget, seed=args.seed,
                time_embedding=args.time_embedding, mask_menu=args.mask_menu,
                loss_weight=args.loss_weight,
                arch=arch, train=train_cfg, seconds=seconds,
                epochs=len(model.trainer.train_loss),
                params=sum(p.numel() for p in model.model.parameters()),
                gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done {out} in {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
