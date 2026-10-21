"""Train one COMPASS network for one model of a WAIC-validation family.

    .COMPASS/bin/python -m validation.a4_waic.train --family noise --model G --budget 65536 --seed 0

Recipe (the likelihood recipe of validation/likelihood_improvement plus the B6
architecture): legacy transformer (hidden 128, depth 4, 8 heads), VESDE sigma 25,
``log_sigma`` time sampling, the "inference" mask menu (1/3 posterior p(g,l|x), 1/3
likelihood p(x|g,l), 1/3 Bernoulli) and EDM high-noise loss weighting. The
final-epoch model is used (validation/c9: best-validation checkpoints are worse).

The network works in standardised coordinates (``Scaler``): parameters by their
exact prior mean/sd, observations by one shared per-family location/scale. The
composition prior of the globals is then N(0, I), and a raw-x WAIC is the
standardised one plus ``2 N sum_d log x_sd[d]`` (the same for every model of a
family, so it cancels in WAIC differences).
"""

import validation.common.runtime_gpu  # noqa: F401  (CPU cap + autocvd, before torch)

import argparse
import json
import os
import time

import numpy as np
import torch

from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm
from compass.schema import ConditionalMenu
from validation.a4_waic.families import family
from validation.a4_waic.scaling import Scaler

MODEL_ROOT = "/export/data/rheinric/compass_validation/a4_waic/models"
ARCH = dict(sde_type="vesde", sigma=25.0, hidden_size=128, depth=4, num_heads=8, mlp_ratio=4)
TRAIN = dict(batch_size=256, max_epochs=400, lr=1e-3, early_stopping_patience=40,
             time_sampling="log_sigma", time_loss_weight="edm")


def model_dir(family_name, model_name, budget, seed):
    return os.path.join(MODEL_ROOT, family_name, model_name, f"n{budget}_s{seed}")


def load(family_name, model_name, budget, seed, device="cpu"):
    out = model_dir(family_name, model_name, budget, seed)
    return SBIm.load(os.path.join(out, "final.pt"), device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=TRAIN["max_epochs"])
    args = parser.parse_args()

    out = model_dir(args.family, args.model, args.budget, args.seed)
    if os.path.exists(os.path.join(out, "final.pt")):
        print(f"exists: {out}")
        return
    os.makedirs(out, exist_ok=True)
    model_def = family(args.family)[args.model]
    scaler = Scaler.for_model(args.family, model_def)
    rng = np.random.default_rng([args.seed, 17, sum(map(ord, args.model))])
    val_rng = np.random.default_rng([args.seed, 9001, sum(map(ord, args.model))])
    train = scaler.forward(model_def.training_data(args.budget, rng))
    val = scaler.forward(model_def.training_data(max(2048, args.budget // 10), val_rng))
    n_theta = model_def.n_global + model_def.n_local
    theta, x = (torch.tensor(a, dtype=torch.float32) for a in (train[:, :n_theta], train[:, n_theta:]))
    theta_val, x_val = (torch.tensor(a, dtype=torch.float32) for a in (val[:, :n_theta], val[:, n_theta:]))

    torch.manual_seed(args.seed)
    net = SBIm(nodes_size=model_def.nodes, device="cuda", **ARCH)
    menu = ConditionalMenu(net.schema, roles={"t": list(range(n_theta)),
                                              "x": list(range(n_theta, model_def.nodes))},
                           patterns=[("p(theta|x)", "x"), ("p(x|theta)", "t")],
                           bernoulli_fraction=1 / 3)
    cfg = dict(TRAIN, max_epochs=args.max_epochs)
    t0 = time.time()
    net.train(theta, x, theta_val=theta_val, x_val=x_val, device="cuda", verbose=False,
              path=out, name="Model", mask_sampler=menu, **cfg)
    seconds = time.time() - t0
    net.save(out, name="final")
    meta = dict(family=args.family, model=args.model, budget=args.budget, seed=args.seed,
                arch=ARCH, train=cfg, seconds=seconds, epochs=len(net.trainer.train_loss),
                scaler=scaler.to_dict(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done {out} in {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
