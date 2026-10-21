#!/usr/bin/env python
"""Experiment 4 -- transfer the winning GM interventions to inverse kinematics.

Only the interventions that *can* transfer are here. The Gaussian mixture's
best arms (`Cbig`, `Dbig`) carry a compositional penalty whose target is the
exact composition, which inverse kinematics cannot supply; the oracle-free
surrogate measured on GM was worse than doing nothing. So what crosses over is:

``residual parameterization``
    ``s_phi = s_base + backbone/sigma`` with ``s_base`` the diffused prior
    score. Inverse kinematics has an independent Gaussian prior
    (``hierarchical.PRIOR_STD``, zero mean) on the four latent angles, so the
    diffused marginal score of any latent subset of them is exactly
    ``-y / (std^2 + lambda^2)`` -- closed form, no oracle, nothing estimated.
    The two observed coordinates are a nonlinear function of the angles and
    have no closed-form marginal, so the baseline is zero there and the network
    predicts those conditionals in full, exactly as before. That is the honest
    scope of the intervention: it applies to the conditionals the composition
    consumes.

``preconditioning``
    The same two derived changes as on GM -- a log-lambda time coordinate
    rescaled onto the sinusoidal embedder's dynamic range, and EDM input
    scaling -- plus the schedule floor lowered so the network is trained at the
    noise scales ModeTree evaluates it at. The time-embedding defect is if
    anything worse here: with ``sigma = 8`` the band lambda in [0.03, 0.2] is
    t in [1.4e-3, 5.7e-2], and every sinusoidal argument in it is below 0.06.

Arms, matched on everything else (same backbone, simulations, menu, optimizer,
budget):

``Ka``    the stock recipe, retrained in this harness as the control
``Kbpre`` residual parameterization + preconditioning + extended schedule

Run:  python ModeTree/experiments/exp4_ik_transfer.py --arms Ka Kbpre
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IK = ROOT / "inverse_kinematics"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(IK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402
import time      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import hierarchical as H  # noqa: E402
import simulator as sim   # noqa: E402
import train_model        # noqa: E402

from compass.Trainer import Trainer          # noqa: E402
from residual_score import PreconditionedScore, ResidualScore  # noqa: E402

MODELS = IK / "artifacts" / "models"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ik_transfer"

ARMS = {
    # name: (wrapper, eps)
    "Ka": ("plain", 1e-3),
    "Kbpre": ("preconditioned", 1e-4),
    "Kb": ("residual", 1e-3),
}


class GaussianPriorScore(torch.nn.Module):
    """``sigma(t) * grad log p_t(theta_L)`` for the independent Gaussian prior.

    Zero on any coordinate whose prior marginal is not available in closed form
    -- here the two observables, which are a nonlinear function of the angles.
    Returned in the network's raw convention (score times ``sigma``), so the
    wrapper simply adds it to the backbone's output.
    """

    def __init__(self, sde, prior_std, latent_indices, nodes):
        super().__init__()
        self.sde = sde
        variance = torch.zeros(nodes, dtype=torch.float64)
        available = torch.zeros(nodes, dtype=torch.float64)
        for position, index in enumerate(latent_indices):
            variance[index] = float(prior_std[position]) ** 2
            available[index] = 1.0
        self.register_buffer("prior_variance", variance)
        self.register_buffer("available", available)

    def forward(self, x, t, c, return_attn_weights=False, **unused):
        lam = self.sde.lambda_t(t).to(x.device).reshape(-1, 1).to(torch.float64)
        state = x.to(torch.float64)
        variance = self.prior_variance.to(x.device).reshape(1, -1) + lam ** 2
        # latent, prior known, and not conditioned
        active = self.available.to(x.device).reshape(1, -1) * (1.0 - c.to(torch.float64))
        score = -active * state / variance
        scale = self.sde.marginal_prob_std(t).to(x.device)
        result = (score * scale).to(x.dtype)
        return (result, None) if return_attn_weights else result


def attach(sbim, kind, eps):
    if kind == "plain":
        return sbim
    baseline = GaussianPriorScore(sbim.sde, H.PRIOR_STD, H.LATENT_INDICES,
                                  H.NODES).to(
        next(sbim.model.parameters()).device)
    if kind == "residual":
        sbim.model = ResidualScore(sbim.model, baseline)
        return sbim
    # sigma_data: the prior sd for the angles, the simulated spread for the
    # observables, measured rather than guessed.
    theta, x = H.simulate(20_000, sim.generator("train", 1234))
    sigma_data = np.concatenate([
        np.asarray(H.PRIOR_STD, dtype=np.float64),
        np.asarray(x, dtype=np.float64).std(axis=0),
    ])
    lam_min = float(sbim.sde.lambda_t(torch.tensor([eps])))
    lam_max = float(sbim.sde.lambda_t(torch.tensor([1.0])))
    sbim.model = PreconditionedScore(sbim.model, baseline, sbim.sde,
                                     sigma_data, (lam_min, lam_max))
    return sbim


class FloorTrainer(Trainer):
    """``Trainer`` with a configurable diffusion-time floor; nothing else."""

    def __init__(self, sbim, eps=1e-3):
        super().__init__(sbim)
        self.eps_override = float(eps)

    def _sample_timesteps(self, batch_size):
        self.eps = self.eps_override
        return super()._sample_timesteps(batch_size)


def build_arm(name, device="cpu", config="h16d2", arch="legacy"):
    wrapper, eps = ARMS[name]
    network = train_model.build(config, arch, device=device)
    attach(network, wrapper, eps)
    network.trainer = FloorTrainer(network, eps=eps)
    return network


def train_arm(name, directory, device="cpu", samples=200_000, seed=7,
              config="h16d2", arch="legacy", max_epochs=None, force=False):
    directory = Path(directory)
    checkpoint = directory / "Model_checkpoint.pt"
    network = build_arm(name, device=device, config=config, arch=arch)
    if checkpoint.exists() and not force:
        backbone = network.model
        while hasattr(backbone, "backbone"):
            backbone = backbone.backbone
        backbone.load_state_dict(
            torch.load(checkpoint, map_location=device)["model_state_dict"])
        print(f"  loaded {checkpoint}", flush=True)
        return network, 0.0

    torch.manual_seed(seed)
    np.random.seed(seed)
    theta, x = H.simulate(samples, sim.generator("train", seed))
    validation = H.simulate(max(2_000, samples // 10),
                            sim.generator("validation", seed))
    epochs, patience = train_model.schedule(samples)
    if max_epochs is not None:
        epochs = int(max_epochs)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    network.train(
        theta=theta, x=x, theta_val=validation[0], x_val=validation[1],
        batch_size=train_model.BATCH_SIZE, max_epochs=epochs,
        early_stopping_patience=patience, lr=train_model.LR,
        time_sampling=train_model.DEFAULT_TIME_SAMPLING, device=device,
        verbose=False, path=str(directory),
        mask_sampler=train_model.build_mask_sampler(network, "six_weighted"),
    )
    return network, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=("Ka", "Kbpre"))
    parser.add_argument("--simulations", type=int, default=200_000)
    parser.add_argument("--config", default="h16d2")
    parser.add_argument("--arch", default="legacy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    record = {}
    for name in args.arms:
        directory = (MODELS / f"ik{name}_{args.config}_n{args.simulations}"
                     f"_s{args.seed}")
        print(f"\n== IK arm {name}: {ARMS[name]} -> {directory.name} ==",
              flush=True)
        network, seconds = train_arm(
            name, directory, device=args.device, samples=args.simulations,
            seed=args.seed, config=args.config, arch=args.arch,
            max_epochs=args.max_epochs, force=args.force)
        if seconds:
            print(f"  trained in {seconds / 60:.1f} minutes", flush=True)
        record[name] = {"directory": str(directory), "seconds": seconds,
                        "arm": ARMS[name], "seed": args.seed}
        with open(OUT / "arms.json", "w") as handle:
            json.dump(record, handle, indent=2)
    print(f"\nwrote {OUT / 'arms.json'}")


if __name__ == "__main__":
    main()
