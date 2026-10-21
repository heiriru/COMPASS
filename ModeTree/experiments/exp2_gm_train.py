#!/usr/bin/env python
"""Experiment 2/3 -- four matched GM arms, graded on the n ladder by integration.

Arms, each a single change on the one before it, all on the same backbone, the
same 200k simulations, the same optimizer, the same mask menu and the same epoch
budget, so a difference is the intervention's:

``A``  the stock recipe, retrained here. A control for the harness, not a new
       result: it must land where the frozen checkpoint lands, or every
       comparison below is measuring the harness instead of the arm.
``B``  ``s_phi = s_base + backbone / sigma`` -- the residual parameterization.
``C``  ``B`` plus the compositional penalty on the summed row error.
``D``  ``C`` plus the derived time/input preconditioning and a schedule that
       reaches the noise scales the composition is actually evaluated at.

Grading uses no sampler. The global dimension is one, so the composed field is
integrated directly along the exact conditional slice (the same slice and the
same closed-form references as ``exp1_gm_residual_error.py``), which makes every
number reference-noise-free.

Run:  python ModeTree/experiments/exp2_gm_train.py --arms A B C D
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "Gaussian_mixtures_compositional_score_modelling"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCHMARK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402
import time      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import datasets    # noqa: E402
import model as M  # noqa: E402
import train       # noqa: E402

import gmresidual  # noqa: E402
from exp1_gm_residual_error import (  # noqa: E402
    ARTIFACTS, SEED, TRAINED_LAMBDA_FLOOR, diffused_global_posterior,
    exact_tall_global_score, integrate, mixture_prior, roots_of, slice_state,
    sources, w1,
)
from modetree.compass_adapter import CompassRowSource  # noqa: E402

MODELS = BENCHMARK / "artifacts" / "models"
OUT = ARTIFACTS.parent / "gm_arms"
N_LADDER = (1, 2, 4, 8, 16, 32, 64, 128)
GRADE_LAMBDAS = (0.5, 0.2, 0.1, 0.05, 0.03167, 0.02, 0.01)

ARMS = {
    # name: (wrapper, comp_weight, comp_target, eps)
    "A": ("plain", 0.0, "oracle", 1e-3),
    "B": ("residual", 0.0, "oracle", 1e-3),
    "C": ("residual", 1.0, "oracle", 1e-3),
    "D": ("preconditioned", 1.0, "oracle", 1e-4),
    # the transferable form of C's penalty: no closed form anywhere in it
    "Csim": ("residual", 1.0, "sim", 1e-3),
    # D changes two things at once. These split them: Deps keeps the stock
    # schedule and adds only the preconditioning, Dpre keeps the stock
    # conditioning and only lowers the schedule floor.
    "Deps": ("preconditioned", 1.0, "oracle", 1e-3),
    "Dpre": ("residual", 1.0, "oracle", 1e-4),
    # B plus D's preconditioning and schedule, *without* the compositional
    # penalty: B has the best-calibrated posterior and D the most accurate
    # field, and nothing so far combines the two.
    "Bpre": ("preconditioned", 0.0, "oracle", 1e-4),
    # Measured on arms B and C, `L_comp` at weight 1 is 0.3% of the DSM loss,
    # so arm C barely applied the penalty at all. This weights the two terms
    # comparably, which is what Part III actually asks to be tested.
    "Cbig": ("residual", 200.0, "oracle", 1e-3),
    # everything that helped, together: the residual parameterization, the
    # compositional penalty at a weight that actually applies it, the derived
    # preconditioning and the extended schedule.
    "Dbig": ("preconditioned", 200.0, "oracle", 1e-4),
    # The penalty that works needs an oracle target, which inverse kinematics
    # does not have. This is the transferable surrogate at a weight between the
    # failed 1 and Cbig's 200: the sim target's loss has an irreducible O(1)
    # noise floor whose gradient is zero, so a weight as large as 200 would buy
    # only gradient variance. Whether any weight recovers the effect is the
    # question that decides whether Cbig/Dbig can reach IK at all.
    "Csimbig": ("residual", 20.0, "sim", 1e-3),
}


def build_arm(name, mixture, device="cpu", config="h32d3wide", seed=7):
    wrapper, comp_weight, comp_target, eps = ARMS[name]
    network = train.build(config, "tokenized", device=device)
    gmresidual.attach(network, mixture, wrapper, eps=eps)
    network.trainer = gmresidual.CompositionalTrainer(
        network, mixture, comp_weight=comp_weight, comp_target=comp_target,
        eps=eps, seed=seed)
    return network


def train_arm(name, mixture, directory, device="cpu", samples=200_000, seed=7,
              force=False, max_epochs=None):
    directory = Path(directory)
    checkpoint = directory / "Model_checkpoint.pt"
    network = build_arm(name, mixture, device=device, seed=seed)
    if checkpoint.exists() and not force:
        # the saved file is an ordinary backbone checkpoint; re-wrap it
        backbone = network.model
        while hasattr(backbone, "backbone"):
            backbone = backbone.backbone
        state = torch.load(checkpoint, map_location=device)
        backbone.load_state_dict(state["model_state_dict"])
        print(f"  loaded {checkpoint}")
        return network, 0.0

    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = mixture.simulate(samples, np.random.default_rng([seed, 1]))
    validation = mixture.simulate(max(2_000, samples // 10),
                                  np.random.default_rng([seed, 2]))

    def split(values):
        return (torch.as_tensor(values[:, M.LATENT_INDICES], dtype=torch.float32),
                torch.as_tensor(values[:, M.OBSERVED_INDICES], dtype=torch.float32))

    theta, x = split(rows)
    theta_validation, x_validation = split(validation)
    epochs, patience = train.schedule(samples)
    if max_epochs is not None:
        epochs = int(max_epochs)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    network.train(
        theta=theta, x=x, theta_val=theta_validation, x_val=x_validation,
        batch_size=train.BATCH_SIZE, max_epochs=epochs,
        early_stopping_patience=patience, lr=train.LR,
        time_sampling=train.DEFAULT_TIME_SAMPLING, device=device, verbose=False,
        path=str(directory),
        mask_sampler=train.build_mask_sampler(network, "six_weighted"),
    )
    return network, time.perf_counter() - started


def grade(network, exact, mixture, prior, grid, lambdas, observations):
    rows = []
    for n in observations:
        rng = np.random.default_rng(
            [SEED, datasets.REGIMES.index("asymmetric"), n, 0])
        data = datasets.make_dataset("asymmetric", n, rng, mixture=mixture,
                                     branch_mass=0.75, spread=1.0)
        x = np.asarray(data["x"], dtype=np.float64).ravel()
        column = x.reshape(-1, 1)
        learned_source = CompassRowSource(
            network, column, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES,
            prior)
        exact_source = CompassRowSource(
            exact, column, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES,
            prior)
        state = slice_state(mixture, grid, x)
        clean = mixture.global_density(grid, x)
        reference_modes = np.sort(mixture.branch_means(x))
        reference_weights = mixture.branch_weights(x)
        posterior_sd = mixture.global_moments(x)[1]
        within_sd = float(np.sqrt(mixture.posterior_variance(n)))
        for lam in lambdas:
            s_learned = learned_source.row_scores(state, lam).numpy()[..., 0]
            s_exact = exact_source.row_scores(state, lam).numpy()[..., 0]
            s_prior = mixture.diffused_prior_score(grid, lam ** 2)
            composed = (1 - n) * s_prior + s_learned.sum(axis=1)
            composed_exact = (1 - n) * s_prior + s_exact.sum(axis=1)
            density = integrate(grid, composed)
            modes = roots_of(grid, composed)
            split = 0.5 * (reference_modes[0] + reference_modes[1])
            mass = float(np.trapezoid(density[grid > split], grid[grid > split]))
            mean = float(np.trapezoid(grid * density, grid))
            variance = float(np.trapezoid((grid - mean) ** 2 * density, grid))
            row = {
                "n": int(n), "lambda": float(lam),
                "extrapolated": bool(lam < TRAINED_LAMBDA_FLOOR),
                "w1": w1(grid, density, clean),
                "w1_over_sd": w1(grid, density, clean) / posterior_sd,
                "rms_network_error": float(np.sqrt(
                    ((composed - composed_exact) ** 2).mean())),
                "relative_network_error": float(np.sqrt(
                    ((composed - composed_exact) ** 2).mean()
                    / max((composed_exact ** 2).mean(), 1e-300))),
                "modes": modes, "modes_reference": reference_modes.tolist(),
                "weight_tv": 0.5 * float(
                    abs(mass - reference_weights[1])
                    + abs((1 - mass) - reference_weights[0])),
                "width_ratio": float(np.sqrt(max(variance, 0.0)) / posterior_sd),
                "posterior_sd": posterior_sd, "within_sd": within_sd,
            }
            tolerance = 0.5 * within_sd + 0.25
            if modes:
                distances = [np.min(np.abs(np.array(modes) - m))
                             for m in reference_modes]
                row["mode_error"] = float(np.max(distances))
                row["mode_error_over_sd"] = float(np.max(distances) / within_sd)
                row["mode_recall"] = float(np.mean(
                    [d < tolerance for d in distances]))
                row["false_modes"] = int(max(0, len(modes) - sum(
                    1 for d in distances if d < tolerance)))
            else:
                row.update({"mode_error": None, "mode_error_over_sd": None,
                            "mode_recall": 0.0, "false_modes": 0})
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=("A", "B", "C", "D"))
    parser.add_argument("--simulations", type=int, default=200_000)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--points", type=int, default=801)
    parser.add_argument("--observations", type=int, nargs="+", default=N_LADDER)
    parser.add_argument("--lambdas", type=float, nargs="+", default=GRADE_LAMBDAS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    mixture = M.DEFAULT
    prior = mixture_prior(mixture)
    _, exact = sources(mixture)
    grid = np.linspace(-6.0, 6.0, args.points)

    # merge rather than replace: arms are often run in separate invocations,
    # and a fresh dict here would drop every arm not named on this command line.
    results = OUT / f"arm_rows{args.tag}.json"
    everything = json.load(open(results)) if results.exists() else {}
    for name in args.arms:
        directory = (MODELS / f"arm{name}_h32d3wide_n{args.simulations}"
                     f"{args.tag}")
        print(f"\n== arm {name}: {ARMS[name]} ==", flush=True)
        network, seconds = train_arm(
            name, mixture, directory, device=args.device,
            samples=args.simulations, max_epochs=args.max_epochs,
            seed=args.seed, force=args.force)
        if seconds:
            print(f"  trained in {seconds / 60:.1f} minutes", flush=True)
        rows = grade(network, exact, mixture, prior, grid, args.lambdas,
                     args.observations)
        for row in rows:
            row["arm"] = name
            row["train_seconds"] = seconds
        everything[name] = rows
        for lam in args.lambdas:
            selected = [r for r in rows if r["lambda"] == lam]
            flag = " (extrapolated)" if selected[0]["extrapolated"] else ""
            print(f"  lam={lam:7.5g}{flag}", flush=True)
            for row in selected:
                print(f"    n={row['n']:4d} W1/sd={row['w1_over_sd']:8.4f} "
                      f"rel_err={row['relative_network_error']:7.3f} "
                      f"recall={row['mode_recall']:.2f} "
                      f"mode_err={row['mode_error']:.3f} "
                      f"TV={row['weight_tv']:.3f} "
                      f"width={row['width_ratio']:.3f}"
                      if row["mode_error"] is not None else
                      f"    n={row['n']:4d} W1/sd={row['w1_over_sd']:8.4f} "
                      f"no modes", flush=True)
        with open(OUT / f"arm_rows{args.tag}.json", "w") as handle:
            json.dump(everything, handle, indent=1)
    print(f"\nwrote {OUT / f'arm_rows{args.tag}.json'}")


if __name__ == "__main__":
    main()
