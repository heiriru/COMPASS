"""Training recipe for the two bimodal benchmarks, and the fidelity grade.

The target is deliberately easy to *represent* and awkward to *sample*: the
single-observation posterior a network has to learn here is a two-component
Gaussian mixture in ``(g, l)`` whose component means move linearly with ``y``.
Nothing about that needs a large model, so the default backbone is the small
``h32d3`` rung of the capacity ladder in
``../New_Attempt/exponential_score_comparison/train.py`` rather than the stock
128-wide, 6-deep one. The interesting difficulty is entirely downstream, in how
``n`` such ambiguous rows are composed.

Because the exact row score is available in closed form (``row_scores.py``),
network quality is graded directly rather than inferred: :func:`score_fidelity`
reports the relative RMS error of the learned score against the exact one, at
the noise levels the composed sampler actually queries, with no composition
anywhere in the loop. That number is what licenses reading a learned-score
failure downstream as a *composition* failure.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

SDE_KWARGS = {"sde_type": "vesde", "sigma": 8.0}

CONFIGS = {
    "h128d6": {"hidden_size": 128, "depth": 6, "num_heads": 8, "mlp_ratio": 4},
    "h32d3": {"hidden_size": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 2},
    "h16d2": {"hidden_size": 16, "depth": 2, "num_heads": 2, "mlp_ratio": 1},
}
DEFAULT_CONFIG = "h32d3"

TRAIN_SAMPLES = 200_000
VALIDATION_FRACTION = 0.1
BATCH_SIZE = 1024
MAX_EPOCHS = 400
# The denoising-score-matching validation loss averages a random diffusion time
# per sample, so it plateaus in visible steps rather than descending smoothly;
# short patience stops on plateau noise rather than on convergence.
PATIENCE = 150
LR = 5e-4
# How diffusion times are drawn during training.
#
# "log_sigma" is the default because it was measured to be much better here and
# costs nothing: COMPASS's own "mixture" default spends half its budget uniformly
# in t, which puts almost nothing near small lambda -- exactly where the sampler
# needs the score and where essentially all of the error was. On the 552K network
# the mean relative row-score error drops 0.166 -> 0.054, and at lambda = 0.05
# from 0.440 to 0.088, which beats the 5.2M network trained the "mixture" way.
# See RESULTS.md, "two free recipe changes".
#
# The main tables in RESULTS.md predate this and were produced with "mixture";
# pass time_sampling="mixture" to reproduce them exactly.
DEFAULT_TIME_SAMPLING = "log_sigma"
# The recipe in use when the shipped checkpoints under models/ were trained. Its
# directories are untagged, from before this knob existed.
LEGACY_TIME_SAMPLING = "mixture"


def model_directory(root, module, config, time_sampling=DEFAULT_TIME_SAMPLING):
    """Checkpoint directory, tagged by the diffusion-time recipe.

    The tag is what stops two recipes from overwriting each other's weights and
    being compared as if they were the same run. It is *not* keyed to the current
    default: flipping the default must never silently re-point an existing
    directory at a checkpoint trained under the other recipe, which is exactly
    what a "tag only when non-default" rule would have done here.

    The one untagged spelling is ``LEGACY_TIME_SAMPLING``, whose directories were
    written before this knob existed and are kept where they are so the shipped
    results stay reproducible.
    """
    name = f"{module.NAME}_{config}"
    if time_sampling != LEGACY_TIME_SAMPLING:
        name = f"{name}_{time_sampling}"
    return Path(root) / "models" / name


def train_or_load(root, module, config=DEFAULT_CONFIG, device="cpu", seed=7,
                  force=False, verbose=True, train_samples=TRAIN_SAMPLES,
                  time_sampling=DEFAULT_TIME_SAMPLING):
    """Load this benchmark's checkpoint, training it with the recipe if absent."""
    from compass import ScoreBasedInferenceModel as SBIm

    directory = model_directory(root, module, config, time_sampling)
    checkpoint = directory / "Model_checkpoint.pt"
    if checkpoint.exists() and not force:
        print(f"Loading checkpoint: {checkpoint}")
        model = SBIm.load(str(checkpoint), device=device)
        # SBIm.load builds the module on the CPU whatever ``device`` says, and
        # only the sampler moves it later; code that touches ``model.model``
        # directly (the fidelity grade below) would otherwise mismatch devices.
        model.model.to(device)
        return model, 0.0

    validation_samples = max(2_000, int(VALIDATION_FRACTION * train_samples))
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    theta_train, x_train = module.simulate(train_samples, generator)
    theta_validation, x_validation = module.simulate(validation_samples, generator)

    model = SBIm(nodes_size=module.NODES, device=device,
                 **SDE_KWARGS, **CONFIGS[config])
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"Training {module.NAME}/{config} ({parameters:,} parameters) on "
          f"{train_samples:,} simulations, time_sampling={time_sampling}")
    directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model.train(
        theta=theta_train, x=x_train, theta_val=theta_validation,
        x_val=x_validation, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
        early_stopping_patience=PATIENCE, lr=LR, time_sampling=time_sampling,
        device=device, verbose=verbose, path=str(directory),
    )
    seconds = time.perf_counter() - started
    model = SBIm.load(str(checkpoint), device=device)
    model.model.to(device)
    return model, seconds


def parameter_count(module, config=DEFAULT_CONFIG):
    """Parameters of a config, without training anything."""
    from compass import ScoreBasedInferenceModel as SBIm

    model = SBIm(nodes_size=module.NODES, device="cpu",
                 **SDE_KWARGS, **CONFIGS[config])
    return sum(parameter.numel() for parameter in model.model.parameters())


@torch.no_grad()
def score_fidelity(model, module, values, lambdas, states=512, device="cpu",
                   seed=99):
    """Relative RMS error of the learned row score against the exact one.

    The exact row score is what the network is trained to output, so this grades
    the network with no composition, no sampler and no KDE in the loop, at
    exactly the noise levels the composed sampler will query. Returns
    ``{lam: relative_rmse}`` plus their mean.
    """
    import mixture_reference as reference

    mask = module.CONDITION_MASK.to(device)
    results = {}
    for lam in lambdas:
        errors, magnitudes = [], []
        for index, value in enumerate(np.asarray(values, dtype=np.float64)):
            shared, local, _, _ = reference.sample_diffused(
                module, [value], lam, states, seed + 1000 * index
            )
            exact_g, exact_l = reference.diffused_score(
                module, shared, local, [value], lam
            )
            state = torch.zeros(states, module.NODES, dtype=torch.float32,
                                device=device)
            state[:, module.GLOBAL_INDEX] = torch.as_tensor(
                shared, dtype=torch.float32, device=device
            )
            state[:, module.LOCAL_INDEX] = torch.as_tensor(
                local[:, 0], dtype=torch.float32, device=device
            )
            state[:, module.OBSERVED_INDEX] = float(value)
            t = model.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1)
            predicted = model.output_scale_function(
                t.to(device),
                model.model(x=state, t=t.to(device),
                            c=mask.unsqueeze(0).repeat(states, 1)),
            ).to(torch.float64)
            exact = torch.stack([
                torch.as_tensor(exact_g, device=device),
                torch.as_tensor(exact_l[:, 0], device=device),
            ], dim=1)
            errors.append((predicted[:, :2] - exact).pow(2).sum().item())
            magnitudes.append(exact.pow(2).sum().item())
        results[float(lam)] = float(
            np.sqrt(sum(errors) / max(sum(magnitudes), 1e-30))
        )
    results["mean"] = float(np.mean([
        value for key, value in results.items() if key != "mean"
    ]))
    return results
