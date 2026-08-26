"""Training recipe and the model-size ladder for the exponential hierarchy.

Every checkpoint in this directory is trained by exactly this function, so a
comparison across rows of the capacity table isolates the architecture and the
simulation budget and nothing else.

The ladder only moves the four knobs ``ScoreBasedInferenceModel`` exposes and
saves (``hidden_size``, ``depth``, ``num_heads``, ``mlp_ratio``). It does *not*
touch ``time_embedding_size``, which is fixed at 256 inside
``ConditionTransformer``: the adaLN modulation head is a ``Linear(256, 6 * nodes
* hidden)`` per block, so that constant is what puts the floor under the
parameter count. At ``hidden_size = 4`` roughly three quarters of the 27K
parameters are that head. Shrinking it would need a checkpoint-format change,
which is out of scope here -- the point of the ladder is which *representable*
model suffices, not how few weights the problem needs in principle.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

import hierarchy

SDE_KWARGS = {"sde_type": "vesde", "sigma": 8.0}

# Ordered large -> small. h128d6 is the stock COMPASS backbone every other
# experiment in New_Attempt/ uses, kept as the accuracy ceiling.
CONFIGS = {
    "h128d6": {"hidden_size": 128, "depth": 6, "num_heads": 8, "mlp_ratio": 4},
    "h32d3": {"hidden_size": 32, "depth": 3, "num_heads": 4, "mlp_ratio": 2},
    "h16d2": {"hidden_size": 16, "depth": 2, "num_heads": 2, "mlp_ratio": 1},
    "h16d1": {"hidden_size": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 1},
    "h8d1": {"hidden_size": 8, "depth": 1, "num_heads": 2, "mlp_ratio": 1},
    "h4d1": {"hidden_size": 4, "depth": 1, "num_heads": 1, "mlp_ratio": 1},
}

# The recipe is defined by a *gradient-step* budget, not an epoch count. An
# epoch on 10K simulations is twenty batches and an epoch on 200K is four
# hundred, so holding epochs fixed across the data ladder would hand the large
# budgets twenty times the optimization and report the difference as a data
# effect. REFERENCE_SAMPLES is the budget at which the epoch numbers below are
# literal; every other budget is rescaled to the same number of steps.
REFERENCE_SAMPLES = 200_000
MAX_EPOCHS = 400
# Patience is deliberately long. The denoising-score-matching validation loss is
# noisy -- it averages a random diffusion time per sample -- so it plateaus in
# visible steps rather than descending smoothly. A first pass of this sweep used
# patience 40 and stopped every run on plateau noise: the resulting table put a
# 109K-parameter network *ahead* of a 552K one on marginal fidelity, which is
# not a capacity ordering, it is truncation. At 150 the ranking is monotone.
PATIENCE = 150
# Throughput on this backbone saturates near 50K samples/s and is bounded by
# per-step launch overhead, not by batch arithmetic: batch 512 runs 38 steps/s
# (20K samples/s) while batch 2048 runs 22 steps/s (46K samples/s). 1024 buys
# most of that back while keeping the step count per epoch high enough for the
# schedule below to stay meaningful.
BATCH_SIZE = 1024
MIN_BATCHES_PER_EPOCH = 10
LR = 5e-4
VALIDATION_FRACTION = 0.1


def schedule(train_samples, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Epoch count and patience giving ``train_samples`` the reference step budget."""
    factor = REFERENCE_SAMPLES / float(train_samples)
    return (max(5, int(round(max_epochs * factor))),
            max(5, int(round(patience * factor))))


def batch_for(train_samples):
    """The recipe batch size, shrunk only when a rung is too small to fill it."""
    return int(min(BATCH_SIZE, max(64, train_samples // MIN_BATCHES_PER_EPOCH)))


def model_directory(root, config, train_samples):
    return Path(root) / "models" / f"{config}_n{train_samples}"


def parameter_count(config):
    """Parameters of a config, without training anything."""
    from compass import ScoreBasedInferenceModel as SBIm

    model = SBIm(nodes_size=hierarchy.NODES, device="cpu",
                 **SDE_KWARGS, **CONFIGS[config])
    return sum(parameter.numel() for parameter in model.model.parameters())


def load_to(checkpoint, device):
    """``SBIm.load``, but with the network actually on ``device``.

    ``SBIm.load`` constructs the ``ScoreBasedInferenceModel`` without forwarding
    its ``device`` argument, so the module is built on the CPU and
    ``map_location`` only decides where the checkpoint tensors are read before
    being copied into those CPU parameters -- the returned model is on the CPU
    whatever you ask for. Code that calls ``model.sample(...)`` first never
    notices, because the sampler moves the module itself; code that touches
    ``model.model(...)`` directly gets a device mismatch. Move it here so the
    result does not depend on which call happens to come first.
    """
    from compass import ScoreBasedInferenceModel as SBIm

    model = SBIm.load(str(checkpoint), device=device)
    model.model.to(device)
    return model


def train_or_load(directory, config, train_samples, device, seed=7, force=False,
                  verbose=True, max_epochs=MAX_EPOCHS):
    """Load ``directory``'s checkpoint, training it with this recipe if absent."""
    from compass import ScoreBasedInferenceModel as SBIm

    directory = Path(directory)
    checkpoint = directory / "Model_checkpoint.pt"
    if checkpoint.exists() and not force:
        print(f"Loading checkpoint: {checkpoint}")
        return load_to(checkpoint, device), 0.0

    validation_samples = max(2000, int(VALIDATION_FRACTION * train_samples))
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    theta_train, x_train = hierarchy.simulate(train_samples, generator)
    theta_validation, x_validation = hierarchy.simulate(validation_samples, generator)

    model = SBIm(nodes_size=hierarchy.NODES, device=device,
                 **SDE_KWARGS, **CONFIGS[config])
    parameters = sum(p.numel() for p in model.model.parameters())
    batch_size = batch_for(train_samples)
    epochs, patience = schedule(train_samples, max_epochs)
    print(f"Training {config} ({parameters:,} parameters) on "
          f"{train_samples:,} simulations, batch {batch_size}, up to {epochs} "
          f"epochs (patience {patience}) -- "
          f"~{epochs * max(1, train_samples // batch_size):,} gradient steps")
    directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model.train(
        theta=theta_train, x=x_train, theta_val=theta_validation,
        x_val=x_validation, batch_size=batch_size,
        max_epochs=epochs, early_stopping_patience=patience, lr=LR,
        time_sampling="mixture", device=device, verbose=verbose,
        path=str(directory),
    )
    seconds = time.perf_counter() - started
    return load_to(checkpoint, device), seconds


# ---------------------------------------------------------------------------
# Network fidelity, single observation only -- no composition can hide in it
# ---------------------------------------------------------------------------

def wasserstein_1d(samples, grid, weights):
    """W1 between draws and a density tabulated on a grid."""
    samples = np.sort(np.asarray(samples, dtype=np.float64))
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    quantiles = (np.arange(len(samples)) + 0.5) / len(samples)
    positions = np.clip(np.searchsorted(cumulative, quantiles), 0, len(grid) - 1)
    return float(np.abs(samples - grid[positions]).mean())


def posterior_fidelity(model, x_observed, num_samples, timesteps, device, chunk=5):
    """Mean W1 of the single-observation marginals, in units of their own std.

    Chunked over observations: the widest config puts observations x samples
    rows through the transformer at once. Chunking changes nothing statistically
    -- these are independent single-observation posteriors.
    """
    x = torch.as_tensor(np.asarray(x_observed, dtype=np.float32)).reshape(-1, 1)
    torch.manual_seed(1208)
    torch.cuda.manual_seed_all(1208)
    blocks = []
    for start in range(0, x.shape[0], chunk):
        blocks.append(model.sample(
            x=x[start:start + chunk], num_samples=num_samples,
            timesteps=timesteps, method="dpm", order=2,
            corrector_steps_interval=1, corrector_steps=10,
            final_corrector_steps=3, snr=0.2, device=device, verbose=False,
        ).detach().cpu().numpy())
    samples = np.concatenate(blocks, axis=0)

    errors_g, errors_l = [], []
    for index, value in enumerate(np.asarray(x_observed, dtype=np.float64)):
        grid_g, weights_g, grid_l, weights_l = \
            hierarchy.single_observation_reference(value)
        mean_g = float((weights_g * grid_g).sum())
        std_g = float(max((weights_g * grid_g**2).sum() - mean_g**2, 0.0) ** 0.5)
        mean_l = float((weights_l * grid_l).sum())
        std_l = float(max((weights_l * grid_l**2).sum() - mean_l**2, 0.0) ** 0.5)
        errors_g.append(wasserstein_1d(samples[index, :, 0], grid_g, weights_g) / std_g)
        errors_l.append(wasserstein_1d(samples[index, :, 1], grid_l, weights_l) / std_l)
    return float(np.mean(errors_g)), float(np.mean(errors_l))


@torch.no_grad()
def score_fidelity(model, x_observed, lambdas, states, device, seed=99):
    """Relative RMS error of the *single-observation* score, per noise level.

    The N = 1 case of :func:`hierarchy.diffused_score` is the exact score of
    ``p_lam(g_t, l_t | x_j)``, which is precisely what the network is trained to
    output -- so this grades the network with no composition anywhere in the
    loop, and at exactly the noise levels the composed sampler will query.

    Returns ``{lam: relative_rmse}`` plus the budget-weighted mean, where the
    error is normalized by the RMS of the exact score at the same states.
    """
    x_observed = np.asarray(x_observed, dtype=np.float64).reshape(-1)
    mask = hierarchy.CONDITION_MASK.to(device)
    results = {}
    for lam in lambdas:
        errors, magnitudes = [], []
        for index, value in enumerate(x_observed):
            grid = hierarchy.quadrature_grid([value], device=device)
            shared, local, _, _ = hierarchy.sample_diffused(
                [value], lam, states, seed + 1000 * index
            )
            exact_g, exact_l = hierarchy.diffused_score(
                shared, local, [value], lam, grid
            )
            state = torch.zeros(states, hierarchy.NODES, dtype=torch.float32,
                                device=device)
            state[:, hierarchy.GLOBAL_INDEX] = torch.as_tensor(shared, device=device)
            state[:, hierarchy.LOCAL_INDEX] = torch.as_tensor(
                local[:, 0], device=device
            )
            state[:, hierarchy.OBSERVED_INDEX] = float(value)
            t = model.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1)
            predicted = model.output_scale_function(
                t.to(device), model.model(
                    x=state, t=t.to(device),
                    c=mask.unsqueeze(0).repeat(states, 1),
                )
            ).to(torch.float64)
            exact = torch.stack([exact_g, exact_l[:, 0]], dim=1).to(device)
            errors.append(
                (predicted[:, :2] - exact).pow(2).sum().item()
            )
            magnitudes.append(exact.pow(2).sum().item())
        results[float(lam)] = math.sqrt(sum(errors) / max(sum(magnitudes), 1e-30))
    results["mean"] = float(np.mean([
        value for key, value in results.items() if key != "mean"
    ]))
    return results
