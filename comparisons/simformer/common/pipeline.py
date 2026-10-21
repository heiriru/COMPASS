"""Stages shared by the experiments: simulate, train, sample, score.

Each experiment's ``run.py`` is a thin wrapper over these so that "the same
training" really is the same training across e01, e03, e04, e05, e06 and e07 --
they differ in what they *measure*, not in what they fit. A checkpoint trained
by e01 is reused by the others via ``checkpoint_path``.

Nothing here is executed at import time.
"""

import json
import time
from pathlib import Path

import torch

from . import paths, tasks as task_module
from .model import build_model, sampling_kwargs, training_kwargs, describe_arm


# --------------------------------------------------------------- simulations
def simulation_path(task_name, budget, split, seed):
    return paths.data_path(task_name, f"{split}_n{budget}_seed{seed}.pt")


def generate_simulations(task_name, budget, seed=0, validation_fraction=0.1,
                         overwrite=False):
    """Draw ``budget`` joint samples, split into train/validation, and save.

    The budget is the *simulation* budget in the paper's sense: the number of
    simulator calls, training and validation together. Splitting inside the
    budget rather than on top of it is what keeps 10^3 / 10^4 / 10^5 comparable
    to their x-axis.
    """
    task = task_module.get_task(task_name)
    train_file = simulation_path(task_name, budget, "train", seed)
    if train_file.exists() and not overwrite:
        return train_file
    generator = torch.Generator().manual_seed(seed)
    theta, x = task.sample_joint(budget, generator=generator)
    n_val = max(1, int(round(validation_fraction * budget)))
    payload = {
        "theta_train": theta[:-n_val], "x_train": x[:-n_val],
        "theta_val": theta[-n_val:], "x_val": x[-n_val:],
        "task": task_name, "budget": budget, "seed": seed,
        "validation_fraction": validation_fraction,
    }
    paths.ensure(train_file.parent)
    torch.save(payload, train_file)
    paths.write_manifest(train_file.with_suffix(".manifest.json"), {
        "stage": "simulate", "task": task_name, "budget": budget, "seed": seed,
        "n_train": int(len(payload["theta_train"])),
        "n_val": int(len(payload["theta_val"])),
        "sha256": paths.file_digest(train_file),
    })
    return train_file


def observation_path(task_name, seed):
    return paths.data_path(task_name, f"observations_seed{seed}.pt")


def generate_observations(task_name, n_observations=10, seed=1000,
                          overwrite=False):
    """The fixed set of ground-truth observations every arm is scored on.

    sbibm ships ten per task; we regenerate ours from the task definition so
    that Tree and HMM (which sbibm does not have) follow the same protocol. The
    seed is deliberately far from the training seeds, and the file is written
    once and then treated as read-only -- an arm scored on a different
    observation set is not comparable to one scored on this one.
    """
    task = task_module.get_task(task_name)
    path = observation_path(task_name, seed)
    if path.exists() and not overwrite:
        return path
    generator = torch.Generator().manual_seed(seed)
    theta, x = task.sample_joint(n_observations, generator=generator)
    paths.ensure(path.parent)
    torch.save({"theta_true": theta, "x_observed": x, "task": task_name,
                "seed": seed}, path)
    paths.write_manifest(path.with_suffix(".manifest.json"), {
        "stage": "observations", "task": task_name,
        "n_observations": n_observations, "seed": seed,
        "sha256": paths.file_digest(path),
    })
    return path


# -------------------------------------------------------------------- training
def checkpoint_dir(task_name, arm, budget, sde, seed, arch):
    return paths.artifacts(
        "e01_benchmark_posterior", "models",
        f"{task_name}__{arm}__{arch}__{sde}__n{budget}__seed{seed}",
    )


def train_arm(task_name, arm, budget, sde="vesde", seed=0, arch="tokenized",
              device="cpu", verbose=True, overwrite=False):
    """Fit one arm. Returns the checkpoint directory.

    Deliberately not called anywhere at import time, and never from a scoring
    stage: training is the expensive, GPU-bound step and must be launched
    explicitly.
    """
    task = task_module.get_task(task_name)
    out = checkpoint_dir(task_name, arm, budget, sde, seed, arch)
    if (out / "Model_checkpoint.pt").exists() and not overwrite:
        return out
    data_file = simulation_path(task_name, budget, "train", seed)
    if not data_file.exists():
        raise FileNotFoundError(
            f"{data_file} is missing. Run the prepare stage first: "
            f"python comparisons/simformer/prepare_data.py --task {task_name}"
        )
    payload = torch.load(data_file, weights_only=False)

    torch.manual_seed(seed)
    model = build_model(task, arm=arm, sde=sde, arch=arch, device=device)
    generator = torch.Generator().manual_seed(seed + 7)
    kwargs = training_kwargs(task, model, arm=arm, generator=generator)

    paths.ensure(out)
    started = time.time()
    model.train(
        payload["theta_train"], payload["x_train"],
        theta_val=payload["theta_val"], x_val=payload["x_val"],
        device=device, path=str(out), name="Model", verbose=verbose, **kwargs,
    )
    elapsed = time.time() - started

    record = describe_arm(task, arm, sde=sde)
    record.update({
        "stage": "train", "budget": budget, "seed": seed, "arch": arch,
        "device": device, "seconds": elapsed,
        "train_rows": int(len(payload["theta_train"])),
        "val_rows": int(len(payload["theta_val"])),
        "training_kwargs": {k: v for k, v in kwargs.items()
                            if isinstance(v, (int, float, str, type(None)))},
        "mask_sampler": type(kwargs["mask_sampler"]).__name__,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "data_sha256": paths.file_digest(data_file),
    })
    paths.write_manifest(out / "manifest.json", record)
    return out


def load_arm(task_name, arm, budget, sde="vesde", seed=0, arch="tokenized",
             device="cpu"):
    from compass import ScoreBasedInferenceModel as SBIm

    out = checkpoint_dir(task_name, arm, budget, sde, seed, arch)
    checkpoint = out / "Model_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Train this arm first."
        )
    return SBIm.load(str(checkpoint), device=device)


# -------------------------------------------------------------------- sampling
def sample_conditional(model, observation, condition_mask, num_samples=10_000,
                       arm="paper_matched", timesteps=None, device="cpu",
                       verbose=False):
    """Draw from an arbitrary conditional of the joint.

    ``observation`` is a full node vector; only the entries under
    ``condition_mask == 1`` are read. This is the single call that e01 (posterior
    mask), e03 (arbitrary masks) and e06 (likelihood mask) all go through.
    """
    kwargs = sampling_kwargs(arm=arm, timesteps=timesteps)
    condition_mask = torch.as_tensor(condition_mask, dtype=torch.float)
    observed = torch.as_tensor(observation, dtype=torch.float)
    data = observed[condition_mask == 1].reshape(1, -1)
    return model.sample(
        x=data, condition_mask=condition_mask.reshape(1, -1),
        num_samples=num_samples, device=device, verbose=verbose, **kwargs,
    )


# --------------------------------------------------------------------- scoring
def write_scores(experiment, name, rows, extra=None):
    """Append a scored table plus its manifest under the experiment's artifacts."""
    out = paths.artifacts(experiment, f"{name}.json")
    paths.ensure(out.parent)
    with open(out, "w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    paths.write_manifest(
        paths.artifacts(experiment, f"{name}.manifest.json"),
        {"stage": "score", "experiment": experiment, "rows": len(rows),
         **(extra or {})},
    )
    return out


def require(path, hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. {hint}")
    return path
