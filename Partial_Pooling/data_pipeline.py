"""Deterministic sharded data generation for every benchmark path."""

from pathlib import Path
import torch

from .artifact_names import (
    normalization_path,
    test_data_path,
    training_index_path,
    training_shard_directory,
    training_shard_filename,
    training_tag,
)
from .io_utils import atomic_json, atomic_torch, manifest, reusable
from .normalization import Normalizer
from .paths import BenchmarkPaths
from .rng import derive_seed, generator
from .schema import (
    GLOBAL_NAMES, JOINT_NAMES, LOCAL_NAMES,
    physical_from_joint, transformed_observations,
)
from .simulators.ddm_sde import simulate_ddm
from .simulators.priors import sample_hierarchical_prior


def _sde_split(config, split, size, subjects):
    rng = generator(config.root_seed, "data", split, "sde")
    globals_, locals_ = sample_hierarchical_prior(rng, size, subjects)
    physical = physical_from_joint(globals_[:, None, :], locals_)
    raw = simulate_ddm(
        physical, rng, config.trials, dt=config.dt,
        max_decision_time=config.max_decision_time,
    )
    return {
        "globals": torch.from_numpy(globals_).float(),
        "locals": torch.from_numpy(locals_).float(),
        "physical": torch.from_numpy(physical).float(),
        "observations_raw": torch.from_numpy(raw).float(),
        "observations": torch.from_numpy(transformed_observations(raw)).float(),
    }


def _slice(value, start, stop):
    if isinstance(value, dict):
        return {key: _slice(item, start, stop) for key, item in value.items()}
    # ``clone`` is required, not cosmetic: ``torch.save`` serializes a view's
    # entire underlying storage, so writing slices directly makes every shard
    # as large as the whole split and the split's disk cost quadratic in
    # ``train_size`` (100,000 samples -> 49 shards of 78 MB = 2.1 GB;
    # 400,000 samples would need ~61 GB). Detaching each slice into its own
    # contiguous storage keeps a shard at its own size.
    return value[start:stop].clone()


def _write_shards(directory, split, payload, config, force):
    directory.mkdir(parents=True, exist_ok=True)
    size = next(value.shape[0] for value in _leaves(payload))
    paths = []
    for start in range(0, size, config.shard_size):
        stop = min(start + config.shard_size, size)
        path = directory / training_shard_filename(
            config, split, start, stop,
        )
        if not reusable(
            path, config.compatible_data_signatures, force,
            signature_key="data_signature",
        ):
            atomic_torch(path, {
                "config_signature": config.data_signature,
                "data_signature": config.data_signature,
                "split": split, "start": start, "stop": stop,
                "seed": derive_seed(config.root_seed, "data", split),
                "payload": _slice(payload, start, stop),
            })
        paths.append(path.name)
    return paths


def _leaves(payload):
    for value in payload.values():
        if isinstance(value, dict):
            yield from _leaves(value)
        else:
            yield value


def _normalizers(training):
    """Per-feature constants, fitted on the training split.

    Subjects are exchangeable draws from the same population, so a split with
    several subjects per row pools them rather than fitting on the first: the
    estimator is the same one, on more samples from the same distribution.
    """
    sde = training["sde"]
    subjects = sde["locals"].shape[1]
    locals_ = sde["locals"].reshape(-1, sde["locals"].shape[-1])
    globals_ = sde["globals"].repeat_interleave(subjects, dim=0)
    observations = sde["observations"].reshape(-1, sde["observations"].shape[-1])
    return {
        "sde_joint_theta": Normalizer.fit(torch.cat((globals_, locals_), 1)).state_dict(),
        "sde_observations": Normalizer.fit(observations).state_dict(),
    }


def generate_training_data(config, paths=None, force=False, cpu_limit=None):
    paths = (paths or BenchmarkPaths.default()).ensure()
    index_path = training_index_path(config, paths)
    normalizer_path = normalization_path(config, paths)
    if reusable(
        index_path, config.compatible_data_signatures, force,
        signature_key="data_signature",
    ) and reusable(
        normalizer_path, config.compatible_data_signatures, force,
        signature_key="data_signature",
    ):
        return index_path
    if config.train_subjects != 1 and not index_path.exists():
        print(
            f"Generating a {config.train_subjects}-subject split "
            f"({training_tag(config)}); normalization and test data are reused "
            "from the one-subject split."
        )
    splits = {}
    training_payload = None
    for split, size in (("train", config.train_size), ("validation", config.validation_size)):
        payload = {
            "sde": _sde_split(config, split, size, subjects=config.train_subjects),
        }
        if split == "train":
            training_payload = payload
        splits[split] = _write_shards(
            training_shard_directory(config, paths),
            split, payload, config, force,
        )
    if config.train_subjects != 1:
        # A grouped split shares the base split's constants rather than fitting
        # its own; see artifact_names.base_training_tag. Refitting here would
        # overwrite the file every published arm was trained under.
        if not normalizer_path.exists():
            raise RuntimeError(
                f"A grouped split reuses {normalizer_path.name}, which does not "
                "exist. Generate the one-subject split for this preset first."
            )
        normalizer_states = torch.load(
            normalizer_path, map_location="cpu", weights_only=False
        )["normalizers"]
    else:
        normalizer_states = _normalizers(training_payload)
        atomic_torch(normalizer_path, {
            "config_signature": config.data_signature,
            "data_signature": config.data_signature,
            "fitted_split": "train",
            "training_method": training_tag(config),
            "training_samples": config.train_size,
            "validation_samples": config.validation_size,
            "normalizers": normalizer_states,
        })
    normalization_json = {
        key: {
            "mean": state["mean"].tolist(), "scale": state["scale"].tolist(),
            "degenerate": state["degenerate"].tolist(),
            "fitted_split": state["fitted_split"],
            "degenerate_policy": state["degenerate_policy"],
        }
        for key, state in normalizer_states.items()
    }
    atomic_json(index_path, {
        **manifest(
            config, "training_data", cpu_limit or {},
            signature=config.data_signature, signature_kind="data",
            splits=splits,
        ),
        "device": "cpu",
        "training_method": training_tag(config),
        "training_samples": config.train_size,
        "validation_samples": config.validation_size,
        "normalization": normalizer_path.name,
        "normalization_statistics": normalization_json,
        "parameter_order": {
            "global": GLOBAL_NAMES, "local": LOCAL_NAMES,
            "joint": JOINT_NAMES,
        },
        "tensor_shapes": {
            "sde_globals": list(training_payload["sde"]["globals"].shape),
            "sde_locals": list(training_payload["sde"]["locals"].shape),
            "sde_observations": list(training_payload["sde"]["observations"].shape),
        },
        "raw_sde_shape": ["instances", "subjects", "trials", 3],
        "observation_order": "trial-major(choice, log_rt, censored)",
    })
    return index_path


def generate_test_data(config, paths=None, force=False, cpu_limit=None):
    paths = (paths or BenchmarkPaths.default()).ensure()
    output = test_data_path(config, paths)
    if reusable(
        output, config.compatible_data_signatures, force,
        signature_key="data_signature",
    ):
        return output
    payload = {
        "sde": _sde_split(config, "test", config.test_datasets, config.subjects),
    }
    dataset_ids = torch.arange(config.test_datasets)
    subject_ids = torch.arange(config.subjects).expand(config.test_datasets, -1)
    atomic_torch(output, {
        "config_signature": config.data_signature,
        "data_signature": config.data_signature,
        "manifest": manifest(
            config, "inference_data", cpu_limit or {},
            signature=config.data_signature, signature_kind="data",
        ),
        "dataset_ids": dataset_ids, "subject_ids": subject_ids,
        "payload": payload,
    })
    return output


def load_split(config, split, paths=None):
    paths = (paths or BenchmarkPaths.default()).ensure()
    index = training_index_path(config, paths)
    reusable(
        index, config.compatible_data_signatures,
        signature_key="data_signature",
    )
    meta = __import__("json").loads(index.read_text())
    directory = training_shard_directory(config, paths)
    shards = []
    for name in meta["splits"][split]:
        path = directory / name
        reusable(
            path, config.compatible_data_signatures,
            signature_key="data_signature",
        )
        shards.append(torch.load(path, map_location="cpu")["payload"])
    return _concatenate(shards)


def _concatenate(payloads):
    first = payloads[0]
    return {
        key: _concatenate([payload[key] for payload in payloads])
        if isinstance(value, dict) else torch.cat([payload[key] for payload in payloads])
        for key, value in first.items()
    }
