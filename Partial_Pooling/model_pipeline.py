"""Native COMPASS model construction, training, and checkpoint reuse."""

from dataclasses import dataclass
import json
import torch

from compass import ScoreBasedInferenceModel

from .artifact_names import (
    checkpoint_directory,
    checkpoint_manifest_path,
    checkpoint_path as named_checkpoint_path,
    checkpoint_tag,
    normalization_path,
    training_tag,
)
from .data_pipeline import load_split
from .io_utils import atomic_json
from .normalization import Normalizer
from .paths import BenchmarkPaths


@dataclass(frozen=True)
class ModelSpec:
    name: str
    theta_key: str
    x_key: str
    theta_dim: int
    x_dim: int


def model_specs(config):
    obs_dim = 3 * config.trials
    return {
        "sde_joint": ModelSpec("sde_joint", "sde_joint_theta", "sde_observations", 10, obs_dim),
    }


def _raw_pair(name, payload):
    if name == "sde_joint":
        source = payload["sde"]
        return torch.cat((source["globals"], source["locals"][:, 0]), 1), source["observations"][:, 0]
    raise KeyError(name)


def load_normalizers(config, paths=None):
    paths = (paths or BenchmarkPaths.default()).ensure()
    state = torch.load(normalization_path(config, paths), map_location="cpu")
    signature = state.get("data_signature", state.get("config_signature"))
    if signature not in config.compatible_data_signatures:
        raise RuntimeError("Normalization configuration mismatch.")
    return {key: Normalizer.from_state_dict(value) for key, value in state["normalizers"].items()}


def checkpoint_path(config, name, paths=None):
    paths = (paths or BenchmarkPaths.default()).ensure()
    return named_checkpoint_path(config, name, paths)


def train_models(config, names=None, paths=None, force=False, device="cpu"):
    paths = (paths or BenchmarkPaths.default()).ensure()
    specs = model_specs(config)
    if names is None or "all" in names:
        names = ["sde_joint"]
    unknown = set(names) - set(specs)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    train = load_split(config, "train", paths)
    validation = load_split(config, "validation", paths)
    normalizers = load_normalizers(config, paths)
    outputs = []
    for name in names:
        spec = specs[name]
        directory = checkpoint_directory(config, name, paths)
        checkpoint = checkpoint_path(config, name, paths)
        signature_path = checkpoint_manifest_path(config, name, paths)
        if checkpoint.exists() and signature_path.exists() and not force:
            saved = json.loads(signature_path.read_text())
            if saved.get("config_signature") != config.signature:
                raise RuntimeError(f"Checkpoint configuration mismatch: {checkpoint}")
            outputs.append(checkpoint)
            continue
        if checkpoint.exists() and not signature_path.exists() and not force:
            raise RuntimeError(
                f"Refusing to overwrite unmanifested checkpoint {checkpoint}; "
                "pass --force to replace it."
            )
        theta, x = _raw_pair(name, train)
        theta_val, x_val = _raw_pair(name, validation)
        theta = normalizers[spec.theta_key].transform(theta)
        x = normalizers[spec.x_key].transform(x)
        theta_val = normalizers[spec.theta_key].transform(theta_val)
        x_val = normalizers[spec.x_key].transform(x_val)
        torch.manual_seed(config.root_seed + sum(map(ord, name)))
        model = ScoreBasedInferenceModel(
            nodes_size=spec.theta_dim + spec.x_dim,
            sde_type=config.sde_type, sigma=config.sigma,
            beta_min=config.beta_min, beta_max=config.beta_max,
            hidden_size=config.hidden_size, depth=config.depth,
            num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
            device=device,
        )
        model.train(
            theta, x, theta_val, x_val,
            batch_size=config.batch_size, max_epochs=config.train_epochs,
            early_stopping_patience=config.patience, device=device,
            verbose=True, path=str(directory), name=checkpoint_tag(config, name),
        )
        atomic_json(signature_path, {
            "config_signature": config.model_signature,
            "model_signature": config.model_signature,
            "data_signature": config.data_signature,
            "model": name,
            "sde_type": config.sde_type,
            "diffusion": config.diffusion_tag,
            "training_method": training_tag(config),
            "training_samples": config.train_size,
            "validation_samples": config.validation_size,
            "checkpoint": checkpoint.name,
            "device": device,
            "theta_dim": spec.theta_dim, "x_dim": spec.x_dim,
            "normalization": {"theta": spec.theta_key, "x": spec.x_key},
            "train_loss": getattr(model.trainer, "train_loss", []),
            "validation_loss": getattr(model.trainer, "val_loss", []),
        })
        outputs.append(checkpoint)
    return outputs


def load_model(config, name, paths=None, device="cpu"):
    checkpoint = checkpoint_path(config, name, paths)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}; run train_models.py first.")
    model = ScoreBasedInferenceModel.load(str(checkpoint), device=device)
    if model.sde_type != config.sde_type:
        raise RuntimeError(
            f"Checkpoint SDE {model.sde_type!r} does not match "
            f"configuration {config.sde_type!r}."
        )
    if config.sde_type == "vpsde" and (
        model.beta_min != config.beta_min or model.beta_max != config.beta_max
    ):
        raise RuntimeError(
            "Checkpoint VPSDE beta schedule does not match the configuration."
        )
    return model
