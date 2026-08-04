"""Central benchmark configuration and presets."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json


REFERENCE_URL = "https://github.com/bayesflow-org/diffusion-experiments/tree/main/case_study4"
REFERENCE_REVISION = "363186e485add4f062b2cf33f356ffa215f07056"
PRESETS = ("smoke", "full", "large")
SDE_TYPES = ("vesde", "vpsde")


def _signature(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class BenchmarkConfig:
    preset: str
    root_seed: int = 20250718
    train_size: int = 2048
    validation_size: int = 256
    test_datasets: int = 10
    subjects: int = 20
    trials: int = 30
    posterior_draws: int = 128
    diffusion_steps: int = 20
    train_epochs: int = 4
    patience: int = 2
    batch_size: int = 128
    shard_size: int = 512
    hidden_size: int = 64
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: int = 2
    sde_type: str = "vesde"
    sigma: float = 25.0
    beta_min: float = 0.1
    beta_max: float = 20.0
    dt: float = 0.001
    max_decision_time: float = 10.0
    correction: str = "prior_corrected_sum"

    def __post_init__(self):
        if self.sde_type not in SDE_TYPES:
            raise ValueError(
                f"Unknown SDE type {self.sde_type!r}; choose one of "
                f"{', '.join(SDE_TYPES)}."
            )
        if self.sigma <= 1.0:
            raise ValueError("VESDE sigma must be greater than 1.")
        if self.beta_min <= 0.0 or self.beta_max <= self.beta_min:
            raise ValueError("VPSDE requires 0 < beta_min < beta_max.")

    def to_dict(self):
        return asdict(self)

    def data_dict(self):
        """Configuration fields that can change simulated data or its layout."""
        values = self.to_dict()
        keys = (
            "preset", "root_seed", "train_size", "validation_size",
            "test_datasets", "subjects", "trials", "shard_size", "dt",
            "max_decision_time",
        )
        return {key: values[key] for key in keys}

    def model_dict(self):
        """Training configuration, retaining legacy VE checkpoint signatures."""
        values = self.to_dict()
        if self.sde_type == "vesde":
            # These fields did not exist in the original VE manifests and do not
            # affect a VE model. Omitting them keeps every existing VE checkpoint
            # reusable after VPSDE support is enabled.
            values.pop("beta_min")
            values.pop("beta_max")
        return values

    @property
    def signature(self):
        """Backward-compatible alias for the model/training signature."""
        return self.model_signature

    @property
    def model_signature(self):
        return _signature(self.model_dict())

    @property
    def data_signature(self):
        return _signature(self.data_dict())

    @property
    def legacy_data_signature(self):
        """Signature used by pre-VPSDE data generated with the same preset."""
        legacy = replace(
            self, sde_type="vesde", sigma=25.0,
            beta_min=0.1, beta_max=20.0,
        ).to_dict()
        legacy.pop("beta_min")
        legacy.pop("beta_max")
        return _signature(legacy)

    @property
    def compatible_data_signatures(self):
        return {self.data_signature, self.legacy_data_signature}

    @property
    def diffusion_tag(self):
        if self.sde_type == "vesde":
            return "vesde"
        beta_min = format(self.beta_min, "g").replace("-", "m").replace(".", "p")
        beta_max = format(self.beta_max, "g").replace("-", "m").replace(".", "p")
        return f"vpsde-b{beta_min}-{beta_max}"


def get_config(
    preset="smoke", root_seed=None, sde_type=None,
    beta_min=None, beta_max=None,
):
    if preset == "smoke":
        config = BenchmarkConfig(preset="smoke")
    elif preset in {"full", "large"}:
        config = BenchmarkConfig(
            preset=preset,
            train_size=100000 if preset == "large" else 32768,
            validation_size=5000 if preset == "large" else 4096,
            test_datasets=100, subjects=100, trials=30,
            posterior_draws=1000, diffusion_steps=100,
            train_epochs=500, patience=20, batch_size=128,
            shard_size=2048, hidden_size=128, depth=6,
            num_heads=16, mlp_ratio=4,
        )
    else:
        raise ValueError(
            f"Unknown preset {preset!r}; choose one of {', '.join(PRESETS)}."
        )
    overrides = {}
    if root_seed is not None:
        overrides["root_seed"] = int(root_seed)
    if sde_type is not None:
        overrides["sde_type"] = str(sde_type).lower()
    if beta_min is not None:
        overrides["beta_min"] = float(beta_min)
    if beta_max is not None:
        overrides["beta_max"] = float(beta_max)
    if overrides:
        config = replace(config, **overrides)
    return config
