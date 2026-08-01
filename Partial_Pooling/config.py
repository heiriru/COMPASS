"""Central benchmark configuration and presets."""

from dataclasses import asdict, dataclass
import hashlib
import json


REFERENCE_URL = "https://github.com/bayesflow-org/diffusion-experiments/tree/main/case_study4"
REFERENCE_REVISION = "363186e485add4f062b2cf33f356ffa215f07056"
PRESETS = ("smoke", "full", "large")


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
    dt: float = 0.001
    max_decision_time: float = 10.0
    correction: str = "prior_corrected_sum"

    def to_dict(self):
        return asdict(self)

    @property
    def signature(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_config(preset="smoke", root_seed=None):
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
    if root_seed is not None:
        config = BenchmarkConfig(**{**config.to_dict(), "root_seed": int(root_seed)})
    return config
