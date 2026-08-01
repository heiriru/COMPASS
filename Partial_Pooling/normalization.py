"""Train-only affine normalization with explicit degenerate-feature policy."""

from dataclasses import dataclass
import torch


@dataclass
class Normalizer:
    mean: torch.Tensor
    scale: torch.Tensor
    degenerate: torch.Tensor
    fitted_split: str = "train"

    @classmethod
    def fit(cls, values, fitted_split="train"):
        values = torch.as_tensor(values, dtype=torch.float32)
        mean = values.mean(0)
        raw_scale = values.std(0, unbiased=False)
        degenerate = (~torch.isfinite(raw_scale)) | (raw_scale <= 1e-8)
        scale = torch.where(degenerate, torch.ones_like(raw_scale), raw_scale)
        return cls(mean, scale, degenerate, fitted_split)

    def transform(self, values):
        return (torch.as_tensor(values, dtype=torch.float32) - self.mean) / self.scale

    def inverse(self, values):
        return torch.as_tensor(values, dtype=torch.float32) * self.scale + self.mean

    def state_dict(self):
        return {
            "mean": self.mean, "scale": self.scale,
            "degenerate": self.degenerate,
            "fitted_split": self.fitted_split,
            "degenerate_policy": "center and use unit scale",
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["scale"], state["degenerate"], state["fitted_split"])
