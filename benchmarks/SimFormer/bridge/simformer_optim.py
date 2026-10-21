"""Simformer's optimizer, as a drop-in for COMPASS's schedule-free AdamW (bridge only).

scoresbibm/methods/score_transformer.py:

    schedule = optax.linear_schedule(lr, min_lr, N // 2, N // 2)   # constant, then linear decay
    optax.chain(optax.adaptive_grad_clip(clip), optax.adam(schedule))

i.e. Adam at ``lr`` for the first half of the ``N`` steps, then linearly down to
``min_lr`` (1e-6), with unit-wise adaptive gradient clipping (factor 10). The
object mimics ``schedulefree.AdamWScheduleFree`` closely enough for
``compass.Trainer`` (``train()`` / ``eval()`` are no-ops: there are no averaged
weights to swap in). ``install`` replaces the class the Trainer instantiates;
it touches nothing in ``src/compass`` and only runs for configs that set
``optimizer: simformer``.
"""

import importlib
import types

import torch


def _unitwise_norm(x):
    # optax.adaptive_grad_clip: scalars/vectors whole; matrices per output unit.
    # torch Linear weights are (out, in), so a unit is a row.
    if x.ndim <= 1:
        return x.norm()
    return x.flatten(1).norm(dim=1).reshape(-1, *([1] * (x.ndim - 1)))


class SimformerAdam:
    TOTAL_STEPS = 10000
    MIN_LR = 1e-6
    CLIP = 10.0

    def __init__(self, params, lr=1e-3, **_):
        self.params = [p for p in params if p.requires_grad]
        self.base_lr = float(lr)
        self.adam = torch.optim.Adam(self.params, lr=self.base_lr)
        self.param_groups = self.adam.param_groups
        self.n = 0

    def lr_at(self, step):
        half = self.TOTAL_STEPS // 2
        if step < half:
            return self.base_lr
        frac = min(1.0, (step - half) / max(1, self.TOTAL_STEPS - half))
        return self.base_lr + frac * (self.MIN_LR - self.base_lr)

    def train(self):
        pass

    def eval(self):
        pass

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        for p in self.params:
            if p.grad is None:
                continue
            max_norm = _unitwise_norm(p).clamp_min(1e-3) * self.CLIP
            g_norm = _unitwise_norm(p.grad)
            p.grad.mul_(torch.where(g_norm > max_norm, max_norm / g_norm.clamp_min(1e-6),
                                    torch.ones_like(g_norm)))
        for group in self.adam.param_groups:
            group["lr"] = self.lr_at(self.n)
        self.adam.step()
        self.n += 1

    def state_dict(self):
        return {"adam": self.adam.state_dict(), "n": self.n}

    def load_state_dict(self, state):
        self.adam.load_state_dict(state["adam"])
        self.n = state["n"]


def install(total_steps, min_lr=1e-6, clip=10.0):
    SimformerAdam.TOTAL_STEPS = int(total_steps)
    SimformerAdam.MIN_LR = float(min_lr)
    SimformerAdam.CLIP = float(clip)
    tm = importlib.import_module("compass.Trainer")
    tm.schedulefree = types.SimpleNamespace(AdamWScheduleFree=SimformerAdam)
