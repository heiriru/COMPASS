#!/usr/bin/env python3
"""Does the shared validation RNG stream drive early stopping? Test the fix.

``docs/group_block_progress.md`` reports that four independent partial-pooling
arms -- K=5, K=10, K=20 and a K=10 repeat at a different init seed -- all took
their best validation loss at *exactly* epoch 48, with epoch-wise validation
curves correlating r = +0.85..0.88 across architectures as different as 172 and
667 tokens. The diagnosis was ``Trainer._run_epoch``: on the validation pass the
diffusion times (``_sample_timesteps``) and the noise (``randn_like``) come from
the global torch RNG with no per-epoch reseed, so every arm consumes the same
draws in lockstep and most of the epoch-to-epoch wobble is a common random draw
rather than model quality.

That was an inference from a correlation. This tests it directly, and tests the
proposed fix, **without editing src**: ``_run_epoch`` is wrapped so that the
validation pass runs under a saved-and-restored RNG state seeded from the epoch
index -- the same save/seed/restore ``Trainer._fixed_validation_masks`` already
uses for the masks, extended to the times and the noise.

Prediction if the diagnosis is right:
  * unpatched: two structurally different models give strongly correlated
    validation curves and tend to agree on the best epoch;
  * patched: the correlation collapses to ~0, because the only thing left in
    common is the data.

Run:
  .COMPASS/bin/python Partial_Pooling/experiments/validation_rng_test.py --device cuda:8
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inverse_kinematics"))

import numpy as np
import torch

import hierarchical as H
import simulator as sim
from compass import ScoreBasedInferenceModel as SBIm
from compass.Trainer import Trainer

SDE = {"sde_type": "vesde", "sigma": 8.0}
ARMS = {
    "h16d2": dict(hidden_size=16, depth=2, num_heads=2, mlp_ratio=2),
    "h32d3": dict(hidden_size=32, depth=3, num_heads=4, mlp_ratio=4),
}


def patch_validation_rng(seed=20250718):
    """Reseed the RNG for every validation pass; restore it afterwards.

    Returns the original method so the caller can put it back. This is the
    candidate fix, applied from outside the library.
    """
    original = Trainer._run_epoch

    def _run_epoch(self, epoch, data_loader, optimizer, is_train):
        if is_train:
            return original(self, epoch, data_loader, optimizer, is_train)
        state = torch.random.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)
        try:
            torch.manual_seed(seed)
            return original(self, epoch, data_loader, optimizer, is_train)
        finally:
            torch.random.set_rng_state(state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    Trainer._run_epoch = _run_epoch
    return original


class Recorder:
    """Capture per-epoch validation loss without depending on the log format."""

    def __init__(self):
        self.train, self.validation = [], []

    def install(self):
        original = Trainer._run_epoch
        recorder = self

        def _run_epoch(self, epoch, data_loader, optimizer, is_train):
            loss = original(self, epoch, data_loader, optimizer, is_train)
            (recorder.train if is_train else recorder.validation).append(float(loss))
            return loss

        Trainer._run_epoch = _run_epoch
        return original


def run(arm, epochs, samples, device, init_seed, patched):
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    model = SBIm(nodes_size=H.NODES, arch="legacy", device=device, **SDE, **ARMS[arm])

    theta, x = H.simulate(samples, sim.generator("train", 7))
    theta_v, x_v = H.simulate(2000, sim.generator("validation", 7))

    # The stream must start identically for every arm, or the comparison
    # measures the init draw rather than the validation draw.
    torch.manual_seed(4242)
    recorder = Recorder()
    original_epoch = recorder.install()      # the library's own method
    if patched:
        patch_validation_rng()               # wraps the recorder, not src
    try:
        with tempfile.TemporaryDirectory() as scratch:
            model.train(theta=theta, x=x, theta_val=theta_v, x_val=x_v,
                        batch_size=1024, max_epochs=epochs,
                        early_stopping_patience=epochs, lr=5e-4,
                        time_sampling="log_sigma", device=device, verbose=False,
                        path=scratch)
    finally:
        Trainer._run_epoch = original_epoch   # unconditionally back to src
    return recorder.validation


def _detrend(y, window=9):
    """Running-median trend and the epoch-to-epoch wobble around it."""
    n = len(y)
    trend = np.array([np.median(y[max(0, i - window // 2):
                                  min(n, i + window // 2 + 1)]) for i in range(n)])
    return y - trend, trend


def summarize(label, a, b, names):
    """Score the pair on statistics that separate shared noise from shared learning.

    A plain correlation of the two curves -- or of their increments -- is not
    the right instrument, and reporting one would have been a mistake: after the
    fix both curves are smooth monotone descents of the same task, so they stay
    correlated at r ~ 0.96 for reasons that have nothing to do with the RNG. The
    discriminating quantities are within a single run:

    * ``up_step_fraction`` -- how often validation loss *rises*. Under a shared
      redrawn stream the curve wobbles; under a fixed one it cannot.
    * ``selection_dip`` -- how far below its own local trend the selected
      minimum sits, in units of the wobble's own sd. This is the optimistic bias
      early stopping buys by picking the luckiest draw, and it is the thing that
      matters for checkpoint selection.
    """
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    out = {"label": label, "arms": names, "epochs": n,
           "r_level": float(np.corrcoef(a, b)[0, 1]),
           "best_epoch": [int(a.argmin()) + 1, int(b.argmin()) + 1],
           "per_arm": []}
    for name, y in zip(names, (a, b)):
        residual, trend = _detrend(y)
        sd = float(residual.std())
        out["per_arm"].append({
            "arm": name,
            "up_step_fraction": float(np.mean(np.diff(y) > 0)),
            "selection_dip_sd": float((trend[y.argmin()] - y.min()) / sd) if sd else 0.0,
            "best_epoch": int(y.argmin()) + 1,
        })
    print(f"{label:10s} {names[0]} / {names[1]}")
    for entry in out["per_arm"]:
        print(f"           {entry['arm']:34s} up-steps {entry['up_step_fraction']:.3f}  "
              f"selection dip {entry['selection_dip_sd']:.2f} sd  "
              f"best epoch {entry['best_epoch']}/{n}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--samples", type=int, default=20_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent
                                       / "validation_rng_test.json"))
    args = p.parse_args(argv)

    results, curves = [], {}
    for patched in (False, True):
        tag = "patched" if patched else "unpatched"
        for arm, seed in (("h16d2", 11), ("h32d3", 11), ("h16d2", 913)):
            key = f"{tag}:{arm}:seed{seed}"
            print(f"running {key} ...", flush=True)
            curves[key] = run(arm, args.epochs, args.samples, args.device,
                              seed, patched)
        results.append(summarize(tag, curves[f"{tag}:h16d2:seed11"],
                                 curves[f"{tag}:h32d3:seed11"],
                                 ["h16d2", "h32d3 (different architecture)"]))
        results.append(summarize(tag, curves[f"{tag}:h16d2:seed11"],
                                 curves[f"{tag}:h16d2:seed913"],
                                 ["h16d2 seed 11", "h16d2 seed 913"]))

    Path(args.out).write_text(json.dumps(
        {"epochs": args.epochs, "samples": args.samples,
         "comparisons": results, "curves": curves}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
