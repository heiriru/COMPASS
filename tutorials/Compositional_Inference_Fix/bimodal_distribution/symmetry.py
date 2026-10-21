#!/usr/bin/env python3
"""Free accuracy: make the learned row score obey the symmetry it should.

The sign model's row posterior has an exact two-fold symmetry that the trained
network only learns approximately, and enforcing it costs no parameters:

    p(g, l | y) = p(-g, -l | y)     the sign is unidentifiable
    p(g, l | y) = p(g, l | -y)      the likelihood is even in y

Both survive VE diffusion (the kernel is symmetric), so at *every* noise level
the exact row score is **odd** in the latent pair and **even** in the
observation:

    s(-g, -l; y) = -s(g, l; y)
    s( g,  l; -y) =  s(g, l; y)

Two ways to impose that on a network that only approximately satisfies it, at
very different costs:

``fold``       feed ``|y|`` instead of ``y``. Exactly imposes the observation
               parity, and is *free* -- one forward pass, same as before.
``symmetrize`` also average the two latent orientations,
               ``[s(x) - s(-x)] / 2``. Exactly imposes the latent oddness, and
               costs one extra forward pass per score evaluation.

Averaging rather than canonicalising ("flip so that g >= 0, evaluate, flip
back") is deliberate. Canonicalisation is also free and also exactly odd, but it
is *discontinuous* at g = 0 -- precisely where the valley is, and precisely
where ``correction="gauss_jacobian"`` has to differentiate the row score. The
average is smooth everywhere.

Neither trick applies to ``control_model``: its row posterior has unequal
component weights and means that are not negatives of each other, so there is no
symmetry to exploit. This is a model-specific win, and it is stated as one.

Usage:
    python symmetry.py                 # grade plain / fold / symmetrize
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"


class SymmetricRowScore(torch.nn.Module):
    """Wrap a trained row-score network so it obeys the sign model's symmetry.

    Drop-in for ``SBIm.model``: same ``forward(x, t, c)`` signature, same output
    shape, no parameters of its own. ``fold`` and ``average`` are independent;
    ``fold`` is free and ``average`` doubles the forward passes.
    """

    def __init__(self, network, module, fold=True, average=True):
        super().__init__()
        self.network = network
        self.problem = module
        self.fold = bool(fold)
        self.average = bool(average)

    def forward(self, x, t, c):
        module = self.problem
        state = x
        if self.fold:
            # The likelihood is even in y, so |y| carries all the information;
            # feeding it removes one direction of the network's error for free.
            state = state.clone()
            state[:, module.OBSERVED_INDEX] = state[
                :, module.OBSERVED_INDEX
            ].abs()
        scores = self.network(x=state, t=t, c=c)
        if not self.average:
            return scores
        # The latent pair flips together; the observed column is left alone, and
        # its score is ignored by the sampler anyway.
        flipped = state.clone()
        latent = [module.GLOBAL_INDEX, module.LOCAL_INDEX]
        flipped[:, latent] = -flipped[:, latent]
        mirrored = self.network(x=flipped, t=t, c=c)
        out = scores.clone()
        out[:, latent] = 0.5 * (scores[:, latent] - mirrored[:, latent])
        return out


class SymmetricModel:
    """The ``ScoreBasedInferenceModel`` surface, with a symmetrised network."""

    def __init__(self, model, module, fold=True, average=True):
        self.sde = model.sde
        self.problem = module
        self.model = SymmetricRowScore(model.model, module, fold, average)
        self.sampler = getattr(model, "sampler", None)
        self._model = model

    def output_scale_function(self, t, scores):
        return self._model.output_scale_function(t, scores)


VARIANTS = {
    "plain": {"fold": False, "average": False,
              "note": "the trained network as-is"},
    "fold": {"fold": True, "average": False,
             "note": "feed |y|; free, one forward pass"},
    "average": {"fold": False, "average": True,
                "note": "average the two latent orientations; two passes"},
    "symmetrize": {"fold": True, "average": True,
                   "note": "both; two passes"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["h32d3", "h128d6"])
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    if arguments.device is None:
        try:
            from autocvd import autocvd

            autocvd(num_gpus=1, interval=1)
        except ImportError:
            pass
        arguments.device = "cuda" if torch.cuda.is_available() else "cpu"

    import run_experiment as experiment
    import sign_model
    import train as recipe

    module = sign_model
    _, _, _, y = module.observations(experiment.OBSERVATIONS,
                                    experiment.DATA_SEEDS["sign"])
    rows = []
    for config in arguments.configs:
        model, _ = recipe.train_or_load(ROOT, module, config,
                                        device=arguments.device, verbose=False)
        for name, settings in VARIANTS.items():
            wrapped = SymmetricModel(model, module, settings["fold"],
                                     settings["average"])
            fidelity = recipe.score_fidelity(
                wrapped, module, y, experiment.FIDELITY_LAMBDAS,
                device=arguments.device,
            )
            rows.append({"config": config, "variant": name,
                         "note": settings["note"],
                         **{f"rel_rmse_lam_{key}": value
                            for key, value in fidelity.items()}})
            print(f"{config:<8}{name:<12}mean {fidelity['mean']:.4f}   "
                  + "  ".join(f"{key:g}:{value:.3f}"
                              for key, value in fidelity.items()
                              if key != "mean"))

    path = ARTIFACTS / "symmetry_fidelity.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
