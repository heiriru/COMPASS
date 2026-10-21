#!/usr/bin/env python3
"""Grade the te sweep with the directory's own harness, te-aware.

``run_all_local_inference.py`` loads every checkpoint through
``train_model.load_to`` -> ``SBIm.load``, which rebuilds a legacy network at
te=256 unconditionally and therefore raises on a rewired checkpoint. Rather than
fork the harness (and lose the guarantee that every arm is scored by identical
code), this patches that one loader to read the width off the weights and
rebuild at it. Everything downstream -- sampling, coverage, SBC ranks, the ABC
comparison, endpoint error -- is the harness's own.

Grading follows the two lessons already paid for in this directory:
  * coverage is read at n>=512 datasets, because the n=128 envelope (+-0.09 at
    the 50% level) cannot resolve the differences involved; and
  * the test seed is varied, because repeating a measurement on the same 128
    datasets proved nothing about reproducibility the last time.

    python experiments/grade_te.py --seeds 812 913 --observations 512
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import torch  # noqa: E402

import te_rewire as TE  # noqa: E402
import train_model as T  # noqa: E402

ROOT = HERE.parent
MODELS = ROOT / "artifacts" / "models"


def te_aware_load_to(checkpoint, device):
    """``train_model.load_to``, but able to read a rewired legacy checkpoint."""
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    if "blocks.0.adaLN_modulation.1.weight" not in state:      # tokenized/grouped
        return _ORIGINAL_LOAD_TO(checkpoint, device)
    te = TE.width_of(state)
    hidden = int(payload["hidden_size"])
    config = next((name for name, spec in T.CONFIGS.items()
                   if spec["hidden_size"] == hidden
                   and spec["depth"] == int(payload["depth"])
                   and spec["mlp_ratio"] == payload["mlp_ratio"]), None)
    if config is None:
        raise ValueError(f"no CONFIGS entry matches {checkpoint}")
    if te == TE.DEFAULT_TIME_EMBEDDING:
        return _ORIGINAL_LOAD_TO(checkpoint, device)            # stock path
    model = T.build(config, "legacy", device=device)
    TE.rewire(model, te)
    model.model.load_state_dict(state)
    model.model.to(device)
    return model


_ORIGINAL_LOAD_TO = T.load_to


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--te", type=int, nargs="+", default=[256, 128, 64, 32])
    p.add_argument("--config", default="h16d2")
    p.add_argument("--simulations", type=int, default=10_000)
    p.add_argument("--menu", default="six_weighted")
    p.add_argument("--time-sampling", default="log_sigma")
    p.add_argument("--seeds", type=int, nargs="+", default=[812, 913])
    p.add_argument("--observations", type=int, default=512)
    p.add_argument("--posterior-draws", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--extra", action="append", default=[],
                   help="NAME=PATH checkpoint to include alongside the sweep.")
    args = p.parse_args(argv)

    T.load_to = te_aware_load_to                      # the one patch
    import run_all_local_inference as R
    R.trainer.load_to = te_aware_load_to

    checkpoints = []
    for te in args.te:
        name = (f"legacy_{args.config}_n{args.simulations}_{args.menu}"
                f"_{args.time_sampling}_te{te}")
        path = MODELS / name / "Model_checkpoint.pt"
        if not path.exists():
            print(f"skipping te={te}: {path} not trained yet")
            continue
        checkpoints.append(f"te{te}={path}")
    checkpoints.extend(args.extra)
    if not checkpoints:
        raise SystemExit("nothing to grade")

    for seed in args.seeds:
        out = ROOT / "artifacts" / f"te_sweep_seed{seed}"
        argv_inner = []
        for entry in checkpoints:
            argv_inner += ["--checkpoint", entry]
        argv_inner += ["--observations", str(args.observations),
                       "--posterior-draws", str(args.posterior_draws),
                       "--seed", str(seed), "--device", args.device,
                       "--artifacts", str(out)]
        print(f"\n=== grading seed {seed} -> {out}")
        R.main(argv_inner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
