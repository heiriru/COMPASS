#!/usr/bin/env python3
"""Freeze the tokenized architecture, the way ``freeze_stage0.py`` froze legacy.

``freeze_stage0.py`` pinned ``arch="legacy"`` before the ladder started. Every
ladder result since then rests on ``arch="tokenized"``, and nothing pins *that*.
The Group/Sequence work edits ``schema.py`` and ``TokenizedTransformer.py``
directly, so this fixture has to exist before any of it lands.

Writes ``tests/fixtures/tokenized_arch_regression.pt`` with two independent
halves:

1. ``cases`` -- freshly constructed, fixed-seed tokenized networks over several
   schemas (atomic-only, trial tokens, per-position head, narrow time
   embedding). Depends on nothing but the source tree, so it fails on a clean
   checkout if ``SchemaEmbedder``/``SchemaHead``/``NodeSchema`` change shape,
   initialization order or arithmetic.
2. ``checkpoint`` -- the trained ``mask6mix`` arm (the 463,364-parameter Stage-4
   winner plus the six-conditional mask menu, currently the best arm measured)
   through all four network entry points: the raw score, ``Sampler`` draws,
   ``PFODE`` log-densities and a ``MultiObsSampler`` ``gauss_jacobian``
   compositional draw.

Run:  python Partial_Pooling/freeze_tokenized.py [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit(3)

import argparse  # noqa: E402
import json  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

# Same reason as freeze_stage0.py: BLAS reduction order, and hence the last bits
# of every matmul, depends on the thread count.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

from compass import IID, ScoreBasedInferenceModel  # noqa: E402

from config import get_config  # noqa: E402
from freeze_stage0 import digest, fixed_inputs  # noqa: E402
from paths import BenchmarkPaths  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
FREEZE = Path(__file__).resolve().parent / "artifacts" / "frozen" / "tokenized"

ARCH_SEED = 20250718
INPUT_SEED = 424242
SAMPLE_SEED = 999

GLOBALS = 7
LOCALS = 3
THETA = GLOBALS + LOCALS
TRIALS = 30
ITEM = 3

# The tokenized shapes worth pinning. "winner" is the Stage-4 architecture the
# whole ladder rests on; the others cover the branches of SchemaEmbedder and
# SchemaHead that the Group/Sequence work is about to touch.
def arch_cases():
    return {
        "scalar_tiny": dict(
            nodes_size=THETA + TRIALS * ITEM, arch="tokenized",
            hidden_size=8, depth=2, num_heads=2, mlp_ratio=2,
        ),
        "trial_tiny": dict(
            theta_size=THETA, x_spec=[IID("trials", item_dim=ITEM, count=TRIALS)],
            hidden_size=8, depth=2, num_heads=2, mlp_ratio=2,
        ),
        "trial_pph": dict(
            theta_size=THETA, x_spec=[IID("trials", item_dim=ITEM, count=TRIALS)],
            hidden_size=8, depth=2, num_heads=2, mlp_ratio=2,
            per_position_head=True,
        ),
        "trial_te64": dict(
            theta_size=THETA, x_spec=[IID("trials", item_dim=ITEM, count=TRIALS)],
            hidden_size=8, depth=2, num_heads=2, mlp_ratio=2,
            time_embedding_size=64,
        ),
        "winner": dict(
            theta_size=THETA, x_spec=[IID("trials", item_dim=ITEM, count=TRIALS)],
            hidden_size=64, depth=2, num_heads=8, mlp_ratio=4,
        ),
    }


def build_arch_fixture():
    entries = {}
    for name, config in arch_cases().items():
        torch.manual_seed(ARCH_SEED)
        model = ScoreBasedInferenceModel(device="cpu", **config)
        model.model.eval()
        generator = torch.Generator().manual_seed(INPUT_SEED)
        x, t, c = fixed_inputs(model.nodes_size, generator=generator)
        with torch.no_grad():
            raw = model.model(x=x, t=t, c=c)
            score = model.output_scale_function(t, raw)
        entries[name] = {
            "config": config,
            "parameters": sum(p.numel() for p in model.model.parameters()),
            "token_count": model.token_count,
            "nodes_size": model.nodes_size,
            "x": x, "t": t, "condition_mask": c,
            "raw_output": raw, "score": score,
        }
        print(f"  {name:12s} nodes={model.nodes_size:3d} tokens={model.token_count:3d} "
              f"params={entries[name]['parameters']:,}")
    return {
        "arch": "tokenized",
        "arch_seed": ARCH_SEED,
        "input_seed": INPUT_SEED,
        "torch_version": torch.__version__,
        "cases": entries,
    }


def mask6mix_config():
    """The best arm measured: Stage-4 winner backbone + the six-conditional menu."""
    return get_config(
        "small", train_subset=100000, arch="tokenized", tokenization="trial",
        hidden_size=64, depth=2, num_heads=8, mlp_ratio=4,
        mask_menu="six_mixed", mask_bernoulli_fraction=0.25,
    )


def build_checkpoint_fixture():
    config = mask6mix_config()
    paths = BenchmarkPaths.default()
    from Partial_Pooling.model_pipeline import checkpoint_path, load_normalizers

    checkpoint = checkpoint_path(config, "sde_joint", paths)
    if not checkpoint.exists():
        print(f"  no checkpoint at {checkpoint}; skipping")
        return None, config

    model = ScoreBasedInferenceModel.load(str(checkpoint), device="cpu")
    model.model.eval()
    nodes_size = model.nodes_size
    generator = torch.Generator().manual_seed(INPUT_SEED)
    x, t, c = fixed_inputs(nodes_size, generator=generator)
    with torch.no_grad():
        raw = model.model(x=x, t=t, c=c)
        score = model.output_scale_function(t, raw)

    normalizers = load_normalizers(config, paths)
    test_blob = torch.load(
        paths.data / f"partial_pooling-train-{config.train_size}-test-{config.test_datasets}.pt",
        map_location="cpu",
    )
    observations = test_blob["payload"]["sde"]["observations"][0, :4]
    observations = normalizers["sde_observations"].transform(observations)

    torch.manual_seed(SAMPLE_SEED)
    draws = model.sample(
        x=observations, timesteps=20, num_samples=16, method="dpm",
        device="cpu", verbose=False, capture_attention=False,
    )

    condition_mask = torch.cat([torch.zeros(THETA), torch.ones(nodes_size - THETA)])
    joint = torch.cat([torch.zeros(4, THETA), observations], dim=1)
    torch.manual_seed(SAMPLE_SEED)
    log_prob = model.log_prob(
        data=joint, condition_mask=condition_mask, timesteps=20,
        divergence="exact", device="cpu", verbose=False,
    )

    torch.manual_seed(SAMPLE_SEED)
    composed = model.sample(
        x=observations, timesteps=20, num_samples=16, method="dpm", order=2,
        multi_obs_inference=True, hierarchy=list(range(GLOBALS)),
        correction="gauss_jacobian", jacobian_refresh=20,
        corrector_steps=0, final_corrector_steps=0,
        device="cpu", verbose=False,
    )

    payload = {
        "arch": "tokenized",
        "arm": "mask6mix",
        "preset": config.preset,
        "checkpoint": str(checkpoint.relative_to(REPO)),
        "checkpoint_sha256": digest(checkpoint),
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "token_count": model.token_count,
        "schema": model.schema.to_dict(),
        "input_seed": INPUT_SEED,
        "sample_seed": SAMPLE_SEED,
        "torch_version": torch.__version__,
        "x": x, "t": t, "condition_mask": c,
        "raw_output": raw, "score": score,
        "observations": observations,
        "posterior_draws": draws,
        "log_prob": log_prob,
        "composed_draws": composed,
    }
    print(f"  mask6mix: params={payload['parameters']:,} tokens={payload['token_count']} "
          f"sha={payload['checkpoint_sha256'][:16]}")
    return payload, config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze the tokenized architecture")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    FREEZE.mkdir(parents=True, exist_ok=True)
    destination = FIXTURES / "tokenized_arch_regression.pt"
    if destination.exists() and not args.force:
        print(f"{destination} exists; pass --force to rebuild.")
        return 0

    print("Building tokenized architecture fixture...")
    fixture = build_arch_fixture()
    print("Building tokenized checkpoint fixture...")
    checkpoint_fixture, config = build_checkpoint_fixture()
    fixture["checkpoint"] = checkpoint_fixture
    torch.save(fixture, destination)
    print(f"Wrote {destination}")

    record = {
        "arch": "tokenized",
        "torch_version": torch.__version__,
        "parameter_counts": {
            name: entry["parameters"] for name, entry in fixture["cases"].items()
        },
        "token_counts": {
            name: entry["token_count"] for name, entry in fixture["cases"].items()
        },
    }
    if checkpoint_fixture is not None:
        record["checkpoint"] = {
            key: checkpoint_fixture[key] for key in
            ("arm", "checkpoint", "checkpoint_sha256", "parameters", "token_count", "schema")
        }
        record["checkpoint"]["model_signature"] = config.model_signature
        record["checkpoint"]["data_signature"] = config.data_signature
    manifest = FREEZE / "manifest.json"
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
