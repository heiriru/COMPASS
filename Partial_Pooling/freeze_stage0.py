#!/usr/bin/env python3
"""Stage 0 of the IID-embedding ladder: freeze the current (legacy) state.

Writes two things:

1. ``tests/fixtures/legacy_arch_regression.pt`` -- outputs of freshly
   constructed, fixed-seed ``arch="legacy"`` networks on fixed
   ``(x, t, condition_mask)`` triples, for the three benchmark presets. This
   fixture depends on nothing but the source tree, so it is the regression test
   that must keep passing on a clean checkout for the whole ladder.

2. ``tests/fixtures/legacy_checkpoint_regression.pt`` -- outputs of the *trained*
   5,957,984-parameter ``small`` checkpoint: raw scores, fixed-seed
   single-observation posterior draws, PF-ODE log-densities, and a fixed-seed
   compositional (``gauss_jacobian``) multi-observation draw. Together these
   cover all nine call sites into the score network.

and a freeze manifest under ``Partial_Pooling/artifacts/frozen/stage0/`` holding
parameter counts, checkpoint digests, normalization constants and a copy of the
baseline recovery metrics.

Run:  python Partial_Pooling/freeze_stage0.py [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit(3)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

# Single-threaded: BLAS reduction order depends on the thread count, so a
# multi-threaded fixture cannot be reproduced bit-for-bit by a test process that
# happens to pick a different number of threads.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

from compass import ScoreBasedInferenceModel  # noqa: E402

from config import get_config  # noqa: E402
from paths import BenchmarkPaths  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
FREEZE = Path(__file__).resolve().parent / "artifacts" / "frozen" / "stage0"

# Fixed seeds; every number in the fixtures is a function of these alone.
ARCH_SEED = 20250718
INPUT_SEED = 424242
SAMPLE_SEED = 999

# (name, hidden_size, depth, num_heads, mlp_ratio) -- the three benchmark presets
# at the partial-pooling layout, plus one deliberately tiny shape so the fixture
# still exercises the code path when nodes_size is small.
ARCH_CASES = (
    ("tiny", 12, 4, 1, 2, 2),
    ("compact", 100, 4, 1, 1, 2),
    ("small", 100, 16, 2, 2, 4),
    ("large", 100, 128, 6, 16, 4),
)

# Baseline recovery runs of the 5.96M `small` checkpoint that the ladder must
# beat: 100 datasets x 20 subjects x 256 draws x 50 DPM steps.
BASELINE_RUNS = {
    "dpm2_gauss_jacobian_newton_churn2_refresh20":
        "churn100/jacobian_newton/artifacts_churn_eta2_refresh20",
    "dpm2_gauss_hierarchical_churn2":
        "churn100/hier_churn2/artifacts_churn_eta2",
    "dpm2_gauss_hierarchical_stock":
        "churn100/hier_stock/artifacts_churn_eta0",
    "langevin_fnpse":
        "churn100/fnpse/artifacts_churn_eta0",
}


def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fixed_inputs(nodes_size, batch=6, generator=None):
    """Deterministic (x, t, condition_mask) triples covering the mask regimes."""
    x = torch.randn(batch, nodes_size, generator=generator)
    t = torch.rand(batch, 1, generator=generator) * 0.9 + 0.05
    masks = torch.zeros(batch, nodes_size)
    # row 0: everything latent (exercises the add_bias_kv escape hatch)
    # row 1: everything conditioned
    # row 2: posterior layout (theta latent, x observed) for the 10/90 split
    # row 3: likelihood layout (theta observed, x latent)
    # rows 4+: random Bernoulli(0.33), the training-time regime
    masks[1] = 1.0
    split = min(10, nodes_size // 2)
    masks[2, split:] = 1.0
    masks[3, :split] = 1.0
    masks[4:] = (torch.rand(batch - 4, nodes_size, generator=generator) < 0.33).float()
    return x, t, masks


def build_arch_fixture():
    entries = {}
    for name, nodes_size, hidden, depth, heads, ratio in ARCH_CASES:
        torch.manual_seed(ARCH_SEED)
        model = ScoreBasedInferenceModel(
            nodes_size=nodes_size, hidden_size=hidden, depth=depth,
            num_heads=heads, mlp_ratio=ratio, device="cpu",
        )
        model.model.eval()
        generator = torch.Generator().manual_seed(INPUT_SEED)
        x, t, c = fixed_inputs(nodes_size, generator=generator)
        with torch.no_grad():
            raw = model.model(x=x, t=t, c=c)
            score = model.output_scale_function(t, raw)
        entries[name] = {
            "config": {
                "nodes_size": nodes_size, "hidden_size": hidden, "depth": depth,
                "num_heads": heads, "mlp_ratio": ratio,
            },
            "parameters": sum(p.numel() for p in model.model.parameters()),
            "x": x, "t": t, "condition_mask": c,
            "raw_output": raw, "score": score,
        }
        print(f"  {name:8s} nodes={nodes_size:3d} params={entries[name]['parameters']:,}")
    return {
        "arch": "legacy",
        "arch_seed": ARCH_SEED,
        "input_seed": INPUT_SEED,
        "torch_version": torch.__version__,
        "cases": entries,
    }


def build_checkpoint_fixture(preset="small"):
    config = get_config(preset)
    paths = BenchmarkPaths.default()
    sys.path.insert(0, str(REPO))
    from Partial_Pooling.model_pipeline import checkpoint_path, load_normalizers

    checkpoint = checkpoint_path(config, "sde_joint", paths)
    if not checkpoint.exists():
        print(f"  no checkpoint at {checkpoint}; skipping")
        return None, None

    model = ScoreBasedInferenceModel.load(str(checkpoint), device="cpu")
    model.model.eval()
    nodes_size = model.nodes_size
    generator = torch.Generator().manual_seed(INPUT_SEED)
    x, t, c = fixed_inputs(nodes_size, generator=generator)
    with torch.no_grad():
        raw = model.model(x=x, t=t, c=c)
        score = model.output_scale_function(t, raw)

    # Posterior layout: 10 latent theta dims, 90 observed trial features.
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

    condition_mask = torch.cat([torch.zeros(10), torch.ones(nodes_size - 10)])
    joint = torch.cat([torch.zeros(4, 10), observations], dim=1)
    torch.manual_seed(SAMPLE_SEED)
    log_prob = model.log_prob(
        data=joint, condition_mask=condition_mask, timesteps=20,
        divergence="exact", device="cpu", verbose=False,
    )

    torch.manual_seed(SAMPLE_SEED)
    composed = model.sample(
        x=observations, timesteps=20, num_samples=16, method="dpm", order=2,
        multi_obs_inference=True, hierarchy=list(range(7)),
        correction="gauss_jacobian", jacobian_refresh=20,
        corrector_steps=0, final_corrector_steps=0,
        device="cpu", verbose=False,
    )

    payload = {
        "arch": "legacy",
        "preset": preset,
        "checkpoint": str(checkpoint.relative_to(REPO)),
        "checkpoint_sha256": digest(checkpoint),
        "parameters": sum(p.numel() for p in model.model.parameters()),
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
    print(f"  {preset}: params={payload['parameters']:,} sha={payload['checkpoint_sha256'][:16]}")
    return payload, config


def freeze_manifest(arch_fixture, checkpoint_fixture, config):
    paths = BenchmarkPaths.default()
    record = {
        "stage": 0,
        "arch": "legacy",
        "torch_version": torch.__version__,
        "parameter_counts": {
            name: entry["parameters"] for name, entry in arch_fixture["cases"].items()
        },
        "parameter_breakdown": {},
        "baseline_metrics": {},
    }
    for name, nodes_size, hidden, depth, heads, ratio in ARCH_CASES:
        torch.manual_seed(ARCH_SEED)
        model = ScoreBasedInferenceModel(
            nodes_size=nodes_size, hidden_size=hidden, depth=depth,
            num_heads=heads, mlp_ratio=ratio, device="cpu",
        ).model
        total = sum(p.numel() for p in model.parameters())
        block_adaln = sum(
            p.numel() for block in model.blocks for p in block.adaLN_modulation.parameters()
        )
        final = sum(p.numel() for p in model.final_layer.parameters())
        record["parameter_breakdown"][name] = {
            "total": total, "block_adaLN": block_adaln, "final_layer": final,
            "everything_else": total - block_adaln - final,
            "position_specific_fraction": (block_adaln + final) / total,
        }

    if checkpoint_fixture is not None:
        record["checkpoint"] = {
            "path": checkpoint_fixture["checkpoint"],
            "sha256": checkpoint_fixture["checkpoint_sha256"],
            "parameters": checkpoint_fixture["parameters"],
            "preset": checkpoint_fixture["preset"],
            "train_size": config.train_size,
            "model_signature": config.model_signature,
            "data_signature": config.data_signature,
        }
        normalization = paths.data / (
            f"partial_pooling-train-{config.train_size}-normalization.pt"
        )
        if normalization.exists():
            state = torch.load(normalization, map_location="cpu")
            record["normalization"] = {
                "path": str(normalization.relative_to(REPO)),
                "sha256": digest(normalization),
                "data_signature": state.get("data_signature"),
                "keys": sorted(state["normalizers"]),
            }

    benchmark = Path(__file__).resolve().parent
    for label, relative in BASELINE_RUNS.items():
        root = benchmark / relative
        summaries = sorted(root.glob("partial_pooling_recovery/*/*/*-summary.json"))
        if not summaries:
            continue
        summary = json.loads(summaries[0].read_text())
        record["baseline_metrics"][label] = summary
        destination = FREEZE / "baseline_metrics" / f"{label}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summaries[0], destination)
        run_config = summaries[0].with_name(
            summaries[0].name.replace("-summary.json", "-run_config.json")
        )
        if run_config.exists():
            shutil.copyfile(run_config, destination.with_name(f"{label}-run_config.json"))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 0 freeze")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preset", default="small")
    args = parser.parse_args(argv)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    FREEZE.mkdir(parents=True, exist_ok=True)

    arch_path = FIXTURES / "legacy_arch_regression.pt"
    if arch_path.exists() and not args.force:
        print(f"{arch_path} exists; pass --force to rebuild.")
        arch_fixture = torch.load(arch_path, map_location="cpu", weights_only=False)
    else:
        print("Building legacy architecture fixture...")
        arch_fixture = build_arch_fixture()
        torch.save(arch_fixture, arch_path)
        print(f"Wrote {arch_path}")

    checkpoint_path_out = FIXTURES / "legacy_checkpoint_regression.pt"
    config = get_config(args.preset)
    if checkpoint_path_out.exists() and not args.force:
        print(f"{checkpoint_path_out} exists; pass --force to rebuild.")
        checkpoint_fixture = torch.load(
            checkpoint_path_out, map_location="cpu", weights_only=False
        )
    else:
        print("Building legacy checkpoint fixture...")
        checkpoint_fixture, config = build_checkpoint_fixture(args.preset)
        if checkpoint_fixture is not None:
            torch.save(checkpoint_fixture, checkpoint_path_out)
            print(f"Wrote {checkpoint_path_out}")

    record = freeze_manifest(arch_fixture, checkpoint_fixture, config)
    manifest = FREEZE / "manifest.json"
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
