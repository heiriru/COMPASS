#!/usr/bin/env python3
"""Parameter budgets for every ladder arm, at the partial-pooling layout.

Stage 2 and Stage 3 buy different things and must not be conflated: Stage 2's
shared trunk is what removes the token-count dependence from the parameter
count, and Stage 3's trial tokens buy permutation symmetry and a 2.5x shorter
sequence at (almost) no parameter cost. This script prints both axes side by
side so the distinction stays visible in the record.

Run:  python Partial_Pooling/report_ladder_budgets.py [--json PATH]
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

from compass.TokenizedTransformer import parameter_breakdown  # noqa: E402

from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.model_pipeline import build_model, model_specs  # noqa: E402

HIDDEN = (16, 32, 48, 64, 96, 128)
DEPTHS = (2, 3, 4, 6)
BUDGET = 1_000_000


def heads_for(hidden):
    """Largest head count up to 8 that keeps head_dim >= 8."""
    for heads in (8, 4, 2, 1):
        if hidden % heads == 0 and hidden // heads >= 8:
            return heads
    return 1


def measure(tokenization, hidden, depth, token_offset=False):
    config = get_config(
        "large", arch="tokenized", tokenization=tokenization,
        hidden_size=hidden, depth=depth, num_heads=heads_for(hidden), mlp_ratio=4,
        token_offset=token_offset or None,
    )
    spec = model_specs(config)["sde_joint"]
    model = build_model(config, spec)
    breakdown = parameter_breakdown(model.model)
    breakdown.update({
        "tokenization": tokenization, "hidden_size": hidden, "depth": depth,
        "num_heads": heads_for(hidden), "tokens": model.token_count,
        "token_offset": bool(token_offset),
        "checkpoint_tag": config.architecture_tag,
        "within_budget": breakdown["total"] < BUDGET,
    })
    return breakdown


def legacy_rows():
    rows = []
    for preset in ("compact", "small", "large"):
        config = get_config(preset)
        model = build_model(config, model_specs(config)["sde_joint"])
        total = sum(p.numel() for p in model.model.parameters())
        block_adaln = sum(
            p.numel() for block in model.model.blocks
            for p in block.adaLN_modulation.parameters()
        )
        final = sum(p.numel() for p in model.model.final_layer.parameters())
        rows.append({
            "preset": preset, "hidden_size": config.hidden_size,
            "depth": config.depth, "tokens": 100, "total": total,
            "block_adaLN": block_adaln, "final_layer": final,
            "position_specific_fraction": (block_adaln + final) / total,
            "train_size": config.train_size,
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ladder parameter budgets")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    legacy = legacy_rows()
    print("Legacy (arch='legacy', 100 scalar tokens, per-node modulation)")
    print(f"  {'preset':8s} {'H':>4s} {'depth':>5s} {'total':>12s} "
          f"{'position-specific':>18s}")
    for row in legacy:
        print(f"  {row['preset']:8s} {row['hidden_size']:4d} {row['depth']:5d} "
              f"{row['total']:12,d} {row['position_specific_fraction']:17.1%}")

    grid = []
    for tokenization in ("scalar", "trial"):
        print()
        tokens = 100 if tokenization == "scalar" else 40
        print(f"Tokenized, tokenization={tokenization!r} ({tokens} tokens), "
              f"shared trunk -- total parameters")
        header = "  " + f"{'H':>4s}" + "".join(f"{f'd={d}':>12s}" for d in DEPTHS)
        print(header)
        for hidden in HIDDEN:
            cells = []
            for depth in DEPTHS:
                row = measure(tokenization, hidden, depth)
                grid.append(row)
                marker = " " if row["within_budget"] else "*"
                cells.append(f"{row['total']:11,d}{marker}")
            print(f"  {hidden:4d}" + "".join(cells))
        print("  (* = at or above the 1,000,000-parameter ceiling)")

    print()
    print("Where the parameters go, at H=16 depth=2:")
    for tokenization in ("scalar", "trial"):
        row = measure(tokenization, 16, 2)
        print(f"  {tokenization:6s} tokens={row['tokens']:3d} total={row['total']:8,d} "
              f"| adaLN {row['block_adaLN']:7,d} | final {row['final_layer']:6,d} "
              f"| embed {row['input_embedding']:6,d} | time {row['time_embedding']:7,d} "
              f"| attn+mlp {row['attention_and_mlp']:6,d}")

    offsets = [measure(t, 16, 2, token_offset=True) for t in ("scalar", "trial")]
    print()
    print("token_offset ablation (per-token offset on the shared modulation):")
    for row in offsets:
        print(f"  {row['tokenization']:6s} total={row['total']:8,d} "
              f"(+{row['token_offsets']:,d} token-dependent parameters)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"legacy": legacy, "tokenized": grid,
             "token_offset_ablation": offsets, "budget": BUDGET},
            indent=2, sort_keys=True,
        ))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
