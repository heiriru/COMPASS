#!/usr/bin/env python3
"""Parameter budget of the two-level ``Group`` trunk, before it is written.

The partial-pooling arm has a hard cap of **1,000,000 parameters**. The Group
architecture stacks two trunks (within-subject and across-subject) where the
ladder's winner had one, so the cap is the binding design constraint and has to
be settled before the class exists rather than discovered after.

This script counts the budget by *constructing the real modules*
(``SharedTransformerBlock``, ``SchemaEmbedder``, ``SchemaHead``,
``SharedFinalLayer``, ``TimestepEmbedder``) and summing them, so it cannot drift
from a hand-derived formula. It self-checks by reassembling the flat Stage-4
winner from the same pieces and asserting it comes to 463,364.

Run:  python Partial_Pooling/report_group_budget.py [--json PATH]
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from compass.ConditionTransformer import TimestepEmbedder  # noqa: E402
from compass.schema import Atomic, IID, NodeSchema  # noqa: E402
from compass.TokenizedTransformer import (  # noqa: E402
    SchemaEmbedder, SharedFinalLayer, SharedTransformerBlock,
)

GLOBALS, LOCALS, TRIALS, ITEM = 7, 3, 30, 3
BUDGET = 1_000_000
WINNER_PARAMETERS = 463_364


def count(module):
    return sum(p.numel() for p in module.parameters())


def flat_winner(hidden=64, time_embedding=256, depth=2, heads=8, mlp_ratio=4):
    """The Stage-4 winner, reassembled from the same pieces as the check."""
    schema = NodeSchema([Atomic("theta", GLOBALS + LOCALS), IID("trials", ITEM, TRIALS)])
    return (
        count(TimestepEmbedder(schema.token_count, hidden, mlp_ratio=mlp_ratio,
                               frequency_embedding_size=time_embedding))
        + count(SchemaEmbedder(schema, hidden))
        + depth * count(SharedTransformerBlock(
            hidden, heads, schema.token_count, mlp_ratio=mlp_ratio,
            time_embedding_size=time_embedding))
        + count(SharedFinalLayer(schema, hidden, time_embedding_size=time_embedding))
    )


def group_budget(subjects, hidden, time_embedding, mlp_ratio, depth_within,
                 depth_across, depth_readout, heads=8, slot="learned",
                 pool="mean", combine="add"):
    """Exact parameter count of the planned two-level trunk.

    The inner sub-block embedders and heads are shared across all ``subjects``
    repeats -- that sharing is what keeps trials exchangeable within a subject,
    and it is also why the total is very nearly independent of K.
    """
    inner = NodeSchema([Atomic("local", LOCALS), IID("trials", ITEM, TRIALS)])
    globals_only = NodeSchema([Atomic("theta", GLOBALS)])
    readout = NodeSchema([Atomic("theta", GLOBALS), Atomic("local", LOCALS),
                          IID("trials", ITEM, TRIALS)])

    def block(tokens):
        return count(SharedTransformerBlock(
            hidden, heads, tokens, mlp_ratio=mlp_ratio,
            time_embedding_size=time_embedding))

    parts = {
        "t_embedder": count(TimestepEmbedder(
            1, hidden, mlp_ratio=mlp_ratio, frequency_embedding_size=time_embedding)),
        "embed_inner": count(SchemaEmbedder(inner, hidden)),
        "embed_global": count(SchemaEmbedder(globals_only, hidden)),
        "within": depth_within * block(LOCALS + TRIALS),
        "across": depth_across * block(GLOBALS + subjects),
        "readout": depth_readout * block(LOCALS + TRIALS),
        "slot": subjects * hidden if slot == "learned" else 0,
        "pool": 0 if pool == "mean" else hidden + count(nn.MultiheadAttention(
            hidden, heads, add_bias_kv=True, batch_first=True)),
        "combine": 0 if combine == "add" else count(nn.Linear(2 * hidden, hidden)),
        "final": count(SharedFinalLayer(
            readout, hidden, time_embedding_size=time_embedding)),
    }
    return sum(parts.values()), parts


# The shapes worth reporting. hidden/time_embedding are held at the winner's
# values wherever possible: docs/followup_results.md measured that shrinking
# time_embedding_size to 64 costs 45% on global NRMSE (0.632 vs 0.436) while
# barely moving validation loss, so it is not a free lever here.
CANDIDATES = (
    ("h64 d2+2+0  (default)", 64, 256, 4, 2, 2, 0),
    ("h64 d2+2+1  (readout)", 64, 256, 4, 2, 2, 1),
    ("h64 d3+2+0", 64, 256, 4, 3, 2, 0),
    ("h64 d2+3+0", 64, 256, 4, 2, 3, 0),
    ("h64 d3+3+0", 64, 256, 4, 3, 3, 0),
    ("h48 d2+2+0", 48, 256, 4, 2, 2, 0),
    ("h48 d3+3+0", 48, 256, 4, 3, 3, 0),
    ("h96 d2+2+0", 96, 256, 4, 2, 2, 0),
)
SUBJECT_COUNTS = (5, 10, 20, 50, 100)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Group trunk parameter budget")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    check = flat_winner()
    assert check == WINNER_PARAMETERS, (
        f"budget calculator disagrees with the known winner: {check:,} != "
        f"{WINNER_PARAMETERS:,}. Fix the calculator before trusting a row below."
    )
    print(f"self-check: flat Stage-4 winner reassembles to {check:,}  OK\n")

    header = f"{'configuration':<24}" + "".join(f"{'K=' + str(k):>12}" for k in SUBJECT_COUNTS)
    print(header)
    print("-" * len(header))
    record = {"budget": BUDGET, "winner_parameters": WINNER_PARAMETERS, "rows": {}}
    for label, hidden, te, ratio, dw, da, dr in CANDIDATES:
        totals = []
        for subjects in SUBJECT_COUNTS:
            total, parts = group_budget(subjects, hidden, te, ratio, dw, da, dr)
            totals.append(total)
            record["rows"].setdefault(label, {})[f"K={subjects}"] = {
                "total": total, "parts": parts, "under_budget": total < BUDGET,
            }
        flag = "" if max(totals) < BUDGET else "   OVER"
        print(f"{label:<24}" + "".join(f"{v:>12,}" for v in totals) + flag)

    total, parts = group_budget(20, 64, 256, 4, 2, 2, 0)
    print(f"\nbreakdown of the default at K=20 ({total:,} parameters):")
    for name, value in sorted(parts.items(), key=lambda kv: -kv[1]):
        if value:
            print(f"  {name:<14}{value:>10,}  {100 * value / total:5.1f}%")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
