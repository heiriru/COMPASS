#!/usr/bin/env python3
import json
import html
from pathlib import Path

GRAPH_JSON = Path("graphify-work/graphify-out/graph.json")
GRAPH_DOT = Path("graphify-work/graphify-out/graph.dot")


def q(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


data = json.loads(GRAPH_JSON.read_text())

nodes = data.get("nodes", [])
edges = data.get("edges", [])

with GRAPH_DOT.open("w") as f:
    f.write("digraph Graphify {\n")
    f.write("  graph [rankdir=LR, overlap=false, splines=true];\n")
    f.write("  node [shape=box, style=rounded, fontsize=10];\n")
    f.write("  edge [fontsize=8];\n\n")

    for node in nodes:
        node_id = node.get("id") or node.get("name")
        label = node.get("name") or node.get("label") or str(node_id)
        kind = node.get("type") or node.get("kind") or ""
        community = node.get("community", "")

        tooltip = "\\n".join(f"{k}: {v}" for k, v in node.items())

        f.write(
            f"  {q(node_id)} "
            f"[label={q(label)}, tooltip={q(tooltip)}, "
            f"group={q(community)}, xlabel={q(kind)}];\n"
        )

    f.write("\n")

    for edge in edges:
        src = edge.get("source") or edge.get("from")
        dst = edge.get("target") or edge.get("to")
        if not src or not dst:
            continue

        label = edge.get("type") or edge.get("label") or ""
        tooltip = "\\n".join(f"{k}: {v}" for k, v in edge.items())

        f.write(
            f"  {q(src)} -> {q(dst)} "
            f"[label={q(label)}, tooltip={q(tooltip)}];\n"
        )

    f.write("}\n")

print(f"Wrote {GRAPH_DOT}")