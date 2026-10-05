"""Print the accuracy and timing tables from the artifacts/*.json results.

    /export/data/rheinric/jax_jacobian_env/bin/python summarize.py > artifacts/summary.md
"""

import json

from common import ARTIFACTS

RUNS = [("torch", "torch_results.json"), ("torch+compile", "torch_results_compiled.json"),
        ("jax", "jax_results.json")]


def load():
    runs = {}
    for label, name in RUNS:
        path = ARTIFACTS / name
        if path.exists():
            runs[label] = json.loads(path.read_text())
    return runs


def main():
    runs = load()
    print("## Jacobian relative error vs float64 exact (float32, GPU)\n")
    for label, result in runs.items():
        accuracy = result["accuracy"]
        keys = [k for k in accuracy[0] if k.endswith("_rel_err")]
        print(f"### {label}\n")
        print("| lambda | " + " | ".join(k[:-8] for k in keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for row in accuracy:
            print(f"| {row['lambda']:g} | " + " | ".join(
                f"{row[k]:.1e}" if k in row else "fail" for k in keys) + " |")
        print()

    print("## Wall clock per Jacobian, median ms (in forward passes of the same run)\n")
    for label, result in runs.items():
        timing = result["timing"]
        names = [k for k in timing[0] if isinstance(timing[0][k], dict)]
        print(f"### {label}\n")
        print("| rows x draws | " + " | ".join(names) + " |")
        print("|---" * (len(names) + 1) + "|")
        for row in timing:
            forward = row["forward"]["median_ms"]
            cells = []
            for name in names:
                if name not in row:
                    cells.append("fail")
                    continue
                ms = row[name]["median_ms"]
                cells.append(f"{ms:.1f}" if name == "forward"
                             else f"{ms:.1f} ({ms / forward:.1f}x)")
            print(f"| {row['rows']} x {row['draws']} | " + " | ".join(cells) + " |")
        print()
        failures = result.get("failures")
        if failures:
            print("Failures: " + "; ".join(
                f"{k}: {v.splitlines()[0][:120]}" for k, v in failures.items()) + "\n")


if __name__ == "__main__":
    main()
