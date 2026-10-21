#!/usr/bin/env python3
"""Summarise bench_toy.py runs: accuracy vs the exact posterior and vs r1, against cost.

Per (n, variant), averaged over seeds:
  W1/sd      to the exact global posterior
  P+         P(g > 0) (exact in the header)
  dW1        W1/sd between the variant's draws and r1's draws of the same seed
  floor      W1/sd between r1 seeds (the Monte-Carlo noise floor for dW1)
  dLoc       mean over rows of W1 between local draws, variant vs r1, in local-sd units
  cost       total network rows relative to r1 (and to legacy)

    python analyze_toy.py artifacts/toy/lean/hierarchical
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


def load(directory, spec, n, seed):
    path = directory / f"{spec.replace('/', 'o')}_n{n}_s{seed}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["g"], data["l"], json.loads(str(data["cost"]))


def local_w1(l_a, l_b):
    sd = l_b.std(axis=0).clip(1e-12)
    return float(np.mean([wasserstein_distance(l_a[:, j], l_b[:, j]) / sd[j]
                          for j in range(l_b.shape[1])]))


def main(directory, baseline="r1"):
    directory = Path(directory)
    records = [json.loads(line) for line in open(directory / "records.jsonl")]
    by = defaultdict(list)
    for r in records:
        by[(r["n"], r["variant"])].append(r)
    for n in sorted({k[0] for k in by}):
        base = {s: load(directory, baseline, n, s) for s in (0, 1, 2)}
        base = {s: v for s, v in base.items() if v is not None}
        seeds = sorted(base)
        g_sd = np.mean([v[0].std() for v in base.values()])
        floor = (wasserstein_distance(base[seeds[0]][0], base[seeds[1]][0]) / g_sd
                 if len(seeds) > 1 else float("nan"))
        floor_l = local_w1(base[seeds[0]][1], base[seeds[1]][1]) if len(seeds) > 1 else float("nan")
        base_cost = np.mean([v[2]["total_rows"] for v in base.values()])
        exact = by[(n, baseline)][0]["p_pos_exact"]
        print(f"\nn = {n}   exact P+ {exact:.3f}   r1 seed floor: dW1 {floor:.3f}  dLoc {floor_l:.3f}")
        print(f"{'variant':<24}{'W1/sd':>7}{'P+':>7}{'sd x':>6}{'dW1':>7}{'dLoc':>7}{'cost':>7}{'refr':>6}")
        rows = []
        for (nn, spec), items in by.items():
            if nn != n:
                continue
            d_g, d_l = [], []
            for item in items:
                mine = load(directory, spec, n, item["seed"])
                ref = base.get(item["seed"])
                if mine is None or ref is None:
                    continue
                d_g.append(wasserstein_distance(mine[0], ref[0]) / g_sd)
                d_l.append(local_w1(mine[1], ref[1]))
            rows.append((np.mean([i["total_rows"] for i in items]) / base_cost, spec, items, d_g, d_l))
        for cost, spec, items, d_g, d_l in sorted(rows, key=lambda r: -r[0]):
            print(f"{spec:<24}{np.mean([i['w1_sd'] for i in items]):7.3f}"
                  f"{np.mean([i['p_pos'] for i in items]):7.3f}"
                  f"{np.mean([i['sd_ratio'] for i in items]):6.2f}"
                  f"{np.mean(d_g) if d_g else float('nan'):7.3f}{np.mean(d_l) if d_l else float('nan'):7.3f}"
                  f"{cost:7.2f}{np.mean([i['refreshes'] for i in items]):6.0f}")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
