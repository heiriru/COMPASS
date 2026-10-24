"""Summarise the COMPASS_PD_PROBE jsonl of the screen by noise band (results/screen/probe_summary.txt)."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "screen"
BANDS = [(2.0, 1e9, "lambda>2"), (0.5, 2.0, "0.5-2"), (0.1, 0.5, "0.1-0.5"), (0.0, 0.1, "<0.1")]

lines = [f"{'toy':5s} {'mode':20s} {'n':>4s} {'band':9s} {'calls':>5s} {'active':>6s} {'k_rel10':>7s} "
         f"{'k_rel50':>7s} {'shift50':>7s} {'dscore50':>8s} {'dscore90':>8s} {'toward0':>7s}"]
for path in sorted(OUT.glob("probe_*.jsonl"), key=lambda p: (p.stem.split("_")[1], p.stem.split("_")[-2],
                                                                int(p.stem.split("_n")[-1]))):
    _, toy, *mode, n = path.stem.split("_")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for lo, hi, name in BANDS:
        band = [r for r in records if lo <= r["lambda"] < hi]
        if not band:
            continue
        mean = lambda key, i=None: np.nanmean([(r[key][i] if i is not None else r[key]) for r in band
                                               if r.get(key) is not None])
        lines.append(f"{toy:5s} {'_'.join(mode):20s} {n[1:]:>4s} {name:9s} {len(band):5d} "
                     f"{mean('active_fraction'):6.2f} {mean('k_rel', 0):7.3f} {mean('k_rel', 1):7.3f} "
                     f"{mean('shift', 1):7.2f} {mean('dscore', 1):8.3f} {mean('dscore', 2):8.3f} "
                     f"{mean('toward_origin'):7.2f}")
text = "\n".join(lines)
print(text)
(OUT / "probe_summary.txt").write_text(text + "\n")
