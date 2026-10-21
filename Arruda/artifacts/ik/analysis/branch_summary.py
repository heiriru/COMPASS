"""Branch-level summary of Arruda samples against the reference, per n and bridge.

Usage: branch_summary.py <run folder with results.json + samples_*.npy> [reference .npz dir]
Reports, for samples and reference: mass on g > 0, mean/sd within each branch, and
the fraction of samples outside the reference grid.
"""
import json, sys
from pathlib import Path
import numpy as np

folder = Path(sys.argv[1])
records = json.loads((folder / "results.json").read_text())
def branches(values, weights=None):
    w = np.ones_like(values) if weights is None else weights
    out = {"mass_pos": float(w[values > 0].sum() / w.sum())}
    for name, sel in (("neg", values < 0), ("pos", values > 0)):
        if w[sel].sum() > 0:
            m = np.average(values[sel], weights=w[sel]); s = np.sqrt(np.average((values[sel] - m) ** 2, weights=w[sel]))
            out[name] = (round(float(m), 3), round(float(s), 3))
        else:
            out[name] = None
    return out
print(f"{'n':>4} {'d':>4} | {'P(g>0)':>7} {'neg branch (mean,sd)':>22} {'pos branch (mean,sd)':>22} {'off':>5}")
for r in records:
    v = np.load(r["samples_path"]); b = branches(v); off = float(np.mean(np.abs(v) > 2.0))
    print(f"{r['n']:>4} {r['bridge']:>4g} | {b['mass_pos']:>7.3f} {str(b['neg']):>22} {str(b['pos']):>22} {off:>5.2f}")
ref = folder / "reference.npz"
if ref.exists():
    z = np.load(ref)
    for n in sorted({r["n"] for r in records}):
        g, d = z["grid"], z[f"n{n}"]; b = branches(g, d)
        print(f"{n:>4}  ref | {b['mass_pos']:>7.3f} {str(b['neg']):>22} {str(b['pos']):>22}")
