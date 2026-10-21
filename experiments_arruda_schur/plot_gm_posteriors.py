"""GM shared posteriors p(g | x_1..x_n): exact vs composition rules, per regime and n."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

sys.path.insert(0, str(common.COMPASS / "Gaussian_mixtures_compositional_score_modelling"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

import model as M  # noqa: E402
from plots import RULE_COLOURS, INK, MUTED  # noqa: E402  (shared style + palette)

mode = sys.argv[1] if len(sys.argv) > 1 else "oracle"
tag = sys.argv[2] if len(sys.argv) > 2 else "posteriors"
root = HERE / "artifacts" / "gm" / mode / "scaling" / tag
draws = np.load(root / "draws.npz")
datasets = json.loads((root / "datasets.json").read_text())
summary = json.loads((root / "summary.json").read_text())
rules = [r for r in ("gauss_jacobian_FD", "arruda_direct", "arruda_schur", "arruda_laplace",
                     "arruda_gauss_direct") if any(k.endswith("_" + r) for k in draws.files)]
regimes = [r for r in ("asymmetric", "symmetric") if any(d["regime"] == r for d in datasets)]
ns = sorted({d["observations"] for d in datasets})

fig, axes = plt.subplots(len(regimes), len(ns), figsize=(3.3 * len(ns), 3.0 * len(regimes)),
                         squeeze=False)
for i, regime in enumerate(regimes):
    for j, n in enumerate(ns):
        ax = axes[i, j]
        x = next(d["x"] for d in datasets if d["regime"] == regime and d["observations"] == n)
        weights, means, var = M.DEFAULT.global_posterior(x)
        sd = np.sqrt(var)
        lo, hi = min(means) - 6 * sd, max(means) + 6 * sd
        grid = np.linspace(lo, hi, 800)
        ax.fill_between(grid, M.DEFAULT.global_density(grid, x), color="#e4e2da", lw=0)
        ax.plot(grid, M.DEFAULT.global_density(grid, x), color=INK, lw=1.6, label="exact")
        for rule in rules:
            key = f"{regime}_n{n}_{rule}"
            if key not in draws.files:
                continue
            g = draws[key]
            inside = g[(g > lo - 4 * sd) & (g < hi + 4 * sd)]
            w1 = summary.get(f"{regime}|{n}|{rule}", {}).get("w1_over_sigma", {})
            w1 = w1.get("median", np.nan) if isinstance(w1, dict) else w1
            if len(inside) > 10 and np.std(inside) > 0:
                kde = gaussian_kde(inside, bw_method=0.08)
                ax.plot(grid, kde(grid) * len(inside) / len(g), color=RULE_COLOURS[rule], lw=1.6,
                        label=f"{rule}  W1/sd {w1:.2f}")
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.set_title(f"{regime}, n = {n}   (exact mass + = {weights[1]:.2f})", fontsize=9)
        ax.set_xlabel("g")
        ax.legend(fontsize=6.3, loc="upper left")
fig.suptitle(f"Gaussian mixture: shared posterior p(g | x₁..xₙ), "
             f"{'exact scores (oracle)' if mode == 'oracle' else mode + ' network'}", fontsize=10.5)
fig.tight_layout()
out = HERE / "artifacts" / "figures" / f"9_gm_posteriors_{mode}.png"
fig.savefig(out, bbox_inches="tight", dpi=150)
print("wrote", out)
