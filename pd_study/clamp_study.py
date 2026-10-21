"""Conservative vs coordinatewise clamping of denoised predictions, on analytic 2-D targets.

Targets (clean posteriors p0, diffused by N(0, lambda^2 I), scores exact by autograd):
  gauss    N(mu, S), corr 0.85                      -> psi quadratic: metric clamp is exact
  mixture  0.6 N(a, S1) + 0.4 N(b, S2), correlated  -> psi not quadratic

Clamps of m(y) = y + lambda^2 s(y) into the box B = [-b, b]^2:
  coordinate  clip(m)                                     (library default)
  metric      argmin_{z in B} (z-m)^T (grad m)^-1 (z-m)   (library clamp_mode="metric")
  conjugate   grad psi_B, psi_B = (psi* + iota_B)*         (exact; conservative_clamp.py)

A. conservativity: ||J - J^T|| / ||J|| of the clamped score field (central differences)
   where the clamp is active, per lambda, plus |metric - conjugate| in score units.
B. sampling with the exact score: reverse VE SDE (Euler-Maruyama, 400 log-spaced levels,
   lambda 5 -> 1e-3) with each clamp, box b in {1, 1.5, 2.5}: mean / covariance /
   correlation error and sliced W1 to exact draws, share outside B.
C. the clamp's purpose: the same with a corrupted score (a growing non-conservative
   rotation beyond radius 2.5, like a network extrapolating in the tails), box b = 3:
   runaway share (|theta| > 6) and W1.

    python clamp_study.py   -> results/clamp_study.json, figures/clamp_*.png
"""
import json
import os
import sys
import time
from pathlib import Path

CPU_THREAD_LIMIT = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(CPU_THREAD_LIMIT)
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT])

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conservative_clamp as cc  # noqa: E402

torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
(HERE / "results").mkdir(exist_ok=True)
(HERE / "figures").mkdir(exist_ok=True)


class Mixture:
    def __init__(self, weights, means, covs):
        self.w = torch.tensor(weights)
        self.mu = torch.tensor(means)
        self.cov = torch.tensor(covs)

    def log_p(self, y, lam2):
        cov = self.cov + lam2 * torch.eye(2)
        d = y[:, None, :] - self.mu[None]
        prec = torch.linalg.inv(cov)
        quad = torch.einsum("nki,kij,nkj->nk", d, prec, d)
        logdet = torch.logdet(cov)
        return torch.logsumexp(torch.log(self.w) - 0.5 * quad - 0.5 * logdet - np.log(2 * np.pi), dim=1)

    def score(self, y, lam2):
        with torch.enable_grad():
            y = y.detach().requires_grad_(True)
            return torch.autograd.grad(self.log_p(y, lam2).sum(), y)[0]

    def denoiser(self, lam2):
        return lambda y: y + lam2 * self.score(y, lam2)

    def denoiser_jacobian(self, lam2):
        def jac(y):
            with torch.enable_grad():
                y = y.detach().requires_grad_(True)
                s = torch.autograd.grad(self.log_p(y, lam2).sum(), y, create_graph=True)[0]
                rows = [torch.autograd.grad(s[:, i].sum(), y, retain_graph=True)[0] for i in range(2)]
            h = torch.stack(rows, dim=-2).detach()
            h = 0.5 * (h + h.mT)
            return torch.eye(2) + lam2 * h
        return jac

    def sample(self, count, generator):
        k = torch.multinomial(self.w, count, replacement=True, generator=generator)
        z = torch.randn(count, 2, generator=generator)
        chol = torch.linalg.cholesky(self.cov)[k]
        return self.mu[k] + (chol @ z.unsqueeze(-1)).squeeze(-1)


TARGETS = {
    "gauss": Mixture([1.0], [[0.4, -0.3]], [[[1.0, 0.85], [0.85, 1.0]]]),
    "mixture": Mixture([0.6, 0.4], [[-1.0, -0.6], [1.3, 1.0]],
                       [[[0.35, 0.25], [0.25, 0.35]], [[0.25, -0.12], [-0.12, 0.25]]]),
}


def clamp(kind, target, lam2, y, m, lower, upper):
    if kind == "none":
        return m
    if kind == "coordinate":
        return cc.coordinate_clamp(m, lower, upper)
    if kind == "metric":
        return cc.metric_clamp(m, target.denoiser_jacobian(lam2)(y), lower, upper)
    psi = lambda v: 0.5 * (v * v).sum(-1) + lam2 * target.log_p(v, lam2)  # noqa: E731
    return cc.conjugate_clamp(target.denoiser(lam2), target.denoiser_jacobian(lam2), y, lower, upper, psi=psi)


def clamped_score(kind, target, lam2, lower, upper, corrupt=None):
    def field(y):
        s = target.score(y, lam2)
        if corrupt is not None:
            s = s + corrupt(y, lam2)
        m = y + lam2 * s
        if kind == "conjugate" and corrupt is not None:
            # the conjugate construction needs a gradient field; with a corrupted score it
            # is run on the corrupted denoiser's symmetric part (what a network would give)
            pass
        return (clamp(kind, target, lam2, y, m, lower, upper) - y) / lam2
    return field


def fd_jacobian(field, y, h=1e-5):
    cols = []
    for j in range(2):
        e = torch.zeros(2)
        e[j] = h
        cols.append((field(y + e) - field(y - e)) / (2 * h))
    return torch.stack(cols, dim=-1)


def part_a():
    out = {}
    b = 1.0
    lower, upper = -b * torch.ones(2), b * torch.ones(2)
    axis = torch.linspace(-3.5, 3.5, 71)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    maps = {}
    for name, target in TARGETS.items():
        for lam in (0.3, 0.7, 1.5):
            lam2 = lam * lam
            m = target.denoiser(lam2)(grid)
            active = ((m < lower) | (m > upper)).any(-1)
            # keep away from the kinks of the clamp (the Jacobian jumps there)
            gap = torch.minimum((m - lower).abs().min(-1).values, (m - upper).abs().min(-1).values)
            pts = grid[active & (gap > 0.02)]
            row = {"active_share": float(active.float().mean()), "points": int(pts.shape[0])}
            fields = {}
            for kind in ("coordinate", "metric", "conjugate"):
                f = clamped_score(kind, target, lam2, lower, upper)
                fields[kind] = f(pts)
                a = cc.antisymmetry(fd_jacobian(f, pts))
                row[kind] = {"median": float(a.median()), "q95": float(a.quantile(0.95)), "max": float(a.max())}
                if name == "mixture" and lam == 0.7:
                    full = torch.full((grid.shape[0],), float("nan"))
                    full[active & (gap > 0.02)] = a
                    maps[kind] = full.reshape(71, 71).numpy()
            s_norm = target.score(pts, lam2).norm(dim=-1).clamp_min(1e-12)
            row["metric_vs_conjugate_rel"] = float(((fields["metric"] - fields["conjugate"]).norm(dim=-1)
                                                    / s_norm).median())
            row["coordinate_vs_conjugate_rel"] = float(((fields["coordinate"] - fields["conjugate"]).norm(dim=-1)
                                                        / s_norm).median())
            out[f"{name}_lam{lam}"] = row
            print(f"A {name:8s} lam {lam:3.1f} active {row['active_share']:.2f}  antisym median "
                  f"coord {row['coordinate']['median']:.3g} metric {row['metric']['median']:.3g} "
                  f"conj {row['conjugate']['median']:.3g}  |metric-conj|/|s| {row['metric_vs_conjugate_rel']:.3g}"
                  f"  |coord-conj|/|s| {row['coordinate_vs_conjugate_rel']:.3g}", flush=True)
    return out, maps, axis.numpy()


def lambdas(steps=400, lam_max=5.0, lam_min=1e-3):
    return torch.exp(torch.linspace(np.log(lam_max), np.log(lam_min), steps + 1))


def reverse_sde(score_fn, count, generator, steps=400):
    lams = lambdas(steps)
    y = lams[0] * torch.randn(count, 2, generator=generator)
    for i in range(steps):
        l0, l1 = float(lams[i]), float(lams[i + 1])
        dvar = l0 * l0 - l1 * l1
        y = y + dvar * score_fn(y, l0 * l0) + np.sqrt(dvar) * torch.randn(count, 2, generator=generator)
        y = torch.nan_to_num(y, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    return y


def sliced_w1(a, b, generator, directions=64):
    theta = torch.rand(directions, generator=generator) * np.pi
    u = torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    pa, pb = (a @ u.T).sort(0).values, (b @ u.T).sort(0).values
    n = min(pa.shape[0], pb.shape[0])
    idx_a = torch.linspace(0, pa.shape[0] - 1, n).long()
    idx_b = torch.linspace(0, pb.shape[0] - 1, n).long()
    return float((pa[idx_a] - pb[idx_b]).abs().mean())


def stats(x, ref, generator, b=None):
    ok = torch.isfinite(x).all(-1) & (x.abs() < 1e5).all(-1)
    x = x[ok]
    c, cr = torch.cov(x.T), torch.cov(ref.T)
    corr = lambda m: float(m[0, 1] / torch.sqrt(m[0, 0] * m[1, 1]))  # noqa: E731
    out = {"mean_err": float((x.mean(0) - ref.mean(0)).norm()), "cov_err": float((c - cr).norm()),
           "corr": corr(c), "corr_ref": corr(cr), "sw1": sliced_w1(x, ref, generator),
           "runaway": float(1 - ok.float().mean() + ((x.abs() > 6).any(-1)).float().sum() / ok.numel())}
    if b is not None:
        out["outside_box"] = float(((x.abs() > b).any(-1)).float().mean())
        out["outside_box_ref"] = float(((ref.abs() > b).any(-1)).float().mean())
    return out


def corruption(y, lam2, radius=2.5, strength=3.0):
    """A non-conservative tail error growing outside `radius` (network extrapolation)."""
    r = y.norm(dim=-1, keepdim=True)
    grow = strength * (r - radius).clamp_min(0.0) ** 2
    rot = torch.stack([-y[:, 1], y[:, 0]], -1) / r.clamp_min(1e-9)
    return grow * (0.5 * y / r.clamp_min(1e-9) + rot)


def part_bc(count=10_000):
    out = {}
    gen = torch.Generator().manual_seed(0)
    for name, target in TARGETS.items():
        ref = target.sample(50_000, gen)
        floor = {}
        for scenario, boxes, corrupt in (("exact", (1.0, 1.5, 2.5), None), ("corrupted", (3.0,), corruption)):
            for b in boxes:
                lower, upper = -b * torch.ones(2), b * torch.ones(2)
                for kind in ("none", "coordinate", "metric", "conjugate"):
                    if kind == "none" and (scenario, "none") in floor:
                        out[f"{name}_{scenario}_b{b}_{kind}"] = floor[(scenario, "none")]
                        continue
                    if kind == "conjugate" and scenario == "corrupted":
                        continue   # the conjugate needs the true potential; not available for a corrupted field

                    def score_fn(y, lam2, kind=kind, lower=lower, upper=upper):
                        s = target.score(y, lam2)
                        if corrupt is not None:
                            s = s + corrupt(y, lam2)
                        if kind == "none":
                            return s
                        m = y + lam2 * s
                        if kind == "metric" and corrupt is not None:
                            # the library's metric is the (symmetrised) Jacobian of the field it has
                            jac = torch.eye(2) + lam2 * cc.jacobian_of(
                                lambda v: target.score(v, lam2) + corrupt(v, lam2), y)
                            jac = 0.5 * (jac + jac.mT)
                            vals, vecs = torch.linalg.eigh(jac)
                            jac = (vecs * vals.clamp(1e-3, 1.0).unsqueeze(-2)) @ vecs.mT
                            z = cc.metric_clamp(m, jac, lower, upper)
                        else:
                            z = clamp(kind, target, lam2, y, m, lower, upper)
                        return (z - y) / lam2

                    t0 = time.time()
                    g = torch.Generator().manual_seed(1)
                    x = reverse_sde(score_fn, count, g)
                    row = stats(x, ref, torch.Generator().manual_seed(2), b)
                    row["seconds"] = time.time() - t0
                    out[f"{name}_{scenario}_b{b}_{kind}"] = row
                    if kind == "none":
                        floor[(scenario, "none")] = row
                    print(f"{'B' if scenario == 'exact' else 'C'} {name:8s} {scenario:9s} b={b:3.1f} {kind:10s} "
                          f"meanErr {row['mean_err']:.3f} covErr {row['cov_err']:.3f} corr {row['corr']:.3f} "
                          f"(ref {row['corr_ref']:.3f}) SW1 {row['sw1']:.3f} outsideB {row['outside_box']:.3f} "
                          f"(ref {row['outside_box_ref']:.3f}) runaway {row['runaway']:.4f}  {row['seconds']:.0f}s",
                          flush=True)
    return out


def figures(a_maps, axis, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#1f1f1e", "#6b6a63"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    vmax = max(np.nanmax(v) for v in a_maps.values())
    for ax, kind in zip(axes, ("coordinate", "metric", "conjugate")):
        im = ax.imshow(a_maps[kind].T, origin="lower", extent=[axis[0], axis[-1], axis[0], axis[-1]],
                       cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{kind} clamp", color=ink, fontsize=10)
        ax.set_xlabel("y1", color=muted)
        ax.set_ylabel("y2", color=muted)
    fig.colorbar(im, ax=axes, label="||J - J^T|| / ||J|| (clamped score)")
    fig.suptitle("mixture target, lambda 0.7, box [-1,1]^2: non-conservativity where the clamp is active",
                 color=ink, fontsize=10)
    fig.savefig(HERE / "figures" / "clamp_antisymmetry_map.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {"coordinate": "#eda100", "metric": "#2a78d6", "conjugate": "#1a9e5c"}
    for ax, name in zip(axes, TARGETS):
        lams = (0.3, 0.7, 1.5)
        for kind in ("coordinate", "metric", "conjugate"):
            med = [results["A"][f"{name}_lam{l}"][kind]["median"] for l in lams]
            q95 = [results["A"][f"{name}_lam{l}"][kind]["q95"] for l in lams]
            ax.plot(lams, med, "o-", color=colors[kind], label=f"{kind} (median)")
            ax.plot(lams, q95, "o:", color=colors[kind], alpha=0.6)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("lambda", color=ink)
        ax.set_ylabel("antisymmetry of the clamped score (median ─, 95% ┄)", color=ink, fontsize=8)
        ax.set_title(name, color=ink)
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(HERE / "figures" / "clamp_antisymmetry_vs_lambda.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    kinds = ("none", "coordinate", "metric", "conjugate")
    for ax, name in zip(axes, TARGETS):
        boxes = (1.0, 1.5, 2.5)
        width = 0.2
        for k, kind in enumerate(kinds):
            vals = [results["BC"][f"{name}_exact_b{b}_{kind}"]["sw1"] for b in boxes]
            ax.bar(np.arange(3) + (k - 1.5) * width, vals, width, label=kind,
                   color={"none": "#bdbbb2", **colors}[kind])
        ax.set_xticks(range(3), [f"b = {b}" for b in boxes])
        ax.set_ylabel("sliced W1 to exact draws", color=ink)
        ax.set_title(f"{name}: exact score, clamped into [-b, b]^2", color=ink, fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(HERE / "figures" / "clamp_sampling_bias.png", dpi=150)


if __name__ == "__main__":
    results = {}
    results["A"], maps, axis = part_a()
    results["BC"] = part_bc()
    (HERE / "results" / "clamp_study.json").write_text(json.dumps(results, indent=2) + "\n")
    figures(maps, axis, results)
    print("wrote results/clamp_study.json and figures/clamp_*.png")
