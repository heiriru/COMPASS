"""Network or rule? Per-group global score of the attn_hier network vs the exact one.

The attn_hier network's g output (g never attends to l) is the per-group marginal score
s_i(g; lam) = grad log p_lam(g | x_i), the diffused single-group posterior. Here it is exact:

    s*_i(g; lam) = E_w[g' - g] / lam^2,   g' = g + lam * eps,  w ~ p(x_i | g') p(g')

(Tweedie; the Gaussian proposal cancels against the convolution kernel), with antithetic
eps pairs and the quadrature likelihood of reference.py. Evaluated at the points the sampler
visits at noise lam: reference global draws + lam * noise, on the max(sigma) <= 0.6 datasets.

Reported per lam, in probit z coordinates:
  rel_err      median ||s_net - s*|| / ||l*||, per group, l* = s* - s_prior,t (the part composed)
  shift_N      ||sum_i (s_net,i - s*_i)|| * sd_post: how far, in composed-posterior sd, the summed
               per-group error would move the composed posterior (>~0.3 = the network limits)
  shift_sys    the same for the mean error times N (the systematic part, grows like N)
  mc_err       the oracle's own Monte-Carlo error (split halves), same units as rel_err
"""
import os, sys, json, math
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:3]))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "3"
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np, torch
import task as T
import compass_arms as CA
from reference import Reference, physical, _log_diff_ndtr, LOG_2PI

DEV = "cuda"
NETWORK = sys.argv[1] if len(sys.argv) > 1 else "attn_hier"
NS = [int(a) for a in sys.argv[2:]] or [10, 25, 100]
LAMS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
POINTS, M = 48, 2048          # evaluation points per dataset, antithetic pairs per point
# SCORE_CHECK_REGIME=full: full-prior datasets; SCORE_CHECK_POINTS=prior: evaluation points from
# the prior N(0,1) instead of the composed posterior (where the training data is dense).
REGIME = os.environ.get("SCORE_CHECK_REGIME", "smax")
POINT_SOURCE = os.environ.get("SCORE_CHECK_POINTS", "posterior")
REF = HERE / "artifacts" / ("reference_smax0.6" if REGIME == "smax" else "reference")
TRAIN_SCALE = CA.NETWORKS[NETWORK].get("train_scale", 1.0)


def per_group_loglik(ref, z):
    """log p(x_i | g(z)) for every group: (N, B)."""
    n, k = ref.logw.shape
    mu, sigma = physical(z)
    log_z = _log_diff_ndtr((-1 - mu) / sigma, (1 - mu) / sigma)
    const = -(torch.log(sigma) + 0.5 * LOG_2PI + log_z).sum(-1) - 0.5 * ((mu / sigma) ** 2).sum(-1)
    coeff = torch.cat([-0.5 / sigma ** 2, mu / sigma ** 2], dim=-1)
    log_tn = ref.features @ coeff.T.to(ref.features.dtype) + ref.logw_flat[:, None]
    return torch.logsumexp(log_tn.reshape(n, k, -1), dim=1).to(z.dtype) + const[None]


def exact_scores(ref, g, lam, gen):
    """s*_i(g; lam) for points g (P, 4): (N, P, 4) plus a split-half MC error estimate."""
    n = ref.logw.shape[0]
    out, halves = [], []
    for p in range(g.shape[0]):
        eps = torch.randn(M, 4, generator=gen, device=DEV, dtype=torch.float64)
        eps = torch.cat([eps, -eps])                                        # antithetic
        gp = g[p] + lam * eps                                               # (2M, 4)
        loglik = torch.cat([per_group_loglik(ref, gp[s:s + 512]) for s in range(0, len(gp), 512)], 1)
        logw = loglik - 0.5 * (gp / TRAIN_SCALE).pow(2).sum(-1)[None]
        w = torch.softmax(logw, dim=1)                                      # (N, 2M)
        out.append((w @ eps) / lam)                                         # (N, 4)
        idx = torch.arange(M, device=DEV)
        half = []
        for sel in (torch.cat([idx[: M // 2], idx[: M // 2] + M]), torch.cat([idx[M // 2:], idx[M // 2:] + M])):
            ws = torch.softmax(logw[:, sel], dim=1)
            half.append((ws @ eps[sel]) / lam)
        halves.append((half[0] - half[1]).norm(dim=-1) / 2)                 # (N,)
    return torch.stack(out, 1), torch.stack(halves, 1)


@torch.no_grad()
def network_scores(model, g, l, xnet, lam):
    """Network g-score for every group at points g (P, 4): (N, P, 4)."""
    n, p = xnet.shape[0], g.shape[0]
    nodes = torch.cat([g[None].expand(n, p, 4), l, xnet[:, None].expand(n, p, 2)], -1).reshape(n * p, 8)
    c = torch.zeros_like(nodes)
    c[:, 6:] = 1.0
    t = model.sde.time_of_lambda(torch.tensor([lam], device=DEV, dtype=torch.float32)).expand(n * p, 1)
    out = model.output_scale_function(t, model.model(x=nodes.float(), t=t, c=c.float()))
    return out[:, :4].double().reshape(n, p, 4)


def main():
    model = CA.load(NETWORK, DEV, 100_000, 200)
    model.model.to(DEV)
    scale = T.normalization()["local_scale"]
    results = {}
    for n in NS:
        for lam in LAMS:
            rows = []
            for ds in range(10):
                d = np.load(REF / f"N{n}_{ds}.npz", allow_pickle=True)
                ref = Reference(d["x"], device=DEV)
                gen = torch.Generator(device=DEV).manual_seed(1000 * ds + n)
                rng = np.random.default_rng(ds)
                pick = rng.choice(len(d["ref_g"]), POINTS, replace=False)
                z_ref = torch.as_tensor(T.globals_to_z(d["ref_g"]), device=DEV)
                sd_post = z_ref.std(0)                                       # composed posterior sd
                base = z_ref[pick] if POINT_SOURCE == "posterior" else \
                    torch.randn(POINTS, 4, generator=gen, device=DEV, dtype=torch.float64)
                g = base + lam * torch.randn(POINTS, 4, generator=gen, device=DEV, dtype=torch.float64)
                l = torch.as_tensor(d["ref_l"][pick].transpose(1, 0, 2) / scale, device=DEV)
                l = l + lam * torch.randn(l.shape, generator=gen, device=DEV, dtype=torch.float64)
                xnet = torch.as_tensor(T.x_to_net(d["x"]), device=DEV)
                s_exact, mc = exact_scores(ref, g, lam, gen)                 # (N, P, 4)
                s_net = network_scores(model, g, l, xnet, lam)
                s_prior = -g / (TRAIN_SCALE ** 2 + lam ** 2)                 # diffused training prior
                lik = s_exact - s_prior[None]                                # the composed part
                err = s_net - s_exact
                rel = (err.norm(dim=-1) / lik.norm(dim=-1).clamp_min(1e-6)).median()
                mc_rel = (mc / lik.norm(dim=-1).clamp_min(1e-6)).median()
                shift_n = (err.sum(0) * sd_post).norm(dim=-1).median()
                shift_sys = (err.mean((0, 1)) * n * sd_post).norm()
                shift_lik = (lik.sum(0) * sd_post).norm(dim=-1).median()
                per_coord = (err.mean((0, 1)) * n * sd_post).tolist()
                rows.append([float(rel), float(mc_rel), float(shift_n), float(shift_sys), float(shift_lik)] + per_coord)
            r = np.median(np.array(rows), 0)
            results[f"N{n}_lam{lam}"] = r.tolist()
            print(f"N={n:<3} lam={lam:<5} rel_err={r[0]:.3f} (oracle mc {r[1]:.3f})  "
                  f"shift_N={r[2]:.2f} sd  shift_sys={r[3]:.2f} sd  [exact composed lik score: {r[4]:.1f} sd]  "
                  f"sys per coord (mu0,mu1,s0,s1)={np.round(r[5:], 2)}", flush=True)
    out = HERE / "artifacts_smax0.6" / f"score_check_{NETWORK}_{REGIME}_{POINT_SOURCE}.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
