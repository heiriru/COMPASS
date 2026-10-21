"""Metrics, grouped by the question each one answers.

Every metric compares a method's joint draws (globals (S, 4) in physical units,
locals (S, N, 2)) with the reference draws for the same dataset. The
``reference`` arm is 500 extra reference draws scored the same way, which is
the Monte-Carlo floor of every number here.

Q1  local posterior p(theta_i | x_1:N), incl. its nonlinear / bimodal structure
    local_c2st         kNN classifier two-sample test per local, 2-D (0.5 = equal)
    local_w1           mean 1-D W1 / reference sd over both coordinates
    branch_err         |P(theta_i on moon +) - P_ref|: the two crescents' mass split
    crescent_dev       median |r - 0.1| / 0.01 when theta_i is mapped back through
                       its own x_i (reference-free; exact posterior: 0.67)
    off_support        fraction of draws that cannot have produced x_i (u < 0 or
                       outside [-1, 1]^2); exact posterior: 0
    local_mean_err / local_width   mean error and sd ratio vs the reference
Q2  population mean mu_g        mu_mean_err, mu_width, mu_w1, mu_cover90
Q3  population spread sigma_g   sigma_mean_err, sigma_width, sigma_w1, sigma_cover90
    (mean_err = |mean - ref mean| / ref sd, width = sd / ref sd, w1 = W1 / ref sd,
     cover90 = truth inside the central 90% interval, averaged over datasets)
    global_c2st        joint 4-D C2ST of the globals
    The task is exactly symmetric under g -> (-mu_1, -mu_0, sigma_1, sigma_0), so the
    global posterior has two mirror modes. In u = (mu_0+mu_1)/sqrt2, v = (mu_1-mu_0)/sqrt2,
    w = sigma_0-sigma_1, s = sigma_0+sigma_1:
    sym_{u,v,w,s}_w1   W1 / ref sd per coordinate (u, w bimodal; v, s invariant)
    sym_inv_w1         mean W1 of the mirror-invariant summaries |u|, v, |w|, s:
                       the global posterior with the mode label removed
    mode_balance       P(u > 0) (exact: 0.5); mode_balance_err = |that - reference|
Q4  dependence between globals and locals
    dep_corr_err       mean |corr(g_k, theta_i,d) - corr_ref| over the 4x2 block and i
    dep_corr_ref       mean |corr_ref| (how much dependence there is to get right)
    joint_c2st         C2ST on (g, theta_i), 6-D, vs the reference joint
    joint_c2st_indep   same, after shuffling theta_i across draws (dependence removed);
                       a method that ignores the dependence scores like this
Q5  behaviour in N: every metric above, per N (tables and plots).
"""
from __future__ import annotations

import numpy as np
import torch

import task as T

KNN_K = 10
FOLDS = 5
MAX_LOCALS = 25          # locals scored by the C2STs (all locals for the rest)


def c2st(a, b, seed=0, k=KNN_K, folds=FOLDS):
    """Balanced kNN two-sample classifier accuracy (5-fold CV)."""
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    X = np.concatenate([a, b]).astype(np.float64)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    order = rng.permutation(2 * n)
    X, y = torch.as_tensor(X[order]), torch.as_tensor(y[order])
    correct = 0
    for fold in range(folds):
        test = torch.arange(2 * n) % folds == fold
        d = torch.cdist(X[test], X[~test])
        idx = d.topk(k, largest=False).indices
        vote = y[~test][idx].mean(1)
        pred = (vote > 0.5).double()
        ties = vote == 0.5
        pred[ties] = torch.as_tensor(rng.integers(0, 2, int(ties.sum())), dtype=torch.double)
        correct += int((pred == y[test]).sum())
    return correct / (2 * n)


def w1(a, b):
    """1-D Wasserstein-1 between two samples (quantile matching)."""
    q = np.linspace(0.0, 1.0, 513)[1:-1]
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def _corr_block(g, l):
    """corr(g_k, l_d): (4, 2)."""
    x = np.concatenate([g, l], axis=1)
    c = np.corrcoef(x.T)
    return np.nan_to_num(c[:4, 4:])


def evaluate(draws_g, draws_l, ref_g, ref_l, true_g, true_l, x, seed=0):
    """All metrics for one dataset. Arrays: (S,4), (S,N,2), (R,4), (R,N,2), (4,), (N,2), (N,2)."""
    out = {}
    finite = np.isfinite(draws_g).all(1) & np.isfinite(draws_l).all((1, 2))
    out["nonfinite"] = float(1 - finite.mean())
    draws_g, draws_l = draws_g[finite], draws_l[finite]
    n_local = ref_l.shape[1]

    # ---- Q2 / Q3: globals ------------------------------------------------
    ref_mean, ref_sd = ref_g.mean(0), ref_g.std(0, ddof=1)
    mean_err = np.abs(draws_g.mean(0) - ref_mean) / ref_sd
    width = draws_g.std(0, ddof=1) / ref_sd
    w1s = np.array([w1(draws_g[:, k], ref_g[:, k]) for k in range(4)]) / ref_sd
    lo, hi = np.quantile(draws_g, [0.05, 0.95], axis=0)
    cover = ((lo <= true_g) & (true_g <= hi)).astype(float)
    for name, sl in (("mu", slice(0, 2)), ("sigma", slice(2, 4))):
        out[f"{name}_mean_err"] = float(mean_err[sl].mean())
        out[f"{name}_width"] = float(width[sl].mean())
        out[f"{name}_w1"] = float(w1s[sl].mean())
        out[f"{name}_cover90"] = float(cover[sl].mean())
        out[f"{name}_rmse_truth"] = float(np.sqrt(((draws_g[:, sl] - true_g[sl]) ** 2).mean()))
    out["global_per_param"] = {"mean_err": mean_err.tolist(), "width": width.tolist(),
                               "w1": w1s.tolist()}
    out["global_c2st"] = c2st(draws_g, ref_g, seed)
    # Mirror-adapted globals: u, w flip sign under the exact symmetry (two modes),
    # v, s are invariant. mode_balance = P(u > 0); the exact posterior has 0.5.
    sym_m, sym_r = T.symmetric_globals(draws_g), T.symmetric_globals(ref_g)
    sym_sd = sym_r.std(0, ddof=1)
    for k, name in enumerate(("u", "v", "w", "s")):
        out[f"sym_{name}_w1"] = float(w1(sym_m[:, k], sym_r[:, k]) / sym_sd[k])
    out["sym_inv_w1"] = float(np.mean([w1(np.abs(sym_m[:, k]), np.abs(sym_r[:, k])) /
                                       np.abs(sym_r[:, k]).std(ddof=1) for k in (0, 2)]
                                      + [out["sym_v_w1"], out["sym_s_w1"]]))
    out["mode_balance"] = float(np.mean(sym_m[:, 0] > 0))
    out["mode_balance_err"] = float(abs(out["mode_balance"] - np.mean(sym_r[:, 0] > 0)))

    # ---- Q1: locals ------------------------------------------------------
    scored = np.arange(n_local) if n_local <= MAX_LOCALS else \
        np.random.default_rng(seed).choice(n_local, MAX_LOCALS, replace=False)
    out["local_c2st"] = float(np.mean([c2st(draws_l[:, i], ref_l[:, i], seed + i)
                                       for i in scored]))
    lref_mean, lref_sd = ref_l.mean(0), np.maximum(ref_l.std(0, ddof=1), 1e-9)   # (N, 2)
    out["local_mean_err"] = float(np.median(np.abs(draws_l.mean(0) - lref_mean) / lref_sd))
    out["local_width"] = float(np.median(draws_l.std(0, ddof=1) / lref_sd))
    out["local_w1"] = float(np.mean([w1(draws_l[:, i, d], ref_l[:, i, d]) / lref_sd[i, d]
                                     for i in range(n_local) for d in range(2)]))
    out["branch_err"] = float(np.mean(np.abs(T.branch(draws_l).mean(0)
                                             - T.branch(ref_l).mean(0))))
    dev, valid = T.crescent_residual(draws_l, x[None])
    out["crescent_dev"] = float(np.median(dev))
    out["off_support"] = float(1 - valid.mean())
    out["local_rmse_truth"] = float(np.sqrt(((draws_l - true_l) ** 2).sum(-1).mean()))

    # ---- Q4: dependence ----------------------------------------------------
    errs, refs, joint, indep = [], [], [], []
    rng = np.random.default_rng(seed + 7)
    for i in range(n_local):
        c_ref = _corr_block(ref_g, ref_l[:, i])
        c_m = _corr_block(draws_g, draws_l[:, i])
        errs.append(np.abs(c_m - c_ref).mean())
        refs.append(np.abs(c_ref).mean())
    for i in scored:
        ref_joint = np.concatenate([ref_g, ref_l[:, i]], 1)
        joint.append(c2st(np.concatenate([draws_g, draws_l[:, i]], 1), ref_joint, seed + i))
        shuffled = np.concatenate([draws_g, draws_l[rng.permutation(len(draws_l)), i]], 1)
        indep.append(c2st(shuffled, ref_joint, seed + i))
    out["dep_corr_err"] = float(np.mean(errs))
    out["dep_corr_ref"] = float(np.mean(refs))
    out["joint_c2st"] = float(np.mean(joint))
    out["joint_c2st_indep"] = float(np.mean(indep))
    return out


SUMMARY_KEYS = (
    "local_c2st", "local_w1", "branch_err", "crescent_dev", "off_support",
    "local_mean_err", "local_width", "local_rmse_truth",
    "mu_mean_err", "mu_width", "mu_w1", "mu_cover90", "mu_rmse_truth",
    "sigma_mean_err", "sigma_width", "sigma_w1", "sigma_cover90", "sigma_rmse_truth",
    "global_c2st", "sym_u_w1", "sym_v_w1", "sym_w_w1", "sym_s_w1", "sym_inv_w1",
    "mode_balance", "mode_balance_err", "dep_corr_err", "dep_corr_ref", "joint_c2st", "joint_c2st_indep",
    "nonfinite", "seconds",
)
MEAN_KEYS = {"mu_cover90", "sigma_cover90"}     # coverage is a rate: average it


def summarize(rows):
    ok = [r for r in rows if "local_c2st" in r]
    summary = {"datasets": len(rows), "failed": len(rows) - len(ok)}
    for key in SUMMARY_KEYS:
        values = [r[key] for r in ok if key in r]
        if values:
            summary[key] = float(np.mean(values) if key in MEAN_KEYS else np.median(values))
    return summary
