"""Analyses and figures of the four WAIC-improvement experiments (IMPROVEMENTS.md).

    .COMPASS/bin/python -m validation.a4_waic.improvements exp1|exp2|exp3a|exp3b|exp4

Reads the saved run records only (results/compass, results/reference); each
sub-command prints its numbers, writes results/improvements_<exp>.json and figures to
results/figures/.
"""

import validation.common.runtime  # noqa: F401

import argparse
import json
import os

import numpy as np

from validation.a4_waic import analyze, common
from validation.a4_waic.families import family
from validation.common import plotting
from validation.common.plotting import EXACT, FOURTH, INK2, LEGACY, NEW, plt

FIG = os.path.join(common.RESULTS, "figures")


def save_json(name, obj):
    with open(os.path.join(common.RESULTS, f"improvements_{name}.json"), "w") as f:
        json.dump(obj, f, indent=1)


def auc(score, label):
    """Area under the ROC curve (probability a positive outranks a negative)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None]).mean()
    ties = (pos[:, None] == neg[None]).mean()
    return float(greater + 0.5 * ties)


def paired(family_name, compass_tag, ref_tag, ref_key="mcmc"):
    """[(true, rep, compass record, reference record)] with both present."""
    out = []
    for (true, rep), d in sorted(analyze.merged(family_name, compass_tag, ref_tag).items()):
        if "compass" in d and "ref" in d and all(
                ref_key in d["ref"]["models"].get(n, {}) for n in d["compass"]["models"]):
            out.append((true, rep, d["compass"], d["ref"]))
    return out


# ---------------------------------------------------------------------------
# shared: COMPASS vs MCMC criterion comparison (exp 1, exp 3b, exp 4)
# ---------------------------------------------------------------------------

def _softmax_weights(ic):
    ic = np.asarray(ic, float)
    w = np.exp(-(ic - ic.min()) / 2)
    return w / w.sum()


def criterion_table(family_name, compass_tag, ref_tag, crit="waic"):
    """Per dataset: COMPASS and MCMC values of ``crit`` (waic | loo) for every model."""
    names = list(family(family_name))
    out = []
    for true, rep, c, r in paired(family_name, compass_tag, ref_tag):
        cv = np.array([c["models"][n][crit] for n in names])
        rv = np.array([r["models"][n]["mcmc"][crit] for n in names])
        out.append(dict(true=true, rep=rep, compass=cv, mcmc=rv,
                        p_compass=np.array([c["models"][n]["p_waic"] for n in names]),
                        p_mcmc=np.array([r["models"][n]["mcmc"]["p_waic"] for n in names]),
                        k_compass=np.array([c["models"][n].get("k_max", np.nan) for n in names]),
                        k_mcmc=np.array([r["models"][n]["mcmc"].get("k_max", np.nan) for n in names]),
                        nk_compass=np.array([c["models"][n].get("n_k_high", np.nan) for n in names]),
                        nk_mcmc=np.array([r["models"][n]["mcmc"].get("n_k_high", np.nan) for n in names]),
                        n_rows=c["n_rows"]))
    return names, out


def criterion_summary(names, table):
    ti = np.array([names.index(t["true"]) for t in table])
    c = np.array([t["compass"] for t in table])
    m = np.array([t["mcmc"] for t in table])
    pc, pm = c.argmin(1), m.argmin(1)
    err = c - m
    true_err = err[np.arange(len(ti)), ti]
    other = ~np.eye(len(names), dtype=bool)[ti]
    gap_m = np.sort(m, 1)[:, 1] - m.min(1)
    wc = np.array([_softmax_weights(x) for x in c])
    wm = np.array([_softmax_weights(x) for x in m])
    hit = np.eye(len(names))[ti]
    per_true = {}
    for j, n in enumerate(names):
        s = ti == j
        if s.any():
            per_true[n] = dict(n=int(s.sum()), acc_compass=float((pc[s] == j).mean()),
                               acc_mcmc=float((pm[s] == j).mean()),
                               agree=float((pc[s] == pm[s]).mean()),
                               true_err_median=float(np.median(true_err[s])),
                               compass_picks={names[k]: int((pc[s] == k).sum()) for k in range(len(names))},
                               mcmc_picks={names[k]: int((pm[s] == k).sum()) for k in range(len(names))})
    rel_c = analyze.reliability(wc.ravel(), hit.ravel())
    rel_m = analyze.reliability(wm.ravel(), hit.ravel())
    return dict(n=len(table), acc_compass=float((pc == ti).mean()), acc_mcmc=float((pm == ti).mean()),
                agree=float((pc == pm).mean()),
                agree_gap_gt4=float((pc == pm)[gap_m > 4].mean()) if (gap_m > 4).any() else None,
                true_err_median=float(np.median(true_err)),
                true_err_q=[float(np.quantile(true_err, q)) for q in (0.1, 0.9)],
                true_err_per_row_median=float(np.median(true_err / np.array([t["n_rows"] for t in table]))),
                other_err_median=float(np.median(err[other])),
                other_err_q=[float(np.quantile(err[other], q)) for q in (0.1, 0.9)],
                p_true_compass_median=float(np.median(np.array([t["p_compass"] for t in table])[np.arange(len(ti)), ti])),
                p_true_mcmc_median=float(np.median(np.array([t["p_mcmc"] for t in table])[np.arange(len(ti)), ti])),
                ece_compass=rel_c["ece"], ece_compass_null95=rel_c["ece_null_q95"],
                ece_mcmc=rel_m["ece"], ece_mcmc_null95=rel_m["ece_null_q95"],
                per_true=per_true), dict(c=c, m=m, ti=ti, pc=pc, pm=pm, wc=wc, wm=wm,
                                         rel_c=rel_c, rel_m=rel_m)


def criterion_figure(names, arrays, title, fname, crit="WAIC"):
    plotting.style()
    c, m, ti, pc, pm = (arrays[k] for k in ("c", "m", "ti", "pc", "pm"))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    ax = axs[0]
    for j in range(len(names)):
        own = ti == j
        ax.scatter(m[own, j], c[own, j], s=12, color=NEW, label="true model" if j == 0 else None)
        ax.scatter(m[~own, j], c[~own, j], s=8, color=LEGACY, alpha=0.5, label="other models" if j == 0 else None)
    lo, hi = min(c.min(), m.min()), max(c.max(), m.max())
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
    ax.set_yscale("symlog", linthresh=100); ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel(f"MCMC {crit} (exact likelihood)"); ax.set_ylabel(f"COMPASS {crit}")
    ax.legend(fontsize=8); ax.set_title(f"{crit} values")
    for ax, pick, lab in ((axs[1], pc, "COMPASS"), (axs[2], pm, "MCMC")):
        mat = np.zeros((len(names), len(names)))
        for t, p in zip(ti, pick):
            mat[t, p] += 1
        mat /= np.maximum(mat.sum(1, keepdims=True), 1)
        ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
        for a in range(len(names)):
            for b in range(len(names)):
                ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[a, b] > 0.6 else "black")
        ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
        ax.set_xlabel(f"{lab} pick"); ax.set_ylabel("true model")
        ax.set_title(f"{lab}: accuracy {np.mean(pick == ti):.2f}")
    ax = axs[3]
    for rel, col, lab in ((arrays["rel_c"], NEW, "COMPASS"), (arrays["rel_m"], EXACT, "MCMC")):
        b = rel["bins"]
        ax.errorbar([x["mean_prob"] for x in b], [x["freq"] for x in b],
                    yerr=[[x["freq"] - x["lo"] for x in b], [x["hi"] - x["freq"] for x in b]],
                    fmt="o-", ms=3, color=col, label=f"{lab} ECE {rel['ece']:.3f} (null95 {rel['ece_null_q95']:.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("model weight"); ax.set_ylabel("frequency true"); ax.legend(fontsize=7)
    ax.set_title("calibration")
    fig.suptitle(title)
    fig.tight_layout()
    print(plotting.save(fig, FIG, fname))


def _clean(obj):
    return json.loads(json.dumps(obj, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


# ---------------------------------------------------------------------------
# exp 1: tutorial with observation noise
# ---------------------------------------------------------------------------

def exp1(compass_tag="trained_n131072_s0_d500_hutch", ref_tag="ref"):
    names, table = criterion_table("tutorial", compass_tag, ref_tag, "waic")
    summary, arrays = criterion_summary(names, table)
    print(json.dumps(summary, indent=1))
    save_json("exp1", _clean(dict(summary=summary, table=table)))
    criterion_figure(names, arrays, "tutorial + noise: COMPASS WAIC vs MCMC WAIC", "fig_exp1_tutorial")


# ---------------------------------------------------------------------------
# exp 2: offset family, seed ensemble and 4x budget
# ---------------------------------------------------------------------------

def _offset_values(tag, ref_tag="ref"):
    """{(true, rep): (compass WAIC [M0, M1], exact WAIC [M0, M1])}."""
    out = {}
    for (true, rep), d in analyze.merged("offset", tag, ref_tag).items():
        if "compass" in d and "ref" in d:
            out[(true, rep)] = (np.array([d["compass"]["models"][n]["waic"] for n in ("M0", "M1")]),
                                np.array([d["ref"]["models"][n]["quad"]["waic"] for n in ("M0", "M1")]))
    return out


def _offset_stats(keys, c, e):
    """c, e: (D, 2) COMPASS and exact WAIC; keys give the true model."""
    true = np.array([k[0] == "M1" for k in keys]).astype(int)
    dc, de = c[:, 1] - c[:, 0], e[:, 1] - e[:, 0]
    err = dc - de
    n = len(err)
    wc = 1 / (1 + np.exp((dc) / 2))           # weight of M1
    wm = 1 / (1 + np.exp((de) / 2))
    rel = analyze.reliability(np.r_[wc, 1 - wc], np.r_[true, 1 - true])
    return dict(n=n, shift=float(err.mean()), shift_se=float(err.std(ddof=1) / np.sqrt(n)),
                sd=float(err.std(ddof=1)), rmse=float(np.sqrt((err ** 2).mean())),
                waic_err_M0=float((c[:, 0] - e[:, 0]).mean()), waic_err_M1=float((c[:, 1] - e[:, 1]).mean()),
                acc=float(((dc < 0).astype(int) == true).mean()),
                acc_exact=float(((de < 0).astype(int) == true).mean()),
                agree=float(((dc < 0) == (de < 0)).mean()),
                ece=rel["ece"], ece_null95=rel["ece_null_q95"])


def exp2(ref_tag="ref"):
    seeds = {s: _offset_values(f"e2_n65536_s{s}", ref_tag) for s in range(5)}
    big = _offset_values("e2_n262144_s0", ref_tag)
    keys = sorted(set.intersection(*(set(v) for v in seeds.values()), set(big)))
    e = np.array([seeds[0][k][1] for k in keys])
    c = {s: np.array([seeds[s][k][0] for k in keys]) for s in seeds}
    out = dict(keys=len(keys), per_seed={s: _offset_stats(keys, c[s], e) for s in seeds})
    out["seed_shift_sd"] = float(np.std([out["per_seed"][s]["shift"] for s in seeds], ddof=1))
    ens = np.mean([c[s] for s in seeds], 0)
    out["ensemble5"] = _offset_stats(keys, ens, e)
    out["budget4x"] = _offset_stats(keys, np.array([big[k][0] for k in keys]), e)
    # Floor: the same sampler and draws with exact scores (oracle level), all 400 datasets.
    orc = _offset_values("oracle_d1000", ref_tag)
    oe = np.array([(v[0][1] - v[0][0]) - (v[1][1] - v[1][0]) for v in orc.values()])
    out["oracle_dWAIC_err"] = dict(mean=float(oe.mean()), sd=float(oe.std(ddof=1)), n=len(oe))
    # Ensemble size curve: average over all subsets of size k.
    from itertools import combinations
    curve = []
    for k in range(1, 6):
        st = [_offset_stats(keys, np.mean([c[s] for s in sub], 0), e) for sub in combinations(range(5), k)]
        curve.append(dict(k=k, shift=float(np.mean([x["shift"] for x in st])),
                          abs_shift=float(np.mean([abs(x["shift"]) for x in st])),
                          sd=float(np.mean([x["sd"] for x in st])), rmse=float(np.mean([x["rmse"] for x in st])),
                          acc=float(np.mean([x["acc"] for x in st]))))
    out["ensemble_curve"] = curve
    # Correlation of per-dataset errors between seeds (shared vs seed-specific part).
    errs = np.array([(c[s][:, 1] - c[s][:, 0]) - (e[:, 1] - e[:, 0]) for s in seeds])
    cc = np.corrcoef(errs)
    out["seed_error_corr_mean"] = float(cc[np.triu_indices(5, 1)].mean())
    print(json.dumps(out, indent=1))
    save_json("exp2", out)

    plotting.style()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    ax = axs[0]
    labels = [f"seed {s}" for s in seeds] + ["5-seed\nensemble", "4x budget\n(seed 0)"]
    stats = [out["per_seed"][s] for s in seeds] + [out["ensemble5"], out["budget4x"]]
    cols = [NEW] * 5 + [FOURTH, EXACT]
    for i, (st, col) in enumerate(zip(stats, cols)):
        ax.errorbar(i, st["shift"], yerr=1.96 * st["shift_se"], fmt="o", color=col)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhspan(0.5, 0.9, color=FOURTH, alpha=0.12, label="pred. 2a ensemble")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("mean ΔWAIC(M1−M0) error (COMPASS − exact)"); ax.legend(fontsize=8)
    ax.set_title(f"shift; seed-to-seed sd {out['seed_shift_sd']:.2f}")
    ax = axs[1]
    ks = [x["k"] for x in curve]
    ax.plot(ks, [x["sd"] for x in curve], "o-", color=NEW, label="per-dataset error sd")
    ax.plot(ks, [x["rmse"] for x in curve], "s-", color=LEGACY, label="RMSE")
    ax.plot(ks, [x["abs_shift"] for x in curve], "^-", color=FOURTH, label="|shift|")
    ax.axhline(out["oracle_dWAIC_err"]["sd"], color=INK2, ls=":", lw=0.8, label="exact-score floor (sd)")
    for name, st, mk in (("4x budget", out["budget4x"], "*"),):
        ax.plot([1], [st["rmse"]], mk, ms=12, color=EXACT, label=f"{name} RMSE")
    ax.set_xlabel("ensemble size (seeds)"); ax.set_ylabel("nats"); ax.legend(fontsize=7)
    ax.set_title("error vs ensemble size")
    ax = axs[2]
    de = e[:, 1] - e[:, 0]
    ax.scatter(de, c[0][:, 1] - c[0][:, 0] - de, s=8, color=NEW, alpha=0.5, label="seed 0")
    ax.scatter(de, ens[:, 1] - ens[:, 0] - de, s=8, color=FOURTH, alpha=0.7, label="ensemble")
    ax.scatter(de, np.array([big[k][0] for k in keys]) @ [-1, 1] - de, s=8, color=EXACT, alpha=0.7, label="4x budget")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("exact ΔWAIC(M1−M0)"); ax.set_ylabel("COMPASS − exact"); ax.legend(fontsize=8)
    ax.set_title("per-dataset error")
    fig.suptitle(f"offset family, {len(keys)} datasets: seed ensemble and budget")
    fig.tight_layout()
    print(plotting.save(fig, FIG, "fig_exp2_ensemble_budget"))


# ---------------------------------------------------------------------------
# exp 3a: out-of-support flag from the recorded posterior draws
# ---------------------------------------------------------------------------

def support_rows(family_name, compass_tag, ref_tag):
    models = family(family_name)
    rows = []
    for true, rep, c, r in paired(family_name, compass_tag, ref_tag):
        for name, m in models.items():
            cm = c["models"][name]
            z = np.abs((np.array(cm["g_mean"]) - m.global_mean) / m.global_std)
            # Draw-level: share of the posterior beyond 3 training sds (Gaussian approx.
            # from the recorded per-dimension mean and sd of the global draws).
            sd = np.array(cm["g_sd"]) / m.global_std
            err = cm["waic"] - r["models"][name]["mcmc"]["waic"]
            rows.append(dict(true=true, rep=rep, model=name, z_max=float(z.max()),
                             z_rms=float(np.sqrt((z ** 2).mean())), sd_max=float(sd.max()),
                             err=float(err), correct=name == true))
    return rows


def exp3a(compass_tag="trained_n131072_s0_d500_hutch", ref_tag="ref_r2", threshold=10.0):
    rows = support_rows("population", compass_tag, ref_tag)
    z = np.array([x["z_max"] for x in rows])
    err = np.array([x["err"] for x in rows])
    big = np.abs(err) > threshold
    out = dict(n=len(rows), threshold=threshold, frac_big=float(big.mean()),
               auc_z_max=auc(z, big), auc_z_rms=auc([x["z_rms"] for x in rows], big),
               by_cut=[])
    for cut in (2.0, 3.0, 4.0):
        flag = z > cut
        out["by_cut"].append(dict(cut=cut, flagged=float(flag.mean()),
                                  recall=float(flag[big].mean()) if big.any() else None,
                                  precision=float(big[flag].mean()) if flag.any() else None,
                                  true_model_flagged=float(flag[[x["correct"] for x in rows]].mean())))
    print(json.dumps(out, indent=1))
    save_json("exp3a", dict(summary=out, rows=rows))

    plotting.style()
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.8))
    corr = np.array([x["correct"] for x in rows])
    ax.scatter(z[corr], np.abs(err[corr]) + 0.1, s=10, color=NEW, label="true model")
    ax.scatter(z[~corr], np.abs(err[~corr]) + 0.1, s=10, color=LEGACY, alpha=0.6, label="other models")
    ax.axhline(threshold, color=INK2, ls="--", lw=0.8)
    ax.axvline(3.0, color=INK2, ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("max_d |posterior mean of g_d − training prior mean| / prior sd")
    ax.set_ylabel("|COMPASS WAIC − MCMC WAIC| + 0.1")
    ax.set_title(f"population: AUC(|err| > {threshold:g}) = {out['auc_z_max']:.3f}")
    ax.legend(fontsize=8)
    print(plotting.save(fig, FIG, "fig_exp3a_support_flag"))


# ---------------------------------------------------------------------------
# exp 3b: wider training prior
# ---------------------------------------------------------------------------

def exp3b(compass_tag="trained_n131072_s0_d500_hutch", ref_tag="ref_r2"):
    out = {}
    arrays = {}
    for fam in ("population", "population_wide"):
        names, table = criterion_table(fam, compass_tag, ref_tag, "waic")
        summ, arr = criterion_summary(names, table)
        rows = support_rows(fam, compass_tag, ref_tag)
        z = np.array([x["z_max"] for x in rows])
        err = np.array([x["err"] for x in rows])
        corr = np.array([x["correct"] for x in rows])
        summ["other_abs_err_median"] = float(np.median(np.abs(err[~corr])))
        summ["other_abs_err_q90"] = float(np.quantile(np.abs(err[~corr]), 0.9))
        summ["true_abs_err_median"] = float(np.median(np.abs(err[corr])))
        summ["frac_abs_err_gt10"] = float((np.abs(err) > 10).mean())
        summ["frac_z_gt3"] = float((z > 3).mean())
        summ["frac_z_gt3_other"] = float((z[~corr] > 3).mean())
        out[fam] = summ
        arrays[fam] = (names, arr, z, err, corr)
        criterion_figure(names, arr, f"{fam}: COMPASS WAIC vs MCMC WAIC (restart references)",
                         f"fig_exp3b_{fam}")
    print(json.dumps({f: {k: v for k, v in s_.items() if k != "per_true"} for f, s_ in out.items()}, indent=1))
    for f in out:
        print(f, {t: (v["acc_compass"], v["acc_mcmc"], v["agree"]) for t, v in out[f]["per_true"].items()})
    save_json("exp3b", _clean(out))

    plotting.style()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, fam in zip(axs[:2], out):
        names, arr, z, err, corr = arrays[fam]
        ax.scatter(z[corr], np.abs(err[corr]) + 0.1, s=10, color=NEW, label="true model")
        ax.scatter(z[~corr], np.abs(err[~corr]) + 0.1, s=10, color=LEGACY, alpha=0.6, label="other models")
        ax.axhline(10, color=INK2, ls="--", lw=0.8); ax.axvline(3, color=INK2, ls=":", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("max_d |posterior mean − training prior mean| / training prior sd")
        ax.set_ylabel("|COMPASS WAIC − MCMC WAIC| + 0.1")
        ax.set_title(f"{fam}: agreement {out[fam]['agree']:.2f}"); ax.legend(fontsize=8)
    ax = axs[2]
    keys = ["acc_compass", "acc_mcmc", "agree"]
    xs = np.arange(len(keys))
    for i, (fam, col) in enumerate(zip(out, (LEGACY, NEW))):
        ax.bar(xs + (i - 0.5) * 0.38, [out[fam][k] for k in keys], 0.38, color=col, label=fam)
    ax.axhline(0.95, color=INK2, ls="--", lw=0.8, label="pred. agreement ≥ 0.95")
    ax.set_xticks(xs, ["COMPASS accuracy", "MCMC accuracy", "agreement"]); ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8); ax.set_title("decisions")
    fig.tight_layout()
    print(plotting.save(fig, FIG, "fig_exp3b_wide_prior"))


# ---------------------------------------------------------------------------
# exp 4: PSIS-LOO
# ---------------------------------------------------------------------------

def _decision(names, recs, true):
    v = np.array(recs)
    return names[int(v.argmin())] == true


def exp4():
    out = {}
    # (a) exact-scores and trained offset: COMPASS PSIS-LOO vs exact quadrature LOO.
    off = {}
    for tag in ["oracle_loo_d1000", "e2_n65536_s0", "e2_n65536_s1", "e2_n65536_s2", "e2_n65536_s3",
                "e2_n65536_s4", "e2_n262144_s0"]:
        d = analyze.merged("offset", tag, "ref_loo")
        el, ew, kk, nk, dl, dw, accl, accw, acce = [], [], [], [], [], [], [], [], []
        for (true, rep), v in d.items():
            if "compass" not in v or "ref" not in v:
                continue
            c, r = v["compass"]["models"], v["ref"]["models"]
            for n in ("M0", "M1"):
                el.append(c[n]["loo"] - r[n]["quad"]["loo"])
                ew.append(c[n]["waic"] - r[n]["quad"]["waic"])
                kk.append(c[n]["k_max"]); nk.append(c[n]["n_k_high"] / v["compass"]["n_rows"])
            dl.append((c["M1"]["loo"] - c["M0"]["loo"]) - (r["M1"]["quad"]["loo"] - r["M0"]["quad"]["loo"]))
            dw.append((c["M1"]["waic"] - c["M0"]["waic"]) - (r["M1"]["quad"]["waic"] - r["M0"]["quad"]["waic"]))
            accl.append((c["M1"]["loo"] < c["M0"]["loo"]) == (true == "M1"))
            accw.append((c["M1"]["waic"] < c["M0"]["waic"]) == (true == "M1"))
            acce.append((r["M1"]["quad"]["loo"] < r["M0"]["quad"]["loo"]) == (true == "M1"))
        el, ew, dl, dw = map(np.array, (el, ew, dl, dw))
        off[tag] = dict(n=len(dl), loo_err_mean=float(el.mean()), loo_err_sd=float(el.std()),
                        waic_err_mean=float(ew.mean()), waic_err_sd=float(ew.std()),
                        dloo_err_mean=float(dl.mean()), dloo_err_sd=float(dl.std()),
                        dwaic_err_mean=float(dw.mean()), dwaic_err_sd=float(dw.std()),
                        k_max_median=float(np.median(kk)), frac_rows_k_high=float(np.mean(nk)),
                        acc_loo=float(np.mean(accl)), acc_waic=float(np.mean(accw)),
                        acc_exact_loo=float(np.mean(acce)))
    out["offset"] = off
    # (b) exact level, offset + noise: exact LOO vs exact WAIC as criteria (quadrature).
    crit = {}
    for fam, names in (("offset", ("M0", "M1")), ("noise", ("G", "T"))):
        d = analyze.merged(fam, "__none__", "ref_loo")
        ev = analyze.merged(fam, "__none__", "ref")
        p_l, p_w, p_z, hit, dlw = [], [], [], [], []
        for k, v in d.items():
            if "ref" not in v or k not in ev:
                continue
            r = v["ref"]["models"]; r2 = ev[k]["ref"]["models"]
            l = np.array([r[n]["quad"]["loo"] for n in names])
            w = np.array([r2[n]["quad"]["waic"] for n in names])
            z = np.array([r2[n]["quad"]["log_z"] for n in names])
            p_l.append(_softmax_weights(l)[1]); p_w.append(_softmax_weights(w)[1])
            p_z.append(1 / (1 + np.exp(z[0] - z[1])))
            hit.append(k[0] == names[1]); dlw.append((l[1] - l[0]) - (w[1] - w[0]))
        hit = np.array(hit, float)
        res = {}
        for lab, pr in (("loo", p_l), ("waic", p_w), ("evidence", p_z)):
            pr = np.array(pr)
            rel = analyze.reliability(np.r_[pr, 1 - pr], np.r_[hit, 1 - hit])
            res[lab] = dict(acc=float(((pr > 0.5) == hit).mean()), ece=rel["ece"], null95=rel["ece_null_q95"])
        res["n"] = len(hit)
        res["dLOO_minus_dWAIC_mean"] = float(np.mean(dlw)); res["dLOO_minus_dWAIC_sd"] = float(np.std(dlw))
        crit[fam] = res
    out["exact_criteria"] = crit
    # (c) MCMC PSIS-LOO Pareto k and COMPASS LOO vs MCMC LOO on the ODE families.
    ode = {}
    for fam, ref_tag in (("population", "ref_r2"), ("population_wide", "ref_r2"), ("tutorial", "ref")):
        names = list(family(fam))
        refs = [r for r in common.load_records("reference", fam, ref_tag)]
        nkm = [r["models"][n]["mcmc"]["n_k_high"] / len(r["models"][n]["mcmc"]["elpd_rows"])
               for r in refs for n in names]
        nkm_true = [r["models"][r["true"]]["mcmc"]["n_k_high"] / len(r["models"][r["true"]]["mcmc"]["elpd_rows"])
                    for r in refs]
        acc_mcmc_loo = np.mean([_decision(names, [r["models"][n]["mcmc"]["loo"] for n in names], r["true"]) for r in refs])
        acc_mcmc_waic = np.mean([_decision(names, [r["models"][n]["mcmc"]["waic"] for n in names], r["true"]) for r in refs])
        ent = dict(n=len(refs), mcmc_frac_rows_k_high=float(np.mean(nkm)),
                   mcmc_frac_rows_k_high_true=float(np.mean(nkm_true)),
                   acc_mcmc_loo=float(acc_mcmc_loo), acc_mcmc_waic=float(acc_mcmc_waic))
        pairs = paired(fam, "trained_n131072_s0_d500_hutch", ref_tag)
        if pairs and "loo" in pairs[0][2]["models"][names[0]]:
            agl, agw, al, aw, el, ew, nkc = [], [], [], [], [], [], []
            for true, rep, c, r in pairs:
                cl = [c["models"][n]["loo"] for n in names]; cw = [c["models"][n]["waic"] for n in names]
                ml = [r["models"][n]["mcmc"]["loo"] for n in names]; mw = [r["models"][n]["mcmc"]["waic"] for n in names]
                agl.append(np.argmin(cl) == np.argmin(ml)); agw.append(np.argmin(cw) == np.argmin(mw))
                al.append(names[int(np.argmin(cl))] == true); aw.append(names[int(np.argmin(cw))] == true)
                j = names.index(true)
                el.append(cl[j] - ml[j]); ew.append(cw[j] - mw[j])
                nkc += [c["models"][n]["n_k_high"] / c["n_rows"] for n in names]
            ent.update(compass_acc_loo=float(np.mean(al)), compass_acc_waic=float(np.mean(aw)),
                       agree_loo=float(np.mean(agl)), agree_waic=float(np.mean(agw)),
                       true_err_loo_median=float(np.median(el)), true_err_waic_median=float(np.median(ew)),
                       compass_frac_rows_k_high=float(np.mean(nkc)))
        ode[fam] = ent
    out["ode"] = ode
    print(json.dumps(out, indent=1))
    save_json("exp4", out)

    plotting.style()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ax = axs[0]
    tags = list(off)
    lab = ["exact\nscores"] + [f"s{t[-1]}" for t in tags[1:-1]] + ["4x"]
    xs = np.arange(len(tags))
    ax.errorbar(xs - 0.12, [off[t]["dwaic_err_mean"] for t in tags], [off[t]["dwaic_err_sd"] for t in tags],
                fmt="o", color=NEW, label="ΔWAIC error (mean ± sd)")
    ax.errorbar(xs + 0.12, [off[t]["dloo_err_mean"] for t in tags], [off[t]["dloo_err_sd"] for t in tags],
                fmt="s", color=FOURTH, label="ΔLOO error vs exact LOO")
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(xs, lab, fontsize=8)
    ax.set_ylabel("COMPASS − exact, M1−M0 (nats)"); ax.legend(fontsize=8)
    ax.set_title(f"offset: rows with k>0.7 {np.mean([off[t]['frac_rows_k_high'] for t in tags]):.0%}")
    ax = axs[1]
    labs = []
    for i, fam in enumerate(crit):
        for j, (cn, col) in enumerate((("evidence", EXACT), ("waic", NEW), ("loo", FOURTH))):
            ax.bar(i * 4 + j, crit[fam][cn]["ece"], color=col, label=cn if i == 0 else None)
            ax.plot([i * 4 + j - 0.4, i * 4 + j + 0.4], [crit[fam][cn]["null95"]] * 2, color=INK2, lw=1)
        labs.append((i * 4 + 1, fam))
    ax.set_xticks([x for x, _ in labs], [f for _, f in labs]); ax.set_ylabel("ECE (line: null95)")
    ax.legend(fontsize=8); ax.set_title("exact criteria: calibration of model weights")
    ax = axs[2]
    fams = list(ode)
    xs = np.arange(len(fams))
    ax.bar(xs - 0.2, [ode[f]["mcmc_frac_rows_k_high"] for f in fams], 0.4, color=EXACT, label="MCMC draws")
    ax.bar(xs + 0.2, [ode[f].get("compass_frac_rows_k_high", np.nan) for f in fams], 0.4, color=NEW, label="COMPASS draws")
    ax.set_xticks(xs, fams); ax.set_ylabel("fraction of rows with Pareto k > 0.7"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.set_title("ODE families: PSIS reliability")
    fig.tight_layout()
    print(plotting.save(fig, FIG, "fig_exp4_psis_loo"))


COMMANDS = {"exp4": exp4, "exp3b": exp3b, "exp1": exp1, "exp2": exp2, "exp3a": exp3a}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=sorted(COMMANDS))
    args = parser.parse_args()
    COMMANDS[args.experiment]()


if __name__ == "__main__":
    main()
