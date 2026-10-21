"""Merge COMPASS and reference results; error, recovery and calibration summaries.

    .COMPASS/bin/python -m validation.a4_waic.analyze --family offset --compass-tag oracle_d1000

Writes results/summary_<family>_<compass-tag>.json (read by plots.py).
"""

import validation.common.runtime  # noqa: F401

import argparse
import json
import math
import os

import numpy as np
from scipy.special import logsumexp, softmax

from validation.a4_waic import common
from validation.a4_waic.families import family


def merged(family_name, compass_tag, ref_tag="ref,ref_ns"):
    """{(true, rep): {"compass": record, "ref": record}}; several comma-separated reference
    tags (e.g. quadrature+MCMC and nested sampling run separately) are merged per model."""
    out = {}
    for rec in common.load_records("compass", family_name, compass_tag):
        out.setdefault((rec["true"], rec["rep"]), {})["compass"] = rec
    for tag in ref_tag.split(","):
        for rec in common.load_records("reference", family_name, tag):
            slot = out.setdefault((rec["true"], rec["rep"]), {})
            if "ref" not in slot:
                slot["ref"] = {"true": rec["true"], "rep": rec["rep"], "models": {}}
            for name, entry in rec["models"].items():
                slot["ref"]["models"].setdefault(name, {}).update(entry)
    return out


def exact_ref(entry):
    """Best available reference for one model: quadrature, else MCMC."""
    return entry.get("quad") or entry.get("mcmc")


def reliability(prob, hit, bins=10, n_boot=500, seed=0):
    """Binned reliability with bootstrap bands, and ECE with a bootstrap 90% interval."""
    prob, hit = np.asarray(prob, float), np.asarray(hit, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)

    def ece(p, h, i):
        total = 0.0
        for b in range(bins):
            m = i == b
            if m.any():
                total += m.mean() * abs(h[m].mean() - p[m].mean())
        return total

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(prob), len(prob))
        boots.append(ece(prob[k], hit[k], idx[k]))
    # ECE of a perfectly calibrated predictor with the same probabilities (finite-n floor).
    null = [ece(prob, (rng.random(len(prob)) < prob).astype(float), idx) for _ in range(n_boot)]
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            freq = hit[m].mean()
            half = 1.645 * math.sqrt(max(freq * (1 - freq), 1e-4) / m.sum())
            rows.append(dict(mean_prob=float(prob[m].mean()), freq=float(freq), n=int(m.sum()),
                             lo=float(max(freq - half, 0)), hi=float(min(freq + half, 1))))
    return dict(ece=ece(prob, hit, idx), ece_ci=[float(np.quantile(boots, 0.05)),
                                                  float(np.quantile(boots, 0.95))],
                ece_null_mean=float(np.mean(null)), ece_null_q95=float(np.quantile(null, 0.95)),
                bins=rows, brier=float(np.mean((prob - hit) ** 2)), n=int(len(prob)))


def summarize(family_name, compass_tag, ref_tag="ref,ref_ns"):
    names = list(family(family_name))
    data = merged(family_name, compass_tag, ref_tag)
    out = dict(family=family_name, compass_tag=compass_tag, models=names, datasets=[])

    # Per-dataset table.
    for (true, rep), d in sorted(data.items()):
        row = dict(true=true, rep=rep)
        c, r = d.get("compass"), d.get("ref")
        if c:
            row["compass"] = {n: c["models"][n]["waic"] for n in names}
            row["compass_mc_se"] = {n: c["models"][n]["mc_se"] for n in names}
            row["compass_p"] = {n: c["models"][n]["p_waic"] for n in names}
            row["compass_weight"] = {n: c["models"][n]["weight"] for n in names}
            row["compass_pbma"] = {n: c["models"][n]["weight_pbma_plus"] for n in names}
            row["compass_n_p_high"] = {n: c["models"][n]["n_p_high"] for n in names}
            row["compass_seconds"] = c["seconds"]
            row["g_true"] = c["g_true"]
        if r:
            for key in ("quad", "mcmc", "ns"):
                if all(key in r["models"][n] for n in names):
                    row[key] = {n: r["models"][n][key]["waic"] for n in names}
                    row[key + "_p"] = {n: r["models"][n][key]["p_waic"] for n in names}
                    if key in ("quad", "ns"):
                        row[key + "_logz"] = {n: r["models"][n][key]["log_z"] for n in names}
                    if key == "ns":
                        row["ns_logz_err"] = {n: r["models"][n]["ns"]["log_z_err"] for n in names}
                    if key == "mcmc":
                        row["mcmc_tau"] = {n: r["models"][n]["mcmc"]["tau_max"] for n in names}
                        row["mcmc_neff"] = {n: r["models"][n]["mcmc"]["n_eff"] for n in names}
        out["datasets"].append(row)

    rows = out["datasets"]
    ref_key = "quad" if any("quad" in r for r in rows) else "mcmc"
    out["ref_key"] = ref_key

    def vec(key, rs, name):
        return np.array([r[key][name] for r in rs])

    # (1) WAIC errors per model and for differences, against the exact reference.
    both = [r for r in rows if "compass" in r and ref_key in r]
    errors = {}
    for n in names:
        e = vec("compass", both, n) - vec(ref_key, both, n)
        errors[n] = dict(bias=float(e.mean()), sd=float(e.std(ddof=1)) if len(e) > 1 else None,
                         rmse=float(np.sqrt((e ** 2).mean())),
                         mean_mc_se=float(np.mean(vec("compass_mc_se", both, n))), n=len(e))
    out["waic_error"] = errors
    diffs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dc = vec("compass", both, a) - vec("compass", both, b)
            dr = vec(ref_key, both, a) - vec(ref_key, both, b)
            diffs[f"{a}-{b}"] = dict(bias=float((dc - dr).mean()), sd=float((dc - dr).std(ddof=1)),
                                     sign_agreement=float(np.mean(np.sign(dc) == np.sign(dr))))
    out["delta_error"] = diffs
    pick_c = [min(names, key=lambda n: r["compass"][n]) for r in both]
    pick_r = [min(names, key=lambda n: r[ref_key][n]) for r in both]
    out["decision_agreement"] = float(np.mean(np.array(pick_c) == np.array(pick_r))) if both else None
    out["n_paired"] = len(both)
    # Agreement by the size of the exact WAIC gap between the two best reference models.
    if both:
        gaps = np.array([sorted(r[ref_key].values())[1] - sorted(r[ref_key].values())[0] for r in both])
        agree = np.array(pick_c) == np.array(pick_r)
        out["agreement_by_gap"] = []
        for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 1e9)):
            m = (gaps >= lo) & (gaps < hi)
            if m.any():
                out["agreement_by_gap"].append(dict(lo=lo, hi=hi, n=int(m.sum()),
                                                    agreement=float(agree[m].mean())))

    # (2) Reference consistency.
    cons = {}
    for key in ("mcmc", "ns"):
        if key == ref_key:
            continue
        rs = [r for r in rows if key in r and ref_key in r]
        if rs:
            e = np.concatenate([vec(key, rs, n) - vec(ref_key, rs, n) for n in names])
            cons[f"{key}_vs_{ref_key}"] = dict(bias=float(e.mean()), sd=float(e.std(ddof=1)),
                                              max_abs=float(np.abs(e).max()), n=len(e))
    rs = [r for r in rows if "ns" in r and "quad_logz" in r]
    if rs:
        z = np.concatenate([(vec("ns_logz", rs, n) - vec("quad_logz", rs, n))
                            / vec("ns_logz_err", rs, n) for n in names])
        cons["ns_logz_z"] = dict(mean=float(z.mean()), sd=float(z.std(ddof=1)), n=len(z))
    out["reference_consistency"] = cons

    # (3) Recovery (confusion matrices: rows = true model, columns = picked).
    def confusion(rs, picker):
        m = np.zeros((len(names), len(names)), int)
        for r in rs:
            m[names.index(r["true"]), names.index(picker(r))] += 1
        return m.tolist()

    methods = {}
    if any("compass" in r for r in rows):
        methods["compass_waic"] = ([r for r in rows if "compass" in r],
                                   lambda r: min(names, key=lambda n: r["compass"][n]),
                                   lambda r: [r["compass_weight"][n] for n in names])
        methods["compass_pbma_plus"] = ([r for r in rows if "compass" in r],
                                        lambda r: max(names, key=lambda n: r["compass_pbma"][n]),
                                        lambda r: [r["compass_pbma"][n] for n in names])
    for key in ("quad", "mcmc", "ns"):
        if any(key in r for r in rows):
            methods[f"{key}_waic"] = ([r for r in rows if key in r],
                                      lambda r, k=key: min(names, key=lambda n: r[k][n]),
                                      lambda r, k=key: softmax([-0.5 * r[k][n] for n in names]).tolist())
    for key in ("quad", "ns"):
        if any(key + "_logz" in r for r in rows):
            methods[f"{key}_evidence"] = ([r for r in rows if key + "_logz" in r],
                                          lambda r, k=key: max(names, key=lambda n: r[k + "_logz"][n]),
                                          lambda r, k=key: softmax([r[k + "_logz"][n] for n in names]).tolist())
    out["recovery"], out["calibration"] = {}, {}
    for label, (rs, picker, prober) in methods.items():
        conf = confusion(rs, picker)
        acc = float(np.trace(np.array(conf)) / max(len(rs), 1))
        out["recovery"][label] = dict(confusion=conf, accuracy=acc, n=len(rs))
        probs = np.array([prober(r) for r in rs])
        hits = np.array([[float(r["true"] == n) for n in names] for r in rs])
        # Two models: calibrate P(first model); more: pool all (dataset, model) pairs.
        if len(names) == 2:
            out["calibration"][label] = reliability(probs[:, 0], hits[:, 0])
        else:
            out["calibration"][label] = reliability(probs.ravel(), hits.ravel())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--compass-tag", required=True)
    parser.add_argument("--ref-tag", default="ref,ref_ns")
    args = parser.parse_args()
    out = summarize(args.family, args.compass_tag, args.ref_tag)
    path = os.path.join(common.RESULTS, f"summary_{args.family}_{args.compass_tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    show = {k: out.get(k) for k in ("n_paired", "ref_key", "waic_error", "delta_error",
                                    "decision_agreement", "agreement_by_gap",
                                    "reference_consistency")}
    print(json.dumps(show, indent=1))
    for label, rec in out["recovery"].items():
        cal = out["calibration"][label]
        print(f"{label:20s} acc {rec['accuracy']:.3f} (n={rec['n']})  ECE {cal['ece']:.3f} "
              f"[{cal['ece_ci'][0]:.3f}, {cal['ece_ci'][1]:.3f}] null95 {cal['ece_null_q95']:.3f}  Brier {cal['brier']:.3f}  "
              f"confusion {rec['confusion']}")


if __name__ == "__main__":
    main()
