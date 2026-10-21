#!/usr/bin/env python
"""Figures for the bridge-topology experiment (section 15).

Seven panels, in the order the experiment asks for them:

``modes``     mode positions against noise -- true, JAC, TOPO-JAC, F-NPSE.
``barrier``   barrier depth against noise, on a log scale.
``mass``      the positive-branch mass each bridge carries, against noise.
``alpha``     the oracle ``alpha*`` against ``tau`` and ``log lambda``, per n,
              with the smooth fits that were actually sampled.
``fnpestep``  the shared-step multiplier of every F-NPSE schedule tested.
``endpoint``  final W1/sd and branch-weight TV against n, all arms.
``snapshots`` bridge snapshots at the genuinely bimodal setting.

Every panel marks diverged arms rather than dropping them, and no panel
selects an arm by its best score: where a ladder exists the *highest-effort
stable* configuration is drawn, because these ladders are non-monotone and
"best" would report a lucky under-converged point.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "gm_topology"

ARMS = ("true", "jac", "topojac", "fnpe", "naive")
COLOURS = {"true": "k", "jac": "tab:purple", "topojac": "tab:green",
           "fnpe": "tab:blue", "naive": "tab:red", "gauss": "tab:orange"}
STYLES = {"true": "-", "jac": "-", "topojac": "-", "fnpe": "--", "naive": ":"}


def _load(name, tag=""):
    path = ARTIFACTS / f"{name}{tag}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _levels(entry):
    rows = sorted(entry["levels"].values(), key=lambda row: row["tau"])
    return np.array([row["tau"] for row in rows]), rows


# ---------------------------------------------------------------------------

def modes_figure(tag=""):
    data = _load("topoprofile", tag)
    if data is None:
        return None
    ns = sorted(data, key=int)
    figure, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.0),
                                squeeze=False)
    for column, n in enumerate(ns):
        axis = axes[0][column]
        taus, rows = _levels(data[n])
        for arm in ARMS:
            for key, marker in (("mode_minus", None), ("mode_plus", None)):
                values = [row[arm].get(key) if row[arm]["num_modes"] >= 2
                          else (row[arm].get("mode_plus")
                                if key == "mode_plus" else np.nan)
                          for row in rows]
                values = [np.nan if v is None else v for v in values]
                axis.plot(taus, values, STYLES[arm], color=COLOURS[arm],
                          lw=1.6 if arm in ("true", "topojac") else 1.2,
                          alpha=0.9,
                          label=arm if key == "mode_plus" else None)
        # The bifurcation levels, as vertical rules.
        for arm in ("true", "jac", "topojac"):
            split = _split_of(rows, arm)
            if split is not None:
                axis.axvline(split, color=COLOURS[arm], lw=0.8, ls="-.",
                             alpha=0.5)
        axis.set_xlabel("tau  (0 = data end, 1 = noise end)")
        axis.set_title(f"n = {n}{tag}  --  mode positions")
        if column == 0:
            axis.set_ylabel("mode location in g")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, "modes", tag)


def _split_of(rows, arm):
    ordered = sorted(rows, key=lambda row: -row["tau"])
    for index, row in enumerate(ordered):
        if row[arm]["num_modes"] >= 2 and all(
                later[arm]["num_modes"] >= 2 for later in ordered[index:]):
            return row["tau"]
    return None


def barrier_figure(tag=""):
    data = _load("topoprofile", tag)
    if data is None:
        return None
    ns = sorted(data, key=int)
    figure, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 3.8),
                                squeeze=False)
    for column, n in enumerate(ns):
        axis = axes[0][column]
        taus, rows = _levels(data[n])
        for arm in ARMS:
            values = [max(row[arm].get("barrier") or 0.0, 1e-3) for row in rows]
            axis.semilogy(taus, values, STYLES[arm], color=COLOURS[arm],
                          lw=1.6 if arm in ("true", "topojac") else 1.2,
                          alpha=0.9, label=arm)
        axis.set_xlabel("tau")
        axis.set_title(f"n = {n}{tag}  --  barrier depth")
        if column == 0:
            axis.set_ylabel("min_k [log q(m_k) - log q(valley)]")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, "barrier", tag)


def mass_figure(tag=""):
    data = _load("topoprofile", tag)
    if data is None:
        return None
    ns = sorted(data, key=int)
    figure, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 3.8),
                                squeeze=False)
    for column, n in enumerate(ns):
        axis = axes[0][column]
        taus, rows = _levels(data[n])
        for arm in ARMS:
            values = [row[arm].get("w_plus") for row in rows]
            values = [np.nan if v is None else v for v in values]
            axis.plot(taus, values, STYLES[arm], color=COLOURS[arm],
                      lw=1.6 if arm in ("true", "topojac") else 1.2,
                      alpha=0.9, label=arm)
        axis.axhline(0.75, color="grey", lw=0.8, ls="--")
        axis.set_xlabel("tau")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"n = {n}{tag}  --  positive-branch mass")
        if column == 0:
            axis.set_ylabel("w_+ (mass right of the bridge's own valley)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    return _save(figure, "mass", tag)


def alpha_figure(tag=""):
    data = _load("alpha_topo", tag)
    if data is None:
        return None
    ns = sorted(data, key=int)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(ns)))
    for colour, n in zip(palette, ns):
        rows = sorted(data[n]["per_level"].values(), key=lambda r: r["tau"])
        taus = np.array([row["tau"] for row in rows])
        alpha = np.array([row["alpha_mid"] for row in rows])
        lows = np.array([row["alpha_plateau"][0] for row in rows])
        highs = np.array([row["alpha_plateau"][1] for row in rows])
        lams = np.array([row["lambda"] for row in rows])
        axes[0].fill_between(taus, lows, highs, color=colour, alpha=0.18)
        axes[0].plot(taus, alpha, "o-", color=colour, ms=3, lw=1.4,
                     label=f"n = {n}")
        axes[1].semilogx(lams, alpha, "o-", color=colour, ms=3, lw=1.4,
                         label=f"n = {n}")
        fits = data[n]["fits"]
        best = min(fits, key=lambda key: fits[key]["rmse"])
        axes[0].plot(fits[best]["tau"], fits[best]["alpha"], "--",
                     color=colour, lw=1.0)
        axes[2].bar([f"{name}\nn={n}" for name in sorted(fits)],
                    [fits[name]["rmse"] for name in sorted(fits)],
                    color=colour, alpha=0.7)
    axes[0].set_xlabel("tau")
    axes[0].set_ylabel("oracle alpha*  (shaded: objective plateau)")
    axes[0].set_title("alpha* against the bridge coordinate"
                      "\n(dashed: best smooth fit)")
    axes[1].set_xlabel("lambda")
    axes[1].set_title("alpha* against lambda")
    axes[2].set_ylabel("fit RMSE against alpha*")
    axes[2].set_title("how simple alpha can be")
    axes[2].tick_params(axis="x", labelsize=6, rotation=90)
    for axis in axes[:2]:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_ylim(-0.05, 1.05)
    figure.suptitle(f"oracle alpha{tag}")
    figure.tight_layout()
    return _save(figure, "alpha", tag)


def fnpe_step_figure(tag=""):
    data = _load("fnpe_step", tag)
    if data is None:
        return None
    ns = sorted(data, key=int)
    figure, axes = plt.subplots(1, len(ns) + 1,
                                figsize=(4.4 * (len(ns) + 1), 3.9),
                                squeeze=False)
    # The schedules themselves, as multipliers of the unscaled step.
    axis = axes[0][0]
    taus = np.linspace(1.0, 0.0, 200)
    for n in (128,):
        for p in (0.5, 1, 2, 4):
            w = (1.0 - taus) ** p
            axis.semilogy(taus, (1.0 - w) / n + w, lw=1.2, label=f"power p={p}")
        for k, tau0 in ((10, 0.5),):
            w = 1.0 / (1.0 + np.exp(-k * (tau0 - taus)))
            axis.semilogy(taus, (1.0 - w) / n + w, "--", lw=1.2,
                          label=f"sigmoid k={k}, tau0={tau0}")
        axis.axhline(1.0 / n, color="grey", ls=":", lw=1.0, label="fixed 1/n")
        axis.axhline(1.0, color="k", ls=":", lw=1.0, label="unscaled")
    axis.set_xlabel("tau")
    axis.set_ylabel("shared-step multiplier (n = 128)")
    axis.set_title("step schedules")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=7)

    for column, n in enumerate(ns, start=1):
        axis = axes[0][column]
        rows = data[n]
        names = sorted(rows)
        widths = [rows[key].get("width_ratio", np.nan) for key in names]
        diverged = [bool(rows[key].get("diverged")) for key in names]
        positions = np.arange(len(names))
        colours = ["tab:red" if d else "tab:blue" for d in diverged]
        axis.barh(positions, [0.0 if not np.isfinite(w) else w
                              for w in widths], color=colours, alpha=0.75)
        axis.axvline(1.0, color="k", lw=1.0, ls="--")
        axis.set_yticks(positions)
        axis.set_yticklabels([key.split("/")[0] for key in names], fontsize=6)
        axis.set_xlabel("width ratio (1 = correctly dispersed)")
        axis.set_title(f"n = {n}{tag}  --  mixing\n(red = diverged)")
        axis.grid(alpha=0.25, axis="x")
    figure.tight_layout()
    return _save(figure, "fnpestep", tag)


def endpoint_figure(tag=""):
    data = _load("cross", tag)
    if data is None:
        return None
    ns = sorted(data, key=int)
    arms = ("TRUE-SCORE", "ORACLE-JAC", "ORACLE-GAUSS", "ORACLE-TOPO-JAC",
            "FNPE-FIXED", "FNPE-ADAPTIVE")
    palette = dict(zip(arms, ["k", "tab:purple", "tab:orange", "tab:green",
                              "tab:blue", "tab:cyan"]))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    width = 0.13
    for index, arm in enumerate(arms):
        offsets = np.arange(len(ns)) + (index - len(arms) / 2) * width
        for axis, key in ((axes[0], "w1_over_sd"), (axes[1], "weight_tv")):
            values, hatches = [], []
            for n in ns:
                row = data[n].get(arm, {})
                value = row.get(key, float("nan"))
                bad = bool(row.get("diverged")) or not np.isfinite(value)
                values.append(0.0 if bad else value)
                hatches.append(bad)
            bars = axis.bar(offsets, values, width, color=palette[arm],
                            alpha=0.85, label=arm if key == "w1_over_sd" else None)
            for bar, bad in zip(bars, hatches):
                if bad:
                    bar.set_hatch("///")
                    bar.set_edgecolor("k")
    for axis, label in ((axes[0], "W1 / sd"), (axes[1], "branch-weight TV")):
        axis.set_xticks(np.arange(len(ns)))
        axis.set_xticklabels([f"n = {n}" for n in ns])
        axis.set_ylabel(label)
        axis.set_yscale("log")
        axis.grid(alpha=0.25, axis="y", which="both")
    axes[0].legend(fontsize=8)
    figure.suptitle(f"endpoint accuracy{tag}  (hatched = diverged)")
    figure.tight_layout()
    return _save(figure, "endpoint", tag)


def snapshot_figure(tag="_mu7.5"):
    data = _load("topoprofile", tag)
    if data is None:
        return None
    n = sorted(data, key=int)[-1]
    rows = data[n]["levels"]
    taus = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
    available = [tau for tau in taus if f"tau={tau:.4f}" in rows]
    figure, axes = plt.subplots(1, len(available),
                                figsize=(3.0 * len(available), 3.4),
                                squeeze=False)
    for column, tau in enumerate(available):
        entry = rows[f"tau={tau:.4f}"]
        axis = axes[0][column]
        grid = np.asarray(entry["grid"])
        # Log density, not density: at intermediate noise the JAC bridge is so
        # much narrower than the truth that on a linear axis it is a vertical
        # line and nothing about its shape is visible.
        for arm in ("true", "jac", "topojac", "fnpe"):
            curve = np.asarray(entry[f"{arm}_curve"])
            axis.plot(grid, curve - curve.max(), STYLES[arm],
                      color=COLOURS[arm],
                      lw=1.7 if arm == "true" else 1.2,
                      alpha=0.95 if arm != "jac" else 0.7, label=arm)
        reference = np.asarray(entry["true_curve"])
        reference = reference - reference.max()
        support = grid[reference > -14.0]
        if support.size:
            pad = 0.35 * (support.max() - support.min() + 1e-6)
            low, high = support.min() - pad, support.max() + pad
            axis.set_xlim(low, high)
            # An arm whose own mode sits outside the true bridge's support
            # would otherwise simply vanish from the panel, which reads as
            # "not plotted" rather than "nowhere near the answer".
            off = [arm for arm in ("jac", "topojac", "fnpe")
                   if entry[arm].get("mode_plus") is not None
                   and not (low <= entry[arm]["mode_plus"] <= high)]
            if off:
                axis.text(0.5, 0.02,
                          "off panel: " + ", ".join(
                              f"{arm} at {entry[arm]['mode_plus']:.2f}"
                              for arm in off),
                          transform=axis.transAxes, fontsize=6, ha="center",
                          color="tab:red")
        axis.set_title(f"tau={tau:g}, lam={entry['lambda']:.3f}\n"
                       f"alpha={entry['alpha']:.2f}", fontsize=9)
        axis.set_ylim(-14.0, 1.0)
        axis.tick_params(labelsize=8)
        if column == 0:
            axis.set_ylabel("slice log density (max-shifted)")
            axis.legend(fontsize=7, loc="lower center")
    figure.suptitle(f"bridge snapshots, n = {n}{tag} "
                    f"(schedule: {data[n]['schedule']})")
    figure.tight_layout()
    return _save(figure, "snapshots", tag)


def _save(figure, name, tag):
    path = ARTIFACTS / f"{name}{tag}.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")
    return path


def main():
    for tag in ("", "_mu7.5"):
        for builder in (modes_figure, barrier_figure, mass_figure,
                        alpha_figure, fnpe_step_figure, endpoint_figure):
            builder(tag)
    snapshot_figure("_mu7.5")
    snapshot_figure("")


if __name__ == "__main__":
    main()
