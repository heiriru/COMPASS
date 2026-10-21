"""Population-dynamics model comparison with WAIC instead of AIC.

The WAIC twin of ``Population_Dynamics.py``: the same four simulators, the same
pretrained SBI models and the same mock observations (one set per true model, each
trajectory with its own perturbed log-parameters), but the models are ranked with
``compare(criterion="waic")``: posterior draws per trajectory, the PF-ODE likelihood
of the trajectory at every draw, and WAIC from those pointwise values. No MAP and no
parameter count are involved.

Every parameter here is trajectory-specific (the tutorial perturbs all four per
trajectory), so there are no shared parameters to compose: ``hierarchy`` is empty for
every model and each trajectory gets its own posterior.

Figures under ``tutorials/output/Population_Dynamics_WAIC/<true_model>/``:
    model_probs_violin.png, model_probs_cumulative.png   (ModelTransfuser.plot_comparison)
    population_evolution.png   prey/predator trajectories of every model from its posterior
                               draws (median and 90% band) against the observed points
"""

import os
import sys
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent
ROOT = TUTORIAL_DIR.parent
for path in (str(ROOT), str(ROOT / "src"), str(TUTORIAL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from Partial_Pooling.runtime import configure_runtime  # noqa: E402  (3-thread cap + autocvd)

configure_runtime(max_threads=3)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from compass import ModelTransfuser as MTf  # noqa: E402
from compass import ScoreBasedInferenceModel as SBIm  # noqa: E402
from Population_Dynamics import (DT, INITIAL_STATE, MODELS, T_MAX,  # noqa: E402
                                 observations_for, solve_ode)

MODEL_DIR = TUTORIAL_DIR / "data" / "population_dynamics"
OUTPUT_DIR = TUTORIAL_DIR / "output" / "Population_Dynamics_WAIC"
COLOURS = ["blue", "orange", "green", "red"]
SEED = 0
WAIC_DRAWS = int(os.environ.get("WAIC_DRAWS", 500))
TRAJECTORY_DRAWS = 200     # posterior draws simulated for the evolution plot


def load_models(device):
    mtf = MTf(path=str(OUTPUT_DIR))
    for model_name in MODELS:
        mtf.add_model(model_name, SBIm.load(str(MODEL_DIR / f"{model_name}.pt"), device=device))
    return mtf


def plot_population_evolution(mtf, observations, path, trajectory=0):
    """Like the notebook's ``plot_results_test``, from posterior draws instead of a MAP."""
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 5), constrained_layout=True, dpi=500)
    observed_time = torch.arange(0, T_MAX)
    for colour, model_name in zip(COLOURS, MODELS):
        draws = mtf.stats[model_name]["posterior_samples"][trajectory]          # (S, 4) log-params
        pick = torch.linspace(0, len(draws) - 1, min(TRAJECTORY_DRAWS, len(draws))).long()
        params = torch.exp(draws[pick].float())
        time, history = solve_ode(MODELS[model_name], INITIAL_STATE.expand(len(pick), -1).clone(),
                                  params, T_MAX, DT)
        waic = float(mtf.stats[model_name]["WAIC"])
        weight = 100 * mtf.stats[model_name]["model_prob"]
        label = f"{model_name} (WAIC {waic:.0f}, {weight:.0f}%)"
        for row, ax in enumerate(axs):
            series = history[:, :, row]
            low, mid, high = (torch.quantile(series, q, dim=0) for q in (0.05, 0.5, 0.95))
            ax.fill_between(time, low, high, color=colour, alpha=0.2, linewidth=0)
            sns.lineplot(x=time, y=mid, color=colour, label=label if row == 0 else None, ax=ax)
    axs[0].scatter(x=observed_time, y=observations[trajectory, ::2] * 100, color="k", s=10, zorder=10)
    axs[1].scatter(x=observed_time, y=observations[trajectory, 1::2] * 100, color="k", s=10, zorder=10)
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=2, frameon=False, fontsize=10)
    plt.xlabel("Time", fontsize=15)
    plt.xlim(0, T_MAX)
    axs[0].set_ylabel("Prey Population", fontsize=15)
    axs[1].set_ylabel("Predator Population", fontsize=15)
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(path / "population_evolution.png")
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    observations = {name: observations_for(func) for name, func in MODELS.items()}
    mtf = load_models(device)
    summary = {}
    for true_model_name in MODELS:
        print(f"WAIC comparison for true model: {true_model_name}")
        mtf.compare(x=observations[true_model_name], device=device, timesteps=50,
                    num_samples=WAIC_DRAWS, criterion="waic", multi_obs_inference=True,
                    hierarchy={name: [] for name in MODELS}, waic_draws=WAIC_DRAWS)
        plot_dir = OUTPUT_DIR / true_model_name.lower().replace(" ", "_")
        plot_dir.mkdir(parents=True, exist_ok=True)
        mtf.plot_comparison(stats_dict=mtf.stats, path=str(plot_dir), show=False, sort="none")
        plot_population_evolution(mtf, observations[true_model_name], plot_dir)
        summary[true_model_name] = {name: dict(waic=float(mtf.stats[name]["WAIC"]),
                                               p_waic=mtf.stats[name]["p_waic"],
                                               mc_se=mtf.stats[name]["waic_mc_se"],
                                               weight=mtf.stats[name]["model_prob"])
                                    for name in MODELS}
        print(f"Saved plots to {plot_dir}")
    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        for true_name, rows in summary.items():
            f.write(f"true {true_name}\n")
            for name, row in rows.items():
                f.write(f"  {name:22s} WAIC {row['waic']:10.2f}  p_waic {row['p_waic']:8.2f}  "
                        f"MC-SE {row['mc_se']:.2f}  weight {row['weight']:.3f}\n")
    print(open(OUTPUT_DIR / "summary.txt").read())


if __name__ == "__main__":
    main()
