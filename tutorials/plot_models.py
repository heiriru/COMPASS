from pathlib import Path

import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import seaborn as sns
import torch

MODEL_COLORS = {
    "Single Gaussian 2p": "#386CB0",
    "Two Blobs 4p": "#F28E2B",
    "Elongated Gaussian 4p": "#59A14F",
}


TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TUTORIAL_DIR / "output" / "Gaussians_baseline_blobs_VP"
MOCK_DATA_PLOT_DIR = OUTPUT_DIR / "mock_data_plots"
MU_RANGE = (-4.0, 4.0)
MIN_BLOB_DISTANCE = 3.0
AXIS_STD_RANGE = (1.0, 4.0)
MOCK_AXIS_STD_RANGE = (2.25, 4.0)
CONFUSION_N = 2
CONFUSION_N_OBSERVATIONS = 100

MODEL_SPECS = [
    {
        "name": "Single Gaussian 2p",
        "theta_dim": 2,
        "family": "single",
        "description": "center mu=(mu_x, mu_y), covariance I",
    },
    {
        "name": "Two Blobs 4p",
        "theta_dim": 4,
        "family": "mixture",
        "description": "two centers mu_1 and mu_2, shared covariance I",
    },
    {
        "name": "Elongated Gaussian 4p",
        "theta_dim": 4,
        "family": "elongated",
        "description": "center, rotation angle, and major-axis standard deviation",
    },
]


def sample_mu(n):
    low, high = MU_RANGE
    return low + (high - low) * torch.rand(n, 2)


def sample_two_centers(n, mock=False):
    mu1 = sample_mu(n)
    mu2 = sample_mu(n)
    if mock:
        too_close = torch.linalg.norm(mu2 - mu1, dim=1) < MIN_BLOB_DISTANCE
        while too_close.any():
            mu2[too_close] = sample_mu(int(too_close.sum()))
            too_close = torch.linalg.norm(mu2 - mu1, dim=1) < MIN_BLOB_DISTANCE
    return mu1, mu2


def sample_axis_params(n, mock=False):
    angle = torch.pi * torch.rand(n)
    low, high = MOCK_AXIS_STD_RANGE if mock else AXIS_STD_RANGE
    axis_std = low + (high - low) * torch.rand(n)
    return angle, axis_std


def gen_single_gaussian(n, theta=None, mock=False):
    mu = sample_mu(n) if theta is None else torch.as_tensor(theta, dtype=torch.float).reshape(n, 2)
    x = mu + torch.randn(n, 2)
    return mu, x


def gen_two_blobs(n, theta=None, mock=False):
    if theta is None:
        mu1, mu2 = sample_two_centers(n, mock=mock)
        theta = torch.cat([mu1, mu2], dim=1)
    else:
        theta = torch.as_tensor(theta, dtype=torch.float).reshape(n, 4)
        mu1, mu2 = theta[:, :2], theta[:, 2:]
    component = torch.randint(0, 2, (n, 1)).bool()
    mu = torch.where(component, mu2, mu1)
    x = mu + torch.randn(n, 2)
    return theta, x


def gen_elongated_gaussian(n, theta=None, mock=False):
    if theta is None:
        mu = sample_mu(n)
        angle, axis_std = sample_axis_params(n, mock=mock)
        theta = torch.cat([mu, angle.unsqueeze(1), axis_std.unsqueeze(1)], dim=1)
    else:
        theta = torch.as_tensor(theta, dtype=torch.float).reshape(n, 4)
        mu, angle, axis_std = theta[:, :2], theta[:, 2], theta[:, 3]
    noise_major = axis_std * torch.randn(n)
    noise_minor = torch.randn(n)
    direction_major = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    direction_minor = torch.stack([-torch.sin(angle), torch.cos(angle)], dim=1)
    x = mu + noise_major.unsqueeze(1) * direction_major + noise_minor.unsqueeze(1) * direction_minor
    return theta, x


def make_generator(family):
    if family == "single":
        return gen_single_gaussian
    if family == "mixture":
        return gen_two_blobs
    if family == "elongated":
        return gen_elongated_gaussian
    raise ValueError(f"Unknown model family: {family}")


def sample_theta_configs(spec, n_configs, mock):
    theta, _ = make_generator(spec["family"])(n_configs, mock=mock)
    return theta


def generate_single_theta_observations(spec, n_observations, mock=False):
    theta_config, _ = make_generator(spec["family"])(1, mock=mock)
    theta = theta_config.repeat(n_observations, 1)
    _, x = make_generator(spec["family"])(n_observations, theta=theta)
    return theta, x


def generate_confusion_mock_sets(spec, n_sets, n_observations):
    return [
        generate_single_theta_observations(spec, n_observations, mock=True)
        for _ in range(n_sets)
    ]


def samples_for_theta(spec, theta, n_samples):
    theta_batch = theta.unsqueeze(0).repeat(n_samples, 1)
    _, x = make_generator(spec["family"])(n_samples, theta=theta_batch)
    return x


def add_covariance_guide(ax, spec, theta, color):
    if spec["family"] == "single":
        mu = theta[:2]
        ax.scatter(mu[0], mu[1], marker="x", s=90, linewidth=2.2, color=color, zorder=5)
        ax.add_patch(Ellipse(mu, width=2.0, height=2.0, fill=False, lw=2.0, color=color, alpha=0.9))
        return

    if spec["family"] == "mixture":
        for mu in (theta[:2], theta[2:]):
            ax.scatter(mu[0], mu[1], marker="x", s=90, linewidth=2.2, color=color, zorder=5)
            ax.add_patch(Ellipse(mu, width=2.0, height=2.0, fill=False, lw=2.0, color=color, alpha=0.9))
        ax.plot([theta[0], theta[2]], [theta[1], theta[3]], color=color, lw=1.2, alpha=0.7)
        return

    if spec["family"] == "elongated":
        mu = theta[:2]
        angle_deg = float(theta[2] * 180.0 / torch.pi)
        axis_std = float(theta[3])
        ax.scatter(mu[0], mu[1], marker="x", s=90, linewidth=2.2, color=color, zorder=5)
        ax.add_patch(
            Ellipse(
                mu,
                width=2.0 * axis_std,
                height=2.0,
                angle=angle_deg,
                fill=False,
                lw=2.0,
                color=color,
                alpha=0.9,
            )
        )



def slugify_model_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def collect_mock_data(n_sets, observations_per_set):
    return {
        spec["name"]: (spec, generate_confusion_mock_sets(spec, n_sets, observations_per_set))
        for spec in MODEL_SPECS
    }


def get_mock_limits(mock_data_by_model):
    all_x = torch.cat(
        [x for _, mock_sets in mock_data_by_model.values() for _, x in mock_sets],
        dim=0,
    )
    pad = 1.0
    x_min, y_min = all_x.min(dim=0).values - pad
    x_max, y_max = all_x.max(dim=0).values + pad
    return float(x_min), float(x_max), float(y_min), float(y_max)


def plot_mock_model_sets(spec, mock_sets, limits, output):
    sns.set_theme(style="ticks", context="notebook")
    colors = sns.color_palette("deep", n_colors=len(mock_sets))
    x_min, x_max, y_min, y_max = limits

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for set_idx, ((theta, x), color) in enumerate(zip(mock_sets, colors), start=1):
        theta_config = theta[0]
        ax.scatter(
            x[:, 0],
            x[:, 1],
            s=16,
            alpha=0.55,
            color=color,
            edgecolors="none",
            rasterized=True,
            label=f"mock set {set_idx}",
        )
        add_covariance_guide(ax, spec, theta_config, color)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="0.88", lw=0.8, zorder=0)
    ax.axvline(0, color="0.88", lw=0.8, zorder=0)
    ax.set_title(f"{spec['name']} mock data")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.grid(True, color="0.92", linewidth=0.8)
    sns.despine(ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], marker="x", color="k", linestyle="None", markersize=8, markeredgewidth=2, label="center")
    )
    labels.append("center")
    handles.append(Line2D([0], [0], color="k", lw=2, label="1-sigma contour"))
    labels.append("1-sigma contour")
    ax.legend(handles=handles, labels=labels, frameon=False, loc="upper right")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mock_data_overview(mock_data_by_model, limits, output):
    sns.set_theme(style="ticks", context="notebook")
    fig, axes = plt.subplots(1, len(MODEL_SPECS), figsize=(4.8 * len(MODEL_SPECS), 4.7), sharex=True, sharey=True)
    if len(MODEL_SPECS) == 1:
        axes = [axes]
    x_min, x_max, y_min, y_max = limits

    for ax, spec in zip(axes, MODEL_SPECS):
        _, mock_sets = mock_data_by_model[spec["name"]]
        colors = sns.color_palette("deep", n_colors=len(mock_sets))
        for set_idx, ((theta, x), color) in enumerate(zip(mock_sets, colors), start=1):
            ax.scatter(
                x[:, 0],
                x[:, 1],
                s=9,
                alpha=0.42,
                color=color,
                edgecolors="none",
                rasterized=True,
                label=f"set {set_idx}",
            )
            add_covariance_guide(ax, spec, theta[0], color)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="0.88", lw=0.8, zorder=0)
        ax.axvline(0, color="0.88", lw=0.8, zorder=0)
        ax.set_title(spec["name"])
        ax.set_xlabel("x_1")
        ax.grid(True, color="0.92", linewidth=0.8)
        sns.despine(ax=ax)

    axes[0].set_ylabel("x_2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.985, 0.5), frameon=False)
    first_mock_sets = next(iter(mock_data_by_model.values()))[1]
    fig.suptitle(
        f"VP Gaussian mock data ({len(first_mock_sets)} sets/model, "
        f"{sum(len(x) for _, x in first_mock_sets)} observations/model)",
        y=0.99,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.065, right=0.86, bottom=0.12, top=0.82, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mock_data_models(n_sets, observations_per_set, output_dir):
    mock_data_by_model = collect_mock_data(n_sets, observations_per_set)
    limits = get_mock_limits(mock_data_by_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for spec in MODEL_SPECS:
        _, mock_sets = mock_data_by_model[spec["name"]]
        output = output_dir / f"{slugify_model_name(spec['name'])}_mock_data.png"
        plot_mock_model_sets(spec, mock_sets, limits, output)
        saved_paths.append(output)

    overview_output = output_dir / "gaussian_vp_mock_data_overview.png"
    plot_mock_data_overview(mock_data_by_model, limits, overview_output)
    saved_paths.append(overview_output)
    return saved_paths

def plot_model_previews(n_configs, samples_per_config, mock, output):
    sns.set_theme(style="ticks", context="notebook")
    palette = sns.color_palette("deep", n_colors=n_configs)
    fig, axes = plt.subplots(1, len(MODEL_SPECS), figsize=(4.7 * len(MODEL_SPECS), 4.8), sharex=True, sharey=True)
    if len(MODEL_SPECS) == 1:
        axes = [axes]

    all_x = []
    plotted = []
    for ax, spec in zip(axes, MODEL_SPECS):
        theta_configs = sample_theta_configs(spec, n_configs, mock=mock)
        for idx, theta in enumerate(theta_configs):
            color = palette[idx]
            x = samples_for_theta(spec, theta, samples_per_config)
            all_x.append(x)
            plotted.append((ax, spec, theta, x, color, idx))

    all_x = torch.cat(all_x, dim=0)
    pad = 1.0
    x_min, y_min = all_x.min(dim=0).values - pad
    x_max, y_max = all_x.max(dim=0).values + pad

    for ax, spec, theta, x, color, idx in plotted:
        ax.scatter(x[:, 0], x[:, 1], s=7, alpha=0.22, color=color, edgecolors="none", rasterized=True)
        add_covariance_guide(ax, spec, theta, color)
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(float(y_min), float(y_max))
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="0.88", lw=0.8, zorder=0)
        ax.axvline(0, color="0.88", lw=0.8, zorder=0)
        ax.set_title(f"{spec['name']}\nk={spec['theta_dim']}", fontsize=12)
        ax.set_xlabel("x_1")
        if ax is axes[0]:
            ax.set_ylabel("x_2")
        ax.grid(True, color="0.92", linewidth=0.8)
        sns.despine(ax=ax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i], markersize=8, label=f"theta draw {i + 1}")
        for i in range(n_configs)
    ]
    legend_handles.append(
        Line2D([0], [0], marker="x", color="k", linestyle="None", markersize=8, markeredgewidth=2, label="center")
    )
    legend_handles.append(
        Line2D([0], [0], color="k", lw=2, label="1-sigma contour")
    )
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(0.985, 0.5),
        frameon=False,
        title="Per-panel parameter settings",
    )

    prior_label = "mock prior" if mock else "training prior"
    fig.suptitle(f"Gaussian model preview ({prior_label})", y=0.99, fontsize=16)
    fig.subplots_adjust(left=0.065, right=0.81, bottom=0.12, top=0.80, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot variable-parameter Gaussian mock data in x-space.")
    parser.add_argument("--num-configs", type=int, default=4, help="Parameter settings to draw per model.")
    parser.add_argument("--samples-per-config", type=int, default=450, help="x samples per parameter setting.")
    parser.add_argument("--seed", type=int, default=7, help="Torch random seed.")
    parser.add_argument("--training-prior", action="store_true", help="Show broad training prior instead of separated/elongated mock prior.")
    parser.add_argument("--mock-sets", type=int, default=CONFUSION_N, help="Mock parameter settings per model.")
    parser.add_argument(
        "--observations-per-set",
        type=int,
        default=CONFUSION_N_OBSERVATIONS,
        help="Observations generated from each mock parameter setting.",
    )
    parser.add_argument(
        "--mock-output-dir",
        type=Path,
        default=MOCK_DATA_PLOT_DIR,
        help="Directory for per-model mock-data plots.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Optional combined model preview path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Deprecated alias for --combined-output.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    mock = not args.training_prior

    print("Model free-parameter counts:")
    for spec in MODEL_SPECS:
        print(f"  {spec['name']}: k={spec['theta_dim']} ({spec['description']})")

    saved_paths = plot_mock_data_models(
        n_sets=args.mock_sets,
        observations_per_set=args.observations_per_set,
        output_dir=args.mock_output_dir,
    )
    for path in saved_paths:
        print(f"Saved {path}")

    combined_output = args.combined_output or args.output
    if combined_output is not None:
        plot_model_previews(
            n_configs=args.num_configs,
            samples_per_config=args.samples_per_config,
            mock=mock,
            output=combined_output,
        )
        print(f"Saved {combined_output}")


if __name__ == "__main__":
    main()
