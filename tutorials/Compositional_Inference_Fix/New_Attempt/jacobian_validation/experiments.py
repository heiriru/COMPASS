"""Validation experiments for correction="gauss_jacobian".

Two experiments, answering two different questions.

``gaussian`` -- the *null test*. Each observation's joint p(g, l_j | x_j) is
exactly Gaussian, which is the one case where the Tweedie construction reduces
algebraically to the pilot form Sigma_0,j^-1 + lambda^-2 I. `gauss_jacobian`
must therefore *tie* `gauss_hierarchical` here; a difference would mean a bug.
What the experiment shows is the cost side: the pilot rule is handed the exact
analytic covariance (an oracle it never has in practice, standing in for a full
extra reverse-diffusion pass), and the Jacobian rule matches it with nothing.

``nongaussian`` -- the *discriminating test*. Each joint is a two-component
Gaussian mixture with unequal weights and opposite-sign g/l correlations, so no
single covariance matrix describes it. This is where the constant-Sigma_0
assumption that every pilot-covariance rule makes is actually wrong, and the
only place an accuracy difference can appear. `gauss_hierarchical` is given the
*exact moment-matched* covariance -- strictly better than any pilot run could
estimate -- so the comparison is deliberately generous to the baseline.

Both use an analytic score, so the network is perfect by construction and every
deviation from the reference posterior is composition error and nothing else.

Rows are appended to CSV as they complete, so a killed run keeps its partial
results and can be resumed by re-running (existing rows are skipped).

Usage:
    python experiments.py --experiment gaussian
    python experiments.py --experiment nongaussian
    python plot.py
"""
import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

from compass.MultiObsSampler import MultiObsSampler  # noqa: E402
from compass.SDE import VESDE  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
NODES = 3                                   # (g, l, x)
CONDITION_MASK = torch.tensor([0.0, 0.0, 1.0])
HIERARCHY = [0]

# Fixed order, used by plot.py for colour assignment. Never reorder.
METHODS = ["uncorrected", "gauss_hierarchical", "gauss_jacobian"]


# ---------------------------------------------------------------------------
# Shared sampling harness
# ---------------------------------------------------------------------------

class PerturbedScore(torch.nn.Module):
    """A imperfect score network, perturbed in *denoiser* space.

    Real network error is an error in the posterior mean it predicts, so the
    honest way to inject it is on the Tweedie denoiser D(y) = y + lambda^2 s(y)
    rather than on the score directly (which would make the perturbation
    lambda-dependent in a way no trained network is):

        D~_j(y) = D_j(y) + eps * ( d_j + C_j (y - mean_j) )

    The constant term ``d_j`` is a per-observation bias -- the error mode that
    matters, because it adds *coherently* across observations while the composed
    posterior contracts like 1/sqrt(n). The linear term ``C_j`` warps the
    denoiser, so it perturbs grad(s_j) as well: that is what tests whether
    reading curvature off the network is more fragile than reading the score,
    which is the main risk `gauss_jacobian` carries over a pilot covariance
    (differentiation amplifies high-frequency error; a pilot averages it away).
    """

    def __init__(self, base, observations, noise, seed):
        super().__init__()
        self.base = base
        self.sde = base.sde
        self.observations = observations
        self.noise = float(noise)
        generator = torch.Generator().manual_seed(seed + 991)
        self.register_buffer(
            "bias", torch.randn(observations, 2, generator=generator,
                                dtype=torch.float64))
        self.register_buffer(
            "warp", torch.randn(observations, 2, 2, generator=generator,
                                dtype=torch.float64) * 0.5)

    def forward(self, x, t, c):
        score = self.base(x, t, c)
        variance = self.sde.lambda_t(t).square().reshape(()).to(torch.float64)
        rows = x.shape[0] // self.observations
        view = score.reshape(self.observations, rows, -1).clone()
        block = x.reshape(self.observations, rows, -1)[:, :, :2].to(torch.float64)
        offset = self.bias[:, None, :] + torch.einsum(
            "nij,nrj->nri", self.warp, block
        )
        view[:, :, :2] = view[:, :, :2] + (
            self.noise * offset / variance
        ).to(x.dtype)
        return view.reshape(x.shape)


def sample_shared(model, observations, correction, num_samples, timesteps,
                  **overrides):
    """Run the compositional sampler and return the shared-coordinate draws."""
    sbim = SimpleNamespace(
        sde=model.sde, model=model,
        output_scale_function=lambda t, value: value,
    )
    sampler = MultiObsSampler(sbim)
    arguments = dict(
        world_size=1, data=torch.zeros(observations, 1),
        condition_mask=CONDITION_MASK, timesteps=timesteps,
        num_samples=num_samples, hierarchy=HIERARCHY,
        prior=(torch.zeros(1), torch.ones(1)),
        local_prior=(torch.zeros(1), torch.ones(1)),
        correction=correction, method="dpm", order=2,
        corrector_steps=0, corrector_steps_interval=10**9,
        final_corrector_steps=0, device="cpu", verbose=False,
        denoise_clamp=None,
    )
    arguments.update(overrides)
    samples = sampler.sample(**arguments)
    return samples[0, :, 0].to(torch.float64), sampler


# ---------------------------------------------------------------------------
# Experiment 1: exactly Gaussian single-observation posteriors (null test)
# ---------------------------------------------------------------------------

JOINT_PRECISION = torch.tensor([[5.0, 4.0], [4.0, 5.0]], dtype=torch.float64)
JOINT_COVARIANCE = torch.linalg.inv(JOINT_PRECISION)


class GaussianScore(torch.nn.Module):
    """Exact diffused score of N(mean_j, Sigma_0) for every observation.

    Model: g ~ N(0,1), l_j ~ N(0,1), x_j = g + l_j + N(0, 0.5^2), whose joint
    posterior precision is [[5,4],[4,5]] with mean (4 x_j / 9) * [1, 1].
    """

    def __init__(self, means, sde):
        super().__init__()
        self.register_buffer("means", means.to(torch.float32))
        self.sde = sde
        self.observations = means.shape[0]

    def forward(self, x, t, c):
        variance = self.sde.lambda_t(t).square().reshape(()).to(torch.float64)
        precision = torch.linalg.inv(
            JOINT_COVARIANCE + variance * torch.eye(2, dtype=torch.float64)
        ).to(torch.float32)
        rows = x.shape[0] // self.observations
        block = x.reshape(self.observations, rows, -1)[:, :, :2]
        score = -torch.einsum(
            "ij,nsj->nsi", precision, block - self.means[:, None, :]
        )
        out = torch.zeros(
            self.observations, rows, x.shape[-1], dtype=x.dtype, device=x.device
        )
        out[:, :, :2] = score
        return out.reshape(x.shape)


def gaussian_case(observations, correction, seed, num_samples, timesteps,
                  score_noise=0.0, **overrides):
    generator = torch.Generator().manual_seed(seed)
    truth = 0.7
    x = truth + torch.randn(observations, generator=generator) \
        + 0.5 * torch.randn(observations, generator=generator)
    means = (4.0 * x / 9.0).unsqueeze(-1).repeat(1, 2)
    model = GaussianScore(means, VESDE(sigma=25.0))
    if score_noise:
        model = PerturbedScore(model, observations, score_noise, seed)

    if correction == "gauss_hierarchical":
        overrides["posterior_covariance"] = JOINT_COVARIANCE.unsqueeze(0)
    shared, sampler = sample_shared(
        model, observations, correction, num_samples, timesteps, **overrides
    )

    # Exact tall posterior: x_j | g ~ N(g, 1 + 0.25).
    precision = 1.0 + observations / 1.25
    reference_mean = float((x.sum() / 1.25) / precision)
    reference_std = float(precision ** -0.5)
    return {
        "mean_error_sigma": abs(float(shared.mean()) - reference_mean) / reference_std,
        "width_ratio": float(shared.std()) / reference_std,
        "wasserstein": float("nan"),
        "network_calls": int(sampler.score_network_calls),
        "jacobian_evaluations": int(getattr(sampler, "_jacobian_evaluations", 0)),
    }


# ---------------------------------------------------------------------------
# Experiment 2: non-Gaussian (mixture) single-observation posteriors
# ---------------------------------------------------------------------------

WEIGHTS = torch.tensor([0.7, 0.3], dtype=torch.float64)
# Different shapes AND opposite-sign g/l correlation: no single covariance
# reproduces this, which is exactly what the pilot rules assume it can.
COMPONENT_COVARIANCE = torch.stack([
    torch.tensor([[0.10, 0.05], [0.05, 0.16]], dtype=torch.float64),
    torch.tensor([[0.26, -0.11], [-0.11, 0.09]], dtype=torch.float64),
])
COMPONENT_OFFSET = torch.tensor([[-0.30, 0.15], [0.55, -0.25]], dtype=torch.float64)


class MixtureScore(torch.nn.Module):
    """Exact diffused score of a per-observation two-component mixture."""

    def __init__(self, centres, sde):
        super().__init__()
        self.register_buffer("centres", centres.to(torch.float64))
        self.sde = sde
        self.observations = centres.shape[0]

    def forward(self, x, t, c):
        variance = self.sde.lambda_t(t).square().reshape(()).to(torch.float64)
        covariance = COMPONENT_COVARIANCE + variance * torch.eye(2, dtype=torch.float64)
        precision = torch.linalg.inv(covariance)
        _, logdet = torch.linalg.slogdet(covariance)

        rows = x.shape[0] // self.observations
        block = x.reshape(self.observations, rows, -1)[:, :, :2].to(torch.float64)
        delta = block[:, :, None, :] - self.centres[:, None, :, :]
        quadratic = torch.einsum("nrki,kij,nrkj->nrk", delta, precision, delta)
        responsibility = torch.softmax(
            torch.log(WEIGHTS)[None, None, :] - 0.5 * quadratic
            - 0.5 * logdet[None, None, :],
            dim=-1,
        )
        component_scores = -torch.einsum("kij,nrkj->nrki", precision, delta)
        score = (responsibility[..., None] * component_scores).sum(dim=-2)
        out = torch.zeros(
            self.observations, rows, x.shape[-1], dtype=x.dtype, device=x.device
        )
        out[:, :, :2] = score.to(x.dtype)
        return out.reshape(x.shape)


def mixture_centres(observations, seed):
    shift = 0.35 * torch.randn(
        observations, generator=torch.Generator().manual_seed(seed + 17),
        dtype=torch.float64,
    )
    return COMPONENT_OFFSET[None] + shift[:, None, None]


def moment_matched_covariance(centres):
    """The best constant Gaussian per observation: exact mixture moments."""
    mean = (WEIGHTS[None, :, None] * centres).sum(dim=1)
    delta = centres - mean[:, None, :]
    scatter = torch.einsum("k,nki,nkj->nij", WEIGHTS, delta, delta)
    within = (COMPONENT_COVARIANCE * WEIGHTS[:, None, None]).sum(0)
    return within[None] + scatter


def exact_shared_marginal(centres, grid, observations):
    """log p(g | x_1..x_n) on a grid, up to a constant.

    p(g | x_1..x_n) propto p(g)^(1-n) prod_j p(g | x_j), and p(g | x_j) is the
    analytic 1-D marginal of observation j's mixture.
    """
    variance = COMPONENT_COVARIANCE[:, 0, 0]
    mean = centres[:, :, 0]
    exponent = -0.5 * (grid[None, None, :] - mean[:, :, None]) ** 2 \
        / variance[None, :, None]
    normalizer = -0.5 * torch.log(2 * math.pi * variance)[None, :, None]
    per_observation = torch.logsumexp(
        torch.log(WEIGHTS)[None, :, None] + normalizer + exponent, dim=1
    )
    log_prior = -0.5 * grid ** 2 - 0.5 * math.log(2 * math.pi)
    return (1 - observations) * log_prior + per_observation.sum(dim=0)


def wasserstein_1d(samples, grid, log_density):
    density = torch.softmax(log_density, dim=0)
    cumulative = torch.cumsum(density, dim=0)
    cumulative = cumulative / cumulative[-1]
    quantiles = (torch.arange(len(samples), dtype=torch.float64) + 0.5) / len(samples)
    positions = torch.searchsorted(
        cumulative.contiguous(), quantiles.clamp(max=1.0)
    ).clamp(max=len(grid) - 1)
    return float((torch.sort(samples).values - grid[positions]).abs().mean())


def nongaussian_case(observations, correction, seed, num_samples, timesteps,
                     score_noise=0.0, **overrides):
    centres = mixture_centres(observations, seed)
    model = MixtureScore(centres, VESDE(sigma=25.0))
    if score_noise:
        model = PerturbedScore(model, observations, score_noise, seed)
    if correction == "gauss_hierarchical":
        overrides["posterior_covariance"] = moment_matched_covariance(centres)
    shared, sampler = sample_shared(
        model, observations, correction, num_samples, timesteps, **overrides
    )

    grid = torch.linspace(-4.0, 4.0, 4001, dtype=torch.float64)
    reference = exact_shared_marginal(centres, grid, observations)
    density = torch.softmax(reference, dim=0)
    reference_mean = float((density * grid).sum())
    reference_std = float(((density * grid ** 2).sum() - reference_mean ** 2) ** 0.5)
    return {
        "wasserstein": wasserstein_1d(shared, grid, reference),
        "mean_error_sigma": abs(float(shared.mean()) - reference_mean) / reference_std,
        "width_ratio": float(shared.std()) / reference_std,
        "network_calls": int(sampler.score_network_calls),
        "jacobian_evaluations": int(getattr(sampler, "_jacobian_evaluations", 0)),
    }


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

FIELDS = ["experiment", "n", "method", "seed", "score_noise", "wasserstein",
          "mean_error_sigma", "width_ratio", "network_calls",
          "jacobian_evaluations"]


def migrate(path):
    """Bring an existing CSV up to the current schema, in place.

    Appending `DictWriter` rows to a file whose header predates a schema change
    silently writes the new column order under the old header, shifting every
    value one place -- the file still parses, and every number in it is wrong.
    So the header is checked against FIELDS on every open, and older rows are
    widened with defaults rather than appended blindly.
    """
    if not path.exists():
        return
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] == FIELDS:
        return
    old_fields = rows[0]
    defaults = {"score_noise": "0.0"}
    migrated = []
    for values in rows[1:]:
        if len(values) == len(FIELDS):
            record = dict(zip(FIELDS, values))       # already new order
        else:
            record = dict(zip(old_fields, values))
        migrated.append({
            field: record.get(field, defaults.get(field, "")) for field in FIELDS
        })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(migrated)
    print(f"migrated {path.name} to the current schema "
          f"({len(migrated)} rows)")


def load_done(path):
    migrate(path)
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["experiment"], row["n"], row["method"], row["seed"],
             str(float(row.get("score_noise") or 0.0)))
            for row in csv.DictReader(handle)
        }


def append(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["gaussian", "nongaussian"],
                        required=True)
    parser.add_argument("--observations", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--score-noise", type=float, nargs="+", default=[0.0],
                        help="Denoiser-space perturbation magnitudes (eps).")
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--timesteps", type=int, default=150)
    arguments = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{arguments.experiment}_metrics.csv"
    done = load_done(path)

    counts = arguments.observations or (
        [2, 8, 32] if arguments.experiment == "gaussian" else [4, 16]
    )
    case = gaussian_case if arguments.experiment == "gaussian" else nongaussian_case

    for noise in arguments.score_noise:
        for observations in counts:
            for method in METHODS:
                for seed in arguments.seeds:
                    key = (arguments.experiment, str(observations), method,
                           str(seed), str(float(noise)))
                    if key in done:
                        print(f"skip {key}", flush=True)
                        continue
                    result = case(
                        observations, method, seed, arguments.num_samples,
                        arguments.timesteps, score_noise=noise,
                    )
                    row = {
                        "experiment": arguments.experiment, "n": observations,
                        "method": method, "seed": seed, "score_noise": noise,
                        **result,
                    }
                    append(path, row)
                    print(
                        f"eps={noise:<5} n={observations:<4} {method:<20} "
                        f"W1={result['wasserstein']:.4f} "
                        f"mean_err={result['mean_error_sigma']:.3f} "
                        f"width={result['width_ratio']:.3f} "
                        f"calls={result['network_calls']}",
                        flush=True,
                    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
