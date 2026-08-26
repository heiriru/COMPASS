#!/usr/bin/env python3
"""The compare.py four-panel figure, with the *exact* single-observation score.

``compare.py`` composes a trained network's single-observation score across the
30 observations. This script runs the identical pipeline -- same observations,
seed, sample count, step count, composition rule, KDE MAP and conditional local
ascent -- with the network replaced by the closed-form score of

    p_lam(g_t, l_t | x_j)

the N = 1 case of ``hierarchy.diffused_score``, i.e. exactly what the network is
trained to output. Every remaining error is therefore the composition rule, the
sampler's initial law and its integrator; nothing in the figure is network error.

Why that is the interesting control for the predictor-only arms: with correctors
switched off, DPM-2 is a deterministic transport, so its output law is exactly
the pushforward of its initial law. Removing the network separates "the rule is
wrong at finite noise" from "the network is wrong" -- the two candidates for the
2.3 sigma shift in ``01_gauss_jacobian_predictor_only.png``.

The exact row score, and why it is a quadrature at all
-----------------------------------------------------
For one observation,

    p(g, l | x) prop exp[-(g - mu_g)^2 / (2 sigma_g^2) + rate g] 1[g <= l <= x]

so under the VESDE kernel N(., lam^2) on both coordinates

    p_lam(g_t, l_t | x) prop int_{-inf}^{x} dg N(g; m, s^2)
                              [Phi((x - l_t)/lam) - Phi((g - l_t)/lam)]

with ``N(g; m, s^2)`` the Gaussian formed by the tilted prior times the kernel:
``s^2 = 1/(1/sigma_g^2 + 1/lam^2)``, ``m = s^2 (mu_g/sigma_g^2 + rate + g_t/lam^2)``.
Without the truncation at ``x`` this integrates in closed form; with it, it is a
bivariate normal probability. So the integral is done numerically -- but on
*nodes placed at* ``min(m, x) + s z``, i.e. exactly where the integrand's mass
is, at every noise level. The integrand is continuous at the truncation (the
bracket vanishes at ``g = x``), so a trapezoid over ``z`` converges fast.

Both score components are then expectations under the same node weights,

    d/dg_t log p = E_w[(g - g_t)] / lam^2
    d/dl_t log p = E_w[ (phi(v) - phi(u)) / (lam D) ],  u = (x - l_t)/lam,
                                                        v = (g - l_t)/lam

so neither depends on how the nodes were chosen, only on their weights. That is
also what makes the module safe under ``torch.func.jvp``, which
``correction="gauss_jacobian"`` needs to differentiate the row score.

Usage:
    python analytic_compare.py --validate          # grade the row score only
    python analytic_compare.py                     # both predictor-only rules
    python analytic_compare.py --methods gauss_jacobian_analytic
    python analytic_compare.py --replot
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare  # noqa: E402
import hierarchy  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# The floor keeps a masked-out node's arithmetic finite: -inf would survive
# softmax as 0 * nan in the weighted sums that follow.
_LOG_ZERO = -1e30


class ExactRowScoreNetwork(torch.nn.Module):
    """Drop-in for ``SBIm.model``: the exact score of ``p_lam(g_t, l_t | x)``.

    Signature and conventions match the trained network exactly -- ``forward``
    takes ``(x, t, c)`` with ``x`` the state ``(g, l, x_obs)``, ``t`` the
    diffusion time and ``c`` the condition mask, and returns a score of the same
    shape. The observed column is returned as zero, as the sampler only ever
    reads the latent columns.

    ``c[:, GLOBAL] == 1`` (the conditional local ascent of stage 3) means ``g``
    is given rather than diffused, and the quadrature collapses to its single
    node at that value.
    """

    def __init__(self, sde, nodes=385, z_max=8.0, chunk=8192):
        super().__init__()
        self.sde = sde
        self.z_max = float(z_max)
        self.chunk = int(chunk)
        self.register_buffer(
            "z", torch.linspace(-z_max, z_max, int(nodes), dtype=torch.float64)
        )

    # -- the two branches ---------------------------------------------------

    def _conditioned(self, g, l_t, obs, lam):
        """``d/dl_t log p_lam(l_t | g, x)``: one node, at the given ``g``."""
        u = (obs - l_t) / lam
        v = (g - l_t) / lam
        v = torch.minimum(v, u - 1e-12)
        log_interval = hierarchy._log_gaussian_interval(v, u)
        log_phi_v = -0.5 * v**2 - hierarchy._LOG_SQRT_2PI
        log_phi_u = -0.5 * u**2 - hierarchy._LOG_SQRT_2PI
        return (torch.exp(log_phi_v - log_interval)
                - torch.exp(log_phi_u - log_interval)) / lam

    def _diffused(self, g_t, l_t, obs, lam):
        """Both score components of the diffused single-observation posterior."""
        variance = lam**2
        prior_variance = hierarchy.SIGMA_G**2
        # Tilted prior exp[-(g-mu)^2/(2 sigma^2) + rate g] = N(g; mu + sigma^2 rate,
        # sigma^2) up to a constant; times the kernel N(g_t; g, lam^2):
        tilted_mean = hierarchy.MU_G + prior_variance * hierarchy.RATE
        s_squared = 1.0 / (1.0 / prior_variance + 1.0 / variance)
        s = math.sqrt(s_squared)
        m = s_squared * (tilted_mean / prior_variance + g_t / variance)

        # Nodes centred on the mass: at m, or against the wall when the kernel
        # would put m past x and the truncated Gaussian piles up below it.
        centre = torch.minimum(m, obs)
        nodes = centre[:, None] + s * self.z[None, :]
        inside = nodes < obs[:, None]

        u = (obs - l_t) / lam
        v = (nodes - l_t[:, None]) / lam
        # Evaluate the interval on admissible arguments only; the inadmissible
        # nodes are removed by the weight, but their arithmetic must stay finite.
        v = torch.where(inside, v, u[:, None] - 1.0)
        log_interval = hierarchy._log_gaussian_interval(v, u[:, None].expand_as(v))

        log_weight = -0.5 * ((nodes - m[:, None]) / s) ** 2 + log_interval
        log_weight = torch.where(
            inside, log_weight, torch.full_like(log_weight, _LOG_ZERO)
        )
        weight = torch.softmax(log_weight, dim=1)

        score_g = (weight * (nodes - g_t[:, None])).sum(dim=1) / variance

        log_phi_v = -0.5 * v**2 - hierarchy._LOG_SQRT_2PI
        log_phi_u = (-0.5 * u**2 - hierarchy._LOG_SQRT_2PI)[:, None]
        derivative = (torch.exp(log_phi_v - log_interval)
                      - torch.exp(log_phi_u - log_interval)) / lam
        return score_g, (weight * derivative).sum(dim=1)

    # -- the network interface ----------------------------------------------

    def forward(self, x, t, c):
        time = torch.as_tensor(t).reshape(-1)[0]
        lam = float(self.sde.lambda_t(time))
        alpha = float(self.sde.alpha_t(time))
        state = x.to(torch.float64)
        mask = c.to(torch.float64)
        # The sampler hands the network alpha * y on latent coordinates and the
        # unscaled value on conditioned ones. The closed forms below live in the
        # y = x_t / alpha coordinate, where the VP kernel *is* the VE kernel at
        # lambda = sigma/alpha -- so one rescale in and one out covers both SDEs.
        # For the VESDE alpha == 1 and every line here is a no-op.
        scale = torch.where(
            mask > 0.5, torch.ones_like(state), torch.full_like(state, alpha)
        )
        state = state / scale
        conditioned = mask[:, hierarchy.GLOBAL_INDEX] > 0.5
        if bool(conditioned.any()) and bool((~conditioned).any()):
            raise NotImplementedError(
                "mixed conditioning on the shared coordinate within one batch"
            )

        g_t = state[:, hierarchy.GLOBAL_INDEX]
        l_t = state[:, hierarchy.LOCAL_INDEX]
        obs = state[:, hierarchy.OBSERVED_INDEX]

        shared_parts, local_parts = [], []
        for start in range(0, state.shape[0], self.chunk):
            stop = min(start + self.chunk, state.shape[0])
            if bool(conditioned.all()):
                shared = torch.zeros_like(g_t[start:stop])
                local = self._conditioned(
                    g_t[start:stop], l_t[start:stop], obs[start:stop], lam
                )
            else:
                shared, local = self._diffused(
                    g_t[start:stop], l_t[start:stop], obs[start:stop], lam
                )
            shared_parts.append(shared)
            local_parts.append(local)

        # d/dx_t = (1/alpha) d/dy on the coordinates that were rescaled above.
        score = torch.stack([
            torch.cat(shared_parts), torch.cat(local_parts),
            torch.zeros_like(g_t),
        ], dim=1) / scale
        return score.to(x.dtype)


class ExactSingleObservationSampler:
    """Stand-in for ``SBIm.sampler``: exact draws from ``p(g, l | x_j)``.

    Only ``gauss_hierarchical`` uses it, and only to estimate the pilot
    covariance. Drawing from the exact single-observation posterior is the
    noiseless limit of the pilot run the learned arm performs, so the pilot
    carries the same Monte-Carlo error at the same ``precision_est_samples`` and
    no network error -- which is the point of this script.
    """

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.calls = 0

    def sample(self, data=None, num_samples=1, device="cpu", **kwargs):
        values = torch.as_tensor(data, dtype=torch.float64).reshape(
            torch.as_tensor(data).shape[0], -1
        )[:, -1].cpu().numpy()
        rng = np.random.default_rng(self.seed + 977 * self.calls)
        self.calls += 1
        draws = np.zeros((len(values), int(num_samples), hierarchy.NODES))
        for index, value in enumerate(values):
            grid, weights, _, _ = hierarchy.single_observation_reference(value)
            cumulative = np.cumsum(weights)
            cumulative /= cumulative[-1]
            g = np.interp(rng.random(int(num_samples)), cumulative, grid)
            local = g + rng.random(int(num_samples)) * (value - g)
            draws[index, :, hierarchy.GLOBAL_INDEX] = g
            draws[index, :, hierarchy.LOCAL_INDEX] = local
            draws[index, :, hierarchy.OBSERVED_INDEX] = value
        return torch.as_tensor(draws, dtype=torch.float32, device=device)


class AnalyticModel:
    """The ``ScoreBasedInferenceModel`` surface ``MultiObsSampler`` actually uses."""

    def __init__(self, sde, device="cpu", nodes=385, seed=0):
        self.sde = sde
        self.model = ExactRowScoreNetwork(sde, nodes=nodes).to(device)
        self.sampler = ExactSingleObservationSampler(seed=seed)

    @staticmethod
    def output_scale_function(t, scores):
        """Identity: the module already returns the score itself."""
        return scores


# ---------------------------------------------------------------------------
# Methods -- the predictor-only arms of compare.py, analytic row score
# ---------------------------------------------------------------------------

METHODS = {
    "gauss_jacobian_analytic_predictor_only": {
        "source": "gauss_jacobian_predictor_only",
        "label": "DPM2 + gauss_jacobian (predictor only, exact row score)",
        "colour": compare.TEAL,
    },
    "gauss_hierarchical_analytic_predictor_only": {
        "source": "gauss_hierarchical_predictor_only",
        "label": "DPM2 + gauss_hierarchical (predictor only, exact row score)",
        "colour": compare.CORAL,
    },
    "gauss_jacobian_analytic": {
        "source": "gauss_jacobian",
        "label": "DPM2 + gauss_jacobian (exact row score)",
        "colour": compare.TEAL,
    },
    "gauss_hierarchical_analytic": {
        "source": "gauss_hierarchical",
        "label": "DPM2 + gauss_hierarchical (exact row score)",
        "colour": compare.CORAL,
    },
    # No predictor-only counterpart: annealed Langevin is corrector-only, so the
    # arm below is the whole method rather than half of it.
    "langevin_fnpe_analytic": {
        "source": "langevin_fnpe",
        "label": "Langevin + F-NPSE (exact row score)",
        "colour": compare.GOLD,
    },
    "gauss_jacobian_analytic_correctors2": {
        "source": "gauss_jacobian_correctors2",
        "label": "DPM2 + gauss_jacobian (2 correctors, exact row score)",
        "colour": compare.TEAL,
    },
    # Note on timing: this arm's pilot covariance comes from exact
    # single-observation draws rather than a reverse-diffusion pilot run, so its
    # wall clock excludes a cost the learned arm really pays. Only the
    # post-pilot sampling is comparable between the two.
    "gauss_hierarchical_analytic_correctors2": {
        "source": "gauss_hierarchical_correctors2",
        "label": "DPM2 + gauss_hierarchical (2 correctors, exact row score)",
        "colour": compare.CORAL,
    },
    # The apples-to-apples question: F-NPSE's sampler, gauss_jacobian's field.
    # With exact row scores the Jacobian rule's composed score is the most
    # accurate of the three at every lambda, yet its DPM2 draws are the worst --
    # which points at the sampler, not the rule. Annealed Langevin is
    # corrector-only, so it relaxes onto the composed density at every level
    # instead of transporting an initial law through it.
    "gauss_jacobian_analytic_langevin": {
        "label": "Langevin + gauss_jacobian (exact row score)",
        "colour": compare.TEAL,
        "sample_kwargs": {
            "method": "langevin", "correction": "gauss_jacobian",
            "corrector_steps": 10, "snr": 0.2,
        },
    },
}

DEFAULT_METHODS = ["gauss_jacobian_analytic_predictor_only",
                   "gauss_hierarchical_analytic_predictor_only"]

for _name, _entry in METHODS.items():
    # "source" mirrors a compare.py arm exactly; "sample_kwargs" defines a
    # configuration that has no counterpart there.
    compare.METHODS[_name] = {
        "label": _entry["label"],
        "colour": _entry["colour"],
        "note": "exact single-observation score, no network",
        "score_note": "exact single-observation score",
        "sample_kwargs": dict(
            _entry["sample_kwargs"] if "sample_kwargs" in _entry
            else compare.METHODS[_entry["source"]]["sample_kwargs"]
        ),
    }


# ---------------------------------------------------------------------------
# Validation: the row score against hierarchy.diffused_score
# ---------------------------------------------------------------------------

def validate(model, x, lambdas, states, device, seed=99):
    """Relative RMS error of the module against the N = 1 exact quadrature.

    ``hierarchy.diffused_score`` reaches the same target by a different route --
    a uniform grid over the clean support rather than mass-placed nodes -- so
    agreement to quadrature precision grades this module rather than restating
    it.
    """
    mask = hierarchy.CONDITION_MASK.to(device)
    worst = 0.0
    for lam in lambdas:
        errors, magnitudes = [], []
        for index, value in enumerate(np.asarray(x, dtype=np.float64)):
            grid = hierarchy.quadrature_grid([value], device=device)
            shared, local, _, _ = hierarchy.sample_diffused(
                [value], lam, states, seed + 1000 * index
            )
            exact_g, exact_l = hierarchy.diffused_score(
                shared, local, [value], lam, grid
            )
            state = torch.zeros(states, hierarchy.NODES, dtype=torch.float64,
                                device=device)
            state[:, hierarchy.GLOBAL_INDEX] = torch.as_tensor(shared, device=device)
            state[:, hierarchy.LOCAL_INDEX] = torch.as_tensor(
                local[:, 0], device=device
            )
            state[:, hierarchy.OBSERVED_INDEX] = float(value)
            t = model.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1)
            with torch.no_grad():
                predicted = model.model(
                    x=state, t=t.to(device),
                    c=mask.unsqueeze(0).repeat(states, 1),
                ).to(torch.float64)
            exact = torch.stack([exact_g, exact_l[:, 0]], dim=1).to(device)
            errors.append((predicted[:, :2] - exact).pow(2).sum().item())
            magnitudes.append(exact.pow(2).sum().item())
        relative = math.sqrt(sum(errors) / max(sum(magnitudes), 1e-30))
        worst = max(worst, relative)
        print(f"  lam={float(lam):<6g} relative RMS vs exact quadrature {relative:.3e}")
    return worst


def validate_conditional(model, x, device, count=512, lam=1e-2, seed=5):
    """The conditioned branch against ``p(l | g, x) = Uniform(g, x)`` diffused."""
    rng = np.random.default_rng(seed)
    value = float(np.asarray(x).reshape(-1)[0])
    g = float(np.asarray(x).reshape(-1).min()) - 0.4
    local = g + rng.random(count) * (value - g) + lam * rng.standard_normal(count)
    state = torch.zeros(count, hierarchy.NODES, dtype=torch.float64, device=device)
    state[:, hierarchy.GLOBAL_INDEX] = g
    state[:, hierarchy.LOCAL_INDEX] = torch.as_tensor(local, device=device)
    state[:, hierarchy.OBSERVED_INDEX] = value
    mask = hierarchy.CONDITION_MASK.clone()
    mask[hierarchy.GLOBAL_INDEX] = 1.0
    t = model.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1)
    with torch.no_grad():
        predicted = model.model(
            x=state, t=t.to(device), c=mask.to(device).unsqueeze(0).repeat(count, 1)
        ).to(torch.float64)[:, hierarchy.LOCAL_INDEX]
    # Closed form: d/dl log[Phi((x-l)/lam) - Phi((g-l)/lam)].
    step = 1e-5
    def log_density(l_value):
        upper = torch.as_tensor((value - l_value) / lam)
        lower = torch.as_tensor((g - l_value) / lam)
        return hierarchy._log_gaussian_interval(lower, upper)
    reference = (log_density(torch.as_tensor(local) + step)
                 - log_density(torch.as_tensor(local) - step)) / (2 * step)
    reference = reference.to(device)
    relative = float((predicted - reference).pow(2).mean().sqrt()
                     / reference.pow(2).mean().sqrt())
    print(f"  conditioned branch (lam={lam:g}) relative RMS "
          f"vs finite differences {relative:.3e}")
    return relative


def validate_jacobian(model, x, device, lam=0.2, count=8):
    """``torch.func.jvp`` through the module against central differences."""
    from torch.func import jvp

    value = float(np.asarray(x).reshape(-1)[0])
    shared, local, _, _ = hierarchy.sample_diffused([value], lam, count, 3)
    state = torch.zeros(count, hierarchy.NODES, dtype=torch.float64, device=device)
    state[:, hierarchy.GLOBAL_INDEX] = torch.as_tensor(shared, device=device)
    state[:, hierarchy.LOCAL_INDEX] = torch.as_tensor(local[:, 0], device=device)
    state[:, hierarchy.OBSERVED_INDEX] = value
    mask = hierarchy.CONDITION_MASK.to(device).unsqueeze(0).repeat(count, 1)
    t = model.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1).to(device)

    worst = 0.0
    for feature in (hierarchy.GLOBAL_INDEX, hierarchy.LOCAL_INDEX):
        tangent = torch.zeros_like(state)
        tangent[:, feature] = 1.0
        _, forward = jvp(lambda s: model.model(x=s, t=t, c=mask), (state,), (tangent,))
        step = 1e-4
        with torch.no_grad():
            plus = model.model(x=state + step * tangent, t=t, c=mask)
            minus = model.model(x=state - step * tangent, t=t, c=mask)
        difference = (plus - minus) / (2 * step)
        relative = float((forward[:, :2] - difference[:, :2]).pow(2).mean().sqrt()
                         / difference[:, :2].pow(2).mean().sqrt())
        worst = max(worst, relative)
        print(f"  jvp column {feature} vs central differences {relative:.3e}")
    return worst


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=list(METHODS))
    parser.add_argument("--observations", type=int, default=compare.OBSERVATIONS)
    parser.add_argument("--num-samples", type=int, default=compare.NUM_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=compare.TIMESTEPS)
    parser.add_argument("--quadrature-nodes", type=int, default=385)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--excursion-sigma", type=float, default=15.0)
    parser.add_argument("--kde-bandwidth", default=None)
    parser.add_argument("--map-timesteps", type=int, default=200)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-eps", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--validate", action="store_true",
                        help="grade the row score and exit")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--replot", action="store_true")
    arguments = parser.parse_args()

    if arguments.kde_bandwidth is not None:
        try:
            arguments.kde_bandwidth = float(arguments.kde_bandwidth)
        except ValueError:
            pass

    compare.configure_style()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    from compass.SDE import VESDE
    sde = VESDE(**{"sigma": compare.recipe.SDE_KWARGS["sigma"]})
    model = AnalyticModel(sde, device=arguments.device,
                          nodes=arguments.quadrature_nodes, seed=arguments.seed)

    _, _, x = hierarchy.observations(arguments.observations, arguments.seed)
    if arguments.validate:
        print("Exact row score vs hierarchy.diffused_score (N = 1):")
        validate(model, x[:4], compare.SCORE_LAMBDAS, 256, arguments.device)
        validate_conditional(model, x, arguments.device)
        print("Forward-mode differentiability (needed by gauss_jacobian):")
        validate_jacobian(model, x, arguments.device)
        return

    # build_problem's only use of the checkpoint is to produce the score model;
    # everything else it returns is an exact reference computed from x alone.
    compare.recipe.train_or_load = lambda *a, **k: (model, 0.0)
    problem = compare.build_problem(
        None, "analytic", 0, arguments.observations, arguments.seed,
        arguments.device,
    )
    print(f"true g = {problem['global_truth']:.5f}   exact posterior: "
          f"mean {problem['global_mean']:.5f}, sd {problem['global_std']:.5f}, "
          f"mode {problem['global_mode']:.5f}, wall at {problem['x'].min():.5f}")

    rows = []
    for name in arguments.methods:
        archive_path = arguments.output_dir / f"{name}.npz"
        if arguments.replot:
            with np.load(archive_path, allow_pickle=True) as archive:
                stored = {key: archive[key] for key in archive.files}
            stored["global_map"] = float(stored["global_map"])
            stored["excluded"] = int(stored["excluded"])
            metrics = json.loads(str(stored["metrics_json"]))
            compare.plot_method(problem, name, stored, metrics,
                                arguments.output_dir / f"01_{name}.png")
            continue

        print(f"\n=== {compare.METHODS[name]['label']} ===")
        kwargs = dict(compare.METHODS[name]["sample_kwargs"])
        samples, sampler, runtime = compare.draw(
            model, problem, kwargs, arguments.num_samples, arguments.timesteps,
            arguments.seed, arguments.denoise_clamp, verbose=arguments.verbose,
        )
        print(f"[{name}] sampled in {runtime:.1f}s, "
              f"{sampler.score_network_calls} score calls")

        result = compare.kde_map_pipeline(model, problem, samples, arguments)
        metrics = compare.evaluate(problem, result, sampler, runtime, [])
        metrics["method"] = name
        scalar = {key: value for key, value in metrics.items()
                  if np.isscalar(value) or isinstance(value, str)}
        print(json.dumps(scalar, indent=2, default=float))
        rows.append(metrics)

        np.savez_compressed(
            archive_path,
            x=problem["x"], global_grid=problem["global_grid"],
            global_density=problem["global_density"],
            global_truth=problem["global_truth"],
            exact_local_mean=problem["local_mean"],
            exact_local_std=problem["local_std"],
            metrics_json=json.dumps(scalar, default=float),
            score_rows_json="[]",
            score_grid=metrics["score_grid"],
            implied_score=metrics["implied_score"],
            exact_score=metrics["exact_score"],
            **{key: value for key, value in result.items()},
        )
        compare.plot_method(problem, name, result, metrics,
                            arguments.output_dir / f"01_{name}.png")

    if rows:
        compare.write_metrics(arguments.output_dir / "methods_analytic.csv", rows)


if __name__ == "__main__":
    main()
