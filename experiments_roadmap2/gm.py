#!/usr/bin/env python3
"""Roadmap 2.8 / 2.9 / 2.10 on the Gaussian-mixture hierarchy, where every score and
every tall posterior is exact.

    python gm.py train   --arm base --seed 7
    python gm.py score   --arm base --seed 7          # per-lambda score error vs the oracle
    python gm.py compose --arm base --seed 7 [--sample-eps 1e-5]

Arms (all on the benchmark's recipe: h32d3wide, 200k simulations, six_weighted menu,
log_sigma time sampling, VESDE sigma 15, batch 1024, lr 5e-4, patience 40, final weights):

    base      the recipe
    tsm_g     2.9: target score matching on g      T_g = grad log p(g) + (l - g)/tau_l^2
    tsm_gl    2.9: ... and on the local l too       T_l = -(l - g)/tau_l^2 + (x - l)/sigma_x^2
    fisher    2.8: Fisher-identity consistency penalty on g (weight 1)
    lo        2.10 control: trained down to t = 1e-5 (lambda 0.0032) instead of 1e-3 (0.032)
    edm_lo    2.10: EDM c_skip/c_out/c_in parameterisation, sigma_data 2.5, t >= 1e-5

``score`` measures, at each lambda, the relative RMSE of the network's posterior score
p(g, l | x) against the exact one, and the part of the g-error that survives averaging
over x at fixed g_t -- the bias b(g_t) that a composition over n observations multiplies
by n. ``compose`` runs the benchmark's own sweep (run_scaling.py) on this network.

The library is imported from ``COMPASS_SRC`` (a frozen snapshot of the roadmap branch).
"""

import os
import sys
from pathlib import Path

CPU_THREAD_LIMIT = 3
_cpus = sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]
os.sched_setaffinity(0, _cpus)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = str(len(_cpus))
print(f"CPU cap: {len(_cpus)} logical CPUs {_cpus}", flush=True)

SRC = os.environ.get("COMPASS_SRC", str(Path(__file__).resolve().parents[1] / "src"))
GM = Path("/export/home/rheinric/COMPASS/Gaussian_mixtures_compositional_score_modelling")
OUT = Path(os.environ.get("ROADMAP_OUT", "/export/data/rheinric/roadmap2/gm"))
sys.path[:0] = [SRC, str(GM)]

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import compass  # noqa: E402
assert Path(compass.__file__).resolve().is_relative_to(Path(SRC).resolve()), compass.__file__
import model as M  # noqa: E402

MIX = M.DEFAULT
RECIPE = dict(config="h32d3wide", simulations=200_000, batch_size=1024, lr=5e-4,
              max_epochs=200, patience=40, sigma=15.0)
ARMS = {
    "base": {},
    "tsm_g": {"tsm": "g"},
    "tsm_gl": {"tsm": "gl"},
    "fisher": {"fisher": 1.0},
    "lo": {"time_eps": 1e-5},
    "edm_lo": {"time_eps": 1e-5, "parameterization": "edm", "sigma_data": 2.5},
    # covariance output on the network (compass.covariance_output), second-order DSM
    "cov": {"covariance": "joint"},          # readout and trunk both trained by the NLL
    "cov_det": {"covariance": "detached"},   # readout only: the score training is unchanged
}
COV_RANK = 4
if os.environ.get("ROADMAP_SMOKE"):
    RECIPE.update(simulations=4000, max_epochs=2, batch_size=256)
LAMBDAS = (0.003, 0.006, 0.01, 0.02, 0.03, 0.06, 0.1, 0.3, 1.0, 3.0)


def directory(arm, seed):
    return OUT / f"{arm}_s{seed}"


def mixture_prior_score(g, lam):
    """Diffused score of the mixture prior on g; ``lam`` broadcasts against ``g``."""
    centres = torch.as_tensor(MIX.components, dtype=g.dtype, device=g.device)
    log_w = torch.log(torch.as_tensor(MIX.weights, dtype=g.dtype, device=g.device))
    var = MIX.tau_g ** 2 + torch.as_tensor(lam, dtype=g.dtype, device=g.device) ** 2
    delta = g.unsqueeze(-1) - centres                          # (..., 1, 2) for g (..., 1)
    terms = log_w - 0.5 * delta.square() / var.unsqueeze(-1)
    resp = torch.softmax(terms, dim=-1)
    return (resp * (-delta / var.unsqueeze(-1))).sum(-1)


def tsm_target(which):
    """Clean joint score of the latent nodes (Trainer global_target_fn)."""
    def target(rows):
        g, l, x = rows[:, 0:1], rows[:, 1:2], rows[:, 2:3]
        t_g = mixture_prior_score(g, torch.zeros_like(g)) + (l - g) / MIX.tau_l ** 2
        if which == "g":
            return t_g
        t_l = -(l - g) / MIX.tau_l ** 2 + (x - l) / MIX.sigma_x ** 2
        return torch.cat([t_g, t_l], 1)
    indices = [0] if which == "g" else [0, 1]
    bound = torch.tensor([1 / MIX.tau_l ** 2 + 1 / MIX.tau_g ** 2,
                          1 / MIX.tau_l ** 2 + 1 / MIX.sigma_x ** 2])[:len(indices)]

    def variance(noisy, lam):
        return bound.to(noisy).expand_as(noisy)
    return target, indices, variance


def build(arm):
    from compass import ScoreBasedInferenceModel
    import train as T
    spec = ARMS[arm]
    return ScoreBasedInferenceModel(
        nodes_size=M.NODES, arch="tokenized", sde_type="vesde", sigma=RECIPE["sigma"],
        parameterization=spec.get("parameterization", "eps"),
        parameterization_sigma_data=spec.get("sigma_data", 1.0),
        **T.CONFIGS[RECIPE["config"]])


def train(arguments):
    import train as T
    from compass.consistency import FisherConsistency
    spec = ARMS[arguments.arm]
    out = directory(arguments.arm, arguments.seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    n = RECIPE["simulations"]
    rows = MIX.simulate(n, np.random.default_rng([arguments.seed, 1]))
    val = MIX.simulate(max(2000, n // 10), np.random.default_rng([arguments.seed, 2]))

    def split(r):
        return (torch.as_tensor(r[:, M.LATENT_INDICES], dtype=torch.float32),
                torch.as_tensor(r[:, M.OBSERVED_INDICES], dtype=torch.float32))

    network = build(arguments.arm)
    extra = {}
    if "tsm" in spec:
        fn, indices, variance = tsm_target(spec["tsm"])
        extra.update(global_target_fn=fn, global_target_indices=indices, tsm_variance_fn=variance)
    if "fisher" in spec:
        extra["fisher_consistency"] = FisherConsistency(
            lambda g, lam: mixture_prior_score(g, lam), indices=M.HIERARCHY,
            observed=M.OBSERVED_INDICES, weight=spec["fisher"], anchors=16, neighbours=64,
            min_ess=8.0)
    if "time_eps" in spec:
        extra["time_eps"] = spec["time_eps"]
    if "covariance" in spec:
        from compass.covariance_output import CovarianceObjective, enable_covariance_output
        scale = torch.as_tensor(rows.std(0), dtype=torch.float32)
        head = enable_covariance_output(network, rank=COV_RANK, node_scale=scale)
        extra["covariance_objective"] = CovarianceObjective(
            head, weight=float(os.environ.get("COV_WEIGHT", "1.0")),
            detach_trunk=spec["covariance"] == "detached")
    started = time.time()
    network.train(*split(rows), *split(val), batch_size=RECIPE["batch_size"],
                  max_epochs=RECIPE["max_epochs"], early_stopping_patience=RECIPE["patience"],
                  lr=RECIPE["lr"], time_sampling="log_sigma", device=arguments.device,
                  verbose=False, path=str(out), mask_sampler=T.six_weighted_menu(network), **extra)
    record = {"arm": arguments.arm, "spec": spec, "seed": arguments.seed, "recipe": RECIPE,
              "minutes": (time.time() - started) / 60, "epochs": len(network.trainer.train_loss),
              "train_loss": network.trainer.train_loss, "val_loss": network.trainer.val_loss,
              "parameters": sum(p.numel() for p in network.model.parameters()),
              "compass": compass.__file__}
    if "fisher" in spec:
        history = np.array(extra["fisher_consistency"].history, dtype=float)
        record["fisher_history_tail"] = history[-200:].tolist()
    (out / "training.json").write_text(json.dumps(record, indent=2))
    print(f"trained {arguments.arm} s{arguments.seed}: {record['epochs']} epochs, "
          f"{record['minutes']:.1f} min, best val {min(network.trainer.val_loss):.4f}", flush=True)


def load(arm, seed, device):
    from compass import ScoreBasedInferenceModel
    if "covariance" in ARMS[arm]:
        from compass.covariance_output import enable_covariance_output
        network = build(arm)
        enable_covariance_output(network, rank=COV_RANK)
        state = torch.load(directory(arm, seed) / "Model.pt", map_location="cpu", weights_only=False)
        network.model.load_state_dict(state.get("model_state_dict", state))
    else:
        network = ScoreBasedInferenceModel.load(str(directory(arm, seed) / "Model.pt"), device=device)
    network.model.to(device).eval()
    return network


NETCOV = "_netcov"


def register_netcov_methods():
    """GM methods with the network's covariance output in place of the Jacobian:
    ``<jacobian method>_netcov``; plus ``gauss_hierarchical_mixture_prior`` (pilot run)."""
    import methods
    from compass.MultiObsSampler import MultiObsSampler
    from compass.covariance_output import NetworkCovarianceHead
    for name in ("gauss_jacobian", "gauss_jacobian_mixture_prior"):
        methods.METHODS[name + NETCOV] = dict(methods.METHODS[name], label=name + NETCOV)
    methods.METHODS["gauss_hierarchical_mixture_prior"] = dict(
        methods.METHODS["gauss_hierarchical"], label="gauss_hierarchical_mixture_prior",
        mixture_prior=True)
    original = methods.draw

    def draw(network, x, method, arguments, device, mixture=M.DEFAULT):
        if not method.endswith(NETCOV):
            return original(network, x, method, arguments, device, mixture)
        head = NetworkCovarianceHead(network, M.LATENT_INDICES, M.OBSERVED_INDICES)
        saved = MultiObsSampler._covariance_head, arguments.jacobian_refresh
        MultiObsSampler._covariance_head = lambda self: head
        arguments.jacobian_refresh = 1        # one cheap forward pass per call
        try:
            return original(network, x, method, arguments, device, mixture)
        finally:
            MultiObsSampler._covariance_head, arguments.jacobian_refresh = saved
    methods.draw = draw


def score(arguments):
    """Per-lambda error of the posterior score p(g, l | x) against the exact one."""
    import oracle
    device = arguments.device
    network = load(arguments.arm, arguments.seed, device)
    exact = oracle.OracleModel(MIX, sigma=RECIPE["sigma"], device=device)
    rng = np.random.default_rng([arguments.seed, 99])
    rows = torch.as_tensor(MIX.simulate(arguments.points, rng), dtype=torch.float32, device=device)
    mask = torch.tensor([0.0, 0.0, 1.0], device=device).expand(len(rows), 3)
    out = {}
    torch.manual_seed(1234)
    for lam in LAMBDAS:
        t = network.sde.time_of_lambda(torch.full((len(rows), 1), lam, device=device))
        noisy = rows.clone()
        noisy[:, :2] += lam * torch.randn_like(noisy[:, :2])
        with torch.no_grad():
            s_net = network.output_scale_function(t, network.model(x=noisy, t=t, c=mask))
            s_ex = exact.output_scale_function(t, exact.model(x=noisy, t=t, c=mask)).to(s_net)
        err = (s_net - s_ex)[:, :2].double().cpu()
        ref = s_ex[:, :2].double().cpu()
        # Bias at fixed g_t: E[s_net - s_exact | g_t], since E_x[s_exact(g_t | x)] is the
        # prior score exactly. Estimated in 50 equal-count bins of g_t.
        g_t = noisy[:, 0].cpu().numpy()
        order = np.argsort(g_t)
        bins = np.array_split(order, 50)
        bias = np.array([err[b, 0].mean().item() for b in bins])
        within = np.array([err[b, 0].var().item() / len(b) for b in bins])
        out[str(lam)] = {
            "rel_rmse_g": float(err[:, 0].square().mean().sqrt() / ref[:, 0].square().mean().sqrt()),
            "rel_rmse_l": float(err[:, 1].square().mean().sqrt() / ref[:, 1].square().mean().sqrt()),
            "rms_bias_g": float(np.sqrt(np.mean(bias ** 2))),
            "rms_bias_g_noise": float(np.sqrt(np.mean(within))),
            "rms_err_g": float(err[:, 0].square().mean().sqrt()),
            "rms_score_g": float(ref[:, 0].square().mean().sqrt()),
        }
        print(f"{arguments.arm} s{arguments.seed} lambda {lam:<6} " + " ".join(
            f"{k} {v:.4g}" for k, v in out[str(lam)].items()), flush=True)
    (directory(arguments.arm, arguments.seed) / "score.json").write_text(json.dumps(out, indent=2))


def info(arguments):
    """Per-lambda error of the backward information K = Cov(theta_0|y,x)^-1 - lam^-2 I against
    the exact one: FD Jacobian of the network (what gauss_jacobian_FD uses) vs the network's
    covariance output (cov arms)."""
    import oracle
    from compass.experimental.multilevel import finite_difference_jacobian, jacobian_precision
    device = arguments.device
    network = load(arguments.arm, arguments.seed, device)
    exact = oracle.OracleModel(MIX, sigma=RECIPE["sigma"], device=device)
    rng = np.random.default_rng([arguments.seed, 77])
    rows = torch.as_tensor(MIX.simulate(4096, rng), dtype=torch.float32, device=device)
    x_obs = rows[:, 2:3]
    head = None
    if "covariance" in ARMS[arguments.arm]:
        from compass.covariance_output import NetworkCovarianceHead
        head = NetworkCovarianceHead(network, M.LATENT_INDICES, M.OBSERVED_INDICES)

    def score_of(model, scale_fn):
        def f(y, lam):
            n = y.shape[0]
            t = network.sde.time_of_lambda(torch.full((n, 1), lam, device=device))
            state = torch.cat([y.float(), x_obs.repeat(n // len(x_obs), 1)], 1)
            c = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
            with torch.no_grad():
                return scale_fn(t, model(x=state, t=t, c=c))[:, :2].double()
        return f
    s_net = score_of(network.model, network.output_scale_function)
    s_ex = score_of(exact.model, exact.output_scale_function)
    eye = torch.eye(2, dtype=torch.float64, device=device)
    out = {}
    torch.manual_seed(4321)
    for lam in (0.03, 0.06, 0.1, 0.3, 1.0, 3.0):
        y = rows[:, :2] + lam * torch.randn_like(rows[:, :2])
        J_ex = finite_difference_jacobian(lambda p, l: s_ex(p, l), y.double(), lam, step=1e-3)
        J_fd = finite_difference_jacobian(lambda p, l: s_net(p, l), y.double(), lam)
        K_ex = jacobian_precision(J_ex, lam) - eye / lam ** 2
        K_fd = jacobian_precision(J_fd, lam) - eye / lam ** 2

        def rel(K):
            return float(((K - K_ex).flatten(1).norm(dim=1) / K_ex.flatten(1).norm(dim=1).clamp_min(1e-9)).median())
        row = {"info_rel_err_fd": rel(K_fd)}
        if head is not None:
            K_net = head.information(y, x_obs, torch.full((len(y),), lam, device=device))
            row["info_rel_err_netcov"] = rel(K_net)
        out[str(lam)] = row
        print(f"{arguments.arm} s{arguments.seed} lambda {lam:<5} " + " ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
    (directory(arguments.arm, arguments.seed) / "info.json").write_text(json.dumps(out, indent=2))


def compose(arguments):
    import methods
    import run_scaling
    import train as T
    network = load(arguments.arm, arguments.seed, arguments.device)
    parameters = sum(p.numel() for p in network.model.parameters())
    # The device is chosen by the caller (CUDA_VISIBLE_DEVICES); never let the sweep's own
    # autocvd search pick another card. The CPU cap is already in place.
    run_scaling.configure_runtime = lambda gpu=True, **_: {
        "active_cpus": len(os.sched_getaffinity(0)), "host_cpus": os.cpu_count(),
        "percent": 100.0 * len(os.sched_getaffinity(0)) / os.cpu_count()}
    run_scaling.load_network = lambda args, train: (network, parameters,
                                                    directory(arguments.arm, arguments.seed))
    methods.COMMON["eps"] = float(arguments.sample_eps)
    tag = f"eps{arguments.sample_eps:g}" + (arguments.tag_suffix or "")
    method_list = arguments.methods or ["gauss_jacobian", "gauss_jacobian_mixture_prior"]
    if any(m.endswith(NETCOV) or m == "gauss_hierarchical_mixture_prior" for m in method_list):
        register_netcov_methods()
    argv = ["--regimes", "asymmetric", "symmetric", "--observations",
            *[str(n) for n in arguments.observations],
            "--methods", *method_list,
            "--replicates", str(arguments.replicates), "--draws", str(arguments.draws),
            "--device", arguments.device, "--tag", tag,
            "--artifacts", str(directory(arguments.arm, arguments.seed))]
    return run_scaling.main(argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "score", "compose", "info"))
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--sample-eps", type=float, default=1e-3)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--observations", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--tag-suffix", default="")
    arguments = parser.parse_args()
    {"train": train, "score": score, "compose": compose, "info": info}[arguments.action](arguments)


if __name__ == "__main__":
    main()
