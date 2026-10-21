"""COMPASS networks and composition rules for hierarchical Two Moons.

Three networks, all trained on the same single-group simulations with the
recipe of ``Arruda/paper_benchmarks`` (VESDE sigma 8, eps floor 1e-5,
``PreconditionedScore`` with the exact N(0, I) prior of the probit globals as the
residual baseline, six_weighted mask menu, log_sigma time sampling, fixed
epochs, final weights):

``legacy``     production per-node network (``nodes_size=8``), the network every
               production rule has been benchmarked on.
``attn_hier``  tokenized trunk with ``hierarchical`` latent attention (g sees g,
               l sees g and l): the g output is the marginal p_t(g | x_j) score.
``attn_full``  tokenized trunk with ``full`` latent attention: the joint score of
               p_t(g, l | x_j).

Every rule then composes the N groups at sampling time. Methods are named
``<network>:<rule>`` in the results; the bare production names run on
``legacy``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import task as T

HERE = Path(__file__).resolve().parent
COMPASS = HERE.parents[1]
for _path in (COMPASS / "src", COMPASS / "Arruda" / "paper_benchmarks",
              COMPASS / "ModeTree" / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

EPS = 1e-5
SDE_KWARGS = {"sde_type": "vesde", "sigma": 8.0}
NETWORK = {"hidden_size": 64, "depth": 4, "num_heads": 4, "mlp_ratio": 2}
NETWORKS = {
    "legacy": {"arch": "legacy", "attention": None},
    "attn_hier": {"arch": "tokenized", "attention": "hierarchical"},
    "attn_full": {"arch": "tokenized", "attention": "full"},
    # Trained on probit globals from N(0, 1.5^2) (1.5x the prior's sd, same 100k simulations);
    # the residual baseline is that training prior, and the sampler divides it out per group
    # (set_training_prior) and applies the target N(0, 1) once. Gaussian-covariance rules only.
    "attn_hier_w1.5": {"arch": "tokenized", "attention": "hierarchical", "train_scale": 1.5},
    "legacy_w1.5": {"arch": "legacy", "attention": None, "train_scale": 1.5},
    # Target score matching on the globals (Trainer global_target_fn): the global target is
    # blended toward grad_z [log p(z) + log p(theta | z)] at low noise (w = 1/(1+lam^2)).
    "attn_hier_tsm": {"arch": "tokenized", "attention": "hierarchical", "tsm": True},
}
TIMESTEPS = 100
MAX_ROWS = 200_000           # network rows (groups x draws) per sampler call
JACOBIAN_MAX_ROWS = 6_000
MARGINAL_MAX_ROWS = 32_000

DPM = {"method": "dpm", "order": 2, "corrector_steps": 0,
       "final_corrector_steps": 0, "terminal_corrector_steps": 0}
FD_SCHEDULE = {"jacobian_refresh": 1, "jacobian_refresh_low_noise": 10,
               "jacobian_refresh_lambda": 0.3}

# name -> (network, MarginalSchurSampler settings or None, sample kwargs)
METHODS = {
    # production rules on the production network
    "uncorrected": ("legacy", None, {**DPM, "correction": "uncorrected"}),
    "gauss": ("legacy", None, {**DPM, "correction": "gauss",
                               "precision_est_samples": 500}),
    "gauss_hierarchical": ("legacy", None, {**DPM, "correction": "gauss_hierarchical",
                                            "precision_est_samples": 500}),
    "gauss_jacobian": ("legacy", None, {**DPM, "correction": "gauss_jacobian",
                                        "jacobian_refresh": 5}),
    "gauss_jacobian_FD": ("legacy", None, {**DPM, "correction": "gauss_jacobian_FD",
                                           **FD_SCHEDULE}),
    "langevin_fnpe": ("legacy", None, {"method": "langevin", "correction": "fnpe",
                                       "corrector_steps": 10, "snr": 0.2}),
    # latent attention: marginal g score from the architecture (Arruda sum)
    "attn_hier:arruda_direct": ("attn_hier", {"estimator": "direct"},
                                {**DPM, "correction": "arruda_schur"}),
    "attn_hier:arruda_gauss_direct": ("attn_hier", {"estimator": "direct"},
                                      {**DPM, "correction": "arruda_schur_gauss",
                                       **FD_SCHEDULE}),
    "attn_hier:gauss_jacobian_FD": ("attn_hier", None,
                                    {**DPM, "correction": "gauss_jacobian_FD", **FD_SCHEDULE}),
    # FD Jacobian refreshed every 5 steps above lambda 0.3 (10 below, as in
    # FD_SCHEDULE), DPM-2 with churn 4 instead of the default 2.
    "attn_hier:gauss_jacobian_FD_r5_churn4": (
        "attn_hier", None,
        {**DPM, "correction": "gauss_jacobian_FD", "churn_eta": 4.0,
         "jacobian_refresh": 5, "jacobian_refresh_low_noise": 10,
         "jacobian_refresh_lambda": 0.3}),
    # wider global training prior (1.5x sd), divided out per group at sampling time
    "attn_hier_w1.5:gauss_jacobian_FD": ("attn_hier_w1.5", None,
                                         {**DPM, "correction": "gauss_jacobian_FD", **FD_SCHEDULE}),
    "legacy_w1.5:gauss_jacobian_FD": ("legacy_w1.5", None,
                                      {**DPM, "correction": "gauss_jacobian_FD", **FD_SCHEDULE}),
    "legacy_w1.5:gauss_hierarchical": ("legacy_w1.5", None,
                                       {**DPM, "correction": "gauss_hierarchical",
                                        "precision_est_samples": 500}),
    # target score matching on the globals (same data, network and recipe as attn_hier)
    "attn_hier_tsm:gauss_jacobian_FD": ("attn_hier_tsm", None,
                                        {**DPM, "correction": "gauss_jacobian_FD", **FD_SCHEDULE}),
    "attn_hier_tsm:arruda_direct": ("attn_hier_tsm", {"estimator": "direct"},
                                    {**DPM, "correction": "arruda_schur"}),
    # latent attention: joint score, production joint-state rules + Schur/Laplace marginals
    "attn_full:gauss_hierarchical": ("attn_full", None,
                                     {**DPM, "correction": "gauss_hierarchical",
                                      "precision_est_samples": 500}),
    "attn_full:gauss_jacobian_FD": ("attn_full", None,
                                    {**DPM, "correction": "gauss_jacobian_FD", **FD_SCHEDULE}),
    "attn_full:arruda_schur": ("attn_full", {"estimator": "schur"},
                               {**DPM, "correction": "arruda_schur", **FD_SCHEDULE}),
    "attn_full:arruda_laplace": ("attn_full", {"estimator": "laplace"},
                                 {**DPM, "correction": "arruda_schur", **FD_SCHEDULE}),
}


def model_directory(network, train_samples, epochs):
    return HERE / "artifacts" / "models" / f"{network}_h{NETWORK['hidden_size']}d{NETWORK['depth']}" \
        f"_n{train_samples}_e{epochs}"


def build(network, device, checkpoint=None):
    """SBIm with the requested trunk, preconditioned; backbone loaded if given."""
    import torch
    import common
    from residual_score import PreconditionedScore
    from compass import ScoreBasedInferenceModel as SBIm
    from compass import hierarchical_roles

    spec = NETWORKS[network]
    model = SBIm(nodes_size=T.NODES, arch=spec["arch"], device=device,
                 **SDE_KWARGS, **NETWORK)
    if spec["attention"]:
        from compass.experimental.latent_attention import enable_latent_attention
        enable_latent_attention(model, spec["attention"],
                                hierarchical_roles(model.schema, T.GLOBALS, T.LOCAL))
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device)
        model.model.load_state_dict(state.get("model_state_dict", state))
    scale = spec.get("train_scale", 1.0)
    baseline = common.GaussianPriorScore(model.sde, [scale] * T.GLOBALS, T.HIERARCHY, T.NODES)
    lam_min = float(model.sde.lambda_t(torch.tensor([EPS])))
    lam_max = float(model.sde.lambda_t(torch.tensor([1.0])))
    model.model = PreconditionedScore(model.model, baseline.to(device), model.sde,
                                      [1.0] * T.NODES, (lam_min, lam_max)).to(device)
    return model


def train(network, device, train_samples, epochs, seed=7):
    import torch
    import common

    directory = model_directory(network, train_samples, epochs)
    if (directory / "Model_final.pt").exists():
        print(f"exists: {directory}")
        return directory
    torch.manual_seed(seed)
    scale = NETWORKS[network].get("train_scale", 1.0)
    data = torch.as_tensor(T.training_set(train_samples, seed=seed, global_scale=scale),
                           dtype=torch.float32)
    validation = torch.as_tensor(T.training_set(max(2000, train_samples // 20), seed=seed + 1,
                                                global_scale=scale), dtype=torch.float32)
    split = T.GLOBALS + T.LOCAL
    tsm = {}
    if NETWORKS[network].get("tsm"):
        import functools
        tsm = {"global_target_fn": functools.partial(T.global_score_target, prior_scale=scale),
               "global_target_indices": T.HIERARCHY, "tsm_lambda_c": 1.0}
    model = build(network, device)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with common.floor_training_eps(EPS):
        model.train(theta=data[:, :split], x=data[:, split:],
                    theta_val=validation[:, :split], x_val=validation[:, split:],
                    batch_size=256, max_epochs=epochs, early_stopping_patience=10 * epochs,
                    lr=5e-4, time_sampling="log_sigma", mask_menu="six",
                    mask_roles=(T.GLOBALS, T.LOCAL), mask_weights=[7, 7, 7, 1, 1, 1],
                    device=device, verbose=True, path=str(directory), **tsm)
    # Final weights: the best-validation checkpoint is a lucky early epoch.
    model.save(path=str(directory), name="Model_final")
    seconds = time.perf_counter() - started
    parameters = sum(p.numel() for p in model.model.parameters())
    (directory / "train.json").write_text(
        __import__("json").dumps({"seconds": seconds, "parameters": parameters,
                                  "train_samples": train_samples, "epochs": epochs}))
    print(f"trained {network} in {seconds:.0f}s ({parameters} parameters) -> {directory}")
    return directory


_LOADED = {}


def load(network, device, train_samples, epochs):
    import common
    key = (network, train_samples, epochs)
    if key not in _LOADED:
        checkpoint = model_directory(network, train_samples, epochs) / "Model_final.pt"
        model = common.freeze(build(network, device, checkpoint))
        model.model.eval()
        _LOADED[key] = model
    return _LOADED[key]


def sample(method, x, num_samples, device, seed, train_samples, epochs):
    """Joint draws for one dataset: (globals physical (S, 4), locals (S, N, 2), seconds)."""
    import torch
    from compass.MultiObsSampler import MultiObsSampler
    from compass.experimental.marginal_schur import MarginalSchurSampler

    network, settings, kwargs = METHODS[method]
    model = load(network, device, train_samples, epochs)
    n = len(x)
    data = torch.as_tensor(T.x_to_net(x), dtype=torch.float32)
    mask = torch.cat([torch.zeros(T.GLOBALS + T.LOCAL), torch.ones(T.DATA)])
    correction = kwargs["correction"]
    if settings is not None:
        max_rows = MARGINAL_MAX_ROWS
    elif correction.startswith("gauss_jacobian"):
        max_rows = JACOBIAN_MAX_ROWS
    else:
        max_rows = MAX_ROWS
    chunk = max(1, min(num_samples, max_rows // n))
    started = time.perf_counter()
    blocks = []
    for k, start in enumerate(range(0, num_samples, chunk)):
        torch.manual_seed(seed * 1000 + k)
        torch.cuda.manual_seed_all(seed * 1000 + k)
        sampler = MarginalSchurSampler(model, **settings) if settings is not None \
            else MultiObsSampler(model)
        scale = NETWORKS[network].get("train_scale", 1.0)
        if scale != 1.0:
            sampler.set_training_prior((torch.zeros(T.GLOBALS), scale * torch.ones(T.GLOBALS)))
        with torch.no_grad():
            s = sampler.sample(
                world_size=1, data=data, condition_mask=mask,
                num_samples=min(chunk, num_samples - start), timesteps=TIMESTEPS,
                eps=EPS, hierarchy=T.HIERARCHY,
                prior=(torch.zeros(T.GLOBALS), torch.ones(T.GLOBALS)),
                local_prior=(torch.zeros(T.LOCAL), torch.ones(T.LOCAL)),
                denoise_clamp=5.0, device=device, verbose=False, **kwargs)
        blocks.append(s.detach().cpu().numpy().astype(np.float64))
        if device == "cuda":
            torch.cuda.empty_cache()
    seconds = time.perf_counter() - started
    samples = np.concatenate(blocks, axis=1)          # (N, S, nodes)
    z = samples[0, :, :T.GLOBALS]
    locals_ = samples[:, :, T.GLOBALS:T.GLOBALS + T.LOCAL].transpose(1, 0, 2) \
        * T.normalization()["local_scale"]
    return T.z_to_globals(z), locals_, seconds
