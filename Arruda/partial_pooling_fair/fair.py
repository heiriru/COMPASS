#!/usr/bin/env python3
"""Fair 2x2 comparison on the upstream partial-pooling case study (Arruda et al., case_study4).

Factors:  composition rule  {upstream compositional_sample, COMPASS gauss_jacobian_FD}
          training prior    {inference prior (upstream), globals' sd x 1.5}
Every cell uses networks trained by upstream's own code (upstream/partial_pooling.py:
SetTransformer(16, dropout .1) + DiffusionModel(), 256 x 128 = 32,768 single-subject
simulations, 500 epochs, batch 128), the same 100 test datasets (100 subjects x 30 trials),
100 posterior draws, and the same local network (consistency model, ancestral sampling).

    fair.py check-sim                       replica simulator vs upstream trial function
    fair.py data                            training sets + test set (fixed seeds, cached)
    fair.py train {global,global_wide,local}
    fair.py check-bridge {global,global_wide}   bridge unit test + n=1 agreement
    fair.py infer {bf,ours} {global,global_wide}
    fair.py score                           metric table (upstream's metrics)

Run through run.sh (KERAS_BACKEND=torch, vendored BayesFlow 2.0.13).
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = Path(__file__).resolve().parent
COMPASS = HERE.parent.parent
sys.path[:0] = [str(HERE), str(COMPASS / "Arruda" / ".vendor"), str(COMPASS), str(COMPASS / "src")]

import Partial_Pooling.runtime as rt  # noqa: E402
rt.configure_cpu_limit(3)

import numpy as np  # noqa: E402
import keras  # noqa: E402
import bayesflow as bf  # noqa: E402
import torch  # noqa: E402

import simulator as sim  # noqa: E402
from upstream import ddm_simulator as up  # noqa: E402
from upstream import settings as S  # noqa: E402

OUT = Path("/export/data/rheinric/compass_pp_fair")
# FAIR_SMOKE=1: 2-epoch networks and posteriors in separate folders, to test the pipeline end to end
SMOKE = os.environ.get("FAIR_SMOKE") == "1"
MODELS = OUT / ("models_smoke" if SMOKE else "models")
POSTERIORS = OUT / ("posteriors_smoke" if SMOKE else "posteriors")
N_TRAIN = S.N_TRAINING_BATCHES * S.BATCH_SIZE      # 32,768
WIDE = 1.5
SEEDS = {"global": 101, "global_wide": 202, "local": 303, "test": 404}
NETWORKS = {"global": 1.0, "global_wide": WIDE}
GLOBALS, LOCALS = sim.GLOBALS, sim.LOCALS
# COMPASS production gauss_jacobian_FD (experiments_arruda_schur/pp.py PLANS), fixed in advance
OURS = dict(method="dpm", order=2, correction="gauss_jacobian_FD", timesteps=50, eps=1e-3,
            jacobian_refresh_low_noise=20, jacobian_refresh_lambda=0.3)


# ----------------------------------------------------------------------------- data
def data_path(name):
    return OUT / "data" / f"{name}.npz"


def load(name):
    with np.load(data_path(name)) as f:
        return {k: f[k] for k in f.files}


def make_data():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    jobs = {"global": (N_TRAIN, 1, 1.0), "global_wide": (N_TRAIN, 1, WIDE), "local": (N_TRAIN, 1, 1.0),
            "test": (S.N_TEST, S.N_SUBJECTS, 1.0)}
    for name, (count, subjects, scale) in jobs.items():
        if data_path(name).exists():
            print("exists", name); continue
        t = time.time()
        out = sim.simulate(count, subjects, S.N_TRIALS, SEEDS[name], prior_scale=scale)
        np.savez(data_path(name), **out)
        print(f"{name}: sim_data {out['sim_data'].shape} in {time.time() - t:.0f}s", flush=True)


def check_sim():
    rows = sim.check_replica()
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # batch structure against upstream's own bf.make_simulator on a tiny batch
    ref = up.simulator_hierarchical.sample(3, n_trials=S.N_TRIALS)
    ours = sim.simulate(3, 1, S.N_TRIALS, 0)
    for k in ref:
        print(f"  {k:22s} upstream {np.shape(ref[k])}  replica {np.shape(ours.get(k))}")


# ----------------------------------------------------------------------------- networks
def global_workflow():
    adapter = (bf.adapters.Adapter().to_array().convert_dtype("float64", "float32")
               .concatenate(GLOBALS, into="inference_variables").rename("sim_data", "summary_variables"))
    return bf.BasicWorkflow(adapter=adapter,
                            summary_network=bf.networks.SetTransformer(summary_dim=16, dropout=0.1),
                            inference_network=bf.networks.DiffusionModel())


def local_workflow():
    adapter = (bf.adapters.Adapter().to_array().convert_dtype("float64", "float32").log(["alpha", "t0"])
               .concatenate(LOCALS, into="inference_variables").concatenate(GLOBALS, into="inference_conditions")
               .rename("sim_data", "summary_variables"))
    return bf.BasicWorkflow(adapter=adapter,
                            summary_network=bf.networks.SetTransformer(summary_dim=16, dropout=0.1),
                            inference_network=bf.networks.StableConsistencyModel())


def model_path(name):
    return MODELS / f"partial_pooling_{name}.keras"


def train(name):
    MODELS.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEEDS[name])
    workflow = local_workflow() if name == "local" else global_workflow()
    t = time.time()
    history = workflow.fit_offline(load(name), epochs=2 if SMOKE else S.EPOCHS, batch_size=S.BATCH_SIZE, verbose=2)
    workflow.approximator.save(model_path(name))
    (MODELS / f"{name}_history.json").write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in history.history.items()} | {"minutes": (time.time() - t) / 60}))
    print(f"saved {model_path(name)} after {(time.time() - t) / 60:.1f} min", flush=True)


def loaded(name):
    workflow = local_workflow() if name == "local" else global_workflow()
    workflow.approximator = keras.models.load_model(model_path(name))
    return workflow


def composed(name):
    """Upstream's compositional workflow for a trained global net (weights and standardiser
    copied by BayesFlow). Both rules read network, summaries and standardiser from this."""
    return bf.CompositionalWorkflow.from_basic_workflow(loaded(name))


# ----------------------------------------------------------------------------- priors
def prior_score(scale=1.0):
    """Upstream prior_global_score for N(m, (scale*s)^2)."""
    return lambda x: {k: up.score_log_norm(x[k], m=m, s=scale * s) for k, (m, s) in sim.PRIOR.items()}


def corrected_prior_score(n):
    """For a net trained under the wide prior p~: upstream's formula is P + sum_j (s_j - P);
    P = (n s~0 - s0)/(n-1) turns it into s0 + sum_j (s_j - s~0), i.e. each factor divides out
    the training prior and the inference prior enters once (exact). Time-free, so upstream's
    (1-t) bridging weight applies to all of P unchanged."""
    s0, sw = prior_score(1.0), prior_score(WIDE)
    return lambda x: {k: (n * sw(x)[k] - s0(x)[k]) / (n - 1) for k in GLOBALS}


# ----------------------------------------------------------------------------- bridge
class Standardiser:
    """BayesFlow's inference-variable standardisation, read off as the affine map it is."""

    def __init__(self, approximator):
        f = lambda x: keras.ops.convert_to_numpy(approximator.standardizer.maybe_standardize(  # noqa: E731
            keras.ops.convert_to_tensor(x), key="inference_variables", stage="inference"))
        z0, z1 = f(np.zeros((1, 7), np.float32)), f(np.ones((1, 7), np.float32))
        self.scale = (1.0 / (z1 - z0))[0].astype(np.float64)   # x = mean + scale * z
        self.mean = (-z0[0] * self.scale).astype(np.float64)

    def prior(self, scale=1.0):
        m = np.array([sim.PRIOR[k][0] for k in GLOBALS]); s = scale * np.array([sim.PRIOR[k][1] for k in GLOBALS])
        return (m - self.mean) / self.scale, s / self.scale


class Bridge(torch.nn.Module):
    """The BayesFlow diffusion net as a COMPASS score network on nodes [g (7, standardised) | summary].

    COMPASS samples a VE SDE (lambda = sigma_t). At noise lambda the VP state is alpha * x with
    log SNR = -2 log lambda, and score_VE(x) = alpha * score_VP(alpha x). The sampler divides the
    output by sigma_t (output_scale_function), so the output is lambda * score_VE on g, 0 on the
    conditioned summary nodes. Upstream's own schedule supplies alpha (variance preserving EDM)."""

    def __init__(self, inference_network, sde, k=7):
        super().__init__()
        self.net, self.sde, self.k = inference_network, sde, k
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # .parameters() non-empty

    def score(self, g, lam, cond):
        log_snr = (-2.0 * torch.log(lam)).reshape(-1, 1).expand(len(g), 1)
        alpha, _ = self.net.noise_schedule.get_alpha_sigma(log_snr_t=log_snr)
        s = self.net.score(alpha * g, log_snr_t=log_snr, conditions=cond, training=False)
        return alpha * s

    def forward(self, x, t, c=None, **_):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        t2 = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1)
        if t2.numel() == 1:
            t2 = t2.expand(len(x2))
        elif t2.numel() != len(x2):
            t2 = t2.repeat_interleave(len(x2) // t2.numel())
        lam = self.sde.lambda_t(t2).reshape(-1, 1)
        out = torch.zeros_like(x2)
        out[:, :self.k] = lam * self.score(x2[:, :self.k], lam, x2[:, self.k:])
        return out.reshape(shape)


def compass_model(workflow):
    from compass import Atomic, ScoreBasedInferenceModel
    conds = summaries_of(workflow, load("test")["sim_data"][:1, :2])
    model = ScoreBasedInferenceModel(theta_size=7, x_spec=[Atomic("summary", conds.shape[-1])],
                                     sde_type="vesde", sigma=25.0, hidden_size=32, depth=1, num_heads=1,
                                     device="cuda")
    model.model = Bridge(workflow.approximator.inference_network, model.sde).to("cuda")
    return model


def summaries_of(workflow, sim_data):
    """(datasets, subjects, trials, 2) -> the resolved conditions BayesFlow's own compositional
    sampler feeds its network (adapter + summary network), (datasets, subjects, d)."""
    resolved, _, _ = workflow.approximator._prepare_compositional_conditions(conditions={"sim_data": sim_data})
    return torch.as_tensor(keras.ops.convert_to_numpy(resolved), dtype=torch.float32)


# ----------------------------------------------------------------------------- inference
def arm_path(rule, net, subjects=S.N_SUBJECTS):
    return POSTERIORS / f"{rule}_{net}_n{subjects}.npz"


def infer_bf(net, subjects=S.N_SUBJECTS):
    test = load("test")
    composed_wf = composed(net)
    prior = prior_score(1.0) if NETWORKS[net] == 1.0 else corrected_prior_score(subjects)
    keras.utils.set_random_seed(SEEDS["test"])
    t = time.time()
    post = composed_wf.compositional_sample(
        num_samples=S.N_SAMPLES, conditions={"sim_data": test["sim_data"][:, :subjects]},
        compute_prior_score=prior, mini_batch_size=3, method=S.METHOD, steps=S.STEPS,
        max_steps=S.MAX_STEP, batch_size=S.BATCH_SIZE, compositional_bridge_d0=0.1)
    seconds = time.time() - t
    save_arm("bf", net, subjects, {k: np.asarray(post[k]).reshape(S.N_TEST, S.N_SAMPLES) for k in GLOBALS},
             seconds, test)


def infer_ours(net, subjects=S.N_SUBJECTS, datasets=None):
    from compass import MultiObsSampler
    test = load("test")
    workflow = composed(net)
    model = compass_model(workflow)
    for p in model.model.parameters():
        p.requires_grad_(False)
    std = Standardiser(workflow.approximator)
    sampler = MultiObsSampler(model)
    if NETWORKS[net] != 1.0:
        sampler.set_training_prior(std.prior(NETWORKS[net]))
    target = tuple(torch.as_tensor(v, dtype=torch.float32) for v in std.prior(1.0))
    mask = torch.cat([torch.zeros(7), torch.ones(model.nodes_size - 7)])
    partial = POSTERIORS / f"ours_{net}_n{subjects}_partial"
    partial.mkdir(parents=True, exist_ok=True)
    ids = range(S.N_TEST) if datasets is None else datasets
    for i in ids:
        f = partial / f"{i:03d}.npz"
        if f.exists():
            continue
        cond = summaries_of(workflow, test["sim_data"][i:i + 1, :subjects])[0]
        torch.manual_seed(SEEDS["test"] + i)
        t = time.time()
        with torch.no_grad():
            draws = sampler.sample(world_size=1, data=cond, condition_mask=mask, num_samples=S.N_SAMPLES,
                                   hierarchy=list(range(7)), prior=target, device="cuda", verbose=False,
                                   **OURS)
        z = draws.detach().cpu().numpy().reshape(-1, draws.shape[-1])[-S.N_SAMPLES:, :7] \
            if draws.dim() == 2 else draws.detach().cpu().numpy()[0, :, :7]
        np.savez(f, z=z, seconds=time.time() - t)
        print(f"[{i}] ours/{net} n={subjects}: {time.time() - t:.0f}s", flush=True)
    if datasets is not None:
        return
    zs = [np.load(partial / f"{i:03d}.npz") for i in range(S.N_TEST)]
    x = np.stack([std.mean + std.scale * z["z"] for z in zs])           # (datasets, draws, 7)
    save_arm("ours", net, subjects, {k: x[:, :, j] for j, k in enumerate(GLOBALS)},
             float(sum(z["seconds"] for z in zs)), test)


def save_arm(rule, net, subjects, globals_, seconds, test):
    """Globals from the arm + locals by upstream's ancestral sampling with the shared local net."""
    local = loaded("local")
    keras.utils.set_random_seed(SEEDS["test"] + 1)
    post = {k: v[..., None] for k, v in globals_.items()}
    samples_local = local.ancestral_sample(conditions={"sim_data": test["sim_data"][:, :subjects]},
                                           ancestral_conditions=post, batch_size=S.BATCH_SIZE * 10)
    path = arm_path(rule, net, subjects)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, seconds=seconds, **{f"g_{k}": v for k, v in globals_.items()},
             **{f"l_{k}": np.asarray(samples_local[k]) for k in LOCALS})
    print(f"wrote {path} ({seconds / 60:.1f} min sampling)", flush=True)


# ----------------------------------------------------------------------------- checks
def check_bridge(net):
    """(1) Bridge score == upstream's own score (VP state, standardised space), at several noise
    levels. (2) n = 1: COMPASS's sampler and upstream's sampler, same network, same subject,
    must give the same posterior (moments over 400 draws, several subjects)."""
    test = load("test")
    workflow = composed(net)
    model = compass_model(workflow)
    net_bf = workflow.approximator.inference_network
    cond = summaries_of(workflow, test["sim_data"][:4, :2]).reshape(8, -1).cuda()
    g = torch.randn(8, 7, device="cuda")
    for lam in (0.05, 0.3, 1.0, 5.0, 20.0):
        log_snr = torch.full((8, 1), -2 * np.log(lam), device="cuda")
        alpha, sigma = net_bf.noise_schedule.get_alpha_sigma(log_snr_t=log_snr)
        ref = alpha * net_bf.score(alpha * g, log_snr_t=log_snr, conditions=cond, training=False)
        t = model.sde.time_of_lambda(torch.tensor(lam, device="cuda"))
        x = torch.cat([g, cond], 1)
        ours = model.output_scale_function(t, model.model(x=x, t=t.expand(8, 1), c=None))[:, :7]
        print(f"lambda {lam:5.2f}: lam check {float(model.sde.lambda_t(t)):.4f}  max|bridge - BF| "
              f"{float((ours - ref).abs().max()):.2e}  |score| {float(ref.abs().mean()):.3f}  "
              f"sigma/alpha {float((sigma / alpha).mean()):.4f}", flush=True)
    # n = 1 agreement: plain (non-compositional) upstream sampling vs COMPASS's single-observation path
    from compass import MultiObsSampler
    std = Standardiser(workflow.approximator)
    sampler = MultiObsSampler(model)
    mask = torch.cat([torch.zeros(7), torch.ones(model.nodes_size - 7)])
    for i in range(3):
        x_i = test["sim_data"][i, :1]
        keras.utils.set_random_seed(i)
        ref = workflow.approximator.sample(num_samples=400, conditions={"sim_data": x_i})
        ref = np.stack([np.asarray(ref[k]).reshape(-1) for k in GLOBALS], 1)
        cond_i = summaries_of(workflow, test["sim_data"][i:i + 1, :2])[0, :1]
        with torch.no_grad():
            d = sampler.sample(world_size=1, data=cond_i, condition_mask=mask, num_samples=400,
                               hierarchy=list(range(7)), prior=tuple(torch.as_tensor(v, dtype=torch.float32)
                                                                     for v in std.prior(NETWORKS[net])),
                               device="cuda", verbose=False, **(OURS | {"correction": "uncorrected"}))
        z = d.detach().cpu().numpy().reshape(-1, d.shape[-1])[:, :7]
        ours = std.mean + std.scale * z
        print(f"subject {i}: mean BF  {np.round(ref.mean(0), 3)}\n           mean us  {np.round(ours.mean(0), 3)}"
              f"\n           sd   BF  {np.round(ref.std(0), 3)}\n           sd   us  {np.round(ours.std(0), 3)}", flush=True)


# ----------------------------------------------------------------------------- scoring
def score():
    from bayesflow.diagnostics.metrics import calibration_error as ece, root_mean_squared_error as nrmse
    test = load("test")
    truth_g = {k: test[k].reshape(-1) for k in GLOBALS}
    truth_g["beta"] = up.beta_from_normal(truth_g.pop("beta_raw"))
    names = GLOBALS[:-1] + ["beta"]
    rows = []
    for path in sorted((POSTERIORS).glob("*.npz")):
        f = np.load(path)
        est = {k: f[f"g_{k}"] for k in GLOBALS}
        est["beta"] = up.beta_from_normal(est.pop("beta_raw"))
        med = {k: np.median(est[k], 1) for k in names}
        r = [np.corrcoef(med[k], truth_g[k])[0, 1] for k in names]
        cov = [np.mean(np.abs(np.mean(est[k] < truth_g[k][:, None], 1) - 0.5) <= 0.45) for k in names]
        # upstream's layout: draws (datasets, samples, 1), targets (datasets, 1)
        e3 = {k: est[k][..., None] for k in names}
        t2 = {k: truth_g[k][:, None] for k in names}
        n_rmse = nrmse(e3, t2, variable_keys=names)["values"]
        cal = ece(e3, t2, variable_keys=names)["values"]
        # locals: draws (datasets, subjects, samples, 1) flattened as upstream does, (-1, samples, 1)
        le = {k: f[f"l_{k}"].reshape(-1, f[f"l_{k}"].shape[2], 1) for k in LOCALS}
        n_sub = f[f"l_{LOCALS[0]}"].shape[1]
        lt = {k: test[k][:, :n_sub].reshape(-1, 1) for k in LOCALS}
        lr = [np.corrcoef(np.median(le[k][..., 0], 1), lt[k][:, 0])[0, 1] for k in LOCALS]
        l_nrmse = nrmse(le, lt, variable_keys=LOCALS)["values"]
        l_cal = ece(le, lt, variable_keys=LOCALS)["values"]
        rows.append((path.stem, np.mean(r), r, np.mean(n_rmse), np.mean(cal), np.mean(cov), float(f["seconds"]),
                     np.mean(lr), np.mean(l_nrmse), np.mean(l_cal)))
    print(f"{'':26s} {'--------- globals ---------':>34s} {'--- locals ---':>22s}")
    print(f"{'arm':26s} {'mean r':>7s} {'NRMSE':>6s} {'ECE':>6s} {'cov90':>6s} {'r':>6s} {'NRMSE':>6s} {'ECE':>6s} "
          f"{'min':>6s}  per-global r ({', '.join(names)})")
    for name, mr, r, nr, ce, cv, s, lr, ln, lc in rows:
        print(f"{name:26s} {mr:7.3f} {nr:6.3f} {ce:6.3f} {cv:6.2f} {lr:6.3f} {ln:6.3f} {lc:6.3f} {s / 60:6.1f}  "
              + " ".join(f"{x:.3f}" for x in r))


def main():
    action, *rest = sys.argv[1:]
    if action == "check-sim":
        return check_sim()
    if action == "data":
        return make_data()
    if action == "train":
        return train(rest[0])
    if action == "check-bridge":
        return check_bridge(rest[0])
    if action == "infer":
        rule, net = rest[:2]
        subjects = int(rest[2]) if len(rest) > 2 else S.N_SUBJECTS
        return (infer_bf if rule == "bf" else infer_ours)(net, subjects)
    if action == "time-ours":          # the first k datasets only (cost estimate / smoke)
        return infer_ours(rest[0], S.N_SUBJECTS, datasets=range(int(rest[1])))
    if action == "score":
        return score()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
