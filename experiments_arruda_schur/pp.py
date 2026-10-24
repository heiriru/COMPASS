"""Partial pooling, encoder-summary lane (K=1, no grouping): latent attention + arruda_schur.

Each subject is one observation. Its 30 trials are summarised by the frozen
SetEncoder into ``z`` (16-D), and the score network models ``p(g, l, z)`` over
26 nodes. The recipe is C2 (``setenc-d16-w64x2s4-h96d6-log_sigma-te64-mask6mix``),
the best K=1 arm on record, and the attention mode is the only change.

    python pp.py train --mode full
    python pp.py infer --mode full --method arruda_laplace --datasets 30
    python pp.py infer --mode none --method gauss_jacobian_FD --datasets 30
    python pp.py infer --mode hierarchical --precondition Rg --datasets 30   # default: gauss_jacobian_FD_r9

The default method is ``gauss_jacobian_FD_r9`` (since 2026-10-23): the FD Jacobian refreshed at
every 9th composed-score call. On this problem it is 3.3-3.75x faster than the production
schedule of ``gauss_jacobian_FD`` (refresh every call above lambda 0.3, every 20th below) with no
accuracy loss beyond the seed floor (docs/jacobian_speedup_results.md). ``gauss_jacobian_FD`` keeps
its old schedule, so earlier result folders under that name stay comparable.

``--mode none`` loads the production C2 checkpoint read-only. Outputs are
written under ``experiments_arruda_schur/artifacts/pp/<mode>/recovery/<method>-...``
in the same per-dataset format as ``Partial_Pooling/infer_encoder_summary.py``,
so ``score_bayesflow_metrics.py`` / ``score_shrinkage.py`` read them unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

sys.path[:0] = [str(common.COMPASS), str(common.COMPASS / "abs_hierarchy")]

import Partial_Pooling.runtime as pp_runtime  # noqa: E402

if os.environ.get("CUDA_VISIBLE_DEVICES"):
    # A deliberately chosen device: cap the CPUs, skip the autocvd search. The
    # benchmark modules imported below call configure_runtime at import time,
    # so the explicit choice has to be made to stick for them too.
    pp_runtime.configure_cpu_limit(3)
    pp_runtime.configure_runtime = lambda max_threads=3, **_: pp_runtime.configure_cpu_limit(max_threads)
else:
    pp_runtime.configure_runtime(max_threads=3)

import torch  # noqa: E402

from compass import Atomic, ScoreBasedInferenceModel, six_conditional_menu  # noqa: E402
from compass.experimental.latent_attention import (  # noqa: E402
    enable_latent_attention, load_latent_attention, save_latent_attention,
)
from compass.experimental.marginal_schur import MarginalSchurSampler  # noqa: E402
from Partial_Pooling import infer_partial_pooling as ipp  # noqa: E402
from Partial_Pooling.artifact_names import test_data_path  # noqa: E402
from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.data_pipeline import load_split  # noqa: E402
from Partial_Pooling.model_pipeline import load_normalizers  # noqa: E402
from Partial_Pooling.normalization import Normalizer  # noqa: E402
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402
from Partial_Pooling.train_encoder_summary import TAG, encode, load_encoder  # noqa: E402

C2 = dict(hidden_size=96, depth=6, heads=8, time_embedding_size=64, bernoulli_fraction=0.25,
          posterior_weight=1.0, time_sampling="log_sigma", train_subset=100_000,
          batch_size=256, epochs=500, patience=20)
C2_NAME = f"{TAG}-h96d6-log_sigma-te64-mask6mix"
MODES = ("none", "global", "hierarchical", "full")
ROLES = {"g": list(range(7)), "l": list(range(7, 10)), "x": list(range(10, 26))}

PLANS = {
    "gauss_hierarchical": ({}, "gauss_hierarchical", {}),
    "gauss_jacobian_FD": ({}, "gauss_jacobian_FD",
                          {"jacobian_refresh": 1, "jacobian_refresh_low_noise": 20, "jacobian_refresh_lambda": 0.3}),
    # The dense schedule: a fresh FD Jacobian at every evaluation, at all noise levels.
    "gauss_jacobian_FD_r1": ({}, "gauss_jacobian_FD", {"jacobian_refresh": 1}),
    # One odd refresh period everywhere (docs/jacobian_speedup_results.md; the fair benchmark's ours-fast).
    "gauss_jacobian_FD_r9": ({}, "gauss_jacobian_FD", {"jacobian_refresh": 9}),
    # The production FD schedule with twice the default Karras churn (eta 2 -> 4).
    "gauss_jacobian_FD_churn4": ({}, "gauss_jacobian_FD",
                                 {"jacobian_refresh": 1, "jacobian_refresh_low_noise": 20, "jacobian_refresh_lambda": 0.3,
                                  "churn_eta": 4.0}),
    "arruda_direct": ({"estimator": "direct"}, "arruda_schur", {}),
    # The log-det gradient (H extra local Jacobians) is refreshed every 5th
    # evaluation (every 20th at low noise); the Newton step runs at every one.
    "arruda_laplace": ({"estimator": "laplace"}, "arruda_schur",
                      {"jacobian_refresh": 5, "jacobian_refresh_low_noise": 20,
                       "jacobian_refresh_lambda": 0.3}),
    "arruda_schur": ({"estimator": "schur"}, "arruda_schur", {}),
    # Arruda's stabiliser: d(t) from 1 at the data end to 1/sqrt(n) at the noise end.
    "arruda_laplace_damped": ({"estimator": "laplace", "damping": True}, "arruda_schur",
                             {"jacobian_refresh": 5, "jacobian_refresh_low_noise": 20,
                              "jacobian_refresh_lambda": 0.3}),
    "arruda_gauss_direct": ({"estimator": "direct"}, "arruda_schur_gauss",
                            {"jacobian_refresh": 1, "jacobian_refresh_low_noise": 20, "jacobian_refresh_lambda": 0.3}),
    "arruda_gauss_laplace": ({"estimator": "laplace"}, "arruda_schur_gauss",
                             {"jacobian_refresh": 5, "jacobian_refresh_low_noise": 20,
                              "jacobian_refresh_lambda": 0.3}),
}


def artifacts(mode, precondition="", wide=1.0, variant=""):
    """``precondition`` ("T", "R", "TR"): abs_hierarchy/precondition.py switches, T = log-lambda
    time input, R = residual output on the latents. ``wide`` != 1: trained on globals drawn from
    the prior with its sd times ``wide``. ``variant``: a free suffix (budget/encoder ablations of
    the same recipe). Each gets its own folder."""
    name = f"{mode}_{precondition}" if precondition else mode
    name = f"{name}_wide{wide:g}" if wide != 1.0 else name
    return HERE / "artifacts" / "pp" / (f"{name}_{variant}" if variant else name)


def checkpoint_of(mode, paths, precondition="", wide=1.0, variant="", final=False):
    """The best-validation checkpoint, or with ``final`` the final-epoch weights the Trainer also
    writes (schedule-free AdamW's averaged evaluation weights at the last epoch)."""
    if mode == "none":
        return paths.data.parent / "encoder_summary" / "checkpoints" / f"{C2_NAME}_checkpoint.pt"
    suffix = (f"-{precondition}" if precondition else "") + (f"-wide{wide:g}" if wide != 1.0 else "")
    kind = "" if final else "_checkpoint"
    return artifacts(mode, precondition, wide, variant) / "checkpoints" / f"{C2_NAME}-{mode}{suffix}{kind}.pt"


WIDE_DATA = Path("/export/data/rheinric/compass_pp_wide")


def wide_split(config, scale, size, split):
    """One-subject training sets with the globals drawn from N(mean, (scale * sd)^2) instead of
    the prior; locals and trials are simulated exactly as in Partial_Pooling.data_pipeline. Same
    simulator, so the network still learns the same likelihood; set_training_prior divides the
    wider prior back out at inference."""
    import numpy as np
    from Partial_Pooling.schema import physical_from_joint, transformed_observations
    from Partial_Pooling.simulators.ddm_sde import simulate_ddm
    from Partial_Pooling.simulators.priors import GLOBAL_PRIOR_MEAN, GLOBAL_PRIOR_STD
    path = WIDE_DATA / f"sde-{split}-wide{scale:g}-n{size}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    rng = np.random.default_rng([config.root_seed, int(round(scale * 1000)), ("train", "validation").index(split)])
    globals_ = rng.normal(GLOBAL_PRIOR_MEAN, scale * GLOBAL_PRIOR_STD, size=(size, 7))
    locals_ = rng.normal(globals_[:, None, :3], np.exp(globals_[:, None, 3:6]), size=(size, 1, 3))
    raw = simulate_ddm(physical_from_joint(globals_[:, None, :], locals_), rng, config.trials,
                       dt=config.dt, max_decision_time=config.max_decision_time)
    out = {"globals": torch.from_numpy(globals_).float(), "locals": torch.from_numpy(locals_).float(),
           "observations": torch.from_numpy(transformed_observations(raw)).float()}
    WIDE_DATA.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def encoder_path_of(arguments, paths):
    return Path(arguments.encoder) if arguments.encoder else paths.data.parent / "encoders" / f"{TAG}.pt"


def train_seed(arguments):
    return arguments.seed if getattr(arguments, "train_seed", None) is None else arguments.train_seed


def build_model(arguments, config, theta_train, theta_normalizer, summary_dim):
    """The C2 score net with ``arguments``' attention mode, precondition and training prior, its
    sidecars written to the checkpoint folder. Returns (model, mask menu, folder, run name)."""
    torch.manual_seed(train_seed(arguments))
    model = ScoreBasedInferenceModel(
        theta_size=10, x_spec=[Atomic("summary", summary_dim)], arch="tokenized", trunk="shared",
        time_embedding_size=C2["time_embedding_size"], sde_type=config.sde_type, sigma=config.sigma,
        hidden_size=C2["hidden_size"], depth=C2["depth"], num_heads=C2["heads"], mlp_ratio=4,
        device="cuda",
    )
    enable_latent_attention(model, arguments.mode, ROLES)
    out = artifacts(arguments.mode, arguments.precondition, arguments.train_prior_scale, arguments.variant) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    if arguments.train_prior_scale != 1.0:
        from Partial_Pooling.simulators.priors import GLOBAL_PRIOR_MEAN, GLOBAL_PRIOR_STD
        mean = (torch.as_tensor(GLOBAL_PRIOR_MEAN, dtype=torch.float32) - theta_normalizer.mean[:7]) / theta_normalizer.scale[:7]
        std = arguments.train_prior_scale * torch.as_tensor(GLOBAL_PRIOR_STD, dtype=torch.float32) / theta_normalizer.scale[:7]
        (out / "training_prior.json").write_text(json.dumps({
            "scale": arguments.train_prior_scale, "global_mean": mean.tolist(), "global_std": std.tolist(),
            # the locals' marginal under the training prior, for the per-subject local clamp
            "local_mean": theta_train[:, 7:].mean(0).tolist(), "local_std": theta_train[:, 7:].std(0).tolist(),
        }, indent=2) + "\n")
    if arguments.precondition:
        import precondition
        # The residual baseline is the diffused Gaussian moment match of each latent's prior
        # marginal; in the normalised theta space these are the training-set moments.
        prior = (theta_train.mean(0).tolist(), theta_train.std(0).tolist())
        if arguments.precondition.endswith("Rg"):
            # Rg: residual on the 7 globals only. The locals' prior marginal is a scale mixture,
            # so its Gaussian moment match is a poor baseline; their output stays the plain net.
            prior = (prior[0][:7], prior[1][:7])
        switches = precondition.attach(model, None, time="T" in arguments.precondition,
                                       residual="R" in arguments.precondition, prior=prior)
        precondition.save(switches, out)
    menu = six_conditional_menu(model.schema, global_size=7, local_size=3,
                                weights=[C2["posterior_weight"]] * 3 + [1.0] * 3,
                                bernoulli_fraction=C2["bernoulli_fraction"])
    save_latent_attention(model, out)
    name = (f"{C2_NAME}-{arguments.mode}" + (f"-{arguments.precondition}" if arguments.precondition else "")
            + (f"-wide{arguments.train_prior_scale:g}" if arguments.train_prior_scale != 1.0 else ""))
    return model, menu, out, name


def train(arguments):
    paths = BenchmarkPaths.default().ensure()
    encoder_path = encoder_path_of(arguments, paths)
    encoder, item_normalizer, z_normalizer = load_encoder(encoder_path, "cuda")
    summary_dim = int(encoder.summary_dim)
    config = get_config("small", train_subset=arguments.train_subset, arch="tokenized",
                        tokenization="scalar", hidden_size=C2["hidden_size"], depth=C2["depth"],
                        num_heads=C2["heads"], mlp_ratio=4, mask_menu="six_mixed",
                        mask_bernoulli_fraction=C2["bernoulli_fraction"],
                        mask_posterior_weight=C2["posterior_weight"],
                        time_sampling=C2["time_sampling"])
    n = arguments.train_subset
    rows = max(1, round(config.validation_size * n / config.train_size))
    if arguments.train_prior_scale != 1.0:
        # Same simulation budget as the stock arm: n training and `rows` validation sets.
        train_split = wide_split(config, arguments.train_prior_scale, n, "train")
        validation = wide_split(config, arguments.train_prior_scale, rows, "validation")
    else:
        train_split = load_split(config, "train", paths)["sde"]
        validation = load_split(config, "validation", paths)["sde"]
    theta_normalizer = load_normalizers(config, paths)["sde_joint_theta"]
    z_train = encode(train_split, encoder, item_normalizer, z_normalizer, "cuda", 1024)[:, 0]
    z_validation = encode(validation, encoder, item_normalizer, z_normalizer, "cuda", 1024)[:, 0]
    theta_train = theta_normalizer.transform(torch.cat((train_split["globals"], train_split["locals"][:, 0]), 1))
    theta_validation = theta_normalizer.transform(torch.cat((validation["globals"], validation["locals"][:, 0]), 1))
    theta_train, z_train = theta_train[:n], z_train[:n]
    theta_validation, z_validation = theta_validation[:rows], z_validation[:rows]

    model, menu, out, name = build_model(arguments, config, theta_train, theta_normalizer, summary_dim)
    print(f"training {name}: {sum(p.numel() for p in model.model.parameters()):,} parameters", flush=True)
    tsm = {}
    if getattr(arguments, "tsm", False):
        # Target score matching on the six globals that reach the data only through the
        # locals (tsm_targets.py), blended with DSM by inverse variance; beta_raw and the
        # locals stay on DSM.
        import tsm_targets
        from Partial_Pooling.simulators.priors import GLOBAL_PRIOR_MEAN, GLOBAL_PRIOR_STD
        fn, var = tsm_targets.make(theta_normalizer, GLOBAL_PRIOR_MEAN,
                                   arguments.train_prior_scale * GLOBAL_PRIOR_STD)
        tsm = {"global_target_fn": fn, "global_target_indices": tsm_targets.INDICES, "tsm_variance_fn": var}
    torch.manual_seed(train_seed(arguments))
    started = time.time()
    model.train(theta_train, z_train, theta_validation, z_validation,
                batch_size=C2["batch_size"], max_epochs=C2["epochs"],
                early_stopping_patience=C2["patience"], device="cuda", verbose=True,
                path=str(out), name=name, mask_sampler=menu, time_sampling=C2["time_sampling"], **tsm)
    (out / f"{name}.json").write_text(json.dumps({
        "recipe": {**C2, "train_subset": n}, "mode": arguments.mode, "variant": arguments.variant, "precondition": arguments.precondition,
        "train_prior_scale": arguments.train_prior_scale, "encoder": str(encoder_path),
        "summary_dim": summary_dim, "seed": train_seed(arguments), "tsm": bool(tsm),
        "train_loss": model.trainer.train_loss, "validation_loss": model.trainer.val_loss,
        "minutes": (time.time() - started) / 60,
    }, indent=2) + "\n")


def truncate_to_support(posterior, runtime, arguments, theta_norm, resample, seed, max_rounds=20):
    """Truncate the prior of the globals to |z| <= ``support_sd`` prior sd (z = normalised theta,
    whose training-set moments are ~N(0, 1)): draws outside are rejected and replaced by fresh
    composed draws until ``draws`` valid ones exist. Only 0.3% of training sets have any global beyond
    3.5 sd (max ~5), and
    the log_sigma_nu collapse is a spurious second mode at 4.3-5.0 sd; 3.5 sd removes 2e-4 of the
    prior mass per coordinate. Returns (posterior, runtime, record of the rejection)."""
    mean, scale = theta_norm.mean[:7], theta_norm.scale[:7]

    def valid(p):
        return (((p["globals"] - mean) / scale).abs() <= arguments.support_sd).all(1)

    keys = ("globals", "locals")
    kept = {k: [posterior[k][valid(posterior)]] for k in keys}
    drawn, rounds = len(posterior["globals"]), 0
    while sum(len(v) for v in kept["globals"]) < arguments.draws and rounds < max_rounds:
        rounds += 1
        torch.manual_seed(seed + 100_000 * rounds)
        extra, _, seconds = resample(rounds)
        runtime += seconds
        drawn += len(extra["globals"])
        ok = valid(extra)
        for k in keys:
            kept[k].append(extra[k][ok])
    accepted = sum(len(v) for v in kept["globals"])
    if accepted == 0:  # nothing inside the support: keep the untruncated draws, flag it
        return posterior, runtime, {"support_sd": arguments.support_sd, "drawn": drawn, "accepted": 0,
                                    "rejected_fraction": 1.0, "rounds": rounds}
    out = dict(posterior)
    for k in keys:
        out[k] = torch.cat(kept[k])[:arguments.draws]
    return out, runtime, {"support_sd": arguments.support_sd, "drawn": drawn, "accepted": accepted,
                          "rejected_fraction": 1 - accepted / drawn, "rounds": rounds}


def infer(arguments):
    paths = BenchmarkPaths.default().ensure()
    config = get_config("small")
    checkpoint = checkpoint_of(arguments.mode, paths, arguments.precondition, arguments.train_prior_scale, arguments.variant,
                               final=arguments.checkpoint == "final")
    model = ScoreBasedInferenceModel.load(str(checkpoint), device="cuda", arch="tokenized")
    if arguments.mode != "none":
        load_latent_attention(model, checkpoint.parent)
    if arguments.precondition:
        import precondition
        assert precondition.load(model, checkpoint.parent, None) is not None, "precondition.json missing"
    model.model.eval()
    settings, correction, sampler_kwargs = PLANS[arguments.method]
    model.multi_obs_sampler = MarginalSchurSampler(model, **settings)
    if arguments.train_prior_scale != 1.0 or arguments.declare_training_prior:
        # The net was trained under a different (wider) prior on the globals: each subject's
        # factor divides that prior out, the target prior enters once (set_training_prior).
        # --declare-training-prior on a stock net declares the target prior itself: a no-op check.
        if arguments.train_prior_scale != 1.0:
            declared = json.loads((checkpoint.parent / "training_prior.json").read_text())
        else:
            from Partial_Pooling.simulators.priors import GLOBAL_PRIOR_MEAN, GLOBAL_PRIOR_STD, LOCAL_PRIOR_MEAN, LOCAL_PRIOR_STD
            tn = load_normalizers(config, paths)["sde_joint_theta"]
            declared = {"global_mean": ((torch.as_tensor(GLOBAL_PRIOR_MEAN).float() - tn.mean[:7]) / tn.scale[:7]).tolist(),
                        "global_std": (torch.as_tensor(GLOBAL_PRIOR_STD).float() / tn.scale[:7]).tolist(),
                        "local_mean": ((torch.as_tensor(LOCAL_PRIOR_MEAN).float() - tn.mean[7:]) / tn.scale[7:]).tolist(),
                        "local_std": (torch.as_tensor(LOCAL_PRIOR_STD).float() / tn.scale[7:]).tolist()}
        model.multi_obs_sampler.set_training_prior((declared["global_mean"], declared["global_std"]))
        local = (torch.tensor(declared["local_mean"]), torch.tensor(declared["local_std"]))
        ipp._normalized_local_prior = lambda normalizer, torch_: local

    original_plan = ipp.inference_plan

    def plan(name):
        if name in PLANS:
            return {"name": name, "sampler": "dpm", "correction": correction, "order": 2,
                    "sampler_kwargs": dict(sampler_kwargs), "label": name}
        return original_plan(name)

    ipp.inference_plan = plan
    # Draws are independent; C2's inherited config gives batch_size // subjects = 1
    # draw per sampler call, which leaves the GPU idle between calls. Batch them.
    ipp.draw_batch_size = lambda draws, training_batch_size, subjects: max(
        1, min(draws, arguments.draw_batch))
    encoder_path = encoder_path_of(arguments, paths)
    encoder, item_norm, z_norm = load_encoder(encoder_path, "cuda")
    summary_dim = int(encoder.summary_dim)
    test = torch.load(test_data_path(config, paths), map_location="cpu", weights_only=False)["payload"]["sde"]
    theta_norm = load_normalizers(config, paths)["sde_joint_theta"]
    identity_z = Normalizer(torch.zeros(summary_dim), torch.ones(summary_dim),
                            torch.zeros(summary_dim, dtype=torch.bool), "encoder_train")
    run_args = SimpleNamespace(
        inference_method=arguments.method, subjects=arguments.subjects, draws=arguments.draws,
        timesteps=arguments.timesteps, gaussian_precision_samples=256,
        gaussian_precision_timesteps=arguments.timesteps, gaussian_precision_batch_size=128,
        langevin_steps_per_level=10, langevin_snr=0.1, dpm_corrector_snr=0.1,
        dpm_corrector_interval=1, dpm_corrector_steps=0, progress_every=5,
    )
    out = (artifacts(arguments.mode, arguments.precondition, arguments.train_prior_scale, arguments.variant) / "recovery" / arguments.tag
           / f"{arguments.method}-s{arguments.subjects}-t{arguments.timesteps}")
    out.mkdir(parents=True, exist_ok=True)
    for dataset_id in range(arguments.dataset_start, arguments.dataset_start + arguments.datasets):
        target = out / f"dataset-{dataset_id:04d}.pt"
        if target.exists():
            continue
        observations = {"observations": test["observations"][dataset_id:dataset_id + 1, :arguments.subjects]}
        z = encode(observations, encoder, item_norm, z_norm, "cuda", 1024)[0]
        torch.manual_seed(arguments.seed + dataset_id)
        posterior, stats, runtime = ipp._sample_dataset(
            model, z, {"theta": theta_norm, "observations": identity_z}, run_args, config, "cuda", torch,
        )
        support = None
        if arguments.support_sd > 0:
            posterior, runtime, support = truncate_to_support(
                posterior, runtime, arguments, theta_norm,
                lambda k: ipp._sample_dataset(model, z, {"theta": theta_norm, "observations": identity_z},
                                              run_args, config, "cuda", torch),
                arguments.seed + dataset_id)
        global_truth = test["globals"][dataset_id]
        local_truth = test["locals"][dataset_id, :arguments.subjects]
        g_err = float((((posterior["globals"].median(0).values - global_truth) / theta_norm.scale[:7]) ** 2).mean().sqrt())
        l_err = float((((posterior["locals"].median(0).values - local_truth) / theta_norm.scale[7:]) ** 2).mean().sqrt())
        torch.save({"dataset_id": dataset_id, "posterior": posterior, "solver_stats": stats, "support": support,
                    "runtime_seconds": runtime, "truth_globals": global_truth,
                    "truth_locals": local_truth, "mode": arguments.mode,
                    "method": arguments.method}, target)
        print(f"[{dataset_id}] {arguments.mode}/{arguments.method}: global std-RMSE {g_err:.4f}; "
              f"local std-RMSE {l_err:.4f}; {runtime:.0f}s", flush=True)
    (out / "summary.json").write_text(json.dumps({
        "mode": arguments.mode, "precondition": arguments.precondition, "method": arguments.method, "checkpoint": str(checkpoint),
        "subjects": arguments.subjects, "draws": arguments.draws, "timesteps": arguments.timesteps,
        "support_sd": arguments.support_sd, "train_prior_scale": arguments.train_prior_scale,
        "declare_training_prior": arguments.declare_training_prior,
        "variant": arguments.variant, "encoder": str(encoder_path),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "infer"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--method", choices=tuple(PLANS) + ("langevin_fnpse",), default="gauss_jacobian_FD_r9")
    parser.add_argument("--subjects", type=int, default=100)
    parser.add_argument("--datasets", type=int, default=30)
    parser.add_argument("--dataset-start", type=int, default=0)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--draw-batch", type=int, default=25)
    parser.add_argument("--precondition", choices=("", "T", "R", "TR", "Rg"), default="",
                        help="T = log-lambda time input, R = residual output, Rg = residual on the globals only "
                             "(abs_hierarchy/precondition.py)")
    parser.add_argument("--support-sd", type=float, default=0.0,
                        help="> 0: truncate the globals' prior to |z| <= this many prior sd by rejection "
                             "(the log_sigma_nu collapse fix); 0 = off")
    parser.add_argument("--train-prior-scale", type=float, default=1.0,
                        help="!= 1: train on globals drawn with the prior sd times this; inference "
                             "divides that training prior out of each subject's factor (exact)")
    parser.add_argument("--declare-training-prior", action="store_true",
                        help="check: run the training-prior composition path with the target prior "
                             "itself declared (must reproduce the stock composition)")
    parser.add_argument("--train-subset", type=int, default=C2["train_subset"],
                        help="training simulations for the score net (validation scales with it)")
    parser.add_argument("--encoder", default="",
                        help="SetEncoder checkpoint; default the production C2 encoder")
    parser.add_argument("--variant", default="",
                        help="artifact-folder suffix, e.g. 'n65536' for a budget ablation")
    parser.add_argument("--train-seed", type=int, default=None,
                        help="training seed (init + batches); default --seed. --seed stays the inference seed")
    parser.add_argument("--tsm", action="store_true",
                        help="target score matching on the 6 globals other than beta_raw (tsm_targets.py)")
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best",
                        help="infer from the best-validation checkpoint or the final-epoch weights")
    parser.add_argument("--tag", default="main", help="recovery/<tag>/...; use 'smoke' for checks")
    arguments = parser.parse_args()
    if arguments.action == "train":
        if arguments.mode == "none":
            raise SystemExit("'none' is the production C2 checkpoint; nothing to train.")
        return train(arguments)
    return infer(arguments)


if __name__ == "__main__":
    main()
