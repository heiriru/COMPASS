#!/usr/bin/env python3
"""K=1 compositional recovery for the jointly trained SetEncoder checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from compass import Atomic, ScoreBasedInferenceModel
from compass.SetEncoder import SetEncoder
from Partial_Pooling.artifact_names import test_data_path
from Partial_Pooling.config import get_config
from Partial_Pooling.infer_partial_pooling import _sample_dataset, inference_plan
from Partial_Pooling.normalization import Normalizer
from Partial_Pooling.paths import BenchmarkPaths
from Partial_Pooling.train_end_to_end_set_k1 import TAG, TRIALS, TRIAL_WIDTH


def load_checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("representation") != "end_to_end_k1_conditional_set_encoder":
        raise ValueError(f"Not an end-to-end K=1 checkpoint: {path}")
    encoder = SetEncoder(**state["encoder_config"]).to(device)
    encoder.load_state_dict(state["encoder_state_dict"])
    encoder.eval()
    # The checkpoint records its schema, but the constructor remains explicit
    # here so this recovery entry point cannot silently load a grouped model.
    model = ScoreBasedInferenceModel(
        theta_size=10, x_spec=[Atomic("summary", state["encoder_config"]["summary_dim"])],
        arch="tokenized", trunk="shared", time_embedding_size=256,
        sde_type="vesde", sigma=25.0, hidden_size=48, depth=2, num_heads=4,
        mlp_ratio=4, device=device,
    )
    model.model.load_state_dict(state["score_state_dict"])
    model.model.eval()
    return model, encoder, Normalizer.from_state_dict(state["item_normalizer"]), \
        Normalizer.from_state_dict(state["theta_normalizer"])


def encode(encoder, observations, item_normalizer):
    items = observations.reshape(-1, TRIALS, TRIAL_WIDTH)
    mean = item_normalizer.mean.to(device=items.device, dtype=items.dtype)
    scale = item_normalizer.scale.to(device=items.device, dtype=items.dtype)
    with torch.no_grad():
        return encoder((items - mean) / scale)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dpm2_gauss_hierarchical", "dpm2_gauss_jacobian_newton", "langevin_fnpse"), required=True)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--datasets", type=int, default=5)
    parser.add_argument("--dataset-start", type=int, default=0)
    parser.add_argument("--shard", default="")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=80)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    paths = BenchmarkPaths.default().ensure()
    config = get_config("small")
    if args.subjects > config.subjects or args.dataset_start < 0 or args.dataset_start + args.datasets > config.test_datasets:
        raise ValueError("Requested more subjects or datasets than the frozen test split has.")
    checkpoint = paths.data.parent / "end_to_end_set_k1" / "checkpoints" / f"{TAG}_checkpoint.pt"
    model, encoder, item_normalizer, theta_normalizer = load_checkpoint(checkpoint, "cuda")
    test = torch.load(test_data_path(config, paths), map_location="cpu", weights_only=False)["payload"]["sde"]
    summary_normalizer = Normalizer(torch.zeros(encoder.summary_dim, device="cuda"),
                                    torch.ones(encoder.summary_dim, device="cuda"),
                                    torch.zeros(encoder.summary_dim, dtype=torch.bool, device="cuda"),
                                    "joint_encoder")
    run_args = SimpleNamespace(
        inference_method=args.method, subjects=args.subjects, draws=args.draws,
        timesteps=args.timesteps, gaussian_precision_samples=256,
        gaussian_precision_timesteps=args.timesteps, gaussian_precision_batch_size=128,
        langevin_steps_per_level=10, langevin_snr=0.1, dpm_corrector_snr=0.1,
        dpm_corrector_interval=1, dpm_corrector_steps=0, progress_every=5,
    )
    out = paths.data.parent / "end_to_end_set_k1" / "recovery" / f"{args.method}-s{args.subjects}-t{args.timesteps}"
    if args.shard:
        out = out / "shards" / args.shard
    out.mkdir(parents=True, exist_ok=True)
    global_errors, local_errors, runtimes = [], [], []
    for dataset_id in range(args.dataset_start, args.dataset_start + args.datasets):
        raw = test["observations"][dataset_id, :args.subjects].to("cuda")
        z = encode(encoder, raw, item_normalizer)
        posterior, stats, runtime = _sample_dataset(
            model, z, {"theta": theta_normalizer, "observations": summary_normalizer},
            run_args, config, "cuda", torch,
        )
        global_estimate = posterior["globals"].median(0).values
        local_estimate = posterior["locals"].median(0).values
        global_truth, local_truth = test["globals"][dataset_id], test["locals"][dataset_id, :args.subjects]
        global_errors.append(float((((global_estimate - global_truth) / theta_normalizer.scale[:7]) ** 2).mean().sqrt()))
        local_errors.append(float((((local_estimate - local_truth) / theta_normalizer.scale[7:]) ** 2).mean().sqrt()))
        runtimes.append(runtime)
        torch.save({"dataset_id": dataset_id, "posterior": posterior, "solver_stats": stats,
                    "runtime_seconds": runtime, "truth_globals": global_truth,
                    "truth_locals": local_truth}, out / f"dataset-{dataset_id:04d}.pt")
        print(f"[{dataset_id + 1}/{args.datasets}] global std-RMSE {global_errors[-1]:.4f}; local std-RMSE {local_errors[-1]:.4f}", flush=True)
    summary = {"representation": "end_to_end_k1_conditional_set_encoder", "checkpoint": str(checkpoint),
               "method": args.method, "plan": inference_plan(args.method), "subjects": args.subjects,
               "dataset_start": args.dataset_start, "datasets": args.datasets, "draws": args.draws,
               "timesteps": args.timesteps, "global_std_rmse": sum(global_errors) / len(global_errors),
               "local_std_rmse": sum(local_errors) / len(local_errors), "total_seconds": sum(runtimes),
               "finished_at": time.time()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
