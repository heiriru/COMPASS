#!/usr/bin/env python3
"""Recovery screen for the frozen 16-D summary-space COMPASS model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from Partial_Pooling.runtime import configure_runtime  # noqa: E402

# Caps this process at three CPUs and claims exactly one genuinely free GPU.
# Must run before torch is imported: the thread caps are read at import time,
# and the host is shared with other users.
configure_runtime(max_threads=3)

import torch

from compass import ScoreBasedInferenceModel
from Partial_Pooling.artifact_names import test_data_path
from Partial_Pooling.config import get_config
from Partial_Pooling.infer_partial_pooling import _sample_dataset, inference_plan
from Partial_Pooling.model_pipeline import load_normalizers
from Partial_Pooling.normalization import Normalizer
from Partial_Pooling.paths import BenchmarkPaths
from Partial_Pooling.train_encoder_summary import TAG, encode, load_encoder


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dpm2_gauss_hierarchical", "dpm2_gauss_jacobian_newton", "langevin_fnpse"), required=True)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--datasets", type=int, default=5)
    parser.add_argument("--dataset-start", type=int, default=0)
    parser.add_argument("--shard", default="")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument(
        "--name", default=f"{TAG}-h48d2-log_sigma",
        help=(
            "Checkpoint stem under artifacts/encoder_summary/checkpoints, as "
            "written by train_encoder_summary.py. The default is the first arm, "
            "whose recovery directories keep their original unsuffixed names."
        ),
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run")
    paths = BenchmarkPaths.default().ensure()
    config = get_config("small")
    if (args.subjects > config.subjects or args.dataset_start < 0
            or args.dataset_start + args.datasets > config.test_datasets):
        raise ValueError("Requested more subjects or datasets than the frozen test split has.")
    checkpoint = (paths.data.parent / "encoder_summary" / "checkpoints"
                  / f"{args.name}_checkpoint.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = ScoreBasedInferenceModel.load(str(checkpoint), device="cuda", arch="tokenized")
    model.model.eval()
    # The encoder is whichever one this checkpoint was trained against; the
    # manifest records it, so a wider-summary arm needs no flag here.
    manifest = json.loads(
        (checkpoint.parent / f"{args.name}.json").read_text()
    )
    encoder, item_norm, z_norm = load_encoder(Path(manifest["encoder"]), "cuda")
    summary_dim = int(manifest["summary_dim"])
    test = torch.load(test_data_path(config, paths), map_location="cpu", weights_only=False)["payload"]["sde"]
    theta_norm = load_normalizers(config, paths)["sde_joint_theta"]
    identity_z = Normalizer(torch.zeros(summary_dim), torch.ones(summary_dim),
                            torch.zeros(summary_dim, dtype=torch.bool), "encoder_train")
    run_args = SimpleNamespace(
        inference_method=args.method, subjects=args.subjects, draws=args.draws,
        timesteps=args.timesteps, gaussian_precision_samples=256,
        gaussian_precision_timesteps=args.timesteps, gaussian_precision_batch_size=128,
        langevin_steps_per_level=10, langevin_snr=0.1, dpm_corrector_snr=0.1,
        dpm_corrector_interval=1, dpm_corrector_steps=0, progress_every=5,
    )
    run = f"{args.method}-s{args.subjects}-t{args.timesteps}"
    if args.name != f"{TAG}-h48d2-log_sigma":
        # Arm-specific subdirectory: the first arm's paths must not move.
        run = f"{args.name}/{run}"
    out = paths.data.parent / "encoder_summary" / "recovery" / run
    if args.shard:
        out = out / "shards" / args.shard
    out.mkdir(parents=True, exist_ok=True)
    global_errors, local_errors, runtimes = [], [], []
    for dataset_id in range(args.dataset_start, args.dataset_start + args.datasets):
        observations = {"observations": test["observations"][dataset_id:dataset_id + 1, :args.subjects]}
        z = encode(observations, encoder, item_norm, z_norm, "cuda", 1024)[0]
        posterior, stats, runtime = _sample_dataset(
            model, z, {"theta": theta_norm, "observations": identity_z},
            run_args, config, "cuda", torch,
        )
        global_estimate = posterior["globals"].median(0).values
        local_estimate = posterior["locals"].median(0).values
        global_truth = test["globals"][dataset_id]
        local_truth = test["locals"][dataset_id, :args.subjects]
        global_errors.append(float((((global_estimate - global_truth) / theta_norm.scale[:7]) ** 2).mean().sqrt()))
        local_errors.append(float((((local_estimate - local_truth) / theta_norm.scale[7:]) ** 2).mean().sqrt()))
        runtimes.append(runtime)
        torch.save({"dataset_id": dataset_id, "posterior": posterior, "solver_stats": stats,
                    "runtime_seconds": runtime, "truth_globals": global_truth,
                    "truth_locals": local_truth}, out / f"dataset-{dataset_id:04d}.pt")
        print(f"[{dataset_id + 1}/{args.datasets}] global std-RMSE {global_errors[-1]:.4f}; local std-RMSE {local_errors[-1]:.4f}", flush=True)
    summary = {"representation": "frozen_predictive_set_summary", "likelihood": "p(z|theta)",
               "name": args.name,
               "checkpoint": str(checkpoint), "method": args.method, "plan": inference_plan(args.method),
               "subjects": args.subjects, "dataset_start": args.dataset_start,
               "datasets": args.datasets, "draws": args.draws,
               "timesteps": args.timesteps, "global_std_rmse": sum(global_errors) / len(global_errors),
               "local_std_rmse": sum(local_errors) / len(local_errors), "total_seconds": sum(runtimes),
               "finished_at": time.time()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
