"""One-time migration from preset-only names to descriptive artifact names."""

import json
import re
import sys
from pathlib import Path


SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def _move(source, target):
    if source == target:
        return target
    if not source.exists():
        return target
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing migration target {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    print(f"renamed {source} -> {target}", flush=True)
    return target


def _inference_method(artifact, run_config):
    method = artifact.get("inference_method") or run_config.get("inference_method")
    if method:
        return method
    correction = artifact.get("correction") or run_config.get("correction")
    sampler = artifact.get("sampler") or run_config.get("sampler")
    if correction == "damped_sum" and sampler in (None, "dpm"):
        return "dpm2_damped_sum"
    if correction == "gauss" and sampler == "dpm":
        return "dpm2_gaussian"
    if correction == "fnpe" and sampler == "langevin":
        return "langevin_fnpse"
    raise RuntimeError(
        f"Cannot infer method from sampler={sampler!r}, correction={correction!r}."
    )


def _migrate_training_data(config, paths, torch, atomic_json, atomic_torch):
    from Partial_Pooling.artifact_names import (
        normalization_path,
        test_data_path,
        training_index_path,
        training_shard_directory,
        training_shard_filename,
        training_tag,
    )

    old_index = paths.data / f"training-{config.preset}.json"
    new_index = training_index_path(config, paths)
    index_path = old_index if old_index.exists() else new_index
    if index_path.exists():
        metadata = json.loads(index_path.read_text())
        old_directory = paths.data / config.preset
        new_directory = training_shard_directory(config, paths)
        if old_directory.exists():
            _move(old_directory, new_directory)
        renamed_splits = {}
        for split, names in metadata["splits"].items():
            renamed_splits[split] = []
            for name in names:
                old_path = new_directory / name
                match = re.search(r"(\d{6})-(\d{6})\.pt$", name)
                if not match:
                    raise RuntimeError(f"Cannot parse shard range from {name}")
                start, stop = map(int, match.groups())
                new_name = training_shard_filename(
                    config, split, start, stop,
                )
                _move(old_path, new_directory / new_name)
                renamed_splits[split].append(new_name)
        metadata.update({
            "splits": renamed_splits,
            "training_method": training_tag(config),
            "training_samples": config.train_size,
            "validation_samples": config.validation_size,
        })
        old_normalizer = paths.data / f"normalization-{config.preset}.pt"
        new_normalizer = normalization_path(config, paths)
        normalizer_path = _move(old_normalizer, new_normalizer)
        metadata["normalization"] = new_normalizer.name
        atomic_json(new_index, metadata)
        if old_index.exists() and old_index != new_index:
            old_index.unlink()
        if normalizer_path.exists():
            normalizers = torch.load(normalizer_path, map_location="cpu")
            normalizers.update({
                "training_method": training_tag(config),
                "training_samples": config.train_size,
                "validation_samples": config.validation_size,
            })
            atomic_torch(normalizer_path, normalizers)

    new_test = test_data_path(config, paths)
    old_tests = (
        paths.data / f"test-{config.preset}.pt",
        paths.data / f"partial_pooling-test-{config.test_datasets}.pt",
    )
    source = next((path for path in old_tests if path.exists()), new_test)
    test_path = _move(source, new_test)
    if test_path.exists():
        test = torch.load(test_path, map_location="cpu")
        test["data_method"] = "partial_pooling"
        test["test_datasets"] = config.test_datasets
        atomic_torch(test_path, test)


def _migrate_checkpoints(config, paths, atomic_json):
    from Partial_Pooling.artifact_names import (
        checkpoint_directory,
        checkpoint_manifest_path,
        checkpoint_path,
        checkpoint_tag,
        training_tag,
    )

    preset_directory = paths.checkpoints / config.preset
    if not preset_directory.exists():
        return
    for old_directory in list(preset_directory.iterdir()):
        if not old_directory.is_dir():
            continue
        old_manifest = old_directory / "manifest.json"
        named_manifests = list(old_directory.glob("*-manifest.json"))
        manifest_path = old_manifest if old_manifest.exists() else (
            named_manifests[0] if len(named_manifests) == 1 else None
        )
        if manifest_path is None:
            continue
        metadata = json.loads(manifest_path.read_text())
        model_name = metadata["model"]
        new_directory = checkpoint_directory(config, model_name, paths)
        if old_directory != new_directory:
            _move(old_directory, new_directory)
            manifest_path = new_directory / manifest_path.name
        old_checkpoints = list(new_directory.glob("*_checkpoint.pt"))
        expected_checkpoint = checkpoint_path(config, model_name, paths)
        if not expected_checkpoint.exists():
            if len(old_checkpoints) != 1:
                raise RuntimeError(
                    f"Expected one checkpoint in {new_directory}, got {old_checkpoints}."
                )
            _move(old_checkpoints[0], expected_checkpoint)
        expected_manifest = checkpoint_manifest_path(config, model_name, paths)
        metadata.update({
            "training_method": training_tag(config),
            "training_samples": config.train_size,
            "validation_samples": config.validation_size,
            "checkpoint": expected_checkpoint.name,
            "checkpoint_tag": checkpoint_tag(config, model_name),
        })
        atomic_json(expected_manifest, metadata)
        if manifest_path.exists() and manifest_path != expected_manifest:
            manifest_path.unlink()


def _migrate_recovery(paths, configs, torch, atomic_json, atomic_torch):
    from Partial_Pooling.artifact_names import (
        checkpoint_tag,
        inference_dataset_filename,
        inference_report_filename,
        recovery_run_name,
    )

    method_by_figure_directory = {}
    recovery_root = paths.root / "partial_pooling_recovery"
    if not recovery_root.exists():
        return method_by_figure_directory
    for preset_directory in recovery_root.iterdir():
        if not preset_directory.is_dir():
            continue
        for old_directory in list(preset_directory.iterdir()):
            if not old_directory.is_dir():
                continue
            signature = old_directory.name.rsplit("-", 1)[-1]
            if not SIGNATURE_PATTERN.fullmatch(signature):
                continue
            artifact_paths = sorted({
                *old_directory.glob("dataset-*.pt"),
                *old_directory.glob("*-dataset-*.pt"),
            })
            if not artifact_paths:
                continue
            config_candidates = sorted(old_directory.glob("*-run_config.json"))
            old_config = old_directory / "run_config.json"
            config_path = old_config if old_config.exists() else (
                config_candidates[0] if len(config_candidates) == 1 else None
            )
            run_config = json.loads(config_path.read_text()) if config_path else {}
            first = torch.load(artifact_paths[0], map_location="cpu")
            method = _inference_method(first, run_config)
            config = configs[preset_directory.name]
            new_directory = preset_directory / recovery_run_name(method, signature)
            if old_directory != new_directory:
                _move(old_directory, new_directory)
                if config_path:
                    config_path = new_directory / config_path.name
            method_by_figure_directory[(preset_directory.name, signature)] = method

            for path in sorted({
                *new_directory.rglob("dataset-*.pt"),
                *new_directory.rglob("*-dataset-*.pt"),
            }):
                artifact = torch.load(path, map_location="cpu")
                artifact["inference_method"] = method
                artifact["training_model"] = "sde_joint"
                artifact["training_samples"] = config.train_size
                artifact["checkpoint_tag"] = checkpoint_tag(config, "sde_joint")
                if method == "dpm2_damped_sum":
                    artifact.setdefault("sampler", "dpm")
                    artifact.setdefault("order", 2)
                target = path.with_name(inference_dataset_filename(
                    method, artifact["dataset_id"],
                ))
                atomic_torch(target, artifact)
                if path != target:
                    path.unlink()

            report_specs = {
                "run_config.json": ("run_config", "json"),
                "global_recovery.csv": ("global_recovery", "csv"),
                "local_recovery.csv": ("local_recovery", "csv"),
                "summary.json": ("summary", "json"),
            }
            for old_name, (report, extension) in report_specs.items():
                source = new_directory / old_name
                target = new_directory / inference_report_filename(
                    method, report, extension,
                )
                _move(source, target)

            config_target = new_directory / inference_report_filename(
                method, "run_config", "json",
            )
            if config_target.exists():
                run_config.update({
                    "inference_method": method,
                    "inference_signature": signature,
                    "run_directory": new_directory.name,
                    "training_model": "sde_joint",
                    "training_samples": config.train_size,
                    "checkpoint_tag": checkpoint_tag(config, "sde_joint"),
                })
                if method == "dpm2_damped_sum":
                    run_config.setdefault("sampler", "dpm")
                    run_config.setdefault("order", 2)
                atomic_json(config_target, run_config)

            summary_target = new_directory / inference_report_filename(
                method, "summary", "json",
            )
            if summary_target.exists():
                summary = json.loads(summary_target.read_text())
                summary["inference_method"] = method
                atomic_json(summary_target, summary)

            sweep_directory = new_directory / "observation_sweep"
            old_manifest = sweep_directory / "manifest.json"
            new_manifest = sweep_directory / inference_report_filename(
                method, "manifest", "json",
            )
            manifest_path = _move(old_manifest, new_manifest)
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest["inference_method"] = method
                manifest["training_model"] = "sde_joint"
                manifest["training_samples"] = config.train_size
                manifest["checkpoint_tag"] = checkpoint_tag(config, "sde_joint")
                if method == "dpm2_damped_sum":
                    manifest.setdefault("sampler", "dpm")
                    manifest.setdefault("order", 2)
                atomic_json(manifest_path, manifest)
    return method_by_figure_directory


def _migrate_figures(paths, method_by_directory, atomic_json):
    figure_root = paths.figures / "partial_pooling"
    for (preset, signature), method in method_by_directory.items():
        old_directory = figure_root / preset / signature
        new_directory = figure_root / preset / f"{method}-{signature}"
        directory = _move(old_directory, new_directory)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["inference_method"] = method
            atomic_json(manifest_path, manifest)


def _migrate_legacy_posteriors(paths, torch, atomic_torch):
    """Prefix the method on tensors from the superseded multi-method pipeline."""
    from Partial_Pooling.artifact_names import inference_dataset_filename

    root = paths.posteriors
    if not root.exists():
        return
    for path in sorted(root.glob("*/*/dataset-*.pt")):
        artifact = torch.load(path, map_location="cpu")
        method = artifact.get("method") or path.parent.name
        artifact["method"] = method
        artifact["inference_method"] = method
        target = path.with_name(inference_dataset_filename(
            method, artifact["dataset_id"],
        ))
        atomic_torch(target, artifact)
        path.unlink()
        print(f"renamed {path} -> {target}", flush=True)


def run():
    import torch

    from Partial_Pooling.config import PRESETS, get_config
    from Partial_Pooling.io_utils import atomic_json, atomic_torch
    from Partial_Pooling.paths import BenchmarkPaths

    paths = BenchmarkPaths.default()
    configs = {preset: get_config(preset) for preset in PRESETS}
    for preset, config in configs.items():
        _migrate_training_data(config, paths, torch, atomic_json, atomic_torch)
        _migrate_checkpoints(config, paths, atomic_json)
    method_by_directory = _migrate_recovery(
        paths, configs, torch, atomic_json, atomic_torch,
    )
    _migrate_figures(paths, method_by_directory, atomic_json)
    _migrate_legacy_posteriors(paths, torch, atomic_torch)


def main():
    from runtime import configure_runtime

    configure_runtime()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
