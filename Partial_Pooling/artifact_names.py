"""Readable, stable names for partial-pooling artifacts."""


DATA_METHOD = "partial_pooling"


def training_tag(config):
    return f"{DATA_METHOD}-train-{config.train_size}"


def training_index_path(config, paths):
    return paths.data / f"{training_tag(config)}.json"


def training_shard_directory(config, paths):
    return paths.data / training_tag(config)


def training_shard_filename(config, split, start, stop):
    return (
        f"{training_tag(config)}-{split}-shard-"
        f"{int(start):06d}-{int(stop):06d}.pt"
    )


def normalization_path(config, paths):
    return paths.data / f"{training_tag(config)}-normalization.pt"


def test_data_path(config, paths):
    return paths.data / (
        f"{training_tag(config)}-test-{config.test_datasets}.pt"
    )


def checkpoint_tag(config, model_name):
    if config.sde_type == "vesde":
        return f"{model_name}-train-{config.train_size}"
    return f"{model_name}-{config.diffusion_tag}-train-{config.train_size}"


def checkpoint_directory(config, model_name, paths):
    return paths.checkpoints / config.preset / checkpoint_tag(config, model_name)


def checkpoint_path(config, model_name, paths):
    tag = checkpoint_tag(config, model_name)
    return checkpoint_directory(config, model_name, paths) / f"{tag}-checkpoint.pt"


def checkpoint_manifest_path(config, model_name, paths):
    tag = checkpoint_tag(config, model_name)
    return checkpoint_directory(config, model_name, paths) / f"{tag}-manifest.json"


def recovery_run_name(inference_method, signature):
    return f"{inference_method}-{signature}"


def recovery_run_directory(paths, preset, inference_method, signature):
    return (
        paths.root / "partial_pooling_recovery" / preset
        / recovery_run_name(inference_method, signature)
    )


def inference_dataset_filename(inference_method, dataset_id):
    return f"{inference_method}-dataset-{int(dataset_id):04d}.pt"


def inference_dataset_glob(inference_method):
    return f"{inference_method}-dataset-*.pt"


def inference_report_filename(inference_method, report, extension):
    return f"{inference_method}-{report}.{extension.lstrip('.')}"
