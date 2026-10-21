"""Readable, stable names for partial-pooling artifacts."""


DATA_METHOD = "partial_pooling"


def training_tag(config):
    """Name of one training split.

    A grouped arm simulates several subjects per instance, so its split is
    genuinely different data and needs its own directory, normalization and
    index. The fragment is omitted at ``train_subjects == 1`` so every split
    generated before grouped arms existed keeps the name it already has.
    """
    tag = f"{DATA_METHOD}-train-{config.train_size}"
    if getattr(config, "train_subjects", 1) != 1:
        tag = f"{tag}-subj{config.train_subjects}"
    return tag


def base_training_tag(config):
    """Tag of the one-subject split a grouped arm shares constants with.

    A grouped split is different simulations and gets its own shards and index,
    but it must **not** get its own normalization constants or its own test
    datasets. Refitting normalizers on the same distribution would move every
    normalized input by sampling noise, and regenerating test data would break
    the byte-identity that makes a new arm comparable to the published ones.
    Both therefore stay on the base tag.
    """
    return f"{DATA_METHOD}-train-{config.preset_split.train_size}"


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
    return paths.data / f"{base_training_tag(config)}-normalization.pt"


def test_data_path(config, paths):
    return paths.data / (
        f"{base_training_tag(config)}-test-{config.test_datasets}.pt"
    )


# pytest collects any module-level name starting with "test_", so importing
# this helper into a test module makes pytest try to run it as a test and fail
# on a missing "config" fixture. It is a path builder, not a test.
test_data_path.__test__ = False


def checkpoint_tag(config, model_name):
    """Checkpoint name for one benchmark arm.

    Legacy VE models keep the historical ``{model}-train-{size}`` name exactly;
    every other axis (diffusion, architecture) contributes a fragment, so ladder
    arms sharing a preset cannot collide.
    """
    parts = [model_name]
    if config.sde_type != "vesde":
        parts.append(config.diffusion_tag)
    if config.architecture_tag:
        parts.append(config.architecture_tag)
    if config.subset_tag:
        parts.append(config.subset_tag)
    parts.append(f"train-{config.train_size}")
    return "-".join(parts)


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
