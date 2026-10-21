"""Shared pieces of the WAIC validation runners: datasets and result files."""

import json
import os

import numpy as np

from validation.a4_waic.families import N_ROWS, family

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
DATA_SEED = 20261017
# Families analysed with other priors but tested on another family's datasets.
DATA_FAMILY = {"population_wide": "population"}


def dataset(family_name, true_model, rep, seed=DATA_SEED):
    """Deterministic test dataset ``rep`` of ``true_model`` (raw coordinates).

    The global and locals are drawn from the true model's prior, so over reps with
    the true model chosen uniformly the data follow the prior predictive of an equal
    mixture of the models: the setting in which posterior model probabilities are
    calibrated.
    """
    family_name = DATA_FAMILY.get(family_name, family_name)
    models = family(family_name)
    names = list(models)
    rng = np.random.default_rng([seed, sum(map(ord, family_name)), names.index(true_model), rep])
    return models[true_model].dataset(N_ROWS[family_name], rng)


def result_path(kind, family_name, tag, true_model, shard=0, shards=1):
    folder = os.path.join(RESULTS, kind, family_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{tag}_true{true_model}_shard{shard}of{shards}.jsonl")


def done_reps(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["rep"] for line in f if line.strip()}


def append(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_records(kind, family_name, tag):
    folder = os.path.join(RESULTS, kind, family_name)
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if name.startswith(tag + "_true") and name.endswith(".jsonl"):
            with open(os.path.join(folder, name)) as f:
                out.extend(json.loads(line) for line in f if line.strip())
    return out
