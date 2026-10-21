"""Native COMPASS model construction, training, and checkpoint reuse."""

from dataclasses import dataclass
import json
import torch

from compass import (
    Atomic, Group, IID, ScoreBasedInferenceModel, SIX_CONDITIONALS,
    six_conditional_menu,
)

from .artifact_names import (
    checkpoint_directory,
    checkpoint_manifest_path,
    checkpoint_path as named_checkpoint_path,
    checkpoint_tag,
    normalization_path,
    training_tag,
)
from .data_pipeline import load_split
from .io_utils import atomic_json
from .normalization import Normalizer
from .paths import BenchmarkPaths


@dataclass(frozen=True)
class ModelSpec:
    name: str
    theta_key: str
    x_key: str
    theta_dim: int
    x_dim: int


def model_specs(config):
    obs_dim = 3 * config.trials
    return {
        "sde_joint": ModelSpec("sde_joint", "sde_joint_theta", "sde_observations", 10, obs_dim),
    }


def group_subjects(config):
    """Subjects per composed factor: ``train_subjects`` for a grouped arm, else 1."""
    return config.train_subjects if config.tokenization == "group" else 1


# The partial-pooling node vector is [7 group-level, 3 subject-level, 3*trials
# observations]; see Partial_Pooling/schema.py:GLOBAL_NAMES / LOCAL_NAMES. Under
# tokenization="group" it becomes [7 group-level, K x (3 subject-level, 3*trials)].
GLOBAL_THETA_DIM = 7
LOCAL_THETA_DIM = 3


def build_model(config, spec, device="cpu"):
    """Construct the score network for one benchmark arm.

    ``arch="legacy"`` reproduces the historical call exactly. ``arch="tokenized"``
    routes through the schema API: ``tokenization="scalar"`` keeps one token per
    scalar (so only the trunk differs from legacy), ``tokenization="trial"``
    groups each trial's ``(choice, log RT, censored)`` into a single token via
    one shared embedder, taking the sequence from 100 tokens to 40 while the
    diffusion state stays the 90 raw trial scalars.

    ``tokenization="group"`` puts ``config.train_subjects`` subjects in one row,
    each an ``Atomic(3)`` local block followed by its own ``IID`` trial block,
    and selects the two-level ``GroupedTransformer``. The diffusion state is
    still the raw scalars, so the likelihood and the composition identity are
    unchanged -- what changes is that one composed factor now holds several
    subjects instead of one.
    """
    common = dict(
        sde_type=config.sde_type, sigma=config.sigma,
        beta_min=config.beta_min, beta_max=config.beta_max,
        hidden_size=config.hidden_size, depth=config.depth,
        num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
        device=device,
    )
    if config.arch == "legacy":
        return ScoreBasedInferenceModel(
            nodes_size=spec.theta_dim + spec.x_dim, **common,
        )
    tokenized = dict(
        arch="tokenized", trunk=config.trunk, token_offset=config.token_offset,
        time_embedding_size=config.time_embedding_size,
        per_position_head=config.per_position_head,
    )
    if config.tokenization == "scalar":
        return ScoreBasedInferenceModel(
            nodes_size=spec.theta_dim + spec.x_dim, **tokenized, **common,
        )
    if spec.x_dim % config.trials:
        raise ValueError(
            f"tokenization={config.tokenization!r} needs x_dim divisible by "
            f"trials; got x_dim={spec.x_dim}, trials={config.trials}."
        )
    item_dim = spec.x_dim // config.trials
    if config.tokenization == "group":
        # arch="grouped", not "tokenized": the nested schema selects the
        # two-level trunk, and trunk/token_offset are flat-trunk options that do
        # not apply to it.
        tokenized.pop("arch")
        tokenized.pop("trunk")
        tokenized.pop("token_offset")
        return ScoreBasedInferenceModel(
            arch="grouped",
            theta_size=GLOBAL_THETA_DIM,
            x_spec=[Group(
                "subjects", count=config.train_subjects,
                blocks=[
                    Atomic("local", LOCAL_THETA_DIM),
                    IID("trials", item_dim=item_dim, count=config.trials),
                ],
                pool=config.group_pool, slot=config.group_slot,
            )],
            depth_readout=config.depth_readout,
            **tokenized, **common,
        )
    return ScoreBasedInferenceModel(
        theta_size=spec.theta_dim,
        x_spec=[IID("trials", item_dim=item_dim, count=config.trials)],
        **tokenized, **common,
    )


def build_mask_sampler(config, model):
    """The training-time condition-mask sampler for one arm.

    ``mask_menu="bernoulli"`` returns None, which leaves the Trainer on its
    historical Bernoulli(0.33) draw. The menus restrict training to the six
    conditionals the samplers actually evaluate; see ``compass.SIX_CONDITIONALS``.
    """
    if config.mask_menu == "bernoulli":
        return None
    fraction = (
        config.mask_bernoulli_fraction if config.mask_menu == "six_mixed" else 0.0
    )
    # The three conditionals with the observations *latent* carry 5-7x the loss
    # of the three with them observed, so a uniform menu spends roughly four
    # fifths of its gradient on generating x rather than on the posteriors the
    # samplers deploy. mask_posterior_weight upweights the observation-conditioned
    # half -- p(g,l|x), p(g|l,x), p(l|g,x) -- to rebalance that without dropping
    # any conditional from the menu.
    weights = [
        config.mask_posterior_weight if "x" in conditioned else 1.0
        for _, conditioned in SIX_CONDITIONALS
    ]
    return six_conditional_menu(
        model.schema, global_size=GLOBAL_THETA_DIM, local_size=LOCAL_THETA_DIM,
        weights=weights, bernoulli_fraction=fraction,
    )


def _grouped_pair(payload, normalizers):
    """Normalized ``(globals, repeats)`` for a grouped arm.

    ``repeats`` is repeat-major ``[(locals_j, obs_j) for j in range(K)]``, which
    is exactly the ``Group`` layout, so concatenating the two gives the node
    vector. The split point is 7 rather than 10 because each subject's own
    latent locals now travel with that subject rather than sitting in a single
    shared theta block.

    The constants are the preset split's, applied per subject: the joint-theta
    normalizer's first 7 dimensions to the group-level parameters, its last 3 to
    every subject's locals, and the observation normalizer to every subject's
    trials. Reusing them rather than refitting is what keeps a grouped arm's
    inputs numerically comparable with the published flat arms.
    """
    source = payload["sde"]
    theta = normalizers["sde_joint_theta"]
    observation = normalizers["sde_observations"]
    split = GLOBAL_THETA_DIM
    globals_ = (source["globals"] - theta.mean[:split]) / theta.scale[:split]
    locals_ = (source["locals"] - theta.mean[split:]) / theta.scale[split:]
    trials = (source["observations"] - observation.mean) / observation.scale
    repeats = torch.cat((locals_, trials), dim=-1)
    return globals_, repeats.reshape(repeats.shape[0], -1)


def _raw_pair(name, payload):
    if name == "sde_joint":
        source = payload["sde"]
        return torch.cat((source["globals"], source["locals"][:, 0]), 1), source["observations"][:, 0]
    raise KeyError(name)


def load_normalizers(config, paths=None):
    paths = (paths or BenchmarkPaths.default()).ensure()
    state = torch.load(normalization_path(config, paths), map_location="cpu")
    signature = state.get("data_signature", state.get("config_signature"))
    if signature not in config.compatible_data_signatures:
        raise RuntimeError("Normalization configuration mismatch.")
    return {key: Normalizer.from_state_dict(value) for key, value in state["normalizers"].items()}


def checkpoint_path(config, name, paths=None):
    paths = (paths or BenchmarkPaths.default()).ensure()
    return named_checkpoint_path(config, name, paths)


def train_models(config, names=None, paths=None, force=False, device="cpu"):
    paths = (paths or BenchmarkPaths.default()).ensure()
    specs = model_specs(config)
    if names is None or "all" in names:
        names = ["sde_joint"]
    unknown = set(names) - set(specs)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    train = load_split(config, "train", paths)
    validation = load_split(config, "validation", paths)
    normalizers = load_normalizers(config, paths)
    outputs = []
    for name in names:
        spec = specs[name]
        directory = checkpoint_directory(config, name, paths)
        checkpoint = checkpoint_path(config, name, paths)
        signature_path = checkpoint_manifest_path(config, name, paths)
        if checkpoint.exists() and signature_path.exists() and not force:
            saved = json.loads(signature_path.read_text())
            if saved.get("config_signature") != config.signature:
                raise RuntimeError(f"Checkpoint configuration mismatch: {checkpoint}")
            outputs.append(checkpoint)
            continue
        if checkpoint.exists() and not signature_path.exists() and not force:
            raise RuntimeError(
                f"Refusing to overwrite unmanifested checkpoint {checkpoint}; "
                "pass --force to replace it."
            )
        grouped = config.tokenization == "group"
        if grouped:
            theta, x = _grouped_pair(train, normalizers)
            theta_val, x_val = _grouped_pair(validation, normalizers)
        else:
            theta, x = _raw_pair(name, train)
            theta_val, x_val = _raw_pair(name, validation)
        # A training subset reuses the split's simulations, normalization and
        # test datasets unchanged; only the number of rows the optimizer sees
        # differs, which is what makes a subset run comparable with the full-size
        # run it is measured against.
        if config.train_subset:
            theta, x = theta[:config.train_subset], x[:config.train_subset]
            validation_rows = config.effective_validation_size
            theta_val, x_val = theta_val[:validation_rows], x_val[:validation_rows]
        if not grouped:
            # The grouped pair is normalized as it is built: its constants are
            # applied per subject, which a single whole-row transform cannot do.
            theta = normalizers[spec.theta_key].transform(theta)
            x = normalizers[spec.x_key].transform(x)
            theta_val = normalizers[spec.theta_key].transform(theta_val)
            x_val = normalizers[spec.x_key].transform(x_val)
        # init_seed varies the weights and the shuffle without touching the
        # simulations, so two arms with different init_seed are the same
        # experiment run twice rather than two different experiments.
        torch.manual_seed((config.init_seed or config.root_seed)
                          + sum(map(ord, name)))
        model = build_model(config, spec, device=device)
        parameters = sum(p.numel() for p in model.model.parameters())
        print(
            f"[{name}] arch={model.arch} tokenization={config.tokenization} "
            f"subjects={group_subjects(config)} nodes={model.nodes_size} "
            f"tokens={model.token_count} parameters={parameters:,} "
            f"train={theta.shape[0]:,} validation={theta_val.shape[0]:,}",
            flush=True,
        )
        mask_sampler = build_mask_sampler(config, model)
        if mask_sampler is not None:
            print(f"[{name}] condition-mask menu:\n{mask_sampler.describe()}",
                  flush=True)
        model.train(
            theta, x, theta_val, x_val,
            batch_size=config.batch_size, max_epochs=config.train_epochs,
            early_stopping_patience=config.patience, device=device,
            verbose=True, path=str(directory), name=checkpoint_tag(config, name),
            mask_sampler=mask_sampler, time_sampling=config.time_sampling,
        )
        # Trainer writes the best-validation checkpoint as "{name}_checkpoint.pt";
        # the benchmark's canonical name is "{tag}-checkpoint.pt". Renaming here
        # keeps a freshly trained arm immediately loadable instead of requiring
        # a separate migration pass.
        trainer_output = directory / f"{checkpoint_tag(config, name)}_checkpoint.pt"
        if not checkpoint.exists() and trainer_output.exists():
            trainer_output.replace(checkpoint)
        if not checkpoint.exists():
            raise RuntimeError(
                f"Training produced no checkpoint at {checkpoint}."
            )
        atomic_json(signature_path, {
            "config_signature": config.model_signature,
            "model_signature": config.model_signature,
            "data_signature": config.data_signature,
            "model": name,
            "sde_type": config.sde_type,
            "diffusion": config.diffusion_tag,
            "training_method": training_tag(config),
            "training_samples": config.effective_train_size,
            "validation_samples": config.effective_validation_size,
            "split_training_samples": config.train_size,
            "train_subset": config.train_subset or None,
            "checkpoint": checkpoint.name,
            "device": device,
            "theta_dim": spec.theta_dim, "x_dim": spec.x_dim,
            "arch": config.arch,
            "tokenization": config.tokenization,
            "trunk": config.trunk if config.arch == "tokenized" else None,
            "token_offset": config.token_offset,
            "time_embedding_size": config.time_embedding_size,
            "per_position_head": config.per_position_head,
            "mask_menu": config.mask_menu,
            "mask_posterior_weight": config.mask_posterior_weight,
            "mask_bernoulli_fraction": (
                config.mask_bernoulli_fraction
                if config.mask_menu == "six_mixed" else None
            ),
            "tokens": model.token_count,
            "parameters": parameters,
            "hidden_size": config.hidden_size, "depth": config.depth,
            "num_heads": config.num_heads, "mlp_ratio": config.mlp_ratio,
            "normalization": {"theta": spec.theta_key, "x": spec.x_key},
            "train_loss": getattr(model.trainer, "train_loss", []),
            "validation_loss": getattr(model.trainer, "val_loss", []),
        })
        outputs.append(checkpoint)
    return outputs


def load_model(config, name, paths=None, device="cpu"):
    checkpoint = checkpoint_path(config, name, paths)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}; run train_models.py first.")
    # Passing arch= makes the dispatch assertive: a legacy checkpoint can never
    # be silently reinterpreted as a tokenized one, or the other way round.
    model = ScoreBasedInferenceModel.load(
        str(checkpoint), device=device, arch=config.network_arch,
    )
    if config.arch == "tokenized":
        expected = build_model(config, model_specs(config)[name], device="cpu")
        if model.schema != expected.schema:
            raise RuntimeError(
                f"Checkpoint schema {model.schema} does not match the "
                f"configured schema {expected.schema}."
            )
    if model.sde_type != config.sde_type:
        raise RuntimeError(
            f"Checkpoint SDE {model.sde_type!r} does not match "
            f"configuration {config.sde_type!r}."
        )
    if config.sde_type == "vpsde" and (
        model.beta_min != config.beta_min or model.beta_max != config.beta_max
    ):
        raise RuntimeError(
            "Checkpoint VPSDE beta schedule does not match the configuration."
        )
    return model
