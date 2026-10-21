"""Central benchmark configuration and presets."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json


REFERENCE_URL = "https://github.com/bayesflow-org/diffusion-experiments/tree/main/case_study4"
REFERENCE_REVISION = "363186e485add4f062b2cf33f356ffa215f07056"
PRESETS = ("smoke", "full", "large", "compact", "small")
SDE_TYPES = ("vesde", "vpsde")
ARCHITECTURES = ("legacy", "tokenized")
# How the 90 observation scalars are read by the score network.
#   "scalar" -- 90 independent scalar tokens (100 tokens in all), the legacy
#               tokenization; with arch="tokenized" this is the Stage-2
#               "tokenized-raw" arm that isolates the trunk change.
#   "trial"  -- 30 trial tokens of (choice, log RT, censored) (40 tokens in
#               all), the Stage-3 lossless IID representation.
#   "group"  -- ``train_subjects`` subjects per row, each an Atomic(3) local
#               block plus its own IID trial block, read by the two-level
#               GroupedTransformer. This is the only tokenization that also
#               changes the *data*: the training split must be generated with
#               that many subjects per instance rather than one.
TOKENIZATIONS = ("scalar", "trial", "group")
# How diffusion times are drawn during training. Under a VESDE with sigma=25,
# "uniform" puts only ~1% of draws below sigma_m = 0.1, i.e. it barely trains the
# score network in the regime where posteriors are actually resolved;
# "log_sigma" is log-uniform in the noise scale and spends the budget evenly
# across scales. "mixture" is the 50/50 default every arm so far was trained
# under.
TIME_SAMPLINGS = ("mixture", "uniform", "log_sigma")
GROUP_POOLINGS = ("mean", "pma")
GROUP_SLOTS = ("learned", "sinusoidal", "none")
TRUNKS = ("shared", "position")
# Which conditionals the training-time condition mask is drawn from.
#   "bernoulli"  -- the historical Bernoulli(0.33) per token. Spreads training
#                   over all 2**tokens conditionals; at 40 tokens the deployed
#                   posterior mask has probability 0.67**10 * 0.33**30 = 6.5e-17
#                   and is never actually drawn.
#   "six"        -- uniform over compass.SIX_CONDITIONALS: p(g,l|x), p(g|l,x),
#                   p(l|g,x), p(x|g,l), p(x,l|g), p(x,g|l).
#   "six_mixed"  -- the same six, mixed with the Bernoulli draw at
#                   mask_bernoulli_fraction, so the general conditional
#                   capability is retained alongside the deployed ones.
MASK_MENUS = ("bernoulli", "six", "six_mixed")


def _signature(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class BenchmarkConfig:
    preset: str
    root_seed: int = 20250718
    train_size: int = 2048
    validation_size: int = 256
    test_datasets: int = 10
    subjects: int = 20
    trials: int = 30
    posterior_draws: int = 128
    diffusion_steps: int = 20
    train_epochs: int = 4
    patience: int = 2
    batch_size: int = 128
    shard_size: int = 512
    hidden_size: int = 64
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: int = 2
    sde_type: str = "vesde"
    sigma: float = 25.0
    beta_min: float = 0.1
    beta_max: float = 20.0
    dt: float = 0.001
    max_decision_time: float = 10.0
    correction: str = "prior_corrected_sum"
    train_subset: int = 0
    arch: str = "legacy"
    tokenization: str = "scalar"
    trunk: str = "shared"
    token_offset: bool = False
    time_embedding_size: int = 256
    per_position_head: bool = False
    mask_menu: str = "bernoulli"
    mask_bernoulli_fraction: float = 0.25
    mask_posterior_weight: float = 1.0
    # Subjects per training row. 1 is every arm built so far: the split is
    # generated one subject at a time and the sampler composes 100 of them.
    # Above 1 this is a *data* axis as well as a model one -- the simulations
    # themselves differ -- which is why it lives here rather than among the
    # architecture overrides, and why it contributes to both signatures.
    # Seed for weight initialisation and the training shuffle only. root_seed is
    # a *data* axis -- changing it regenerates the simulations -- so it cannot be
    # used to ask "how much does this arm vary between training runs on the same
    # data?", which is the question a seed-variance check has to answer. 0 means
    # "use root_seed", which is what every arm trained so far did.
    time_sampling: str = "mixture"
    init_seed: int = 0
    train_subjects: int = 1
    # Grouped-trunk options. depth_within and depth_across both come from
    # ``depth``, so depth=2 gives the recommended 2+2+0 at 761,797 parameters;
    # the readout stack is extra capacity and defaults to empty.
    depth_readout: int = 0
    group_pool: str = "mean"
    group_slot: str = "learned"

    def __post_init__(self):
        if self.train_subset < 0 or self.train_subset > self.train_size:
            raise ValueError(
                f"train_subset must be in [0, {self.train_size}], got "
                f"{self.train_subset}."
            )
        if self.arch not in ARCHITECTURES:
            raise ValueError(
                f"Unknown arch {self.arch!r}; choose one of "
                f"{', '.join(ARCHITECTURES)}."
            )
        if self.tokenization not in TOKENIZATIONS:
            raise ValueError(
                f"Unknown tokenization {self.tokenization!r}; choose one of "
                f"{', '.join(TOKENIZATIONS)}."
            )
        if self.trunk not in TRUNKS:
            raise ValueError(
                f"Unknown trunk {self.trunk!r}; choose one of "
                f"{', '.join(TRUNKS)}."
            )
        if self.arch == "legacy" and (
            self.tokenization != "scalar" or self.trunk != "shared"
            or self.token_offset or self.time_embedding_size != 256
            or self.per_position_head
        ):
            raise ValueError(
                "arch='legacy' is the per-node network; tokenization, trunk, "
                "token_offset, time_embedding_size and per_position_head are "
                "arch='tokenized' settings."
            )
        if self.mask_menu not in MASK_MENUS:
            raise ValueError(
                f"Unknown mask_menu {self.mask_menu!r}; choose one of "
                f"{', '.join(MASK_MENUS)}."
            )
        if self.train_subjects < 1:
            raise ValueError(
                f"train_subjects must be >= 1; got {self.train_subjects}."
            )
        if self.tokenization == "group":
            if self.subjects % self.train_subjects:
                raise ValueError(
                    f"train_subjects={self.train_subjects} must divide "
                    f"subjects={self.subjects}, so a test dataset's subjects "
                    "regroup evenly into composed units."
                )
        elif self.train_subjects != 1:
            raise ValueError(
                f"train_subjects={self.train_subjects} needs "
                "tokenization='group'; any other tokenization would flatten the "
                "extra subjects into one undifferentiated observation block."
            )
        if self.time_sampling not in TIME_SAMPLINGS:
            raise ValueError(
                f"Unknown time_sampling {self.time_sampling!r}; choose one of "
                f"{', '.join(TIME_SAMPLINGS)}."
            )
        if self.arch == "legacy" and self.time_sampling != "mixture":
            raise ValueError(
                "time_sampling is recorded in the architecture tag, which is "
                "empty for arch='legacy' so that legacy artifact names stay "
                "byte-identical. Use a tokenized arch to vary it."
            )
        if self.group_pool not in GROUP_POOLINGS:
            raise ValueError(
                f"Unknown group_pool {self.group_pool!r}; choose one of "
                f"{', '.join(GROUP_POOLINGS)}."
            )
        if self.group_slot not in GROUP_SLOTS:
            raise ValueError(
                f"Unknown group_slot {self.group_slot!r}; choose one of "
                f"{', '.join(GROUP_SLOTS)}."
            )
        if self.depth_readout < 0:
            raise ValueError(
                f"depth_readout must be >= 0; got {self.depth_readout}."
            )
        if self.tokenization != "group" and (
            self.depth_readout or self.group_pool != "mean"
            or self.group_slot != "learned"
        ):
            raise ValueError(
                "depth_readout, group_pool and group_slot are "
                "tokenization='group' settings."
            )
        if self.mask_posterior_weight <= 0.0:
            raise ValueError(
                "mask_posterior_weight must be positive; got "
                f"{self.mask_posterior_weight}."
            )
        if not 0.0 <= self.mask_bernoulli_fraction <= 1.0:
            raise ValueError(
                "mask_bernoulli_fraction must be in [0, 1]; got "
                f"{self.mask_bernoulli_fraction}."
            )
        if self.time_embedding_size < 2 or self.time_embedding_size % 2:
            raise ValueError(
                "time_embedding_size must be a positive even number; got "
                f"{self.time_embedding_size}."
            )
        if self.sde_type not in SDE_TYPES:
            raise ValueError(
                f"Unknown SDE type {self.sde_type!r}; choose one of "
                f"{', '.join(SDE_TYPES)}."
            )
        if self.sigma <= 1.0:
            raise ValueError("VESDE sigma must be greater than 1.")
        if self.beta_min <= 0.0 or self.beta_max <= self.beta_min:
            raise ValueError("VPSDE requires 0 < beta_min < beta_max.")

    def to_dict(self):
        return asdict(self)

    def data_dict(self):
        """Configuration fields that can change simulated data or its layout."""
        values = self.to_dict()
        keys = (
            "preset", "root_seed", "train_size", "validation_size",
            "test_datasets", "subjects", "trials", "shard_size", "dt",
            "max_decision_time",
        )
        payload = {key: values[key] for key in keys}
        # Popped at its default so every split generated before grouped arms
        # existed keeps its signature, exactly as model_dict does for the
        # options added after the ladder arms were trained.
        if self.train_subjects != 1:
            payload["train_subjects"] = self.train_subjects
        return payload

    def model_dict(self):
        """Training configuration, retaining legacy VE checkpoint signatures."""
        values = self.to_dict()
        if self.arch == "legacy":
            # These fields did not exist before the IID-embedding ladder and do
            # not describe a legacy model. Omitting them keeps every existing
            # legacy checkpoint signature -- and therefore every existing
            # checkpoint -- valid.
            for key in ("arch", "tokenization", "trunk", "token_offset"):
                values.pop(key)
        if not self.train_subset:
            values.pop("train_subset")
        # Fields added after the ladder arms were trained. Popping them at their
        # default keeps every existing checkpoint signature -- legacy and
        # tokenized alike -- valid, so no already-trained arm is invalidated by
        # the option merely existing.
        for key, default in (
            ("time_embedding_size", 256), ("per_position_head", False),
            ("mask_menu", "bernoulli"),
            # train_subjects is in both dicts on purpose: it changes the
            # simulations and it is the Group's count.
            ("train_subjects", 1), ("depth_readout", 0), ("init_seed", 0),
            ("time_sampling", "mixture"),
            ("group_pool", "mean"), ("group_slot", "learned"),
        ):
            if values.get(key) == default:
                values.pop(key, None)
        if values.get("mask_menu") != "six_mixed":
            values.pop("mask_bernoulli_fraction", None)
        if self.mask_menu == "bernoulli" or self.mask_posterior_weight == 1.0:
            values.pop("mask_posterior_weight", None)
        if self.sde_type == "vesde":
            # These fields did not exist in the original VE manifests and do not
            # affect a VE model. Omitting them keeps every existing VE checkpoint
            # reusable after VPSDE support is enabled.
            values.pop("beta_min")
            values.pop("beta_max")
        return values

    @property
    def signature(self):
        """Backward-compatible alias for the model/training signature."""
        return self.model_signature

    @property
    def model_signature(self):
        return _signature(self.model_dict())

    @property
    def data_signature(self):
        return _signature(self.data_dict())

    @property
    def legacy_data_signature(self):
        """Signature used by pre-VPSDE data generated with the same preset."""
        legacy = replace(
            self, sde_type="vesde", sigma=25.0,
            beta_min=0.1, beta_max=20.0,
        ).to_dict()
        legacy.pop("beta_min")
        legacy.pop("beta_max")
        return _signature(legacy)

    @property
    def network_arch(self):
        """The ``arch`` the network is actually built with.

        ``arch`` stays "tokenized" for a grouped arm -- it is the benchmark's
        legacy-vs-schema switch, and every tokenized-only validation applies --
        while the *network* is "grouped", because the nested schema selects the
        two-level trunk. Anything that talks to
        ``ScoreBasedInferenceModel(arch=...)`` wants this one.
        """
        if self.tokenization == "group":
            return "grouped"
        return self.arch

    @property
    def preset_split(self):
        """This preset as shipped, before any override.

        Its split is the one whose normalization constants and test datasets
        every arm of the preset shares, whatever row count or subject count the
        arm itself was generated with.
        """
        return get_config(self.preset)

    @property
    def base_data_signature(self):
        """Signature of the preset's own split, which this arm borrows from.

        A grouped arm generates its own simulations but reuses the preset
        split's normalization and test datasets verbatim (see
        ``artifact_names.base_training_tag``), so those two files validate
        against this signature rather than the arm's own.
        """
        preset = self.preset_split
        return _signature(replace(
            self, train_subjects=1,
            tokenization="trial" if self.tokenization == "group" else self.tokenization,
            train_size=preset.train_size,
            validation_size=preset.validation_size,
        ).data_dict())

    @property
    def compatible_data_signatures(self):
        return {
            self.data_signature, self.legacy_data_signature,
            self.base_data_signature,
        }

    @property
    def effective_train_size(self):
        """Simulations actually shown to the network."""
        return self.train_subset or self.train_size

    @property
    def effective_validation_size(self):
        """Validation simulations, scaled with the training subset."""
        if not self.train_subset:
            return self.validation_size
        return max(1, round(
            self.validation_size * self.train_subset / self.train_size
        ))

    @property
    def subset_tag(self):
        """Filename fragment for a training-set subset; empty when unused.

        A subset reuses an existing split's simulations, normalization constants
        and test datasets verbatim -- only the number of rows the optimizer sees
        changes -- so it is a model-side axis, never a data-side one.
        """
        return f"sub{self.train_subset}" if self.train_subset else ""

    @property
    def architecture_tag(self):
        """Filename fragment distinguishing one ladder arm from another.

        Empty for a legacy model, so legacy artifact names are byte-identical to
        what they have always been.
        """
        if self.arch == "legacy":
            return ""
        parts = [
            "tok", self.tokenization,
            f"h{self.hidden_size}d{self.depth}x{self.num_heads}m{self.mlp_ratio}",
        ]
        if self.time_sampling != "mixture":
            parts.append(f"ts{self.time_sampling.replace('_', '')}")
        if self.init_seed:
            parts.append(f"i{self.init_seed}")
        if self.tokenization == "group":
            parts.append(f"gK{self.train_subjects}")
            if self.depth_readout:
                parts.append(f"rd{self.depth_readout}")
            if self.group_pool != "mean":
                parts.append(self.group_pool)
            if self.group_slot != "learned":
                parts.append(f"slot{self.group_slot}")
        if self.trunk != "shared":
            parts.append(self.trunk)
        if self.token_offset:
            parts.append("offset")
        if self.time_embedding_size != 256:
            parts.append(f"te{self.time_embedding_size}")
        if self.per_position_head:
            parts.append("pph")
        if self.mask_menu == "six":
            parts.append("mask6")
        if self.mask_menu != "bernoulli" and self.mask_posterior_weight != 1.0:
            weight = format(self.mask_posterior_weight, "g").replace(".", "p")
            parts.append(f"pw{weight}")
        elif self.mask_menu == "six_mixed":
            fraction = format(self.mask_bernoulli_fraction, "g").replace(".", "p")
            parts.append(f"mask6b{fraction}")
        return "-".join(parts)

    @property
    def diffusion_tag(self):
        if self.sde_type == "vesde":
            return "vesde"
        beta_min = format(self.beta_min, "g").replace("-", "m").replace(".", "p")
        beta_max = format(self.beta_max, "g").replace("-", "m").replace(".", "p")
        return f"vpsde-b{beta_min}-{beta_max}"


def get_config(
    preset="smoke", root_seed=None, sde_type=None,
    beta_min=None, beta_max=None, train_subset=None, arch=None, tokenization=None,
    trunk=None, token_offset=None, time_embedding_size=None,
    per_position_head=None, mask_menu=None, mask_bernoulli_fraction=None,
    mask_posterior_weight=None, train_subjects=None, depth_readout=None,
    group_pool=None, group_slot=None, train_size=None, validation_size=None,
    init_seed=None, time_sampling=None,
    hidden_size=None, depth=None,
    num_heads=None, mlp_ratio=None, train_epochs=None, patience=None,
    batch_size=None, lr=None,
):
    """Build a benchmark configuration.

    The architecture overrides are deliberately *not* new presets: they leave
    ``data_dict`` untouched, so every ladder arm reuses the same simulations,
    the same normalization constants and the same test datasets as the legacy
    models it is being compared against.
    """
    if preset == "smoke":
        config = BenchmarkConfig(preset="smoke")
    elif preset in {"full", "large"}:
        config = BenchmarkConfig(
            preset=preset,
            train_size=100000 if preset == "large" else 32768,
            validation_size=5000 if preset == "large" else 4096,
            test_datasets=100, subjects=100, trials=30,
            posterior_draws=1000, diffusion_steps=100,
            train_epochs=500, patience=20, batch_size=128,
            shard_size=2048, hidden_size=128, depth=6,
            num_heads=16, mlp_ratio=4,
        )
    elif preset == "compact":
        # Sub-1M-parameter capacity ablation of "large": same 800,000-sample
        # scale of training data as "large" uses 100,000 (8x), but a backbone
        # thinned to hidden_size=4/depth=1/num_heads=1/mlp_ratio=2 (~867K
        # parameters for this problem's nodes_size=100). depth=1 is a hard
        # requirement here, not a preference: nodes_size=100 (10 shared+local
        # theta dims + 90 flattened trial features) means each transformer
        # layer's per-node adaLN-modulation cost alone already dominates the
        # budget, so depth=2 at any hidden_size below the "full"/"large"
        # regime exceeds 1M parameters (verified empirically: hidden_size=4,
        # depth=2 -> ~1.48M).
        config = BenchmarkConfig(
            preset=preset,
            train_size=800000, validation_size=40000,
            test_datasets=100, subjects=100, trials=30,
            posterior_draws=1000, diffusion_steps=100,
            train_epochs=500, patience=20, batch_size=128,
            shard_size=2048, hidden_size=4, depth=1,
            num_heads=1, mlp_ratio=2,
        )
    elif preset == "small":
        # Reduced-capacity counterpart of "large": 5,957,984 parameters
        # (hidden_size=16/depth=2/num_heads=2/mlp_ratio=4) trained on 400,000
        # simulations, versus 127,748,608 parameters on 100,000 simulations.
        #
        # The two settings are floors, not tuning choices. num_heads=2 keeps
        # head_dim=8: attention must route 90 flattened trial nodes into the 3
        # subject-local and 7 group-level parameter nodes, and a smaller
        # query-key subspace cannot express that selection. depth=2 gives the
        # two rounds of routing the hierarchy needs (trials -> local, local ->
        # global) and makes the score nonlinearly composed in x, which
        # "gauss_jacobian" composition and "newton_map_estimate" curvature both
        # consume -- a near-affine score yields an x-independent Jacobian.
        #
        # Note that the total count understates the reduction: 97% of the
        # parameters here are the per-node adaLN modulation, Linear(256,
        # 6*nodes_size*hidden_size), which is a function of the diffusion time
        # only. The data-dependent pathway is 168,096 parameters against
        # 2,480,896 for "large".
        config = BenchmarkConfig(
            preset=preset,
            train_size=400000, validation_size=20000,
            test_datasets=100, subjects=100, trials=30,
            posterior_draws=1000, diffusion_steps=100,
            train_epochs=500, patience=20, batch_size=128,
            shard_size=2048, hidden_size=16, depth=2,
            num_heads=2, mlp_ratio=4,
        )
    else:
        raise ValueError(
            f"Unknown preset {preset!r}; choose one of {', '.join(PRESETS)}."
        )
    overrides = {}
    for key, value, cast in (
        ("train_subset", train_subset, int),
        ("arch", arch, str), ("tokenization", tokenization, str),
        ("trunk", trunk, str), ("token_offset", token_offset, bool),
        ("time_embedding_size", time_embedding_size, int),
        ("per_position_head", per_position_head, bool),
        ("mask_menu", mask_menu, str),
        ("mask_bernoulli_fraction", mask_bernoulli_fraction, float),
        ("mask_posterior_weight", mask_posterior_weight, float),
        # Data-side, unlike every other override here: a grouped arm simulates
        # its own split and sizes it to the experiment (100,000 groups for the
        # globals-matched arm, 100,000/K for the budget-matched one) rather than
        # to the preset. It still borrows the preset's normalization constants
        # and test datasets; see base_data_signature.
        ("train_size", train_size, int),
        ("validation_size", validation_size, int),
        ("init_seed", init_seed, int),
        ("time_sampling", time_sampling, str),
        ("train_subjects", train_subjects, int),
        ("depth_readout", depth_readout, int),
        ("group_pool", group_pool, str), ("group_slot", group_slot, str),
        ("hidden_size", hidden_size, int), ("depth", depth, int),
        ("num_heads", num_heads, int), ("mlp_ratio", mlp_ratio, int),
        ("train_epochs", train_epochs, int), ("patience", patience, int),
        ("batch_size", batch_size, int),
    ):
        if value is not None:
            overrides[key] = cast(value)
    if root_seed is not None:
        overrides["root_seed"] = int(root_seed)
    if sde_type is not None:
        overrides["sde_type"] = str(sde_type).lower()
    if beta_min is not None:
        overrides["beta_min"] = float(beta_min)
    if beta_max is not None:
        overrides["beta_max"] = float(beta_max)
    if overrides:
        config = replace(config, **overrides)
    return config
