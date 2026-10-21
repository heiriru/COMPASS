"""Building the COMPASS arms for a Simformer comparison.

Three arms per task, and the distinction between them is the whole point of the
directory:

``paper_matched``
    COMPASS configured as close to the published Simformer as the code allows:
    their SDE constants, their token width / depth / head count, their five-way
    condition-mask law drawn per batch, their batch size, and Euler-Maruyama
    sampling at 500 steps. Every place COMPASS *cannot* match is listed in
    ``paper.MISMATCHES`` and re-stated by ``describe_arm``. This arm answers
    "how does COMPASS do on their protocol".

``compass_native``
    COMPASS's own measured recipe: ``time_sampling="log_sigma"``, a
    posterior-weighted structured mask menu, DPM-2 with churn at far fewer
    function evaluations. Same data, same budget, same metric. This arm answers
    "how good is COMPASS actually", and the gap between the two arms is a
    recipe effect, not an architecture effect.

``posterior_only``
    The paper's own control (Fig. A5/A6, "Simformer (posterior only)"), trained
    on the posterior mask alone. It is also the arm that settles the open
    COMPASS question in ``docs/followup_results.md``: their free-lunch result
    says posterior-only ties the full menu, while our IK numbers say a 7:1
    posterior weighting halves endpoint error. Both cannot be regime-independent.

Nothing here trains anything; these are constructors and configuration records.
"""

import torch

from compass import Atomic, ConditionalMenu
from compass import ScoreBasedInferenceModel as SBIm

from . import paper
from .masks import SimformerMaskSampler, posterior_mask

#: Their token dim is 50 with 4 heads of attention size 10, i.e. the attention
#: width is decoupled from the token width. ``nn.MultiheadAttention`` splits
#: hidden_size across heads and requires divisibility, so 50 with 4 heads is not
#: constructible. 52 is the nearest multiple of 4 to their token dim; 40 would
#: instead match their total attention width (4 x 10) and shrink the token. We
#: default to preserving the token dim and record the choice.
PAPER_HIDDEN_SIZE = 52
PAPER_HIDDEN_SIZE_ALTERNATIVE = 40

ARMS = ("paper_matched", "compass_native", "posterior_only")


def build_model(task, arm="paper_matched", sde="vesde", deep=False,
                hidden_size=None, arch="tokenized", device="cpu", x_spec=None):
    """Construct an untrained ``ScoreBasedInferenceModel`` for one arm.

    Args:
        task: a ``common.tasks.Task``.
        arm: one of ``ARMS``. Only ``compass_native`` departs from the paper's
            architecture settings.
        sde: ``"vesde"`` (the paper's default, Fig. 4/A5) or ``"vpsde"`` (A6).
        deep: 8 layers instead of 6 -- what they used for Lotka-Volterra, SIR
            and Hodgkin-Huxley.
        arch: ``"tokenized"`` is the closer analogue of Simformer (one shared
            trunk, per-variable identity embedding). ``"legacy"`` is COMPASS's
            per-node trunk, which ``inverse_kinematics/README.md`` measured as
            far better at small heterogeneous node counts. Run both before
            reading anything into a single number.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; choose from {ARMS}.")
    spec = paper.ARCHITECTURE_DEEP if deep else paper.ARCHITECTURE
    hidden_size = hidden_size or PAPER_HIDDEN_SIZE
    if hidden_size % spec["heads"]:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by "
            f"{spec['heads']} heads."
        )

    kwargs = dict(
        depth=spec["layers"],
        num_heads=spec["heads"],
        mlp_ratio=spec["widening_factor"],
        hidden_size=hidden_size,
        time_embedding_size=spec["time_embedding_dim"],
        device=device,
    )
    if sde == "vesde":
        kwargs.update(sde_type="vesde", sigma=paper.VESDE["sigma_max"])
    elif sde == "vpsde":
        kwargs.update(sde_type="vpsde", beta_min=paper.VPSDE["beta_min"],
                      beta_max=paper.VPSDE["beta_max"])
    else:
        raise ValueError(f"Unknown sde {sde!r}; choose 'vesde' or 'vpsde'.")

    if arch == "legacy":
        # time_embedding_size and the tokenized-only switches do not apply.
        kwargs.pop("time_embedding_size", None)
        return SBIm(nodes_size=task.nodes_size, arch="legacy", **kwargs)
    # One token per scalar, which is what Simformer does: every variable -- each
    # parameter and each data element -- is its own token. An IID block over
    # SLCP's four exchangeable 2-vectors would be the COMPASS-native layout and
    # a different experiment; it is not what the paper tokenizes.
    x_spec = x_spec or Atomic("x", task.x_dim)
    return SBIm(theta_size=task.theta_dim, x_spec=x_spec, arch=arch,
                token_offset=True, **kwargs)


def training_kwargs(task, model, arm="paper_matched", generator=None):
    """Keyword arguments for ``model.train`` implementing one arm's recipe.

    Unstated-in-the-paper settings are marked below; they are choices, and any
    table produced from them has to say so.
    """
    if arm == "paper_matched":
        sampler = SimformerMaskSampler(
            model.schema, task.theta_dim, per_batch=True, generator=generator)
        return {
            "batch_size": paper.TRAINING["batch_size"],
            "mask_sampler": sampler,
            # COMPASS's library default. The paper states "Adam optimizer" and
            # "early stopping based on validation loss" and no learning rate,
            # epoch cap or patience -- these three are ours.
            "lr": 1e-3,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            # Their loss samples t uniformly on [1e-5, 1]; COMPASS's "uniform"
            # is the matching scheme. "log_sigma" is the COMPASS-native choice
            # and belongs to the other arm.
            "time_sampling": "uniform",
        }
    if arm == "posterior_only":
        mask = posterior_mask(task.nodes_size, task.theta_dim)
        return {
            "batch_size": paper.TRAINING["batch_size"],
            "mask_sampler": _FixedMaskSampler(mask),
            "lr": 1e-3,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            "time_sampling": "uniform",
        }
    if arm == "compass_native":
        # The structured menu over (theta, x) -- with only two roles the six
        # conditionals collapse to the two non-degenerate ones, so the weights
        # are what carries the posterior emphasis measured in
        # docs/followup_results.md.
        menu = ConditionalMenu(
            model.schema,
            roles={"t": task.roles["theta"], "x": task.roles["x"]},
            patterns=[("posterior", "x"), ("likelihood", "t")],
            weights=[7.0, 1.0],
            bernoulli_fraction=0.25,
        )
        return {
            "batch_size": paper.TRAINING["batch_size"],
            "mask_sampler": menu,
            "lr": 5e-4,
            "max_epochs": 500,
            "early_stopping_patience": 60,
            "time_sampling": "log_sigma",
        }
    raise ValueError(f"Unknown arm {arm!r}; choose from {ARMS}.")


class _FixedMaskSampler:
    """Always returns one mask -- the posterior-only control."""

    def __init__(self, mask):
        self.mask = torch.as_tensor(mask, dtype=torch.float)

    def sample(self, shape):
        shape = torch.Size(shape)
        if shape[-1:] != self.mask.shape:
            raise ValueError(
                f"Expected trailing dimension {tuple(self.mask.shape)}, "
                f"got {tuple(shape)}."
            )
        return self.mask.expand(shape).clone()


def sampling_kwargs(arm="paper_matched", timesteps=None):
    """Sampler settings per arm.

    The paper: Euler-Maruyama, 500 steps, no churn, no correctors. COMPASS
    native: DPM-2 with churn at eta=2.0, which ``docs/unified_method.md``
    measured as the better predictor and which needs far fewer steps -- so the
    two arms must always be reported with their step counts attached, or the
    comparison silently trades accuracy against compute.
    """
    if arm == "compass_native":
        return {"method": "dpm", "order": 2, "timesteps": timesteps or 50,
                "churn_eta": 2.0, "corrector_steps": 0,
                "eps": paper.TIME_INTERVAL[0]}
    return {"method": "euler",
            "timesteps": timesteps or paper.SAMPLING["steps"],
            "churn_eta": 0.0, "corrector_steps": 0,
            "eps": paper.TIME_INTERVAL[0]}


def describe_arm(task, arm, sde="vesde"):
    """A record to write into every artifact manifest."""
    return {
        "task": task.name,
        "arm": arm,
        "sde": sde,
        "nodes_size": task.nodes_size,
        "theta_dim": task.theta_dim,
        "x_dim": task.x_dim,
        "paper_architecture": paper.ARCHITECTURE,
        "hidden_size_used": PAPER_HIDDEN_SIZE,
        "known_mismatches": sorted(paper.MISMATCHES),
    }
