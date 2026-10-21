"""Every published Simformer setting this comparison has to match, in one place.

Source: Gloeckler, Deistler, Weilbach, Wood, Macke, *All-in-one simulation-based
inference*, ICML 2024 (arXiv:2404.09636v3). Local copy:
``~/Papers/Gloeckler et al. - 2024 - All-in-one simulation-based inference.pdf``.
Code: https://github.com/mackelab/simformer.

Section references below are to that PDF. Nothing here is a COMPASS default and
nothing here may be "improved" -- an apples-to-apples arm is defined by using
these numbers even where COMPASS would choose differently. Where COMPASS cannot
express a setting, the mismatch is recorded in ``MISMATCHES`` rather than being
silently papered over.
"""

# ---------------------------------------------------------------- SDEs (A2.1)
# "We set sigma_max = 15, sigma_min = 0.0001, beta_min = 0.01, and beta_max = 10
#  for all experiments. Both for the time interval [1e-5, 1.]."
VESDE = {"sigma_max": 15.0, "sigma_min": 1e-4}
VPSDE = {"beta_min": 0.01, "beta_max": 10.0}
TIME_INTERVAL = (1e-5, 1.0)

# --------------------------------------------------------- Architecture (A2.1)
# "token dimension of 50 ... diffusion time through a 128-dimensional random
#  Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an
#  attention size of 10, and a widening factor of 3".
ARCHITECTURE = {
    "token_dim": 50,
    "layers": 6,
    "heads": 4,
    "attention_size": 10,
    "widening_factor": 3,
    "time_embedding_dim": 128,
}
# "For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the
#  number of layers to 8."
ARCHITECTURE_DEEP = dict(ARCHITECTURE, layers=8)

# ------------------------------------------------------------- Training (A2.1)
TRAINING = {
    "batch_size": 1000,
    "optimizer": "adam",
    "early_stopping": "validation loss",
}

# The condition-mask law (Sec. 3.1 and A2.1): "At every training batch, we
# selected uniformly at random a mask corresponding to the joint, the posterior,
# the likelihood or two random masks. The random masks were drawn from a
# Bernoulli distribution with p = 0.3 and p = 7."  ("p = 7" is a typo for 0.7;
# the released code uses 0.7.)  Note this is a *uniform* five-way draw, so the
# deployed posterior mask is seen 20% of the time by construction -- Simformer
# never has COMPASS's Bernoulli-only pathology, and never tested any other
# weighting.  See ``common/masks.py``.
MASK_MENU = {
    "patterns": ("joint", "posterior", "likelihood", "bernoulli_0.3", "bernoulli_0.7"),
    "weights": (1, 1, 1, 1, 1),
    "bernoulli_rates": (0.3, 0.7),
}

# ------------------------------------------------------------ Sampling (A2.1)
# "we solved the reverse SDE using an Euler-Maruyama discretization. We use 500
#  steps by default".  Fig. A7 shows 50 steps already saturate C2ST.
SAMPLING = {"solver": "euler_maruyama", "steps": 500, "steps_saturating": 50}

# ----------------------------------------------------------- Protocol (4.1/A2)
SIMULATION_BUDGETS = (1_000, 10_000, 100_000)
N_OBSERVATIONS = 10          # sbibm ground-truth observations per task
N_CONDITIONALS = 100         # random conditionals scored in the all-cond. tasks
NLL_EVAL_SAMPLES = 5_000     # A3.1, average NLL of the true parameter

# Reference-sample recipes for the non-sbibm conditionals (A2.2), per task.
REFERENCE_MCMC = {
    "two_moons": {"slice_steps": 1000, "mh_steps": 3000, "mh_step_size": 0.01},
    "slcp": {"slice_steps": 600, "mh_steps": 2000, "mh_step_size": 0.1},
    "tree": {"hmc_steps": 5000},
    "hmm": {"hmc_steps": 5000},
}

# Baselines they report. NPE/NLE/NRE come from the ``sbi`` library with default
# parameters "but opting for a more expressive neural spline flow for NPE and
# NLE"; NPSE is their own conditional-MLP score model.
BASELINES = ("NPE", "NLE", "NRE", "NPSE", "Simformer (posterior only)")


# --------------------------------------------------------------------------
# Where COMPASS cannot be made identical. Each entry is a real confound and
# belongs in the write-up of any arm that touches it.
MISMATCHES = {
    "attention_scope": (
        "Simformer attends densely over all tokens and uses the edge mask M_E "
        "only to *restrict* attention. COMPASS sets key_padding_mask = 1 - c "
        "(ConditionTransformer.py:167, TokenizedTransformer.py:205), so only "
        "*conditioned* tokens are keys and latent tokens never attend to each "
        "other. This is the single largest architectural difference and is "
        "itself the subject of e03."
    ),
    "joint_mask": (
        "Simformer's five-way menu includes the all-latent joint mask "
        "M_C = 0. Under COMPASS's key-padding convention that mask leaves the "
        "attention with no keys at all; it survives only because of "
        "add_bias_kv. compass.ConditionalMenu refuses it outright ('conditions "
        "nothing'). common/masks.py therefore implements the law directly."
    ),
    "attention_head_width": (
        "Their attention size is 10 per head with a token dim of 50 (heads do "
        "not tile the token dimension). torch.nn.MultiheadAttention splits "
        "hidden_size across heads, so 4 heads of a 50-dim token cannot be 10 "
        "wide. Matched on token dim, layers, heads and widening; head width "
        "differs by construction."
    ),
    "time_embedding": (
        "Theirs is a 128-dim random Gaussian Fourier embedding. COMPASS uses a "
        "sinusoidal embedding whose width is the time_embedding_size argument; "
        "set it to 128 to match the width, not the family."
    ),
    "edge_masks": (
        "COMPASS has no edge/adjacency mask M_E at all, so the 'directed graph' "
        "and 'undirected graph' arms -- their headline 10x simulation-efficiency "
        "claim -- cannot be reproduced without a trunk change. See e02."
    ),
    "functional_parameters": (
        "No Fourier index embedding and no random-time subsampling in COMPASS, "
        "so infinite-dimensional parameters (SIRD contact rate) are out of "
        "reach today. See e09."
    ),
    "guidance": (
        "COMPASS's sampler has cfg_alpha but no general constraint-guidance "
        "term s + grad log sigma(-s(t) c(x)). Needed for interval conditioning. "
        "See e10."
    ),
}
