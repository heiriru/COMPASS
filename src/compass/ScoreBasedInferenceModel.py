import torch
import torch.nn as nn

from .ConditionTransformer import ConditionTransformer
from .GroupedTransformer import GroupedTransformer
from .TokenizedTransformer import TokenizedTransformer
from .schema import (
    Atomic, IID, NodeSchema, ConditionalMenu, SIX_CONDITIONALS,
    hierarchical_roles, nested_hierarchical_roles,
)
from .SDE import VESDE, VPSDE
from .churn import (
    DEFAULT_ETA as DEFAULT_CHURN_ETA, GAMMA_CAP as DEFAULT_CHURN_GAMMA_CAP,
)
from .Sampler import Sampler
from .Trainer import Trainer
from .MultiObsSampler import MultiObsSampler
from .PFODE import PFODE

#################################################################################################
# ///////////////////////////////////// Diffusion Model /////////////////////////////////////////
#################################################################################################

# Bumped whenever a checkpoint's architecture payload changes shape. Legacy
# checkpoints predate the field entirely and are recognised by its absence.
# Version 2 adds the latent-attention rule; it is written only by models that use
# one, so a version-1 reader refuses them instead of silently dropping the rule,
# and every other checkpoint keeps writing 1.
ARCH_VERSION = 2
ARCHITECTURES = ("legacy", "tokenized", "grouped")


def _resolve_architecture(nodes_size, theta_size, x_spec, arch, schema):
    """Decide the architecture and node schema from the constructor signature.

    ``nodes_size=`` alone keeps meaning the legacy per-node network, exactly as
    before. ``theta_size=`` / ``x_spec=`` select the schema-driven network. A
    legacy model must never enter the new implementation by accident, so the
    combinations are checked rather than coerced.
    """
    if schema is not None:
        if not isinstance(schema, NodeSchema):
            schema = NodeSchema.from_dict(schema)
        arch = arch or ("grouped" if schema.has_group else "tokenized")
    elif theta_size is not None or x_spec is not None:
        if theta_size is None or x_spec is None:
            raise ValueError(
                "The schema signature needs both theta_size= and x_spec=; got "
                f"theta_size={theta_size!r}, x_spec={x_spec!r}."
            )
        if nodes_size is not None:
            raise ValueError(
                "Pass either nodes_size= (legacy) or theta_size=/x_spec= "
                "(schema), not both."
            )
        if arch == "legacy":
            raise ValueError(
                "arch='legacy' is not available for the schema signature; the "
                "legacy network is built from nodes_size= alone."
            )
        schema = NodeSchema.from_spec(theta_size, x_spec)
        arch = arch or ("grouped" if schema.has_group else "tokenized")
    elif nodes_size is not None:
        schema = NodeSchema.legacy(nodes_size)
        arch = arch or "legacy"
    else:
        raise ValueError(
            "ScoreBasedInferenceModel needs either nodes_size= or "
            "theta_size=/x_spec=."
        )
    if arch not in ARCHITECTURES:
        raise ValueError(
            f"Unknown arch {arch!r}; choose one of {', '.join(ARCHITECTURES)}."
        )
    if arch == "legacy" and not schema.is_atomic:
        raise ValueError(
            "arch='legacy' is one token per scalar; this schema groups "
            f"{schema.nodes_size} nodes into {schema.token_count} tokens."
        )
    # A nested schema selects the two-level trunk, and nothing else may. The
    # rule runs both ways on purpose: a Group schema in the flat trunk would
    # flatten every repeat into one context, silently discarding the nesting the
    # block exists to express, and a flat schema in the grouped trunk has no
    # repeats to iterate. Neither fails loudly on its own, so both are refused
    # here rather than discovered in a training curve.
    if schema.has_group and arch != "grouped":
        raise ValueError(
            f"A schema with a Group block is arch='grouped', not {arch!r}. "
            "Drop the explicit arch= and it is selected automatically."
        )
    if arch == "grouped" and not schema.has_group:
        raise ValueError(
            "arch='grouped' is the two-level trunk and needs a Group block; "
            f"this schema has none ({[b.kind for b in schema.blocks]})."
        )
    return arch, schema


class ScoreBasedInferenceModel(nn.Module):
    def __init__(
            self,
            nodes_size=None,    # Number of features in joint (theta, x) -- legacy signature
            sde_type="vesde",   # Type of Stochastic Differential Equation (SDE)
            sigma=25.0,         # Variance for VESDE
            beta_min=0.1,       # Minimum beta for VPSDE
            beta_max=20.0,      # Maximum beta for VPSDE
            hidden_size=128,    # Hidden size of the transformer
            depth=6,            # Number of transformer blocks
            num_heads=16,       # Number of attention heads
            mlp_ratio=4,        # Ratio of MLP hidden size to embedding size
            device="cpu",       # Device to initialize the model on
            theta_size=None,    # Schema signature: number of parameter dimensions
            x_spec=None,        # Schema signature: observation blocks (Atomic / IID / Group)
            arch=None,          # "legacy", "tokenized" or "grouped"; per signature
            trunk="shared",     # Tokenized trunk: "shared" or "position"
            token_offset=False, # Tokenized ablation: per-token modulation offset
            time_embedding_size=256,  # Width of the diffusion-time embedding
            per_position_head=False,  # Per-scalar readout for atomic blocks
            schema=None,        # Prebuilt NodeSchema (used by load)
            depth_within=None,  # Grouped trunk: blocks inside one repeat
            depth_across=None,  # Grouped trunk: blocks over globals + repeats
            depth_readout=None, # Grouped trunk: blocks after the broadcast back
            time_embedding="raw",  # "raw" t or "log_lambda" (see _install_time_map)
            latent_attention="none",  # "none", "global", "hierarchical" or "full"
            latent_roles=None,  # (global_size, local_size) or {"g","l","x": indices}
            parameterization="eps",  # "eps" or "edm" (experimental, compass.parameterization)
            parameterization_sigma_data=1.0,
            ):

        super(ScoreBasedInferenceModel, self).__init__()

        arch, schema = _resolve_architecture(
            nodes_size, theta_size, x_spec, arch, schema,
        )
        self.arch = arch
        self.schema = schema
        self.trunk = trunk if arch == "tokenized" else None
        self.token_offset = bool(token_offset) if arch == "tokenized" else False
        # The diffusion-time embedding is a function of t alone, but it sets the
        # input width of every adaLN modulation, so at small hidden_size it is
        # the single largest block in the budget (131,584 of 463,364 parameters
        # for the trial-h64/d2 arm). Both schema trunks expose it; only the
        # legacy network is pinned at 256, so its checkpoints stay loadable.
        self.time_embedding_size = (
            256 if arch == "legacy" else int(time_embedding_size)
        )
        self.per_position_head = (
            False if arch == "legacy" else bool(per_position_head)
        )
        # The two-level trunk has three depths where the flat one has a single
        # ``depth``. Both level depths fall back to ``depth`` so a config that
        # already says depth=2 gets the recommended 2+2+0 without needing a new
        # field, and the readout stack defaults to empty because its blocks are
        # extra capacity rather than part of the routing.
        self.depth_within = depth if depth_within is None else int(depth_within)
        self.depth_across = depth if depth_across is None else int(depth_across)
        self.depth_readout = 0 if depth_readout is None else int(depth_readout)
        self.nodes_size = schema.nodes_size
        self.token_count = schema.token_count
        self.sde_type = sde_type
        self.sigma = sigma
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Init SDE
        self.sigma = sigma
        if sde_type == "vesde":
            self.sde = VESDE(sigma=self.sigma)
        elif sde_type == "vpsde":
            self.sde = VPSDE(beta_min=beta_min, beta_max=beta_max)
        else:
            raise ValueError("Invalid SDE type")
        
        # Init Model
        if self.arch == "legacy":
            self.model = ConditionTransformer(
                nodes_size=self.nodes_size, hidden_size=hidden_size,
                depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            )
        elif self.arch == "grouped":
            self.model = GroupedTransformer(
                schema=self.schema, hidden_size=hidden_size,
                time_embedding_size=self.time_embedding_size,
                depth_within=self.depth_within, depth_across=self.depth_across,
                depth_readout=self.depth_readout, num_heads=num_heads,
                mlp_ratio=mlp_ratio, per_position_head=self.per_position_head,
            )
        else:
            self.model = TokenizedTransformer(
                schema=self.schema, hidden_size=hidden_size, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, trunk=self.trunk,
                token_offset=self.token_offset,
                time_embedding_size=self.time_embedding_size,
                per_position_head=self.per_position_head,
            )
        self.model.to(device)
        self._install_time_map(time_embedding)
        from .parameterization import install as _install_parameterization
        _install_parameterization(self, parameterization, parameterization_sigma_data)
        self.latent_attention = "none"
        if latent_attention not in (None, "none"):
            # Latent-to-latent attention (compass.latent_attention): parameter-free,
            # so the state dict is the plain tokenized one; the rule lives in the
            # architecture payload.
            if self.arch != "tokenized" or self.trunk != "shared":
                raise ValueError(
                    "latent_attention needs arch='tokenized' with trunk='shared'; "
                    f"got arch={self.arch!r}, trunk={self.trunk!r}."
                )
            if latent_roles is None:
                raise ValueError(
                    "latent_attention needs latent_roles=(global_size, local_size) "
                    "or a {'g': ..., 'l': ..., 'x': ...} map of node indices."
                )
            from .latent_attention import enable_latent_attention
            enable_latent_attention(self, latent_attention, latent_roles)

        # Init Trainer
        self.trainer = Trainer(self)

        # Init Sampler
        self.sampler = Sampler(self)
        self.multi_obs_sampler = MultiObsSampler(self)

        # Init probability-flow ODE engine (log-probability evaluation, MAP)
        self.pfode = PFODE(self)
        
    #############################################
    # ----- Forward Diffusion -----
    #############################################

    def forward_diffusion_sample(self, x_0, t, x_1=None, condition_mask=None):
        # Diffusion process for time t with defined SDE
        
        if condition_mask is None:
            condition_mask = torch.zeros_like(x_0)

        if x_1 is None:
            x_1 = torch.randn_like(x_0)*(1-condition_mask)+(condition_mask)*x_0

        # Perturbation kernel p_0t(x_t|x_0) = N(alpha_t x_0, sigma_t^2) on the
        # latent dims; conditioned dims stay clean. VESDE: alpha_t = 1.
        std = self.sde.marginal_prob_std(t).reshape(-1, 1).to(x_0.device)
        alpha = self.sde.alpha_t(t).reshape(-1, 1).to(x_0.device)
        x_t = x_0 * (alpha * (1-condition_mask) + condition_mask) + std * x_1 * (1-condition_mask)
        return x_t
    
    #############################################
    # ----- Output Scaling -----
    #############################################

    def output_scale_function(self, t, x):
        scale = self.sde.marginal_prob_std(t).to(x.device)
        return x / scale
    
    #############################################
    # ----- Training -----
    #############################################
    
    def train(self, theta, x, theta_val=None, x_val=None,
                batch_size=128, max_epochs=500, lr=1e-3, device="cpu",
                verbose=True, path=None, name="Model", early_stopping_patience=100,
                time_sampling="mixture", mask_loss_weights=None,
                mask_sampler=None, mask_menu="auto",
                mask_roles=None, mask_weights=None,
                mask_bernoulli_fraction=0.0, mask_bernoulli_probability=0.33,
                global_target_fn=None, global_target_indices=None, tsm_lambda_c=1.0,
                tsm_variance_fn=None,
                time_loss_weight="auto", sigma_data=1.0,
                validation_noise_seed=20250719, restore="final", time_eps=1e-3,
                fisher_consistency=None, covariance_objective=None):
        """
        Train the model on the provided data

        Args:
            theta: Training Parameters 
            x: Training observations
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            lr: Learning rate
            device: Device to run training on
                    if "cuda", training will be distributed across all available GPUs
            theta_val: Validation Parameters
            x_val: Validation observations
            verbose: Whether to show training progress
            path: Path to save model
            early_stopping_patience: Number of epochs to wait before early stopping
                    (default 100; 20 stopped measurably too early, see Trainer.train)
            validation_noise_seed: Fixed seed for the validation pass's diffusion
                    times and noise (default); ``None`` is the historical
                    unseeded draw, for reproducing older runs. See Trainer.train.
            time_eps: Smallest diffusion time trained on (default 1e-3, i.e. lambda 0.032
                    at sigma 15-25). Lower it together with the sampling ``eps``.
            fisher_consistency: Optional ``compass.consistency.FisherConsistency``
                    penalty (experimental); see Trainer.train.
            covariance_objective: Optional ``compass.covariance_output.CovarianceObjective``
                    (second-order DSM for the network's covariance output, opt-in); added to
                    the loss on training passes only.
            restore: ``"final"`` (default) keeps the last epoch's averaged weights,
                    ``"best"`` reloads the best-validation checkpoint at the end.
            time_sampling: Diffusion-time sampling scheme during training
                    ("mixture" (default), "uniform" or "log_sigma"); see Trainer.train
            mask_loss_weights: Optional list of (mask, weight) loss-balancing
                    pairs. Masks are still sampled by the ordinary Bernoulli
                    sampler; this only reweights examples after drawing them.
            mask_sampler: Optional object with ``sample(shape)`` returning a
                    condition mask, replacing the default Bernoulli(0.33) draw.
                    ``compass.ConditionalMenu`` restricts training to a named
                    menu of conditionals (see ``compass.SIX_CONDITIONALS``).
                    Supplying this overrides ``mask_menu``.
            mask_menu: Which condition-mask distribution to train under.
                    ``"auto"`` (default) follows ``mask_roles``: none given is
                    ``"bernoulli"`` (the historical default, unchanged), an int
                    is ``"inference"`` and a tuple/dict is ``"six"``.
                    "inference" (needs mask_roles=<number of theta dims>): 1/3
                    posterior mask, 1/3 likelihood mask, 1/3 Bernoulli -- the
                    directions model comparison uses; see _resolve_mask_sampler.

                    ``"bernoulli"`` draws every node's conditioning
                    flag independently at ``mask_bernoulli_probability``, which
                    spreads the signal over all ``2**tokens`` conditionals.

                    ``"six"`` trains on the six non-degenerate conditionals over
                    a (global, local, observation) partition -- ``p(g,l|x)``,
                    ``p(g|l,x)``, ``p(l|g,x)``, ``p(x|g,l)``, ``p(x,l|g)``,
                    ``p(x,g|l)`` -- **drawn uniformly**, so each is seen 1/6 =
                    16.7% of the time and the deployed posterior ``p(g,l|x)``
                    goes from never-drawn to one draw in six. Requires
                    ``mask_roles``.

                    Caveat, measured rather than assumed: under a uniform menu
                    the three conditionals with the observations *latent* carry
                    5-7x the loss of the three with them observed, so roughly
                    four fifths of the gradient goes to generating ``x``. If the
                    posterior conditionals are what matter, pass ``mask_weights``
                    to upweight them, or mix a Bernoulli fraction back in.
            mask_roles: ``(global_size, local_size)`` for the ``"six"`` menu, i.e.
                    the node vector is ``[g (global_size), l (local_size), x]``.
                    A ``dict`` of explicit ``{role: node indices}`` is also
                    accepted for layouts that are not in that order.
            mask_weights: Optional relative probabilities of the six patterns.
                    ``None`` (the default) is the uniform draw described above.
            mask_bernoulli_fraction: Probability that a ``"six"`` draw falls back
                    to the Bernoulli sampler, so the general conditional
                    capability is not lost. ``0.0`` (default) is menu-only.
            mask_bernoulli_probability: Rate for the Bernoulli draws.
            time_loss_weight: "auto" (default), None or "edm". "edm" weights the
                    epsilon loss by (1 + sigma^2 / sigma_data^2), which puts training
                    signal on high noise levels (see Trainer.train). "auto" is "edm"
                    whenever training uses a mask menu (a ``mask_menu`` other than
                    Bernoulli, or an explicit ``mask_sampler``) and None otherwise:
                    the two were measured as a pair. On partial pooling (six-mixed
                    menu, 3 seeds x 30 datasets) EDM cut global RMSE 0.666 -> 0.434
                    in every seed and raised mean r 0.871 -> 0.903; on likelihoods
                    (inference menu) it cut PF-ODE log p error 32-37%. EDM alone,
                    under the Bernoulli draw, cost the posterior a little, so the
                    Bernoulli default keeps the plain epsilon loss.
            sigma_data: Data scale for time_loss_weight="edm".
            global_target_fn, global_target_indices, tsm_lambda_c: optional target
                    score matching on the globals of a hierarchical model; see
                    Trainer.train. ``global_target_fn`` maps clean node vectors to
                    grad_g [log p(g) + log p(l | g)] in network coordinates.
        """
        if mask_menu == "auto":
            mask_menu = ("bernoulli" if mask_roles is None or mask_sampler is not None
                         else "six" if isinstance(mask_roles, (tuple, list, dict)) else "inference")
        mask_sampler = self._resolve_mask_sampler(
            mask_sampler, mask_menu, mask_roles, mask_weights,
            mask_bernoulli_fraction, mask_bernoulli_probability,
        )
        if time_loss_weight == "auto":
            time_loss_weight = "edm" if mask_sampler is not None else None

        # Combine theta and x into a single tensor
        # train_data: (num_samples, node_size)
        train_data = torch.cat([theta, x], dim=1)
        val_data = None
        if theta_val is not None and x_val is not None:
            val_data = torch.cat([theta_val, x_val], dim=1)

        if device == "cuda":
            world_size = torch.cuda.device_count()
        else :
            world_size = 1

        self.trainer.train(world_size=world_size, train_data=train_data, val_data=val_data,
                            max_epochs=max_epochs, early_stopping_patience=early_stopping_patience, batch_size=batch_size, lr=lr,
                            path=path, name=name, device=device, verbose=verbose,
                            time_sampling=time_sampling,
                            mask_loss_weights=mask_loss_weights,
                            mask_sampler=mask_sampler,
                            global_target_fn=global_target_fn,
                            global_target_indices=global_target_indices,
                            tsm_lambda_c=tsm_lambda_c, tsm_variance_fn=tsm_variance_fn,
                            time_loss_weight=time_loss_weight, sigma_data=sigma_data,
                            validation_noise_seed=validation_noise_seed, restore=restore,
                            time_eps=time_eps, fisher_consistency=fisher_consistency,
                            covariance_objective=covariance_objective)


    def _resolve_mask_sampler(self, mask_sampler, mask_menu, mask_roles,
                              mask_weights, bernoulli_fraction,
                              bernoulli_probability):
        """Turn the ``mask_menu`` API call into a concrete mask sampler."""
        if mask_sampler is not None:
            if mask_menu not in (None, "bernoulli"):
                raise ValueError(
                    "Pass either mask_sampler or mask_menu, not both; got "
                    f"mask_menu={mask_menu!r} alongside an explicit sampler."
                )
            return mask_sampler
        if mask_menu in (None, "bernoulli"):
            # None means "the Trainer's own default", which is the Bernoulli draw.
            return None
        if mask_menu == "inference":
            # The two directions model comparison uses: the posterior p(theta | x)
            # and the likelihood p(x | theta), 1/3 each, plus 1/3 Bernoulli draws so
            # general conditioning is kept. Under Bernoulli(0.33) alone these exact
            # masks are drawn only a few percent of the time. With time_loss_weight=
            # "edm" this cut the PF-ODE log-likelihood error by 32-37% on non-Gaussian
            # likelihoods without hurting the posterior
            # (validation/likelihood_improvement).
            if mask_roles is None or isinstance(mask_roles, (tuple, list, dict)):
                raise ValueError(
                    "mask_menu='inference' needs mask_roles=<number of theta dims>: "
                    "the leading nodes are the parameters, the rest the observations."
                )
            theta_size = int(mask_roles)
            if not 0 < theta_size < self.nodes_size:
                raise ValueError(f"mask_roles={theta_size} must be in (0, {self.nodes_size}).")
            frac = 1 / 3 if bernoulli_fraction in (None, 0.0) else bernoulli_fraction
            return ConditionalMenu(
                self.schema,
                {"t": list(range(theta_size)), "x": list(range(theta_size, self.nodes_size))},
                [("p(theta|x)", "x"), ("p(x|theta)", "t")], weights=mask_weights,
                bernoulli_fraction=frac, bernoulli_probability=bernoulli_probability,
            )
        if mask_menu != "six":
            raise ValueError(
                f"Unknown mask_menu {mask_menu!r}; choose 'bernoulli', 'six' or "
                "'inference', or pass an explicit mask_sampler."
            )
        if mask_roles is None:
            raise ValueError(
                "mask_menu='six' needs mask_roles=(global_size, local_size) to "
                "say which nodes are group-level, subject-level and observed. "
                "The six conditionals are defined over that partition."
            )
        if isinstance(mask_roles, dict):
            roles = mask_roles
        else:
            global_size, local_size = mask_roles
            roles = (
                nested_hierarchical_roles(self.schema, global_size, local_size)
                if self.schema.has_group
                else hierarchical_roles(self.schema, global_size, local_size)
            )
        return ConditionalMenu(
            self.schema, roles, SIX_CONDITIONALS, weights=mask_weights,
            bernoulli_fraction=bernoulli_fraction,
            bernoulli_probability=bernoulli_probability,
        )

    #############################################
    # ----- Sample -----
    #############################################
        
    def sample(self, theta=None, x=None, err=None, condition_mask=None,
               timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None, multi_obs_inference=False, hierarchy=None,
               prior=None, local_prior=None, correction="gauss_jacobian_FD", posterior_precision=None,
               posterior_covariance=None, posterior_mean=None,
               global_posterior_mean=None, global_posterior_covariance=None,
               precision_est_samples=500, precision_est_timesteps=None, denoise_clamp=5.0,
               damping_at_data=1.0, damping_at_noise=None,
               composition_temperature=1.0,
               composition_batch_size=None,
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=None, final_corrector_steps=3, terminal_corrector_steps=0,
               churn_eta=DEFAULT_CHURN_ETA, churn_s_noise=1.0,
               churn_gamma_cap=DEFAULT_CHURN_GAMMA_CAP,
               adaptive_abs_tol=0.002576, adaptive_rel_tol=0.1,
               adaptive_safety=0.9, adaptive_exponent=0.9,
               adaptive_max_evals=10000, adaptive_initial_step=None,
               device="cpu", verbose=True, method="dpm", save_trajectory=False,
               capture_attention=True, terminal_denoise=False,
               precision_est_batch_size=128,
               covariance_shrinkage=0.01, covariance_nugget=1e-6,
               pd_epsilon=1e-8, pd_floor=1e-3,
               jacobian_refresh=9, jacobian_curvature_floor=1e-3,
               jacobian_fd_step=1e-2, jacobian_refresh_low_noise=None,
               jacobian_refresh_lambda=0.3):
        """
        Sample from the model using the specified method

        Args:
            data: Input data : Tensor of shape (num_samples, num_observed_features)
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (num_samples, num_total_features)
            timesteps: Number of diffusion steps
            eps: End time for diffusion process. Forwarded to both samplers; it
                    used to be dropped, so every sample stopped at the sampler
                    default 1e-3 whatever was passed (benchmarks/SimFormer: that
                    residual noise was the whole C2ST gap to Simformer).
            num_samples: Number of samples to generate
            cfg_alpha: Classifier-free guidance strength

            - Multi-observation (compositional score modeling) parameters -
            multi_obs_inference: Treat the rows of x as i.i.d. observations of the same
                    underlying parameters and sample the joint posterior p(theta | x_1..x_n)
            hierarchy: Indices of the shared (global) parameters composed across
                    observations (defaults to all latent variables)
            prior: Gaussian prior over the hierarchy dimensions as a tuple (mean, std),
                    each of length len(hierarchy). Defaults to N(0, 1).
            correction: Score composition rule for multi_obs_inference; default
                    "gauss_jacobian_FD" with jacobian_refresh=9 (the standard COMPASS
                    method). Existing values are preserved;
                    benchmark names also include "legacy_mean",
                    "prior_corrected_sum", "damped_sum" and "minibatch_damped".
                    "gauss_jacobian" is the pilot-free hierarchical GAUSS rule:
                    it takes every single-observation backward covariance from
                    the score network's own Jacobian (Tweedie second order)
                    instead of from an estimated pilot covariance, so it needs
                    no precision-estimation run, makes no Gaussian assumption
                    on the single-observation posteriors, and reduces exactly to
                    the pilot form when they happen to be Gaussian.
                    "gauss_jacobian_FD" is the same rule with the Jacobian taken
                    by one batched finite-difference network call instead of
                    forward-mode AD (~4x cheaper, same posterior within
                    Monte-Carlo noise).
            jacobian_refresh: Composed-score evaluations between Jacobian
                    refreshes for correction="gauss_jacobian" (curvature varies
                    slowly in t, so 5-20 is cheap and accurate).
            jacobian_curvature_floor: Lower bound on the eigenvalues of
                    I - lambda^2 H_j for correction="gauss_jacobian".
            jacobian_fd_step, jacobian_refresh_low_noise,
            jacobian_refresh_lambda: Finite-difference step (relative to
                    lambda_t) for "gauss_jacobian_FD", and an optional separate
                    refresh period below noise level jacobian_refresh_lambda
                    for both Jacobian corrections; see MultiObsSampler.sample.
            posterior_precision: Optional precision estimate of the single-observation
                    posteriors on the hierarchy dimensions (for correction="gauss");
                    estimated automatically if not provided
            posterior_covariance: Optional full covariance estimate with shape
                    H x H for ``full_gaussian`` or (H + L) x (H + L) for the
                    Schur modes, with one matrix or one per observation; estimated
                    automatically if omitted.
            posterior_mean: Optional single-observation posterior means paired
                    with posterior_covariance. Supplying them enables Gaussian
                    moment projection before covariance-aware composition.
            precision_est_samples: Samples per observation for the automatic estimate
            precision_est_timesteps: Diffusion steps for the automatic estimate
            precision_est_batch_size: Maximum flattened transformer rows per
                    automatic precision-estimation batch
            damping_at_data: Damping endpoint d(0) at the data end
            damping_at_noise: Damping endpoint d(1) at the noise end; defaults
                    to 1/sqrt(number of observations)
            composition_temperature: Likelihood-ratio temperature for the
                    shared compositional score; 1.0 preserves the original.
            composition_batch_size: Optional observation mini-batch size for
                    the unbiased "damping" or "minibatch_damped" estimator.
                    True mini-batching is restricted to all-global latent models.

            - DPM-Solver parameters -
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration. ``None``
                    (the default) means the method's own default: 0 under
                    "dpm", where the churned predictor replaces them, and 5
                    under "langevin", which is corrector-only by definition.
                    Asking for both churn and correctors is an error.
            final_corrector_steps: Extra correction steps at the end
            churn_eta: Karras churn strength for the DPM predictor (default
                    2.0). Each level steps up in noise and then takes the
                    deterministic step down, at no extra score evaluations.
                    0.0 recovers the stock probability-flow predictor.
                    See ``compass.churn`` for the measurements behind the default.
            churn_s_noise: Scale on the injected churn noise (Karras S_noise).
            churn_gamma_cap: Per-step noise-up cap; the schedule is geometric,
                    so this clamp is what makes churn_eta saturate above
                    ~gamma_cap/(1-r). Raise it to sweep past that.

            - Adaptive reverse-SDE parameters -
            adaptive_abs_tol: Absolute local-error tolerance
            adaptive_rel_tol: Relative local-error tolerance
            adaptive_safety: Step-size safety multiplier
            adaptive_exponent: Error-controller exponent
            adaptive_max_evals: Maximum score evaluations
            adaptive_initial_step: Optional initial diffusion-time step

            - Other parameters -
            device: Device to run sampling on
            verbose: Whether to show progress bar
            method: Sampling method to use (euler, dpm, langevin, adaptive)
            save_trajectory: Whether to save the intermediate denoising trajectory
            capture_attention: Retain midpoint attention weights during standard
                    single-observation sampling. Defaults to True for compatibility.
            terminal_denoise: For standard single-observation sampling, apply
                    one deterministic probability-flow step from ``eps`` to
                    zero noise. Defaults to False for backward compatibility.
        """

        # Combine data and create condition mask
        # 0 for latent values, 1 for observed values
        data = None
        # The convenience masks below assume the node vector splits contiguously
        # into [theta, x]. A nested schema does not: each repeat carries its own
        # latent locals *inside* the observed block, so
        # [zeros(nodes - width), ones(width)] would condition the wrong nodes.
        # Refuse rather than build a plausible-looking wrong mask.
        if self.schema.has_group and condition_mask is None and (
                theta is not None or x is not None):
            raise ValueError(
                "A grouped schema interleaves latent per-repeat parameters with "
                "the observations, so there is no contiguous [theta, x] split to "
                "infer a condition_mask from. Pass the full "
                f"{self.nodes_size}-wide node vector as x= together with an "
                "explicit condition_mask=."
            )
        if theta is None and x is not None:
            # If only x is provided, use it as the data
            data = x
            if condition_mask is None:
                condition_mask = torch.cat([torch.zeros(self.nodes_size-data.shape[-1]),torch.ones(data.shape[-1])])
        if x is None and theta is not None:
            # If only theta is provided, use it as the data
            data = theta
            if condition_mask is None:
                condition_mask = torch.cat([torch.ones(data.shape[-1]),torch.zeros(self.nodes_size-data.shape[-1])])
        if theta is not None and x is not None:
            # Both theta and x are provided, raise an error
            raise ValueError("Both theta and x are provided!\nPlease provide theta to sample from the Likelihood OR x to sample from the Posterior.")
        if data is None:
            # If neither theta nor x is provided, infer both
            condition_mask = torch.zeros(self.nodes_size)
            data = torch.zeros(1, self.nodes_size)
            

        # Choose device        
        if device == "cuda":
            # Run sampling on all available GPUs
            world_size = torch.cuda.device_count()
        else:
            # Run sampling on specified device
            world_size = 1
            
        if multi_obs_inference == False:
            samples = self.sampler.sample(world_size=world_size, data=data, err=err, condition_mask=condition_mask, timesteps=timesteps, eps=eps, num_samples=num_samples, device=device, cfg_alpha=cfg_alpha,
                                    order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                    churn_eta=churn_eta, churn_s_noise=churn_s_noise, churn_gamma_cap=churn_gamma_cap,
                                    verbose=verbose, method=method, save_trajectory=save_trajectory,
                                    capture_attention=capture_attention,
                                    terminal_denoise=terminal_denoise)
            
        elif multi_obs_inference == True:
            # Hierarchical Compositional Score Modeling
            samples = self.multi_obs_sampler.sample(world_size=world_size, data=data, condition_mask=condition_mask, timesteps=timesteps, eps=eps, num_samples=num_samples, device=device, cfg_alpha=cfg_alpha, hierarchy=hierarchy,
                                      prior=prior, local_prior=local_prior,
                                      correction=correction,
                                      posterior_precision=posterior_precision,
                                      posterior_covariance=posterior_covariance,
                                      posterior_mean=posterior_mean,
                                      global_posterior_mean=global_posterior_mean,
                                      global_posterior_covariance=global_posterior_covariance,
                                      precision_est_samples=precision_est_samples, precision_est_timesteps=precision_est_timesteps,
                                      precision_est_batch_size=precision_est_batch_size,
                                      denoise_clamp=denoise_clamp,
                                      damping_at_data=damping_at_data,
                                      damping_at_noise=damping_at_noise,
                                      composition_temperature=composition_temperature,
                                      composition_batch_size=composition_batch_size,
                                      order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                      churn_eta=churn_eta, churn_s_noise=churn_s_noise, churn_gamma_cap=churn_gamma_cap,
                                      terminal_corrector_steps=terminal_corrector_steps,
                                      covariance_shrinkage=covariance_shrinkage,
                                      covariance_nugget=covariance_nugget,
                                      pd_epsilon=pd_epsilon,
                                      pd_floor=pd_floor,
                                      jacobian_refresh=jacobian_refresh,
                                      jacobian_curvature_floor=jacobian_curvature_floor,
                                      jacobian_fd_step=jacobian_fd_step,
                                      jacobian_refresh_low_noise=jacobian_refresh_low_noise,
                                      jacobian_refresh_lambda=jacobian_refresh_lambda,
                                      adaptive_abs_tol=adaptive_abs_tol,
                                      adaptive_rel_tol=adaptive_rel_tol,
                                      adaptive_safety=adaptive_safety,
                                      adaptive_exponent=adaptive_exponent,
                                      adaptive_max_evals=adaptive_max_evals,
                                      adaptive_initial_step=adaptive_initial_step,
                                      verbose=verbose, method=method, save_trajectory=save_trajectory)

        # just return the sampled values
        samples = samples[:,:,(1-condition_mask).bool()] 
        
        return samples
    
    #############################################
    # ----- Log-probability (PF-ODE) -----
    #############################################

    def log_prob(self, data, condition_mask, timesteps=100, eps=1e-3,
                 divergence="exact", hutchinson_samples=32,
                 device="cpu", batch_size=4096, verbose=False, differentiable=False,
                 endpoint="isotropic"):
        """
        Evaluate the log-probability of the latent dimensions of `data` given its
        conditioned dimensions with the probability-flow ODE (no KDE involved).

        With the parameters theta conditioned (condition_mask 1 on theta, 0 on x)
        this returns the model likelihood log p(x | theta); with the observations
        conditioned it returns the posterior log p(theta | x).

        Args:
            data:           Joint (theta, x) node vectors, shape (num_points, nodes_size).
                            Conditioned dims hold the conditioning values, latent dims
                            the point the density is evaluated at.
            condition_mask: Binary mask (1 = conditioned, 0 = latent),
                            shape (nodes_size,) or (num_points, nodes_size).
            timesteps:      Integration nodes of the 2nd-order Heun solver.
            eps:            Diffusion end time (density is smoothed by sigma(eps)).
            divergence:     "exact" (default) or "hutchinson".
            hutchinson_samples: Probe vectors if divergence="hutchinson".
            device:         Device to run on.
            batch_size:     Points per integration batch.
            verbose:        Show a progress bar.
            differentiable: Keep the autograd graph so the result can be
                            differentiated w.r.t. ``data`` (used for likelihood
                            maximization, see compass.likelihood_maximization).
            endpoint:       "isotropic" (default), "gaussian" or "fixed" density at
                            t = 1; see PFODE.log_prob.

        Returns:
            Tensor (num_points,) of log-probabilities on CPU (on ``device``
            and attached to the graph if ``differentiable``).
        """
        return self.pfode.log_prob(data=data, condition_mask=condition_mask,
                                   timesteps=timesteps, eps=eps,
                                   divergence=divergence, hutchinson_samples=hutchinson_samples,
                                   device=device, batch_size=batch_size, verbose=verbose,
                                   differentiable=differentiable, endpoint=endpoint)

    def map_estimate(self, data, condition_mask, **kwargs):
        """
        KDE-free MAP estimate of the latent dimensions of `data` given its
        conditioned dimensions via annealed score ascent (see PFODE.map_estimate).
        """
        return self.pfode.map_estimate(data=data, condition_mask=condition_mask, **kwargs)

    def hierarchical_map_estimate(self, data, condition_mask, init=None,
                                  hierarchy=None, prior=None, local_prior=None,
                                  correction="gauss",
                                  posterior_precision=None,
                                  posterior_covariance=None, posterior_mean=None,
                                  denoise_clamp=5.0,
                                  cfg_alpha=None, sigma_start=None,
                                  damping_at_data=1.0,
                                  damping_at_noise=None,
                                  timesteps=100, eps=1e-3,
                                  iterations_per_level=3,
                                  max_iterations_per_level=None,
                                  convergence_tol=1e-6,
                                  jacobian_refresh=1,
                                  jacobian_curvature_floor=1e-3, device="cpu"):
        """Refine shared and local parameters with a compositional joint score.

        Unlike :meth:`map_estimate`, this method treats rows as one hierarchical
        posterior: hierarchy coordinates remain synchronized and receive the
        multi-observation composed score.
        """
        return self.multi_obs_sampler.map_estimate(
            data=data, condition_mask=condition_mask, init=init,
            hierarchy=hierarchy, prior=prior, local_prior=local_prior,
            correction=correction,
            posterior_precision=posterior_precision,
            posterior_covariance=posterior_covariance,
            posterior_mean=posterior_mean,
            denoise_clamp=denoise_clamp, cfg_alpha=cfg_alpha,
            damping_at_data=damping_at_data,
            damping_at_noise=damping_at_noise,
            sigma_start=sigma_start, timesteps=timesteps, eps=eps,
            iterations_per_level=iterations_per_level,
            max_iterations_per_level=max_iterations_per_level,
            convergence_tol=convergence_tol,
            jacobian_refresh=jacobian_refresh,
            jacobian_curvature_floor=jacobian_curvature_floor, device=device,
        )

    #############################################
    # ----- Composition diagnostics -----
    #############################################

    def certify_composition(self, samples, x, condition_mask=None, **kwargs):
        """Score composed samples against the exact tall-data target.

        The multi-observation posterior factorizes as
        ``p(theta)^(1-n) prod_j p(theta | x_j)``, and every factor is a
        single-observation conditional density the PF-ODE evaluates exactly.
        This gives the unnormalized target at every returned sample, plus a
        self-normalized effective sample size and optional reweighted draws --
        the only measurement in COMPASS that judges a composition rule without
        already knowing the right answer. See
        :meth:`MultiObsSampler.certify_composition` for the full contract.
        """
        if condition_mask is None:
            condition_mask = torch.cat([
                torch.zeros(self.nodes_size - x.shape[-1]),
                torch.ones(x.shape[-1]),
            ])
        return self.multi_obs_sampler.certify_composition(
            samples=samples, data=x, condition_mask=condition_mask, **kwargs
        )

    def composition_curl(self, samples, x, condition_mask=None, **kwargs):
        """Relative antisymmetry of the composed score field's Jacobian.

        A genuine score is a gradient, so a conservative composition has a
        symmetric Jacobian. Large antisymmetry means Langevin correctors -- whose
        invariant distribution assumes a gradient field -- will circulate
        samples rather than refine them. Requires a configured sampler: call
        after :meth:`sample` with the same settings.
        """
        if condition_mask is None:
            condition_mask = torch.cat([
                torch.zeros(self.nodes_size - x.shape[-1]),
                torch.ones(x.shape[-1]),
            ])
        return self.multi_obs_sampler.composition_curl(
            samples=samples, data=x, condition_mask=condition_mask, **kwargs
        )

    #############################################
    # ----- Save & Load -----
    #############################################

    def architecture_payload(self):
        """The fields a checkpoint needs to rebuild this network exactly.

        Legacy checkpoints carry no ``arch_version``; writing one for a legacy
        model would make old and new checkpoints indistinguishable by absence,
        so the legacy payload is left byte-identical to what it always was.
        """
        if self.arch == "legacy":
            return {}
        mode = getattr(self.model, "latent_attention", "none")
        payload = {
            "arch_version": 1 if mode == "none" else 2,
            "arch": self.arch,
            "schema": self.schema.to_dict(),
            "trunk": self.trunk,
            "token_offset": self.token_offset,
        }
        # Written only when they differ from the historical defaults, so every
        # checkpoint trained before these options existed still round-trips
        # byte-for-byte and keeps arch_version 1.
        if self.time_embedding_size != 256:
            payload["time_embedding_size"] = self.time_embedding_size
        if self.per_position_head:
            payload["per_position_head"] = True
        if self.arch == "grouped":
            # Always written for a grouped model: unlike the options above there
            # is no historical default to fall back on, and a wrong depth would
            # load silently into a differently shaped state dict.
            payload["depth_within"] = self.depth_within
            payload["depth_across"] = self.depth_across
            payload["depth_readout"] = self.depth_readout
        if mode != "none":
            payload["latent_attention"] = mode
            payload["latent_roles"] = self.model.latent_roles
        return payload

    #############################################
    # ----- Diffusion-time input map -----
    #############################################

    TIME_EMBEDDINGS = ("raw", "log_lambda")

    def _install_time_map(self, time_embedding):
        """Optionally feed the network log-lambda instead of the raw diffusion time.

        The sinusoidal ``TimestepEmbedder`` (DiT-style) has frequencies <= 1 and was
        designed for integer timesteps up to ~1000. COMPASS passes ``t`` in
        (eps, 1], so every embedding argument is below 1: the whole low-noise band
        lambda < 0.2 sits in t < 0.036 and the top band lambda 5..9.8 in
        t 0.79..1 (VESDE, sigma = 25). ``"log_lambda"`` feeds

            tau = 1000 * (log lambda(t) - log lambda(1e-3)) / (log lambda(1) - log lambda(1e-3))

        so noise levels are spread evenly over the embedder's design range. It is a
        forward pre-hook on the network: no parameters, no state-dict keys and no
        call sites change, and the mapping is stored in the checkpoint
        (validation/likelihood_improvement).
        """
        if time_embedding not in self.TIME_EMBEDDINGS:
            raise ValueError(f"time_embedding must be one of {self.TIME_EMBEDDINGS}, "
                             f"got {time_embedding!r}")
        self.time_embedding = time_embedding
        handle = getattr(self, "_time_map_handle", None)
        if handle is not None:
            handle.remove()
            self._time_map_handle = None
        if time_embedding == "raw":
            return
        sde = self.sde
        one = torch.ones(1)
        log_lo = float(torch.log(sde.lambda_t(1e-3 * one)))
        log_hi = float(torch.log(sde.lambda_t(one)))

        def to_tau(t):
            t = torch.as_tensor(t)
            lam = sde.lambda_t(t.clamp_min(1e-6))
            return 1000.0 * (torch.log(lam) - log_lo) / (log_hi - log_lo)

        def hook(module, args, kwargs):
            if "t" in kwargs:
                kwargs = dict(kwargs)
                kwargs["t"] = to_tau(kwargs["t"]).to(kwargs["t"].dtype)
            elif len(args) >= 2:
                args = list(args)
                args[1] = to_tau(args[1]).to(args[1].dtype)
                args = tuple(args)
            return args, kwargs

        self._time_map_handle = self.model.register_forward_pre_hook(hook, with_kwargs=True)

    def time_payload(self):
        """Optional checkpoint keys (time map, PF-ODE base moments); empty for the
        defaults, so old checkpoints stay byte-identical."""
        out = {} if getattr(self, "time_embedding", "raw") == "raw" else {
            "time_embedding": self.time_embedding}
        if getattr(self, "base_moments", None) is not None:
            out["base_moments"] = [m.detach().cpu() for m in self.base_moments]
        from .parameterization import payload as _parameterization_payload
        out.update(_parameterization_payload(self))
        return out

    def set_base_moments(self, data):
        """Per-node mean and variance of training data (num, nodes_size) for the
        PF-ODE ``endpoint="fixed"`` base density; stored in the checkpoint."""
        data = torch.as_tensor(data, dtype=torch.float32)
        self.base_moments = (data.mean(0), data.var(0))

    def save(self, path, name="Model"):
        state_dict = {
                'model_state_dict' : self.model.state_dict(),
                'nodes_size': self.nodes_size,
                'sde_type': self.sde_type,
                'sigma': self.sigma,
                'beta_min': self.beta_min,
                'beta_max': self.beta_max,
                'hidden_size': self.hidden_size,
                'depth': self.depth,
                'num_heads': self.num_heads,
                'mlp_ratio': self.mlp_ratio,
                **self.architecture_payload(),
                **self.time_payload(),
            }

        torch.save(state_dict, f"{path}/{name}.pt")

    @staticmethod
    def load(path, device=None, arch=None):
        """Rebuild a saved model, dispatching on the checkpoint's architecture.

        A checkpoint written before schemas existed carries no ``arch_version``
        and is loaded as ``arch="legacy"``. Asking for a different architecture
        than the checkpoint records is an error, never a conversion: the
        parameter shapes of the two trunks are incompatible by construction, and
        silently reinterpreting one as the other is the failure mode this
        dispatch exists to prevent.
        """

        if device is None:
            # Automatically detect device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(path, map_location=device)

        stored_arch = (
            checkpoint.get('arch', 'tokenized')
            if 'arch_version' in checkpoint else 'legacy'
        )
        if arch is not None and arch != stored_arch:
            raise RuntimeError(
                f"{path} is an arch={stored_arch!r} checkpoint"
                + (" (it predates arch_version)" if stored_arch == 'legacy' else "")
                + f"; arch={arch!r} was requested. Architectures are not "
                "convertible -- retrain instead."
            )
        version = checkpoint.get('arch_version')
        if version is not None and version > ARCH_VERSION:
            raise RuntimeError(
                f"{path} was written with arch_version {version}; this COMPASS "
                f"supports up to {ARCH_VERSION}."
            )

        model = ScoreBasedInferenceModel(
            nodes_size=checkpoint['nodes_size'] if stored_arch == 'legacy' else None,
            schema=(
                NodeSchema.from_dict(checkpoint['schema'])
                if stored_arch != 'legacy' else None
            ),
            arch=stored_arch,
            trunk=checkpoint.get('trunk', 'shared'),
            token_offset=checkpoint.get('token_offset', False),
            time_embedding_size=checkpoint.get('time_embedding_size', 256),
            per_position_head=checkpoint.get('per_position_head', False),
            # None for every non-grouped checkpoint, which puts the constructor
            # back on its own fallbacks; a grouped checkpoint always carries all
            # three, so a shape mismatch is impossible rather than merely
            # unlikely.
            depth_within=checkpoint.get('depth_within'),
            depth_across=checkpoint.get('depth_across'),
            depth_readout=checkpoint.get('depth_readout'),
            sde_type=checkpoint['sde_type'],
            sigma=checkpoint['sigma'],
            beta_min=checkpoint.get('beta_min', 0.1),
            beta_max=checkpoint.get('beta_max', 20.0),
            hidden_size=checkpoint['hidden_size'],
            depth=checkpoint['depth'],
            num_heads=checkpoint['num_heads'],
            mlp_ratio=checkpoint['mlp_ratio'],
            time_embedding=checkpoint.get('time_embedding', 'raw'),
            latent_attention=checkpoint.get('latent_attention', 'none'),
            latent_roles=checkpoint.get('latent_roles'),
            parameterization=checkpoint.get('parameterization', 'eps'),
            parameterization_sigma_data=checkpoint.get('parameterization_sigma_data', 1.0),
        )

        if 'base_moments' in checkpoint:
            model.base_moments = tuple(torch.as_tensor(m) for m in checkpoint['base_moments'])
        incompatible = model.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        unexpected = set(incompatible.unexpected_keys)
        legacy_divergence = {
            key for key in unexpected if key.startswith("divergence_head.")
        }
        if incompatible.missing_keys or unexpected != legacy_divergence:
            raise RuntimeError(
                "Checkpoint architecture does not match the current score model: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        return model
