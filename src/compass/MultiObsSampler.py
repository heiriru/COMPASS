import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import tqdm
import datetime
import math
import os

from .churn import (
    DEFAULT_ETA as DEFAULT_CHURN_ETA, GAMMA_CAP as DEFAULT_CHURN_GAMMA_CAP,
    apply_churn, resolve_corrector_steps, schedule_ceiling,
)

class TensorTupleDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        assert len(tensor1) == len(tensor2), "Tensors must have the same length"

    def __len__(self):
        return len(self.tensor1)

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx], idx

#################################################################################################
# ///////////////////////////////// Multi-Observation Sampling //////////////////////////////////
#################################################################################################
class MultiObsSampler():
    """
    Compositional score modeling for inference with multiple i.i.d. observations
    (F-NPSE, Geffner et al. 2023, https://arxiv.org/abs/2209.14249).

    The posterior given n observations factorizes over the single-observation posteriors:

        p(theta | x_1, ..., x_n)  ∝  p(theta)^(1-n) * prod_j p(theta | x_j)

    so a score network trained on single (theta, x) pairs can be composed at inference
    time. The `hierarchy` indices select the *shared* (global) parameters that are
    composed across observations; latent dimensions not in `hierarchy` are treated as
    per-observation (local) latents and keep their individual scores.

    Fifteen composition rules are implemented (`correction` argument):

    - "gauss" (default): Gaussian-corrected composition (Gloeckler et al. 2024,
      "Compositional simulation-based inference for time series"). The composed score
      approximates the score of the *diffused* multi-observation posterior, so it is
      valid inside the standard reverse-diffusion samplers (euler / dpm). Requires an
      estimate of the single-observation posterior precision on the hierarchy
      dimensions, which is estimated automatically if not provided.
    - "full_gaussian": the full-covariance GAUSS correction from Algorithm 2 of
      Gloeckler et al. 2024. It estimates a covariance matrix for every
      single-observation posterior and composes the scores with full precision
      matrices and a linear solve. Unlike "gauss", it retains correlations
      between hierarchy coordinates.
    - "uncorrected": (1-n) * diffused-prior score + sum of individual scores.
      Cheap, exact as t -> 0, but biased at large t.
    - "Gauss_schur_global": estimate each observation's joint global/local
      covariance, Schur-reduce it at the current diffusion time, and correct only
      the shared score.
    - "Gauss_global_local": use the same Schur-reduced shared correction and also
      apply its implied global/local cross-score correction to each local latent.
      Optionally accepts ``posterior_mean``/``global_posterior_mean`` to replace
      the learned score by the score of an externally supplied Gaussian (a
      diagnostic "moment projection" ablation, exact only if that Gaussian is
      itself the true single-observation posterior -- it is NOT compositional
      score modeling and must never be fed oracle/analytic moments in a genuine
      inference run).
    - "gauss_hierarchical": the same hierarchical GAUSS correction as
      "Gauss_global_local", but hardened so it can only ever compose the
      network's own learned scores: passing ``posterior_mean`` or
      ``global_posterior_mean`` raises. This is the paper's GAUSS algorithm
      (Linhart et al. 2024, "Diffusion posterior sampling for simulation-based
      inference in tall data settings") generalized to a global/local hierarchy
      by exact block (arrow-precision) elimination of its Lemma 3.2: composing
      the network's real per-observation joint score with the marginal precision
      Λ_j(t) = Var(g_t | x_j)^-1 is algebraically *identical* to the full
      global+local block solve (the local-dependent terms cancel), so no
      marginal-score approximation or oracle moment is required -- only the
      pilot covariance (never the mean), estimated from the network's own DDIM
      draws, exactly as GAUSS estimates Σ_0,j. See
      New_Attempt/derivation.md for the full derivation.
    - "gauss_jacobian": the same hierarchical (arrow-elimination) composition as
      "gauss_hierarchical", but the per-observation backward covariance is taken
      from the network's own Jacobian by Tweedie's second-order identity instead
      of from a pilot covariance. In the sampler's y = x/alpha coordinates,

          Sigma_t,j(y) = lambda_t^2 ( I + lambda_t^2 grad s_j(y) )

      holds *exactly* for any p(theta_0 | x_j) -- no Gaussian assumption. It
      reduces identically to Sigma_0,j^-1 + lambda_t^-2 I (the GAUSS/pilot form
      used by every other covariance-aware rule) when the single-observation
      posterior really is the Gaussian N(., Sigma_0,j), so this is a strict
      generalization rather than a competing approximation. Consequences:

      * No pilot run. `posterior_covariance`, `posterior_precision`,
        `precision_est_*`, `covariance_shrinkage` and `covariance_nugget` are
        all unused (and rejected), removing a whole second reverse-diffusion
        pass and five hyperparameters.
      * The weighting is state- and time-dependent, so it tracks
        heteroscedasticity, skew and multimodality of the single-observation
        posteriors, which the constant-Sigma_0,j rules cannot.
      * Cost is `len(hierarchy) + len(local latents)` forward-mode JVPs per
        refresh -- the size of the latent block (single digits), independent of
        the observation count and of `nodes_size`. `jacobian_refresh` amortizes
        it further, since curvature varies far more slowly in t than the score.

      Like "gauss_hierarchical" it composes only the network's own scores and
      refuses `posterior_mean`/`global_posterior_mean` outright. It is the
      composition counterpart of `newton_map_estimate(curvature="jacobian")`
      and shares its curvature construction, so the two can be combined for a
      fully pilot-free hierarchical pipeline. See
      New_Attempt/jacobian_curvature.md for the derivation, the conditioning
      argument and the cost accounting.
    - "gauss_jacobian_FD": "gauss_jacobian" with the score Jacobian taken by
      forward finite differences instead of forward-mode AD. All D perturbed
      copies of the state (plus the base point) go through one no-grad network
      call, which keeps the fused attention kernel; the D forward-mode JVPs of
      "gauss_jacobian" cost ~7 forward passes each and run on the slow math
      attention backend. The step is ``jacobian_fd_step * lambda_t`` in y
      coordinates, since the lambda-smoothed score varies on the scale
      lambda_t or slower. Everything downstream -- clamping, Schur reduction,
      local cross-correction, refresh cache, guards -- is shared with
      "gauss_jacobian", and `newton_map_estimate(curvature="jacobian")` with
      this correction uses the same finite-difference Jacobian.

      Measured on Partial_Pooling (small preset, 20 and 100 subjects) against
      exact "gauss_jacobian": relative Jacobian error 1e-3 (2e-2 at
      lambda = 0.003), posterior means within 0.001-0.007 posterior SD of the
      exact rule at ~4x less wall clock. Combine with
      `jacobian_refresh_low_noise` for a further ~1.5x (see :meth:`sample`).

    Certifying a composition:
        `certify_composition` evaluates the *exact* tall-data target
        p(theta)^(1-n) prod_j p(theta | x_j) at the returned samples with the
        probability-flow ODE, giving an effective sample size and optional
        reweighted draws. It is the only measurement here that judges a
        composition rule without already knowing the right answer.
        `composition_curl` reports how far the composed field is from being a
        gradient, which is what decides whether Langevin correctors refine a
        composed estimate or circulate it.
    - "fnpe": the exact Eq. 7 of Geffner et al.: (1-n)(1-t) * prior score + sum of
      individual scores. This is the score of the paper's bridging densities, which are
      NOT the diffusion marginals of the posterior: it is only consistent with annealed
      Langevin sampling (use method="langevin"). The initial noise and Langevin step
      sizes on the hierarchy dimensions are scaled by 1/n internally (reference
      N(0, sigma_max^2 / n)).
    - "damping": the error-damped compositional estimator of Arruda et al. 2026,
      d(t) * [(1-n)(1-t) prior_score + n/m sum_(j in batch) score_j].
    - "gauss_damping": the Gaussian-corrected score multiplied by d(t).
    - "hybrid": the Gaussian correction with the prior precision and score
      contribution additionally weighted by (1-t), without damping.
    - "hybrid_damping": the Gaussian correction with the prior precision and score
      contribution additionally weighted by (1-t), followed by d(t).
    - "legacy_mean": historical prior-score-plus-mean aggregation.
    - "prior_corrected_sum": explicit name for the uncorrected prior-corrected sum.
    - "damped_sum": the prior-corrected sum multiplied by 1/n.
    - "minibatch_damped": the same 1/n-damped sum using an unbiased observation
      mini-batch, selected before evaluating the score network.

    Validity of the covariance-aware corrections:
        Every GAUSS-style rule composes the backward kernels as
        p(theta_0 | theta_t)^(1-n) prod_j p(theta_0 | theta_t, x_j), whose
        precision is Lambda(t) = sum_j Lambda_j(t) + (1-n) Lambda_prior(t).
        Lemma 3.1 of Linhart et al. 2024 is valid only while Lambda(t) is
        positive definite, which (their Eq. 21) requires

            sum_j Sigma_{t,j}^-1  >  (n-1) Sigma_{t,prior}^-1,

        i.e. each single-observation posterior must be *narrower* than the
        prior -- "normally the case in Bayesian approaches", as the paper puts
        it. Hierarchical models sit right on that boundary: one observation is
        nearly uninformative about a population-scale parameter, so Lambda_j ~=
        Lambda_prior and a few percent of pilot-covariance error flips the sign.

        Rather than repair Lambda after the fact, we make the condition hold by
        construction. Define observation j's *information* about the shared
        parameters, I_j(t) = Lambda_j(t) - Lambda_prior(t). Then

            Lambda(t) = Lambda_prior(t) + sum_j I_j(t)

        identically, so Lambda(t) is positive definite for every t and every n
        as soon as each I_j(t) is positive semi-definite -- which is a property
        of every real posterior, not a numerical convenience. Projecting the
        *estimated* I_j onto that cone therefore corrects an inadmissible
        estimate instead of distorting the target, is a no-op on admissible
        ones, and does not grow with n. See `_project_information`.

        Repairing the composed matrix instead is unsound: the composed score is
        obtained by solving against Lambda, so lifting a negative eigenvalue to
        a small epsilon divides by that epsilon and amplifies the deviation
        from the subject-score mean by |lambda_min| / epsilon.

    Practical notes:
    - Hierarchical models (per-observation local latents alongside the shared
      parameters, e.g. hierarchy=[0, 1] with further latent dimensions) benefit from
      dense Langevin correction: corrector_steps_interval=1, corrector_steps=10,
      snr=0.2. With the default settings the posterior width can come out conservative
      (up to ~1.5x too wide at n ~ 200) because the diagonal Gaussian correction
      ignores the global-local coupling; posterior means are unaffected.
    - For large n the composed posterior contracts like 1/sqrt(n), so its quality is
      limited by the score network's systematic error: a network bias of delta in
      parameter units stays a bias of ~delta, which becomes more visible relative to
      the shrinking posterior width. Training with more simulations/epochs (and the
      default "mixture" time sampling of the Trainer) directly improves this.
    """

    def __init__(self, SBIm):
        self.SBIm = SBIm
        # Get SDE from model for calculations
        self.sde = self.SBIm.sde
        self.prior_object = None
        self.training_prior = None  # see set_training_prior
        # Project each observation's information I_j = K_j - P_0 onto the PSD cone
        # (_project_information). On by default; COMPASS_INFORMATION_PROJECTION=0 or
        # set_information_projection(False) turns it off (the diagnostics still record
        # what it would have lifted).
        self.information_projection = os.environ.get("COMPASS_INFORMATION_PROJECTION", "1") != "0"
        # gauss_jacobian: clip the Jacobian curvature eigenvalues below at 0 (kappa~ = max(kappa, 0)).
        # On by default; COMPASS_CURVATURE_CLIP=0 or set_curvature_lower_clip(False) keeps
        # negative curvature (precision 1/(v(1 - v kappa)) < 1/v, still positive). The upper
        # clip (1 - floor)/v always stays: past 1/v the precision is infinite or negative.
        self.curvature_lower_clip = os.environ.get("COMPASS_CURVATURE_CLIP", "1") != "0"
        # How the composed precision Lambda_S = Lambda_prior(t) + sum_j I_j is kept positive
        # definite (set_pd_strategy). "row_projection" (default) is the per-row PSD projection
        # above, bit for bit. Every strategy records whether the *raw* Lambda_S was PD.
        self.pd_strategy = os.environ.get("COMPASS_PD_STRATEGY", "row_projection")
        self.pd_fallback = os.environ.get("COMPASS_PD_FALLBACK", "repair")
        self.pd_nugget = float(os.environ.get("COMPASS_PD_NUGGET", "0.1"))
        # jsonl path for _probe_row_lift (diagnostics only; does not change the draws)
        self.pd_probe = os.environ.get("COMPASS_PD_PROBE") or None
        # Denoised-prediction clamp: "coordinate" (default, per-coordinate clip) or
        # "metric" (conservative: projection in the denoiser's own metric; set_clamp_mode).
        self.clamp_mode = os.environ.get("COMPASS_CLAMP_MODE", "coordinate")
        self.covariance_shrinkage = 0.01
        self.covariance_nugget = 1e-6
        self.pd_epsilon = 1e-8
        self.pd_floor = 1e-3
        self.jacobian_refresh = 1
        self.jacobian_curvature_floor = 1e-3
        # Opt-in speed-up (set_cholesky_screen): skip the eigendecompositions of matrices a
        # Cholesky factorisation already proves need no eigenvalue clamping.
        self.cholesky_screen = False
        # Opt-in speed-up (set_eigh_precision): dtype and chunk of the two per-row
        # eigendecompositions on the gauss_jacobian path.
        self.eigh_dtype = torch.float64
        self.eigh_chunk = 2048
        self.jacobian_fd_step = 1e-2
        self.jacobian_refresh_low_noise = None
        self.jacobian_refresh_lambda = 0.3
        self.local_prior_mean = None
        self.local_prior_std = None
        self._reset_jacobian_state()
        # Counters live here as well as in sample()/map_estimate() so a freshly
        # constructed sampler is usable by helpers that do not run a full setup.
        self.score_network_calls = 0
        self.evaluated_subject_rows = 0
        self.evaluated_row_samples = 0
        self.jacobian_row_samples = 0
        self.map_diagnostics = None
        self._reset_covariance_diagnostics()

    DAMPING_CORRECTIONS = frozenset(
        {"damping", "gauss_damping", "hybrid_damping"}
    )
    DIAGONAL_GAUSSIAN_CORRECTIONS = frozenset(
        {"gauss", "gauss_damping", "hybrid", "hybrid_damping"}
    )
    FULL_GAUSSIAN_CORRECTIONS = frozenset({"full_gaussian"})
    # Corrections whose per-observation backward covariance comes from the
    # network's own Jacobian (Tweedie second order) rather than from a pilot
    # covariance. They share every downstream step with the Schur family.
    JACOBIAN_GAUSSIAN_CORRECTIONS = frozenset({"gauss_jacobian", "gauss_jacobian_FD"})
    # Jacobian corrections whose score Jacobian is taken by finite differences
    # rather than forward-mode AD; see `_row_score_jacobian`.
    FINITE_DIFFERENCE_JACOBIAN_CORRECTIONS = frozenset({"gauss_jacobian_FD"})
    JACOBIAN_CORRECTIONS = JACOBIAN_GAUSSIAN_CORRECTIONS | frozenset({"bp_jac"})
    SCHUR_GAUSSIAN_CORRECTIONS = frozenset(
        {"Gauss_schur_global", "Gauss_global_local", "gauss_hierarchical"}
    ) | JACOBIAN_GAUSSIAN_CORRECTIONS
    # Corrections that also propagate the shared-score composition into a
    # linear cross-correction of each observation's local latent score.
    LOCAL_CROSS_CORRECTIONS = frozenset(
        {"Gauss_global_local", "gauss_hierarchical"}
    ) | JACOBIAN_GAUSSIAN_CORRECTIONS
    # Corrections that must only ever compose real, learned network scores:
    # posterior_mean / global_posterior_mean (which substitute an externally
    # supplied Gaussian's score for the network's) are rejected outright.
    NO_MOMENT_SUBSTITUTION_CORRECTIONS = frozenset(
        {"gauss_hierarchical"}
    ) | JACOBIAN_GAUSSIAN_CORRECTIONS
    COVARIANCE_GAUSSIAN_CORRECTIONS = (
        FULL_GAUSSIAN_CORRECTIONS | SCHUR_GAUSSIAN_CORRECTIONS
    )
    GAUSSIAN_CORRECTIONS = (
        DIAGONAL_GAUSSIAN_CORRECTIONS | COVARIANCE_GAUSSIAN_CORRECTIONS
    )
    VALID_CORRECTIONS = frozenset(
        {
            "gauss", "full_gaussian", "Gauss_schur_global",
            "Gauss_global_local", "gauss_hierarchical", "gauss_jacobian",
            "gauss_jacobian_FD", "bp_jac",
            "hybrid", "uncorrected",
            "fnpe", "legacy_mean",
            "prior_corrected_sum", "damped_sum", "minibatch_damped",
        } | DAMPING_CORRECTIONS
    )
    TRUE_MINIBATCH_CORRECTIONS = frozenset({"damping", "minibatch_damped"})
    DAMPED_CORRECTIONS = DAMPING_CORRECTIONS | frozenset(
        {"damped_sum", "minibatch_damped"}
    )
    TEMPERABLE_CORRECTIONS = frozenset(
        {"fnpe", "gauss_hierarchical", "gauss_jacobian", "gauss_jacobian_FD",
         "bp_jac"}
    )

    #############################################
    # ----- Main Sampling Loop -----
    #############################################

    def sample(self, world_size, data, condition_mask=None, timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None, hierarchy=None,
               prior=None, local_prior=None,
               correction="gauss", posterior_precision=None,
               posterior_covariance=None, posterior_mean=None,
               global_posterior_mean=None, global_posterior_covariance=None,
               precision_est_samples=500, precision_est_timesteps=None, denoise_clamp=5.0,
               damping_at_data=1.0, damping_at_noise=None,
               damping_schedule="exponential",
               composition_temperature=1.0,
               composition_batch_size=None,
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=None, final_corrector_steps=3, terminal_corrector_steps=0, terminal_corrector_counts=None,
               churn_eta=DEFAULT_CHURN_ETA, churn_s_noise=1.0,
               churn_gamma_cap=DEFAULT_CHURN_GAMMA_CAP,
               adaptive_abs_tol=0.002576, adaptive_rel_tol=0.1,
               adaptive_safety=0.9, adaptive_exponent=0.9,
               adaptive_max_evals=10000, adaptive_initial_step=None,
               device="cpu", verbose=True, method="dpm", save_trajectory=False, result_dict=None,
               precision_est_batch_size=128, covariance_shrinkage=0.01,
               covariance_nugget=1e-6, pd_epsilon=1e-8, pd_floor=1e-3,
               jacobian_refresh=1, jacobian_curvature_floor=1e-3,
               jacobian_block_size=2, jacobian_pooling="contiguous",
               jacobian_fd_step=1e-2, jacobian_refresh_low_noise=None,
               jacobian_refresh_lambda=0.3, jacobian_refresh_high_noise=None,
               jacobian_refresh_high_lambda=None, jacobian_refresh_band=None,
               jacobian_refresh_outside_band=None, jacobian_secant=False,
               start_lambda=None, init_state=None):
        """
        Sample from the multi-observation posterior via compositional score modeling.

        Args:
            data: Input data
                    - Tensor of observed values, one row per observation
                        - Shape data: (num_observations, num_observed_features)
                    - condition_mask must be provided (or inferable)
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (num_total_features,) or (num_observations, num_total_features)
            timesteps: Number of diffusion steps
            eps: End time for diffusion process
            num_samples: Number of samples to generate
            cfg_alpha: Classifier-free guidance strength
            hierarchy: Indices of the *shared* (global) latent variables that are composed
                    across observations. Defaults to all latent variables.
            prior: Gaussian prior over the hierarchy dimensions, as a tuple
                    (mean, std) or, for "full_gaussian", (mean, covariance_matrix).
                    Defaults to a standard normal N(0, 1) (correct if the model was
                    trained on parameters standardized to zero mean and unit variance).
            correction: Composition rule. In addition to the existing "gauss",
                    "uncorrected", "fnpe", "damping", "gauss_damping", "hybrid"
                    and "hybrid_damping" values, accepts "legacy_mean",
                    "prior_corrected_sum", "damped_sum", "minibatch_damped" and
                    "gauss_jacobian" (pilot-free hierarchical GAUSS),
                    "gauss_jacobian_FD" (the same rule with a finite-difference
                    Jacobian, ~4x cheaper) and "bp_jac" (block-pooled Jacobian
                    composition; see the class docstring).
            jacobian_refresh: For correction="gauss_jacobian" only. Recompute
            jacobian_block_size: Number of observations K pooled by ``bp_jac``
                    before its Jacobian is inverted. The final short block is
                    weighted by its actual size.
                    the score Jacobian every this many composed-score
                    evaluations and reuse it in between. Curvature varies far
                    more slowly in t than the score does, so values of 5-20 cut
                    the overhead to a few percent at essentially no accuracy
                    cost. 1 (the default) refreshes at every evaluation.
                    Use an odd period with the order-2 DPM sampler: it makes
                    two calls per step, so an even period always refreshes at
                    the same one of them, which measurably biases the result
                    (docs/jacobian_speedup_results.md). 9 was 3.3x faster than
                    the old low-noise schedule on Partial_Pooling at no
                    measurable accuracy cost.
                    What is held between refreshes is only the network's
                    implied Sigma_0,j; the exact lambda(t)-dependent part of
                    Lambda_j(t) is rebuilt at every evaluation, so a lagged
                    refresh degrades gracefully towards ordinary GAUSS with a
                    periodically re-derived pilot covariance rather than
                    mismatching noise scales.
            jacobian_curvature_floor: For correction="gauss_jacobian" only.
                    Lower bound on the eigenvalues of I - lambda^2 H_j, i.e. how
                    close a single-observation backward kernel may come to a
                    point mass. Caps Lambda_j at (floor * lambda^2)^-1.
            jacobian_fd_step: For correction="gauss_jacobian_FD" only. Relative
                    finite-difference step; the absolute step is
                    ``jacobian_fd_step * lambda_t`` in y coordinates. 1e-2 keeps
                    float32 rounding and truncation error both near 1e-3.
            jacobian_refresh_low_noise: For the Jacobian corrections
                    ("gauss_jacobian", "gauss_jacobian_FD"). Refresh period
                    used once lambda_t < ``jacobian_refresh_lambda``;
                    ``jacobian_refresh`` then governs only the high-noise part
                    of the trajectory. None (the default) uses
                    ``jacobian_refresh`` throughout. A lagged refresh is nearly
                    free at low noise and costly at high noise: on
                    Partial_Pooling, 20 below lambda 0.3 and 1 above stayed
                    within Monte-Carlo noise of refreshing every evaluation,
                    while the reverse moved posterior means by ~0.2 SD.
            jacobian_refresh_lambda: Noise level lambda_t at which
                    ``jacobian_refresh_low_noise`` takes over.
            jacobian_refresh_high_noise, jacobian_refresh_high_lambda: The
                    mirror image at the noise end: refresh period used while
                    lambda_t > ``jacobian_refresh_high_lambda``. Both None (the
                    default) leaves the high-noise end on ``jacobian_refresh``.
            jacobian_refresh_band: ``(r_low, r_high)`` switches to an adaptive
                    band instead of fixed lambda edges, on
                    ``r = lambda^2 * median_j(min eig K_j)`` with K_j the cached
                    lambda-free information: the noise variance against the
                    widest direction of a single-observation posterior. The
                    Jacobian is refreshed every ``jacobian_refresh`` calls while
                    r_low <= r <= r_high and every
                    ``jacobian_refresh_outside_band`` calls otherwise (None:
                    never, after the first). Outside the band the information
                    barely moves with the state: far above it Cov(theta_0 | y)
                    tends to the posterior covariance, below it the smoothed
                    density is locally Gaussian. Overrides the fixed-lambda
                    edges when given.
            jacobian_secant: Between refreshes, update each row's cached
                    information K_j by a damped BFGS step from the secant pair
                    of consecutive evaluations, ``K v = u`` with ``u = s2 - s1``
                    and ``v = D1 - D2``, ``D = y + lambda^2 s`` the Tweedie
                    denoiser. The relation holds exactly for a Gaussian
                    posterior at any pair of noise levels and locally at a
                    single level for any posterior, and costs no network calls.
            start_lambda, init_state: Start the reverse diffusion at noise level
                    ``start_lambda`` from ``init_state`` (shape
                    ``(observations, num_samples, nodes)`` in the sampler's
                    y = x/alpha coordinates, shared coordinates synchronised)
                    instead of from pure noise at the top of the schedule. The
                    levels below ``start_lambda`` are the ones the full schedule
                    would use, so the step density is unchanged. See
                    :meth:`laplace_initial_state`.
            posterior_precision: Optional estimate of the single-observation posterior
                    precision on the hierarchy dimensions, used by the "gauss" correction.
                    Shape (len(hierarchy),) or (num_observations, len(hierarchy)).
                    If None, it is estimated by sampling the single-observation
                    posteriors once with the standard sampler.
            posterior_covariance: Optional covariance estimate for covariance-aware
                    corrections. ``full_gaussian`` expects H x H global covariance;
                    Schur modes expect (H + L) x (H + L) joint global/local covariance,
                    with either one matrix or one per observation. If None, empirical
                    covariances are estimated from ordinary single-observation draws.
            posterior_mean: Optional mean estimate paired with
                    ``posterior_covariance``. When supplied, covariance-aware
                    corrections project the learned single-observation score onto
                    the Gaussian defined by these moments before composition. This
                    removes systematic network bias for genuinely Gaussian
                    posteriors; omit it to retain the learned Tweedie mean.
            precision_est_samples: Number of samples per observation for the automatic
                    precision estimation.
            precision_est_timesteps: Number of diffusion steps for the automatic
                    precision estimation (defaults to `timesteps`).
            precision_est_batch_size: Maximum flattened transformer rows evaluated
                    at once while estimating precision. Batching does not change
                    `precision_est_samples`.
            denoise_clamp: Clamp the denoised predictions theta_t + sigma_t^2 * score
                    to within this many prior standard deviations of the prior mean
                    (stabilizes the score composition in the tails of the reference
                    distribution). None disables clamping.
            damping_at_data: Damping endpoint d(0), where t=0 is the data end.
            damping_at_noise: Damping endpoint d(1), where t=1 is the noise end.
                    Defaults to 1/sqrt(num_observations).
            composition_temperature: Likelihood-ratio temperature beta for
                    ``fnpe``, ``gauss_hierarchical`` and ``gauss_jacobian``.
                    The shared score becomes
                    ``s_prior + beta * (s_composed - s_prior)``. beta=1
                    preserves the original method exactly.
            composition_batch_size: Optional number of observation scores used in
                    each unbiased damping update. None uses every observation except
                    for "minibatch_damped", whose default is min(3, R). True
                    mini-batching requires every latent coordinate to be shared;
                    models with row-specific locals must evaluate all rows.

            - DPM-Solver parameters -
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            final_corrector_steps: Extra correction steps at the end

            - Adaptive reverse-SDE parameters -
            adaptive_abs_tol: Absolute error tolerance.
            adaptive_rel_tol: Relative error tolerance.
            adaptive_safety: Step-size safety multiplier.
            adaptive_exponent: Step-size controller exponent.
            adaptive_max_evals: Maximum score evaluations (two per proposal).
            adaptive_initial_step: Optional initial step in diffusion time.

            - Other parameters -
            device: Device to run sampling on
            verbose: Whether to show progress bar
            method: Sampling method to use (euler, dpm, langevin, adaptive)
            save_trajectory: Whether to save the intermediate denoising trajectory
        """

        # Sampler ablation without editing every experiment: COMPASS_SAMPLER_OVERRIDE is a JSON
        # object whose keys (churn_eta, corrector_steps, corrector_steps_interval,
        # final_corrector_steps, terminal_corrector_steps, snr) replace the caller's arguments,
        # for method="dpm" only, e.g. Langevin correctors instead of churn.
        self.sampler_override = None
        override = os.environ.get("COMPASS_SAMPLER_OVERRIDE")
        if override and method == "dpm":
            import json
            self.sampler_override = json.loads(override)
            allowed = {"churn_eta", "corrector_steps", "corrector_steps_interval",
                       "final_corrector_steps", "terminal_corrector_steps", "snr"}
            unknown = set(self.sampler_override) - allowed
            if unknown:
                raise ValueError(f"COMPASS_SAMPLER_OVERRIDE: unknown keys {sorted(unknown)}")
            churn_eta = self.sampler_override.get("churn_eta", churn_eta)
            corrector_steps = self.sampler_override.get("corrector_steps", corrector_steps)
            corrector_steps_interval = self.sampler_override.get("corrector_steps_interval",
                                                                 corrector_steps_interval)
            final_corrector_steps = self.sampler_override.get("final_corrector_steps", final_corrector_steps)
            terminal_corrector_steps = self.sampler_override.get("terminal_corrector_steps",
                                                                 terminal_corrector_steps)
            snr = self.sampler_override.get("snr", snr)

        # Set parameters
        self.world_size = world_size
        self.timesteps = timesteps
        self.eps = eps
        self.num_samples = int(num_samples)
        self.cfg_alpha = cfg_alpha
        self.verbose = verbose
        self.method = method
        self.save_trajectory = save_trajectory
        self.hierarchy = hierarchy
        self.correction = correction
        self.composition_temperature = float(composition_temperature)
        if not math.isfinite(self.composition_temperature) or self.composition_temperature <= 0.0:
            raise ValueError("composition_temperature must be finite and positive.")
        self.denoise_clamp = denoise_clamp
        self.score_network_calls = 0
        self.evaluated_subject_rows = 0
        self.evaluated_row_samples = 0
        self.jacobian_row_samples = 0
        self.jacobian_block_size = int(jacobian_block_size)
        self.jacobian_pooling = str(jacobian_pooling)
        self.solver_stats = None
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.covariance_nugget = float(covariance_nugget)
        self.pd_epsilon = float(pd_epsilon)
        self.pd_floor = float(pd_floor)
        self.jacobian_refresh = int(jacobian_refresh)
        self.jacobian_curvature_floor = float(jacobian_curvature_floor)
        if self.jacobian_refresh < 1:
            raise ValueError("jacobian_refresh must be at least 1.")
        self._configure_jacobian_schedule(
            jacobian_fd_step, jacobian_refresh_low_noise, jacobian_refresh_lambda,
            refresh_high_noise=jacobian_refresh_high_noise,
            refresh_high_lambda=jacobian_refresh_high_lambda,
            refresh_band=jacobian_refresh_band,
            refresh_outside_band=jacobian_refresh_outside_band,
            secant=jacobian_secant,
        )
        self.start_lambda = None if start_lambda is None else float(start_lambda)
        if self.start_lambda is not None and not (
                math.isfinite(self.start_lambda) and self.start_lambda > 0.0):
            raise ValueError("start_lambda must be finite and positive.")
        if (init_state is None) != (self.start_lambda is None):
            raise ValueError("start_lambda and init_state must be given together.")
        self.init_state = init_state
        if self.jacobian_block_size < 1:
            raise ValueError("jacobian_block_size must be at least 1.")
        if self.jacobian_pooling not in {"contiguous", "score_sorted"}:
            raise ValueError("jacobian_pooling must be 'contiguous' or 'score_sorted'.")
        if not 0.0 < self.jacobian_curvature_floor < 1.0:
            raise ValueError("jacobian_curvature_floor must lie in (0, 1).")
        self._reset_jacobian_state()
        if not 0.0 <= self.covariance_shrinkage < 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1).")
        if self.covariance_nugget < 0.0 or self.pd_epsilon <= 0.0:
            raise ValueError(
                "covariance_nugget must be nonnegative and pd_epsilon positive."
            )
        if self.pd_floor <= 0.0:
            raise ValueError("pd_floor must be positive.")
        self._reset_covariance_diagnostics()

        # The terminal-corrector settings are read by _sample_loop, which every
        # method shares, so they must exist for every method. Initializing them
        # only for "dpm"/"langevin" made "euler" and "adaptive" raise
        # AttributeError on the shared path -- which took the damping and hybrid
        # corrections down with them, since those are exercised through euler.
        self.terminal_corrector_steps = int(terminal_corrector_steps)
        if self.terminal_corrector_steps < 0:
            raise ValueError("terminal_corrector_steps must be nonnegative.")
        self.terminal_corrector_counts = tuple(sorted({
            int(count) for count in (terminal_corrector_counts or ())
        }))
        if any(count < 1 for count in self.terminal_corrector_counts):
            raise ValueError("terminal_corrector_counts must contain positive integers.")
        if self.terminal_corrector_counts and self.terminal_corrector_steps:
            raise ValueError("Specify either terminal_corrector_steps or terminal_corrector_counts, not both.")

        # Churn is the default stochastic predictor; see compass/churn.py for the
        # measurements that made it the default and for why it and the Langevin
        # correctors are alternatives rather than complements.
        self.churn_eta = float(churn_eta)
        self.churn_s_noise = float(churn_s_noise)
        self.churn_gamma_cap = float(churn_gamma_cap)
        if self.churn_eta < 0.0:
            raise ValueError("churn_eta must be nonnegative.")
        if self.churn_gamma_cap <= 0.0:
            raise ValueError("churn_gamma_cap must be positive.")

        if method in ("dpm", "langevin"):
            self.corrector_steps_interval = corrector_steps_interval
            self.corrector_steps = resolve_corrector_steps(
                corrector_steps, method, self.churn_eta,
            )
            self.final_corrector_steps = final_corrector_steps
            self.snr = snr
            self.order = order

        if correction not in self.VALID_CORRECTIONS:
            choices = "', '".join(sorted(self.VALID_CORRECTIONS))
            raise ValueError(
                f"Unknown correction '{correction}'. Choose from '{choices}'."
            )
        if (self.composition_temperature != 1.0
                and correction not in self.TEMPERABLE_CORRECTIONS):
            choices = "', '".join(sorted(self.TEMPERABLE_CORRECTIONS))
            raise ValueError(
                "composition_temperature is supported only for correction="
                f"'{choices}'."
            )
        if method not in ("euler", "dpm", "langevin", "adaptive"):
            raise ValueError(
                f"Sampling method {method} not recognized. Choose from "
                "'euler', 'dpm', 'langevin' or 'adaptive'."
            )
        if correction == "fnpe" and method != "langevin":
            print("WARNING: correction='fnpe' composes the scores of the F-NPSE bridging densities, "
                  "which are NOT the diffusion marginals of the posterior. Reverse-diffusion samplers "
                  "(euler/dpm) diverge with it; use method='langevin' (annealed Langevin dynamics).")
        if save_trajectory and method == "adaptive":
            raise ValueError(
                "save_trajectory is not supported by the adaptive sampler because "
                "its accepted time grid has dynamic length."
            )

        data_tensor = torch.as_tensor(data)
        num_observations = 1 if data_tensor.dim() == 1 else int(data_tensor.shape[0])
        if correction == "minibatch_damped" and composition_batch_size is None:
            composition_batch_size = min(3, num_observations)
        self._configure_damping(
            num_observations, damping_at_data, damping_at_noise,
            composition_batch_size, damping_schedule,
        )
        if (composition_batch_size is not None
                and correction not in self.TRUE_MINIBATCH_CORRECTIONS):
            raise ValueError(
                "composition_batch_size is supported only for "
                "correction='damping' or 'minibatch_damped'."
            )
        if (
            correction in self.GAUSSIAN_CORRECTIONS
            and self.composition_batch_size < num_observations
        ):
            raise ValueError(
                "composition_batch_size currently applies only to "
                "correction='damping'; Gaussian precision ratios require all "
                "observations."
            )
        if world_size > 1 and (
            method == "adaptive"
            or self.composition_batch_size < num_observations
        ):
            raise NotImplementedError(
                "Adaptive sampling and observation-score mini-batching currently "
                "require world_size=1 so all stochastic decisions remain synchronized."
            )
        if world_size > 1 and correction in self.JACOBIAN_CORRECTIONS:
            # The Jacobian is taken w.r.t. the *whole* synchronized state, so a
            # rank that owns only a shard of the observations cannot form it;
            # the shared-coordinate tangent would be missing the other shards.
            raise NotImplementedError(
                "Jacobian composition requires world_size=1: the score "
                "Jacobian is taken with respect to the full synchronized state, "
                "which no single rank holds under the sharded sampler."
            )
        self._configure_adaptive(
            adaptive_abs_tol=adaptive_abs_tol,
            adaptive_rel_tol=adaptive_rel_tol,
            adaptive_safety=adaptive_safety,
            adaptive_exponent=adaptive_exponent,
            adaptive_max_evals=adaptive_max_evals,
            adaptive_initial_step=adaptive_initial_step,
        )

        # Resolve hierarchy before spawning workers (needed for prior & precision setup)
        if self.hierarchy is None:
            if condition_mask is None:
                raise ValueError("Either hierarchy or condition_mask must be provided.")
            cm = condition_mask if condition_mask.dim() == 1 else condition_mask[0]
            self.hierarchy = torch.where(cm == 0)[0].tolist()

        cm = torch.as_tensor(condition_mask)
        cm = cm if cm.dim() == 1 else cm[0]
        latent_indices = torch.where(cm == 0)[0].tolist()
        missing_hierarchy = sorted(set(self.hierarchy) - set(latent_indices))
        if missing_hierarchy:
            raise ValueError(
                "hierarchy indices must refer to latent coordinates; observed "
                f"indices were supplied: {missing_hierarchy}."
            )
        hierarchy_set = set(self.hierarchy)
        self.local_latent_indices = [
            index for index in latent_indices if index not in hierarchy_set
        ]
        # A complete covariance is essential when the learned single-observation
        # score couples a shared variable to row-local nuisance variables.
        self.full_gaussian_features = (
            self.hierarchy + self.local_latent_indices
        )
        self.covariance_features = (
            self.full_gaussian_features if correction in self.SCHUR_GAUSSIAN_CORRECTIONS
            else self.hierarchy
        )
        self.local_prior_mean, self.local_prior_std = self._resolve_local_prior(
            local_prior
        )

        if self.composition_batch_size < num_observations:
            cm = torch.as_tensor(condition_mask)
            cm = cm if cm.dim() == 1 else cm[0]
            latent_indices = set(torch.where(cm == 0)[0].tolist())
            if latent_indices != set(self.hierarchy):
                raise ValueError(
                    "True observation mini-batching is supported only when every "
                    "latent coordinate is shared. Native global/local inference "
                    "must evaluate every subject to update its local state."
                )

        if posterior_precision is not None and posterior_covariance is not None:
            raise ValueError(
                "posterior_precision and posterior_covariance are mutually exclusive."
            )
        if (posterior_mean is not None
                and correction not in self.COVARIANCE_GAUSSIAN_CORRECTIONS):
            raise ValueError(
                "posterior_mean is supported only for covariance-aware Gaussian "
                "corrections."
            )
        if posterior_mean is not None and correction in self.NO_MOMENT_SUBSTITUTION_CORRECTIONS:
            raise ValueError(
                f"correction={correction!r} composes only the network's real "
                "scores; posterior_mean would substitute an externally supplied "
                "Gaussian's score instead. Use correction='Gauss_global_local' "
                "for that (diagnostic, non-compositional) ablation."
            )

        if ((global_posterior_mean is None) != (global_posterior_covariance is None)):
            raise ValueError(
                "global_posterior_mean and global_posterior_covariance must be supplied together."
            )
        if (global_posterior_mean is not None
                and correction != "Gauss_global_local"):
            raise ValueError(
                "global posterior moments are supported only for correction="
                "'Gauss_global_local'."
            )

        # Resolve the Gaussian prior over the hierarchy dimensions.
        self.prior_mean, self.prior_std, self.prior_covariance = self._resolve_prior(prior)
        self.prior_precision_matrix = self._precision_from_covariance(
            self.prior_covariance.unsqueeze(0)
        )[0]

        # Covariance-aware GAUSS setup. Full mode stores HxH matrices; Schur
        # modes store one (H+L)x(H+L) matrix per observation.
        if correction in self.JACOBIAN_CORRECTIONS:
            # No pilot moments at all: the per-observation backward precision is
            # rebuilt from the network's Jacobian at every refresh.
            if posterior_covariance is not None or posterior_precision is not None:
                raise ValueError(
                    f"correction={correction!r} derives every single-observation "
                    "covariance from the score Jacobian and takes no pilot "
                    "estimate; drop posterior_covariance/posterior_precision."
                )
            self.posterior_covariance = None
            self.posterior_precision = None
            self.posterior_precision_matrix = None
            self.posterior_mean = None
            self.global_posterior_mean = None
            self.global_posterior_covariance = None
            self.global_posterior_precision_matrix = None
            self._covariance_time_cache = {}
        elif correction in self.COVARIANCE_GAUSSIAN_CORRECTIONS:
            if posterior_precision is not None:
                raise ValueError(
                    f"correction={correction!r} requires posterior_covariance, not "
                    "posterior_precision."
                )
            if posterior_covariance is None:
                if verbose:
                    print(
                        "Estimating single-observation posterior covariances for "
                        f"the {correction!r} correction ..."
                    )
                if correction == "Gauss_global_local":
                    global_posterior_mean, posterior_covariance = (
                        self.estimate_posterior_moments(
                            data, condition_mask,
                            num_samples=precision_est_samples,
                            timesteps=precision_est_timesteps or timesteps,
                            eps=eps, batch_size=precision_est_batch_size,
                            device=device if world_size <= 1 else "cuda:0",
                            feature_indices=self.covariance_features,
                        )
                    )
                    global_posterior_mean = global_posterior_mean[:, :len(self.hierarchy)]
                    global_posterior_covariance = posterior_covariance[:, :len(self.hierarchy), :len(self.hierarchy)]
                else:
                    posterior_covariance = self._estimate_posterior_covariance(
                        data, condition_mask,
                        num_samples=precision_est_samples,
                        timesteps=precision_est_timesteps or timesteps,
                        eps=eps, batch_size=precision_est_batch_size,
                        device=device if world_size <= 1 else "cuda:0",
                        feature_indices=self.covariance_features,
                    )
            self.posterior_covariance = self._validate_covariance(
                posterior_covariance, num_observations=num_observations,
                dimension=len(self.covariance_features),
                name="posterior_covariance",
            )
            self.posterior_mean = self._validate_posterior_mean(
                posterior_mean, num_observations=num_observations,
                dimension=len(self.covariance_features),
            )
            self.posterior_precision_matrix = self._precision_from_covariance(
                self.posterior_covariance
            )
            self._covariance_time_cache = {}
            self.posterior_precision = None
            if correction == "Gauss_global_local":
                if global_posterior_mean is None:
                    raise ValueError(
                        "Gauss_global_local requires marginal global pilot moments. "
                        "Omit posterior_covariance to estimate them automatically, or "
                        "supply global_posterior_mean and global_posterior_covariance."
                    )
                self.global_posterior_mean = self._validate_posterior_mean(
                    global_posterior_mean, num_observations=num_observations,
                    dimension=len(self.hierarchy),
                )
                self.global_posterior_covariance = self._validate_covariance(
                    global_posterior_covariance, num_observations=num_observations,
                    dimension=len(self.hierarchy),
                    name="global_posterior_covariance",
                )
                self.global_posterior_precision_matrix = self._precision_from_covariance(
                    self.global_posterior_covariance
                )
            else:
                self.global_posterior_mean = None
                self.global_posterior_covariance = None
                self.global_posterior_precision_matrix = None
        elif correction in self.DIAGONAL_GAUSSIAN_CORRECTIONS:
            if posterior_covariance is not None:
                raise ValueError(
                    "posterior_covariance is supported only for covariance-aware "
                    "Gaussian corrections."
                )
            if posterior_precision is None:
                if verbose:
                    print(
                        "Estimating single-observation posterior precisions for "
                        f"the '{correction}' correction ..."
                    )
                posterior_precision = self._estimate_posterior_precision(
                    data, condition_mask,
                    num_samples=precision_est_samples,
                    timesteps=precision_est_timesteps or timesteps,
                    eps=eps, batch_size=precision_est_batch_size,
                    device=device if world_size <= 1 else "cuda:0")
            self.posterior_precision = self._validate_precision(torch.as_tensor(posterior_precision, dtype=torch.float32))
            self.posterior_covariance = None
            self.posterior_precision_matrix = None
            self.posterior_mean = None
            self.global_posterior_mean = None
            self.global_posterior_covariance = None
            self.global_posterior_precision_matrix = None
        else:
            if posterior_covariance is not None:
                raise ValueError(
                    "posterior_covariance is supported only for "
                    "correction='full_gaussian'."
                )
            self.posterior_precision = None
            self.posterior_covariance = None
            self.posterior_precision_matrix = None
            self.posterior_mean = None
            self.global_posterior_mean = None
            self.global_posterior_covariance = None
            self.global_posterior_precision_matrix = None

        if self.world_size > 1:
            manager = mp.Manager()
            result_dict = manager.dict()
            mp.spawn(self._sample_loop, args=(data, condition_mask, num_samples, result_dict), nprocs=self.world_size, join=True)
            samples = result_dict.get('samples', None)
            manager.shutdown()

        else:
            rank = 0
            self.device = device
            samples = self._sample_loop(rank, data, condition_mask, num_samples)

        return samples

    def _configure_damping(self, num_observations, damping_at_data,
                           damping_at_noise, composition_batch_size,
                           damping_schedule="exponential"):
        """Validate and store damping endpoints and mini-batch size."""
        n = int(num_observations)
        if n < 1:
            raise ValueError("At least one observation is required.")
        at_data = float(damping_at_data)
        at_noise = n ** -0.5 if damping_at_noise is None else float(damping_at_noise)
        if not (0.0 < at_noise <= at_data <= 1.0):
            raise ValueError(
                "Damping endpoints must satisfy "
                "0 < damping_at_noise <= damping_at_data <= 1."
            )
        batch_size = n if composition_batch_size is None else int(composition_batch_size)
        if batch_size < 1 or batch_size > n:
            raise ValueError(
                "composition_batch_size must be between 1 and the number "
                f"of observations ({n}), got {batch_size}."
            )
        self.damping_at_data = at_data
        self.damping_at_noise = at_noise
        if damping_schedule not in {"exponential", "linear", "cosine", "quadratic", "sqrt"}:
            raise ValueError(f"Unknown damping_schedule={damping_schedule!r}.")
        self.damping_schedule = damping_schedule
        self.composition_batch_size = batch_size

    def _configure_adaptive(self, adaptive_abs_tol, adaptive_rel_tol,
                            adaptive_safety, adaptive_exponent,
                            adaptive_max_evals, adaptive_initial_step):
        """Validate and store adaptive reverse-SDE controller settings."""
        values = {
            "adaptive_abs_tol": float(adaptive_abs_tol),
            "adaptive_rel_tol": float(adaptive_rel_tol),
            "adaptive_safety": float(adaptive_safety),
            "adaptive_exponent": float(adaptive_exponent),
        }
        if any(value <= 0.0 for value in values.values()):
            raise ValueError("Adaptive tolerances and controller factors must be positive.")
        max_evals = int(adaptive_max_evals)
        if max_evals < 2:
            raise ValueError("adaptive_max_evals must be at least 2.")
        initial_step = (
            None if adaptive_initial_step is None else float(adaptive_initial_step)
        )
        if initial_step is not None and initial_step <= 0.0:
            raise ValueError("adaptive_initial_step must be positive when provided.")
        self.adaptive_abs_tol = values["adaptive_abs_tol"]
        self.adaptive_rel_tol = values["adaptive_rel_tol"]
        self.adaptive_safety = values["adaptive_safety"]
        self.adaptive_exponent = values["adaptive_exponent"]
        self.adaptive_max_evals = max_evals
        self.adaptive_initial_step = initial_step

    def _damping_factor(self, t):
        """Endpoint-preserving damping schedule, with t=0 at the data end."""
        t = torch.as_tensor(t)
        at_data = torch.as_tensor(
            self.damping_at_data, dtype=t.dtype, device=t.device
        )
        at_noise = torch.as_tensor(
            self.damping_at_noise, dtype=t.dtype, device=t.device
        )
        if self.damping_schedule == "exponential":
            return at_data * torch.exp(torch.log(at_noise / at_data) * t)
        if self.damping_schedule == "linear":
            return at_data + (at_noise - at_data) * t
        if self.damping_schedule == "cosine":
            return at_noise + (at_data - at_noise) * torch.cos(torch.pi * t / 2)
        if self.damping_schedule == "quadratic":
            return at_data + (at_noise - at_data) * t.square()
        return at_data + (at_noise - at_data) * torch.sqrt(t)

    @torch.no_grad()
    def map_estimate(self, data, condition_mask, init=None, hierarchy=None,
                     prior=None, local_prior=None,
                     correction="gauss", posterior_precision=None,
                     posterior_covariance=None, posterior_mean=None,
                     global_posterior_mean=None,
                     global_posterior_covariance=None,
                     denoise_clamp=5.0, cfg_alpha=None, sigma_start=None,
                     damping_at_data=1.0, damping_at_noise=None,
                     timesteps=100, eps=1e-3, iterations_per_level=3,
                     max_iterations_per_level=None, convergence_tol=1e-6,
                     jacobian_refresh=1, jacobian_curvature_floor=1e-3,
                     device="cpu"):
        """Refine hierarchical MAP candidates by annealed compositional score ascent.

        ``init`` may have shape ``(observations, nodes)`` or
        ``(observations, candidates, nodes)``. Hierarchy coordinates use the
        composed score; all other latent coordinates retain their row-specific
        score. Each annealing level alternates local and hierarchy updates until
        their normalized update is below ``convergence_tol``. Gaussian precision
        or covariance estimates must be reused from posterior sampling, except
        for ``correction="gauss_jacobian"``, which needs none.
        """
        state = self._configure_map_estimate(
            data=data, condition_mask=condition_mask, init=init,
            hierarchy=hierarchy, prior=prior, local_prior=local_prior,
            correction=correction, posterior_precision=posterior_precision,
            posterior_covariance=posterior_covariance,
            posterior_mean=posterior_mean,
            global_posterior_mean=global_posterior_mean,
            global_posterior_covariance=global_posterior_covariance,
            denoise_clamp=denoise_clamp, cfg_alpha=cfg_alpha,
            damping_at_data=damping_at_data, damping_at_noise=damping_at_noise,
            timesteps=timesteps, iterations_per_level=iterations_per_level,
            max_iterations_per_level=max_iterations_per_level,
            convergence_tol=convergence_tol, device=device,
            jacobian_refresh=jacobian_refresh,
            jacobian_curvature_floor=jacobian_curvature_floor,
        )
        n_obs = state["n_obs"]
        hierarchy = state["hierarchy"]
        candidates = state["candidates"]
        expanded_mask = state["expanded_mask"]
        expanded_data = state["expanded_data"]
        latent = state["latent"]
        shared_latent = state["shared_latent"]
        local_latent = state["local_latent"]
        has_local_latents = state["has_local_latents"]
        min_iterations = state["min_iterations"]
        max_iterations = state["max_iterations"]

        times, lams = self._map_schedule(sigma_start, timesteps, eps, device)
        indices = torch.arange(n_obs, device=device)
        z = candidates
        for index in range(int(timesteps)):
            t = times[index].reshape(1, 1)
            lam = lams[index]
            for iteration in range(max_iterations):
                # Gauss-Seidel block ascent: first relax the row-specific
                # coordinates while the hierarchy is fixed, then recompute the
                # score before moving the shared coordinates. Updating both
                # blocks from the same stale score causes an observation-count
                # dependent lag in tightly concentrated shared posteriors.
                max_update = torch.zeros((), device=device)
                if has_local_latents:
                    score = self._get_score(
                        z, t, expanded_mask, indices, cfg_alpha
                    )
                    local_update = lam**2 * score * local_latent
                    z = z + local_update
                    max_update = local_update.abs().max()
                    z = z * (1 - expanded_mask) + expanded_data * expanded_mask

                score = self._get_score(
                    z, t, expanded_mask, indices, cfg_alpha
                )
                shared_update = lam**2 * score * shared_latent
                z = z + shared_update
                max_update = torch.maximum(max_update, shared_update.abs().max())
                z[:, :, hierarchy] = z[:1, :, hierarchy]
                z = z * (1 - expanded_mask) + expanded_data * expanded_mask

                if iteration + 1 >= min_iterations:
                    state_scale = (z * latent).abs().max().clamp_min(1.0)
                    if max_update <= float(convergence_tol) * state_scale:
                        break
        return self._finalize_map(
            z, times[-1], latent, expanded_mask, expanded_data, hierarchy, device,
        )
    def _configure_map_estimate(
            self, data, condition_mask, init, hierarchy, prior, local_prior,
            correction, posterior_precision, posterior_covariance,
            posterior_mean, global_posterior_mean, global_posterior_covariance,
            denoise_clamp, cfg_alpha, damping_at_data, damping_at_noise,
            timesteps, iterations_per_level, max_iterations_per_level,
            convergence_tol, device, jacobian_refresh=1,
            jacobian_curvature_floor=1e-3):
        """Validate arguments and build the shared state for MAP refinement.

        This is the configuration block ``map_estimate`` used to run inline.
        ``newton_map_estimate`` reuses it unchanged, so both estimators accept
        identical arguments and enforce identical guards -- an A/B between them
        therefore isolates the ascent step and nothing else.
        """
        if correction == "fnpe":
            raise ValueError(
                "correction='fnpe' is not supported by deterministic hierarchical "
                "MAP ascent because its bridging scores are not diffusion-posterior scores."
            )
        allowed = {"uncorrected"} | self.GAUSSIAN_CORRECTIONS | self.DAMPING_CORRECTIONS
        if correction not in allowed:
            raise ValueError(f"Unknown correction '{correction}'.")
        min_iterations = int(iterations_per_level)
        max_iterations = (
            max(min_iterations, 50) if max_iterations_per_level is None
            else int(max_iterations_per_level)
        )
        if (int(timesteps) < 1 or min_iterations < 1
                or max_iterations < min_iterations):
            raise ValueError(
                "timesteps and iterations_per_level must be at least 1, and "
                "max_iterations_per_level must not be smaller than "
                "iterations_per_level."
            )
        if float(convergence_tol) <= 0:
            raise ValueError("convergence_tol must be positive.")

        data = torch.as_tensor(data, dtype=torch.float32)
        if data.dim() != 2:
            raise ValueError(
                f"data must have shape (num_observations, nodes_size), got {tuple(data.shape)}."
            )
        n_obs, nodes_size = data.shape
        mask = torch.as_tensor(condition_mask, dtype=torch.float32)
        if mask.dim() == 1:
            if mask.numel() != nodes_size:
                raise ValueError(f"condition_mask must have {nodes_size} entries.")
            mask = mask.unsqueeze(0).repeat(n_obs, 1)
        elif mask.dim() != 2 or tuple(mask.shape) != tuple(data.shape):
            raise ValueError(
                "condition_mask must have shape (nodes_size,) or "
                f"(num_observations, nodes_size), got {tuple(mask.shape)}."
            )
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("condition_mask must contain only 0 and 1.")

        if hierarchy is None:
            hierarchy = torch.where(mask[0] == 0)[0].tolist()
        hierarchy = [int(index) for index in hierarchy]
        if not hierarchy:
            raise ValueError("hierarchy must contain at least one shared latent coordinate.")
        if len(set(hierarchy)) != len(hierarchy):
            raise ValueError("hierarchy indices must be unique.")
        if min(hierarchy) < 0 or max(hierarchy) >= nodes_size:
            raise ValueError(f"hierarchy indices are out of range: {hierarchy}.")
        if torch.any(mask[:, hierarchy] != 0):
            raise ValueError("Every hierarchy coordinate must be latent in condition_mask.")

        candidates = data.unsqueeze(1) if init is None else torch.as_tensor(init, dtype=torch.float32)
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(1)
        if (candidates.dim() != 3 or candidates.shape[0] != n_obs
                or candidates.shape[2] != nodes_size or candidates.shape[1] < 1):
            raise ValueError(
                "init must have shape (observations, nodes) or "
                f"(observations, candidates, nodes), got {tuple(candidates.shape)}."
            )
        sync_error = (candidates[:, :, hierarchy] - candidates[:1, :, hierarchy]).abs().max().item()
        if sync_error > 1e-6:
            raise ValueError(
                "Hierarchy coordinates must be synchronized across observations; "
                f"maximum deviation is {sync_error:.3e}."
            )

        self.world_size = 1
        self.rank = 0
        self.device = device
        self.model = self.SBIm.model.to(device)
        self.model.eval()
        self.num_observations = n_obs
        self.num_samples = candidates.shape[1]
        self.cfg_alpha = cfg_alpha
        self.hierarchy = hierarchy
        self.correction = correction
        self.denoise_clamp = denoise_clamp
        self.score_network_calls = 0
        self.evaluated_subject_rows = 0
        self.evaluated_row_samples = 0
        self.jacobian_row_samples = 0
        latent_indices = torch.where(mask[0] == 0)[0].tolist()
        hierarchy_set = set(hierarchy)
        self.local_latent_indices = [
            index for index in latent_indices if index not in hierarchy_set
        ]
        self.full_gaussian_features = hierarchy + self.local_latent_indices
        self.covariance_features = (
            self.full_gaussian_features if correction in self.SCHUR_GAUSSIAN_CORRECTIONS
            else hierarchy
        )
        self.covariance_shrinkage = 0.01
        self.covariance_nugget = 1e-6
        self.pd_epsilon = 1e-8
        self.pd_floor = 1e-3
        self.jacobian_refresh = int(jacobian_refresh)
        self.jacobian_curvature_floor = float(jacobian_curvature_floor)
        if self.jacobian_refresh < 1:
            raise ValueError("jacobian_refresh must be at least 1.")
        if not 0.0 < self.jacobian_curvature_floor < 1.0:
            raise ValueError("jacobian_curvature_floor must lie in (0, 1).")
        # MAP refinement always refreshes on the uniform `jacobian_refresh`
        # schedule and uses the default finite-difference step.
        self._configure_jacobian_schedule(1e-2, None, 0.3)
        self._reset_jacobian_state()
        self.local_prior_mean, self.local_prior_std = self._resolve_local_prior(
            local_prior
        )
        self._reset_covariance_diagnostics()
        self._configure_damping(
            n_obs, damping_at_data, damping_at_noise,
            composition_batch_size=n_obs,
        )
        if posterior_precision is not None and posterior_covariance is not None:
            raise ValueError(
                "posterior_precision and posterior_covariance are mutually exclusive."
            )
        if (posterior_mean is not None
                and correction not in self.COVARIANCE_GAUSSIAN_CORRECTIONS):
            raise ValueError(
                "posterior_mean is supported only for covariance-aware Gaussian "
                "corrections."
            )
        if posterior_mean is not None and correction in self.NO_MOMENT_SUBSTITUTION_CORRECTIONS:
            raise ValueError(
                f"correction={correction!r} composes only the network's real "
                "scores; posterior_mean would substitute an externally supplied "
                "Gaussian's score instead. Use correction='Gauss_global_local' "
                "for that (diagnostic, non-compositional) ablation."
            )
        if ((global_posterior_mean is None)
                != (global_posterior_covariance is None)):
            raise ValueError(
                "global_posterior_mean and global_posterior_covariance must "
                "be supplied together."
            )
        if (global_posterior_mean is not None
                and correction != "Gauss_global_local"):
            raise ValueError(
                "global posterior moments are supported only for correction="
                "'Gauss_global_local'."
            )
        self.prior_mean, self.prior_std, self.prior_covariance = self._resolve_prior(prior)
        self.prior_mean = self.prior_mean.to(device)
        self.prior_std = self.prior_std.to(device)
        self.prior_covariance = self.prior_covariance.to(device)
        self.prior_precision_matrix = self._precision_from_covariance(
            self.prior_covariance.unsqueeze(0)
        )[0]
        if self.local_prior_mean is not None:
            self.local_prior_mean = self.local_prior_mean.to(device)
            self.local_prior_std = self.local_prior_std.to(device)
        if correction in self.JACOBIAN_GAUSSIAN_CORRECTIONS:
            # Pilot-free: the per-observation backward precision is rebuilt from
            # the score Jacobian, so nothing has to be carried over from
            # posterior sampling. This pairs with
            # ``newton_map_estimate(curvature="jacobian")``, which builds its
            # Newton curvature the same way, for an end-to-end pilot-free run.
            if posterior_covariance is not None or posterior_precision is not None:
                raise ValueError(
                    f"correction={correction!r} derives every single-observation "
                    "covariance from the score Jacobian and takes no pilot "
                    "estimate; drop posterior_covariance/posterior_precision."
                )
            self.posterior_covariance = None
            self.posterior_precision = None
            self.posterior_precision_matrix = None
            self.posterior_mean = None
            self.global_posterior_mean = None
            self.global_posterior_covariance = None
            self.global_posterior_precision_matrix = None
            self._covariance_time_cache = {}
        elif correction in self.COVARIANCE_GAUSSIAN_CORRECTIONS:
            if posterior_precision is not None:
                raise ValueError(
                    f"correction={correction!r} requires posterior_covariance, not "
                    "posterior_precision."
                )
            if posterior_covariance is None:
                raise ValueError(
                    "posterior_covariance is required for "
                    f"correction={correction!r}; reuse it from posterior sampling."
                )
            self.posterior_covariance = self._validate_covariance(
                posterior_covariance, num_observations=n_obs,
                name="posterior_covariance",
                dimension=len(self.covariance_features),
            ).to(device)
            self.posterior_mean = self._validate_posterior_mean(
                posterior_mean, num_observations=n_obs,
                dimension=len(self.covariance_features),
            )
            if self.posterior_mean is not None:
                self.posterior_mean = self.posterior_mean.to(device)
            self.posterior_precision_matrix = self._precision_from_covariance(
                self.posterior_covariance
            )
            self._covariance_time_cache = {}
            self.posterior_precision = None
            if correction == "Gauss_global_local":
                if global_posterior_mean is None:
                    raise ValueError(
                        "Gauss_global_local MAP refinement requires marginal "
                        "global moments saved during posterior sampling."
                    )
                self.global_posterior_mean = self._validate_posterior_mean(
                    global_posterior_mean, num_observations=n_obs,
                    dimension=len(hierarchy),
                ).to(device)
                self.global_posterior_covariance = self._validate_covariance(
                    global_posterior_covariance, num_observations=n_obs,
                    dimension=len(hierarchy),
                    name="global_posterior_covariance",
                ).to(device)
                self.global_posterior_precision_matrix = (
                    self._precision_from_covariance(
                        self.global_posterior_covariance
                    )
                )
            else:
                self.global_posterior_mean = None
                self.global_posterior_covariance = None
                self.global_posterior_precision_matrix = None
        elif correction in self.DIAGONAL_GAUSSIAN_CORRECTIONS:
            if posterior_precision is None:
                raise ValueError(
                    f"posterior_precision is required for correction='{correction}'; "
                    "reuse it from posterior sampling."
                )
            precision = torch.as_tensor(
                posterior_precision, dtype=torch.float32, device=device
            )
            self.posterior_precision = self._validate_precision(precision)
            if self.posterior_precision.shape[0] not in (1, n_obs):
                raise ValueError("posterior_precision must have one row or one row per observation.")
            if self.posterior_precision.shape[1] != len(hierarchy):
                raise ValueError("posterior_precision width must equal len(hierarchy).")
            self.posterior_covariance = None
            self.posterior_precision_matrix = None
            self.posterior_mean = None
            self.global_posterior_mean = None
            self.global_posterior_covariance = None
            self.global_posterior_precision_matrix = None
        else:
            if posterior_covariance is not None:
                raise ValueError(
                    "posterior_covariance is supported only for "
                    "correction='full_gaussian'."
                )
            self.posterior_precision = None
            self.posterior_covariance = None
            self.posterior_precision_matrix = None
            self.posterior_mean = None
            self.global_posterior_mean = None
            self.global_posterior_covariance = None
            self.global_posterior_precision_matrix = None

        data = data.to(device)
        candidates = candidates.to(device).clone()
        mask = mask.to(device)
        expanded_mask = mask.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        expanded_data = data.unsqueeze(1).expand_as(candidates)
        candidates = candidates * (1 - expanded_mask) + expanded_data * expanded_mask
        latent = 1 - expanded_mask
        shared_latent = torch.zeros_like(latent)
        shared_latent[:, :, hierarchy] = 1
        local_latent = latent - shared_latent
        has_local_latents = bool(torch.any(local_latent).item())
        return {
            "n_obs": n_obs,
            "nodes_size": nodes_size,
            "hierarchy": hierarchy,
            "candidates": candidates,
            "mask": mask,
            "expanded_mask": expanded_mask,
            "expanded_data": expanded_data,
            "latent": latent,
            "shared_latent": shared_latent,
            "local_latent": local_latent,
            "has_local_latents": has_local_latents,
            "min_iterations": min_iterations,
            "max_iterations": max_iterations,
        }

    def _map_schedule(self, sigma_start, timesteps, eps, device):
        """Log-spaced annealing schedule in noise scale, from sigma_start to eps."""
        one = torch.ones(1, device=device)
        lam_min = self.sde.lambda_t(eps * one).item()
        lam_max = self.sde.lambda_t(one).item()
        sigma_start = 1.0 if sigma_start is None else float(sigma_start)
        lam_hi = min(max(sigma_start, 2 * lam_min), lam_max)
        lams = torch.logspace(
            torch.log10(torch.tensor(lam_hi, device=device)),
            torch.log10(torch.tensor(lam_min, device=device)),
            int(timesteps), device=device,
        )
        times = self.sde.time_of_lambda(lams)
        return times, lams

    def _finalize_map(self, z, final_time, latent, expanded_mask, expanded_data,
                      hierarchy, device):
        """Rescale y -> x, resynchronize the shared block, restore observed columns."""
        alpha_final = self.sde.alpha_t(final_time).to(device)
        result = z * (alpha_final * latent + expanded_mask)
        result[:, :, hierarchy] = result[:1, :, hierarchy]
        result = result * (1 - expanded_mask) + expanded_data * expanded_mask
        return result.cpu()

    #############################################
    # ----- Arrow-Newton MAP refinement -----
    #############################################

    def newton_map_estimate(
            self, data, condition_mask, init=None, hierarchy=None,
            prior=None, local_prior=None,
            correction="gauss", posterior_precision=None,
            posterior_covariance=None, posterior_mean=None,
            global_posterior_mean=None, global_posterior_covariance=None,
            denoise_clamp=None, cfg_alpha=None, sigma_start=None,
            damping_at_data=1.0, damping_at_noise=None,
            timesteps=100, eps=1e-3, iterations_per_level=1,
            max_iterations_per_level=None, convergence_tol=1e-6,
            curvature="jacobian", curvature_refresh=1, trust_radius=3.0,
            max_backtracks=2, richardson=True,
            jacobian_refresh=1, jacobian_curvature_floor=1e-3, device="cpu"):
        """Hierarchical MAP by annealed arrow-structured Newton ascent.

        Same target as :meth:`map_estimate`, same arguments, same guards -- only
        the ascent step differs, so a comparison between the two isolates the
        optimizer. Both seek the stationary point ``s(z, t) = 0`` of the
        ``lambda``-smoothed log-posterior at each annealing level, so they share
        a fixed point; they differ only in how fast, and how reliably, they
        reach it.

        Why the step changes
        --------------------
        :meth:`map_estimate` takes ``z <- z + lambda**2 * s``, which is the
        Newton step for a Gaussian of variance ``lambda**2``. The curvature of
        the smoothed log-posterior is ``H = (Sigma_0 + lambda**2)^-1``, so that
        step undershoots by ``1 + Sigma_0 / lambda**2`` -- one to two orders of
        magnitude at the final levels for a local latent. Separately, the
        alternating global/local sweep is block Gauss-Seidel on an arrow-shaped
        system, whose contraction degrades as the observation count grows.

        This method solves the arrow system directly. With ``H_j`` the joint
        ``(g, l_j)`` curvature of observation ``j`` and ``H_prior`` the diffused
        prior curvature on the shared block, the exact Newton step is

            S      = sum_j M_j + (1 - n) H_prior,
                     M_j = H_j[gg] - H_j[gl] H_j[ll]^-1 H_j[lg]
            d_g    = S^-1 (s_g - sum_j H_j[gl] H_j[ll]^-1 s_l,j)
            d_l,j  = H_j[ll]^-1 (s_l,j - H_j[lg] d_g)

        i.e. the same block elimination as the score composition in
        ``New_Attempt/derivation.md``, applied to the Newton system instead.
        Both blocks move together and consistently, so the Gauss-Seidel lag
        disappears by construction. Cost is one ``H x H`` solve plus ``n`` small
        local solves per iteration -- negligible beside a network call.

        Args:
            curvature: ``"jacobian"`` (default) builds ``H_j = -grad(s_j)``
                exactly by forward-mode AD, needs no pilot covariance and works
                for every correction. ``"gaussian"`` reuses the already
                validated ``posterior_covariance`` as
                ``H_j = (Sigma_0,j + lambda**2 I)^-1``; it needs a correction
                whose covariance spans the shared *and* local coordinates
                (see ``SCHUR_GAUSSIAN_CORRECTIONS``).
                Pairing ``curvature="jacobian"`` with
                ``correction="gauss_jacobian"`` makes the whole estimate
                pilot-free: both the composition weights and the Newton
                curvature then come from the same network Jacobian, and nothing
                has to be carried over from a posterior-sampling run.
            jacobian_refresh, jacobian_curvature_floor: Passed through to
                ``correction="gauss_jacobian"``; see :meth:`sample`. They govern
                the *composition* weights and are independent of
                ``curvature_refresh``, which governs the Newton step.
            curvature_refresh: Recompute the curvature every this many
                annealing levels. Curvature varies slowly in ``lambda``, so the
                default of every level is already conservative.
            trust_radius: Cap on the max-abs step, in units of ``lambda``. This
                is what stops a lagging iterate from running out to the
                ``denoise_clamp`` boundary.
            max_backtracks: Halvings of the step allowed while the merit
                ``||s||**2`` fails to decrease. ``0`` disables the safeguard.
            richardson: Extrapolate the last two levels to ``lambda -> 0``.
                The fixed point at level ``lambda`` is the mode of the
                *smoothed* posterior, biased by ``O(lambda**2)`` for any skewed
                posterior; both iterates already exist, so this is free.
            denoise_clamp: Defaults to ``None`` here, unlike
                :meth:`map_estimate`. With a positive-definite curvature, a
                trust region and backtracking, the clamp is no longer
                load-bearing, and on a mode that legitimately sits in the prior
                tail it is a pure bias. Pass ``5.0`` to match the old default.

        After the call, :attr:`map_diagnostics` reports per-level convergence,
        backtracks, trust-region hits and the Richardson shift.
        """
        if curvature not in ("jacobian", "gaussian"):
            raise ValueError("curvature must be 'jacobian' or 'gaussian'.")
        if float(trust_radius) <= 0:
            raise ValueError("trust_radius must be positive.")
        if int(max_backtracks) < 0:
            raise ValueError("max_backtracks must be nonnegative.")
        if int(curvature_refresh) < 1:
            raise ValueError("curvature_refresh must be at least 1.")
        if max_iterations_per_level is None:
            max_iterations_per_level = max(int(iterations_per_level), 10)

        state = self._configure_map_estimate(
            data=data, condition_mask=condition_mask, init=init,
            hierarchy=hierarchy, prior=prior, local_prior=local_prior,
            correction=correction, posterior_precision=posterior_precision,
            posterior_covariance=posterior_covariance,
            posterior_mean=posterior_mean,
            global_posterior_mean=global_posterior_mean,
            global_posterior_covariance=global_posterior_covariance,
            denoise_clamp=denoise_clamp, cfg_alpha=cfg_alpha,
            damping_at_data=damping_at_data, damping_at_noise=damping_at_noise,
            timesteps=timesteps, iterations_per_level=iterations_per_level,
            max_iterations_per_level=max_iterations_per_level,
            convergence_tol=convergence_tol, device=device,
            jacobian_refresh=jacobian_refresh,
            jacobian_curvature_floor=jacobian_curvature_floor,
        )
        n_obs = state["n_obs"]
        hierarchy = state["hierarchy"]
        expanded_mask = state["expanded_mask"]
        expanded_data = state["expanded_data"]
        latent = state["latent"]
        shared_latent = state["shared_latent"]
        local_latent = state["local_latent"]
        min_iterations = state["min_iterations"]
        max_iterations = state["max_iterations"]

        local_indices = list(self.local_latent_indices)
        feature_order = list(hierarchy) + local_indices
        width = len(hierarchy)
        identity = torch.eye(width, dtype=torch.float64, device=device)
        prior_covariance = self.prior_covariance.to(
            device=device, dtype=torch.float64
        )

        times, lams = self._map_schedule(sigma_start, timesteps, eps, device)
        indices = torch.arange(n_obs, device=device)
        z = state["candidates"]

        diagnostics = {
            "levels": int(timesteps), "converged_levels": 0,
            "iterations": 0, "backtracks": 0, "unimproved_steps": 0,
            "trust_region_hits": 0, "richardson_shift": 0.0,
            "curvature": curvature,
        }
        blocks = None
        previous_z = None
        previous_lambda_squared = None

        for index in range(int(timesteps)):
            t = times[index].reshape(1, 1)
            lam = float(lams[index].item())
            lambda_squared = lam * lam
            if blocks is None or index % int(curvature_refresh) == 0:
                blocks = self._map_curvature_blocks(
                    z, t, expanded_mask, lambda_squared, curvature, feature_order,
                )
            prior_curvature = torch.linalg.inv(
                prior_covariance + lambda_squared * identity
            )

            score = self._get_score(z, t, expanded_mask, indices, cfg_alpha)
            merit = self._map_merit(score, shared_latent, local_latent)
            for iteration in range(max_iterations):
                update = self._arrow_newton_update(
                    score, blocks, prior_curvature, n_obs, width, hierarchy,
                    local_indices, lambda_squared, z,
                )
                update, factor = self._trust_region_scale(
                    update, trust_radius * lam
                )
                diagnostics["trust_region_hits"] += int((factor < 1.0).sum())

                scale = torch.ones(
                    update.shape[1], dtype=update.dtype, device=update.device
                )
                for attempt in range(int(max_backtracks) + 1):
                    candidate = self._apply_map_update(
                        z, update * scale.view(1, -1, 1), expanded_mask,
                        expanded_data, hierarchy,
                    )
                    candidate_score = self._get_score(
                        candidate, t, expanded_mask, indices, cfg_alpha
                    )
                    candidate_merit = self._map_merit(
                        candidate_score, shared_latent, local_latent
                    )
                    # A relative slack. Near the fixed point ||s|| -> 0, so a
                    # strict comparison backtracks on pure float noise: the
                    # safeguard exists to catch genuine divergence, not to
                    # enforce monotonicity to the last bit.
                    improved = candidate_merit <= merit * (1.0 + 1e-3)
                    if bool(improved.all()) or attempt == int(max_backtracks):
                        break
                    scale = torch.where(improved, scale, scale * 0.5)
                    diagnostics["backtracks"] += 1
                diagnostics["unimproved_steps"] += int((~improved).sum())

                applied = update * scale.view(1, -1, 1)
                z, score, merit = candidate, candidate_score, candidate_merit
                diagnostics["iterations"] += 1

                if iteration + 1 >= min_iterations:
                    state_scale = (z * latent).abs().max().clamp_min(1.0)
                    if applied.abs().max() <= float(convergence_tol) * state_scale:
                        diagnostics["converged_levels"] += 1
                        break

            if index == int(timesteps) - 2:
                previous_z = z.clone()
                previous_lambda_squared = lambda_squared

        if (richardson and previous_z is not None
                and previous_lambda_squared > lambda_squared):
            # z(lambda) = z* + c lambda^2  =>  z* = z + lambda^2 (z - z_prev)
            #                                        / (lambda_prev^2 - lambda^2)
            weight = lambda_squared / (previous_lambda_squared - lambda_squared)
            correction_step, _ = self._trust_region_scale(
                (z - previous_z) * weight, trust_radius * lam
            )
            diagnostics["richardson_shift"] = float(correction_step.abs().max())
            z = self._apply_map_update(
                z, correction_step, expanded_mask, expanded_data, hierarchy,
            )

        diagnostics["score_network_calls"] = int(self.score_network_calls)
        self.map_diagnostics = diagnostics
        return self._finalize_map(
            z, times[-1], latent, expanded_mask, expanded_data, hierarchy, device,
        )

    def laplace_initial_state(
            self, data, condition_mask, num_samples, start_lambda, hierarchy,
            prior=None, local_prior=None, correction="gauss_jacobian_FD",
            denoise_clamp=5.0, newton_levels=8, newton_iterations=3,
            sigma_start=None, generator=None, device="cpu", fit=None):
        """Draws from a Laplace approximation of p_lambda(theta | X) at ``start_lambda``.

        Feed the result to :meth:`sample` as ``init_state`` together with the
        same ``start_lambda``; the reverse diffusion then skips every level
        above it. ``fit`` reuses a :meth:`laplace_fit` result (e.g. across draw
        batches of one dataset); otherwise one is computed and kept in
        :attr:`laplace_last_fit`. Returns ``(observations, num_samples, nodes)``
        in y = x/alpha coordinates at ``start_lambda``.
        """
        if fit is None:
            fit = self.laplace_fit(
                data, condition_mask, start_lambda, hierarchy, prior=prior,
                local_prior=local_prior, correction=correction,
                denoise_clamp=denoise_clamp, newton_levels=newton_levels,
                newton_iterations=newton_iterations, sigma_start=sigma_start,
                device=device,
            )
        return self.laplace_draw(fit, num_samples, generator=generator)

    def laplace_fit(
            self, data, condition_mask, start_lambda, hierarchy, prior=None,
            local_prior=None, correction="gauss_jacobian_FD", denoise_clamp=5.0,
            newton_levels=8, newton_iterations=3, sigma_start=None, device="cpu"):
        """Laplace approximation of the composed diffused posterior at ``start_lambda``.

        1. :meth:`newton_map_estimate` (same correction, so the same composed
           score) annealed down to ``start_lambda`` only, no Richardson step:
           the mode m of the composed diffused posterior at that level.
        2. One Jacobian at m gives each row's lambda-free information
           K_j = Lambda_j - lambda^-2 I over (g, l_j) -- the quantity the
           Jacobian corrections cache. Composed like the rule composes it:
           S = P_prior + sum_j [K_j,gg - K_j,gl K_j,ll^-1 K_j,lg - P_ref]^+,
           with P_ref the training prior's precision if one is declared.

        :meth:`laplace_draw` then samples theta_0 - m ~ N(0, S^-1) on g, l_j | g
        from the conditional of K_j, plus the diffusion noise lambda * eps on
        every latent coordinate (shared on g). Exact when the composed posterior
        is Gaussian. The fit touches one state, not the draws; its network rows
        are in the returned ``diagnostics`` (and :attr:`laplace_diagnostics`).
        """
        lam = float(start_lambda)
        if not (math.isfinite(lam) and lam > 0.0):
            raise ValueError("start_lambda must be finite and positive.")
        if correction not in self.JACOBIAN_GAUSSIAN_CORRECTIONS:
            raise ValueError("laplace_fit needs a Jacobian correction.")
        t_s = self.sde.time_of_lambda(
            torch.tensor([lam], dtype=torch.float32, device=device)
        ).reshape(1, 1)
        if sigma_start is None:
            sigma_start = max(1.0, 3.0 * lam)
        # sample() takes only the observed values; the MAP routine wants the
        # full node vector with the observed columns filled in.
        data = torch.as_tensor(data, dtype=torch.float32).cpu()
        flat_mask = torch.as_tensor(condition_mask, dtype=torch.float32).cpu()
        if flat_mask.dim() == 1 and data.shape[-1] != flat_mask.numel():
            full = torch.zeros(data.shape[0], flat_mask.numel())
            full[:, flat_mask == 1] = data
            data = full
        mode = self.newton_map_estimate(
            data, flat_mask, hierarchy=hierarchy, prior=prior,
            local_prior=local_prior, correction=correction,
            denoise_clamp=denoise_clamp, timesteps=int(newton_levels),
            max_iterations_per_level=int(newton_iterations),
            convergence_tol=1e-3, max_backtracks=1,
            eps=float(t_s.reshape(())), sigma_start=sigma_start,
            richardson=False, curvature="jacobian", device=device,
        ).to(device)
        n_obs, _, nodes = mode.shape
        mask = flat_mask.to(device=device, dtype=mode.dtype)
        mask = mask.expand(n_obs, nodes).unsqueeze(1)
        latent = 1 - mask
        alpha = self.sde.alpha_t(t_s).to(device)
        y_mode = mode * (latent / alpha + mask)

        variance = lam * lam
        joint = self._jacobian_joint_precision(y_mode, t_s, mask, variance)[:, 0]
        h = len(hierarchy)
        order = list(self.full_gaussian_features)
        dimension = len(order)
        eye = torch.eye(dimension, dtype=torch.float64, device=device)
        information = joint - eye / variance
        information = 0.5 * (information + information.mT)
        values, vectors = torch.linalg.eigh(information)
        floor = 1e-8 * values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        information = (vectors * values.clamp_min(0.0).unsqueeze(-2)) @ vectors.mT

        k_gg = information[:, :h, :h]
        regression, local_factor = None, None
        if dimension > h:
            k_gl = information[:, :h, h:]
            k_ll = information[:, h:, h:]
            values, vectors = torch.linalg.eigh(k_ll)
            values = values.clamp_min(float(floor.min()))
            k_ll = (vectors * values.unsqueeze(-2)) @ vectors.mT
            regression = torch.linalg.solve(k_ll, k_gl.mT)          # K_ll^-1 K_lg
            marginal = k_gg - k_gl @ regression
            local_factor = torch.linalg.cholesky(k_ll)
        else:
            marginal = k_gg
        prior_precision = self.prior_precision_matrix.to(device=device,
                                                         dtype=torch.float64)
        reference = prior_precision
        if self.training_prior is not None:
            reference = torch.diag(1.0 / self.training_prior[1].to(
                device=device, dtype=torch.float64) ** 2)
        gain = 0.5 * ((marginal - reference) + (marginal - reference).mT)
        values, vectors = torch.linalg.eigh(gain)
        gain = (vectors * values.clamp_min(0.0).unsqueeze(-2)) @ vectors.mT
        composed = prior_precision + gain.sum(dim=0)
        diagnostics = {
            "start_lambda": lam,
            "row_samples": int(getattr(self, "evaluated_row_samples", 0)),
            "jacobian_row_samples": int(getattr(self, "jacobian_row_samples", 0)),
            "global_sd": torch.linalg.inv(composed).diagonal().sqrt().tolist(),
            "mode_global": y_mode[0, 0, hierarchy].tolist(),
            "map": self.map_diagnostics,
        }
        self.laplace_diagnostics = diagnostics
        fit = {"lambda": lam, "mode": y_mode, "order": order, "hierarchy": list(hierarchy),
               "global_factor": torch.linalg.cholesky(composed),
               "regression": regression, "local_factor": local_factor,
               "device": device, "diagnostics": diagnostics}
        self.laplace_last_fit = fit
        return fit

    @staticmethod
    def laplace_draw(fit, num_samples, generator=None):
        """``num_samples`` draws at ``fit['lambda']`` from a :meth:`laplace_fit` result."""
        device = fit["device"]
        y_mode, order, hierarchy = fit["mode"], fit["order"], fit["hierarchy"]
        n_obs, _, nodes = y_mode.shape
        h, dimension, samples = len(hierarchy), len(order), int(num_samples)

        def solve(factor, z):
            # x = L^-T z has covariance (L L^T)^-1.
            return torch.linalg.solve_triangular(
                factor.mT, z.unsqueeze(-1), upper=True).squeeze(-1)

        offset = torch.zeros(n_obs, samples, dimension, dtype=torch.float64,
                             device=device)
        z = torch.randn(samples, h, dtype=torch.float64, device=device,
                        generator=generator)
        xi_g = solve(fit["global_factor"].expand(samples, h, h), z)
        offset[:, :, :h] = xi_g.unsqueeze(0)
        if fit["regression"] is not None:
            mean_shift = -torch.einsum("nlh,sh->nsl", fit["regression"], xi_g)
            z = torch.randn(n_obs, samples, dimension - h, dtype=torch.float64,
                            device=device, generator=generator)
            local = solve(fit["local_factor"].unsqueeze(1).expand(
                -1, samples, -1, -1), z)
            offset[:, :, h:] = mean_shift + local
        noise = torch.randn(n_obs, samples, dimension, dtype=torch.float64,
                            device=device, generator=generator)
        noise[:, :, :h] = noise[:1, :, :h]
        offset = offset + fit["lambda"] * noise

        state = y_mode.expand(n_obs, samples, nodes).clone().to(torch.float64)
        state[:, :, order] = state[:, :, order] + offset
        state = state.to(y_mode.dtype)
        state[:, :, hierarchy] = state[:1, :, hierarchy]
        return state

    def _raw_row_scores(self, z, t, condition_mask):
        """Uncomposed per-observation network score in y = x/alpha coordinates.

        Mirrors the network call inside :meth:`_get_score` but stops before the
        composition, and stays outside ``no_grad`` so it can be differentiated.
        """
        alpha = self.sde.alpha_t(t).to(z.device)
        rows, samples, features = z.shape
        x = z * (alpha * (1 - condition_mask) + condition_mask)
        scores = self.SBIm.model(
            x=x.reshape(rows * samples, features), t=t,
            c=condition_mask.reshape(rows * samples, features),
        )
        scores = self.SBIm.output_scale_function(t, scores)
        self.score_network_calls += 1
        self.evaluated_subject_rows += int(rows)
        self.evaluated_row_samples = getattr(self, "evaluated_row_samples", 0) + int(rows) * int(samples)
        return (alpha * scores).reshape(rows, samples, features)

    def _row_score_jacobian(self, z, t, condition_mask, feature_order):
        """Exact per-observation Jacobian grad(s_j) on the latent block, via JVPs.

        Observation rows are independent inside the network, so a tangent that
        perturbs one coordinate in *every* row returns that column of *every*
        row's Jacobian in a single forward-mode pass. ``len(feature_order)``
        JVPs therefore give the whole block, and that count is the size of the
        latent state (single digits here), not the number of observations.

        The shared coordinate is synchronized across rows, so its tangent is
        legitimately nonzero everywhere; a local tangent is too, but rows do not
        mix, so row ``j`` still receives exactly ``d s_j / d l_j``.

        For ``correction="gauss_jacobian_FD"`` the columns come from
        :meth:`_row_score_jacobian_fd` instead.
        """
        if (getattr(self, "correction", None)
                in self.FINITE_DIFFERENCE_JACOBIAN_CORRECTIONS):
            return self._row_score_jacobian_fd(
                z, t, condition_mask, feature_order
            )
        try:
            from torch.func import jvp
        except ImportError as error:                       # pragma: no cover
            raise RuntimeError(
                "curvature='jacobian' needs torch.func (PyTorch >= 2.0); "
                "pass curvature='gaussian' instead."
            ) from error

        def raw_scores(state):
            return self._raw_row_scores(state, t, condition_mask)

        # A JVP costs several forward passes; counted here as one per column.
        self.jacobian_row_samples = (getattr(self, "jacobian_row_samples", 0)
                                     + z.shape[0] * z.shape[1] * len(feature_order))
        columns = []
        # The fused scaled-dot-product-attention kernels have no forward-mode
        # AD rule, so the transformer raises NotImplementedError under jvp
        # unless attention is routed through the math backend. Only these D
        # tangent passes pay the (modest) cost; ordinary score evaluation still
        # gets the fused kernel.
        with self._forward_ad_attention():
            for feature in feature_order:
                tangent = torch.zeros_like(z)
                tangent[:, :, feature] = 1.0
                _, column = jvp(raw_scores, (z,), (tangent,))
                # Nothing differentiates *through* the curvature, and the jvp
                # runs with grad enabled, so keeping the tangent attached would
                # grow a graph across every annealing level / diffusion step.
                columns.append(
                    column[:, :, feature_order].detach().to(torch.float64)
                )
        return torch.stack(columns, dim=-1)

    # Upper bound on (observation rows x samples x copies) per network call in
    # the finite-difference Jacobian; bounds its peak memory to that of one
    # ordinary forward pass over this many rows.
    FD_JACOBIAN_MAX_ROWS = 16384

    def _row_score_jacobian_fd(self, z, t, condition_mask, feature_order):
        """Forward-difference counterpart of :meth:`_row_score_jacobian`.

        Column ``k`` is ``(s(z + h e_k) - s(z)) / h`` with ``h =
        jacobian_fd_step * lambda_t``. The perturbation of coordinate ``k`` is
        applied in every observation row at once, exactly like the JVP tangent,
        so a shared coordinate moves synchronously and a local one only ever
        reaches its own row. The base point and all D perturbed copies are
        stacked along the sample axis and evaluated in as few no-grad network
        calls as ``FD_JACOBIAN_MAX_ROWS`` allows (usually one), which keeps the
        fused attention kernel and is ~4-5x cheaper than D JVPs.

        The step is accurate because a lambda-smoothed score is smooth on the
        scale lambda_t; a network much rougher than that (e.g. untrained with
        large random weights) needs a smaller ``jacobian_fd_step``.
        """
        rows, samples, features = z.shape
        lam = float(self.sde.lambda_t(t).reshape(-1)[0])
        step = float(self.jacobian_fd_step) * lam
        copies = [z]
        for feature in feature_order:
            shifted = z.clone()
            shifted[:, :, feature] += step
            copies.append(shifted)
        self.jacobian_row_samples = (getattr(self, "jacobian_row_samples", 0)
                                     + rows * samples * len(copies))
        per_call = max(1, self.FD_JACOBIAN_MAX_ROWS // max(1, rows * samples))
        mask_per_sample = (
            condition_mask.dim() == 3 and condition_mask.shape[1] == samples
        )
        outputs = []
        with torch.no_grad():
            for start in range(0, len(copies), per_call):
                chunk = copies[start:start + per_call]
                mask = (
                    condition_mask.repeat(1, len(chunk), 1) if mask_per_sample
                    else condition_mask
                )
                scores = self._raw_row_scores(torch.cat(chunk, dim=1), t, mask)
                outputs.append(
                    scores.reshape(rows, len(chunk), samples, features)
                )
        scores = torch.cat(outputs, dim=1)[..., feature_order].to(torch.float64)
        columns = (scores[:, 1:] - scores[:, :1]) / step
        # (rows, D_in, samples, D_out) -> (rows, samples, D_out, D_in), the
        # layout `_row_score_jacobian` returns.
        return columns.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _forward_ad_attention():
        """Context manager selecting an attention backend that supports jvp."""
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:                                # pragma: no cover
            from contextlib import nullcontext
            return nullcontext()
        return sdpa_kernel([SDPBackend.MATH])

    def _map_curvature_blocks(self, z, t, condition_mask, lambda_squared,
                              curvature, feature_order):
        """Per-observation curvature -grad(s_j) over the latent block.

        Returns shape ``(observations, candidates, D, D)`` in float64, with the
        coordinate order ``hierarchy + local_latent_indices``.
        """
        dimension = len(feature_order)
        if curvature == "jacobian":
            jacobian = self._row_score_jacobian(
                z, t, condition_mask, feature_order
            )
            blocks = -jacobian
        else:
            if self.posterior_covariance is None:
                raise ValueError(
                    "curvature='gaussian' requires posterior_covariance; pass "
                    "it, or use curvature='jacobian' which needs no pilot "
                    "covariance."
                )
            if list(self.covariance_features) != list(feature_order):
                raise ValueError(
                    "curvature='gaussian' needs a covariance spanning the "
                    "shared and local coordinates, but correction="
                    f"{self.correction!r} estimates it over "
                    f"{list(self.covariance_features)} while the latent block "
                    f"is {list(feature_order)}. Use a correction in "
                    "SCHUR_GAUSSIAN_CORRECTIONS, or curvature='jacobian'."
                )
            covariance = self.posterior_covariance.to(
                device=z.device, dtype=torch.float64
            )
            eye = torch.eye(dimension, dtype=torch.float64, device=z.device)
            blocks = torch.linalg.inv(covariance + lambda_squared * eye)
            blocks = blocks.unsqueeze(1).expand(-1, z.shape[1], -1, -1)
            if blocks.shape[0] == 1:
                blocks = blocks.expand(z.shape[0], -1, -1, -1)

        # A lambda-smoothed density obeys 0 <= H <= lambda^-2 I exactly, since
        # H = (Sigma_0 + lambda^2 I)^-1. Imposing it costs one small eigh and
        # bounds any damage an imperfect score can do to the preconditioner.
        blocks = 0.5 * (blocks + blocks.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(blocks)
        eigenvalues = eigenvalues.clamp(min=0.0, max=1.0 / lambda_squared)
        return (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.mT

    def _project_map_information(self, marginal, prior_curvature):
        """Project each observation's curvature information onto the PSD cone.

        The same argument as :meth:`_project_information`, for curvature rather
        than backward precision: writing ``H_j = H_prior + I_j`` makes

            S = sum_j H_j + (1 - n) H_prior = H_prior + sum_j I_j

        an identity, so ``S`` is positive definite for every ``n`` as soon as
        each ``I_j`` is positive semi-definite -- which every real posterior
        satisfies, since one more observation cannot leave you less certain
        about the shared parameters than the prior alone. An estimated ``I_j``
        with a negative eigenvalue is outside the model class, not a hard case.

        This makes the Newton direction an ascent direction unconditionally, so
        no eigenvalue floor on ``S`` is load-bearing for correctness.
        """
        information = marginal - prior_curvature
        information = 0.5 * (information + information.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(information)
        eigenvalues = eigenvalues.clamp_min(0.0)
        return (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.mT

    def _arrow_newton_update(self, score, blocks, prior_curvature, n_obs, width,
                             hierarchy, local_indices, lambda_squared, z):
        """Solve the arrow-structured Newton system for shared and local blocks."""
        gradient_global = score[:1, :, hierarchy].to(torch.float64)
        block_gg = blocks[..., :width, :width]

        if local_indices:
            block_gl = blocks[..., :width, width:]
            block_lg = blocks[..., width:, :width]
            block_ll = blocks[..., width:, width:]
            gradient_local = score[:, :, local_indices].to(torch.float64)

            # Floor the local curvature so the elimination stays well posed.
            # The bound is non-binding whenever the local posterior is at all
            # informative; it only catches a degenerate estimated block.
            block_ll = 0.5 * (block_ll + block_ll.mT)
            values, vectors = torch.linalg.eigh(block_ll)
            values = values.clamp(
                min=self.pd_floor / lambda_squared, max=1.0 / lambda_squared
            )
            block_ll = (vectors * values.unsqueeze(-2)) @ vectors.mT

            solved_cross = torch.linalg.solve(block_ll, block_lg)
            solved_gradient = torch.linalg.solve(
                block_ll, gradient_local.unsqueeze(-1)
            )
            marginal = block_gg - block_gl @ solved_cross
            numerator = gradient_global[0] - (
                block_gl @ solved_gradient
            ).squeeze(-1).sum(dim=0)
        else:
            marginal = block_gg
            numerator = gradient_global[0]

        composed = prior_curvature + self._project_map_information(
            marginal, prior_curvature
        ).sum(dim=0)
        delta_global = torch.linalg.solve(
            composed, numerator.unsqueeze(-1)
        ).squeeze(-1)

        update = torch.zeros(z.shape, dtype=torch.float64, device=z.device)
        update[:, :, hierarchy] = delta_global.unsqueeze(0).expand(n_obs, -1, -1)
        if local_indices:
            update[:, :, local_indices] = solved_gradient.squeeze(-1) - (
                solved_cross @ delta_global.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1)
        return update.to(z.dtype)

    @staticmethod
    def _map_merit(score, shared_latent, local_latent):
        """Per-candidate ||s||^2 over the latent block; shared block counted once."""
        shared = (score[:1] * shared_latent[:1]).square().sum(dim=(0, 2))
        return shared + (score * local_latent).square().sum(dim=(0, 2))

    @staticmethod
    def _trust_region_scale(update, radius):
        """Cap each candidate's max-abs step at `radius`; return the step and factor."""
        magnitude = update.abs().amax(dim=2).amax(dim=0)
        factor = torch.clamp(
            radius / magnitude.clamp_min(torch.finfo(update.dtype).tiny), max=1.0
        )
        return update * factor.view(1, -1, 1), factor

    @staticmethod
    def _apply_map_update(z, update, expanded_mask, expanded_data, hierarchy):
        """Apply a latent-only update, resynchronize shared coords, restore data."""
        moved = z + update * (1 - expanded_mask)
        moved[:, :, hierarchy] = moved[:1, :, hierarchy]
        return moved * (1 - expanded_mask) + expanded_data * expanded_mask

    def _make_sampling_schedule(self, device):
        """Build the reverse schedule; covariance modes always run through eps."""
        one = torch.ones(1, device=device)
        sigma_max = self.sde.lambda_t(one)
        sigma_min = self.sde.lambda_t(self.eps * one)
        if self.correction in self.DIAGONAL_GAUSSIAN_CORRECTIONS:
            n = self.num_observations
            prior_precision = 1.0 / self.prior_std**2
            if self.posterior_precision.shape[0] == 1:
                posterior_precision = n * self.posterior_precision[0]
            else:
                posterior_precision = self.posterior_precision.sum(dim=0)
            composed_precision = (
                (1 - n) * prior_precision + posterior_precision
            )
            sigma_stop = composed_precision.reciprocal().sqrt().min()
            sigma_min = torch.maximum(
                sigma_min, sigma_stop.reshape(1).to(device)
            )
        sigmas = torch.logspace(
            torch.log10(sigma_max).item(), torch.log10(sigma_min).item(),
            self.timesteps, device=device,
        )
        start = getattr(self, "start_lambda", None)
        if start is not None:
            # Warm start: keep the full schedule's levels below start_lambda,
            # so the step density there is exactly that of a cold run.
            below = sigmas[sigmas < start]
            sigmas = torch.cat([torch.tensor([start], device=device,
                                             dtype=sigmas.dtype), below])
            self.timesteps = int(sigmas.numel())
        return self.sde.time_of_lambda(sigmas)

    def _sample_loop(self, rank, data, condition_mask, num_samples, result_dict=None):
        # Set rank
        self.rank = rank
        self.verbose = self.verbose if self.rank == 0 else False

        # Set distributed parameters
        if self.world_size > 1:
            self._ddp_setup(rank, self.world_size)
        else:
            self.model = self.SBIm.model.to(self.device)

        # Check data structure
        data_loader, self.num_observations = self._check_data_structure(data, condition_mask)

        # Move prior (and precision estimates) to device
        self.prior_mean = self.prior_mean.to(self.device)
        self.prior_std = self.prior_std.to(self.device)
        self.prior_covariance = self.prior_covariance.to(self.device)
        self.prior_precision_matrix = self.prior_precision_matrix.to(self.device)
        if self.local_prior_mean is not None:
            self.local_prior_mean = self.local_prior_mean.to(self.device)
            self.local_prior_std = self.local_prior_std.to(self.device)
        if self.posterior_precision is not None:
            self.posterior_precision = self.posterior_precision.to(self.device)
        if self.posterior_covariance is not None:
            self.posterior_covariance = self.posterior_covariance.to(self.device)
            self.posterior_precision_matrix = self.posterior_precision_matrix.to(
                self.device
            )
        if self.posterior_mean is not None:
            self.posterior_mean = self.posterior_mean.to(self.device)
        if self.global_posterior_mean is not None:
            self.global_posterior_mean = self.global_posterior_mean.to(self.device)
            self.global_posterior_covariance = self.global_posterior_covariance.to(
                self.device
            )
            self.global_posterior_precision_matrix = (
                self.global_posterior_precision_matrix.to(self.device)
            )

        self.timesteps_list = self._make_sampling_schedule(self.device)

        # Loop over data samples. A terminal-corrector sweep shares this one
        # predictor trajectory and branches only at its final state.
        if self.terminal_corrector_counts and self.world_size > 1:
            raise NotImplementedError("terminal_corrector_counts requires world_size=1.")
        all_samples = []
        sweep_samples = {count: [] for count in self.terminal_corrector_counts}
        indices = []
        self.terminal_corrector_seconds = {count: 0.0 for count in self.terminal_corrector_counts}
        for batch in data_loader:
            data_batch, condition_mask_batch, idx = self._prepare_data(batch, num_samples, self.device)
            if getattr(self, "init_state", None) is not None:
                data_batch = self._warm_start_state(data_batch, condition_mask_batch)
            else:
                data_batch = self._initial_sample(data_batch, condition_mask_batch)

            if self.method == "euler":
                samples = self._basic_sampler(data_batch, condition_mask_batch, idx)
            elif self.method == "dpm":
                samples = self._dpm_sampler(data_batch, condition_mask_batch, idx,
                                            order=self.order, snr=self.snr, corrector_steps_interval=self.corrector_steps_interval,
                                            corrector_steps=self.corrector_steps, final_corrector_steps=self.final_corrector_steps,
                                            terminal_corrector_steps=self.terminal_corrector_steps)
            elif self.method == "langevin":
                samples = self._langevin_sampler(data_batch, condition_mask_batch, idx,
                                                 snr=self.snr, steps_per_level=self.corrector_steps)
            elif self.method == "adaptive":
                samples = self._adaptive_sampler(data_batch, condition_mask_batch, idx)
            else:
                raise ValueError(f"Sampling method {self.method} not recognized.")

            variants = {None: samples}
            if self.terminal_corrector_counts:
                variants = {}
                terminal_time = self.timesteps_list[-1].reshape(-1, 1)
                previous_count = 0
                cumulative_seconds = 0.0
                for count in self.terminal_corrector_counts:
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.synchronize(self.device)
                    started = datetime.datetime.now().timestamp()
                    samples = self._corrector_step(
                        samples, terminal_time, condition_mask_batch, idx,
                        count - previous_count, self.snr, self.cfg_alpha,
                    )
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.synchronize(self.device)
                    cumulative_seconds += datetime.datetime.now().timestamp() - started
                    self.terminal_corrector_seconds[count] += cumulative_seconds
                    variants[count] = samples.detach().clone()
                    previous_count = count

            for count, result in variants.items():
                if self.correction in self.DIAGONAL_GAUSSIAN_CORRECTIONS:
                    result = self._final_denoise(result, condition_mask_batch, idx)
                alpha_end = self.sde.alpha_t(self.timesteps_list[-1]).to(result.device)
                result = result * (alpha_end * (1 - condition_mask_batch) + condition_mask_batch)
                if not torch.all(torch.isfinite(result)):
                    raise RuntimeError(f"Sampling with correction={self.correction!r} produced non-finite values.")
                result[:, :, self.hierarchy] = result[:1, :, self.hierarchy]
                if count is None:
                    all_samples.append(result)
                else:
                    sweep_samples[count].append(result)
            indices.append(idx)

        if self.world_size > 1:
            dist.barrier()
            self._gather_samples(all_samples, indices, result_dict)
            dist.barrier()
            dist.destroy_process_group()
        else:
            stats = dict(self.solver_stats or {})
            stats["score_network_calls"] = int(self.score_network_calls)
            stats["evaluated_subject_rows"] = int(self.evaluated_subject_rows)
            stats["evaluated_row_samples"] = int(getattr(self, "evaluated_row_samples", 0))
            stats["jacobian_row_samples"] = int(getattr(self, "jacobian_row_samples", 0))
            stats["jacobian_refreshes"] = int(getattr(self, "_jacobian_refreshes", 0))
            stats["jacobian_secant_updates"] = int(getattr(self, "_secant_updates", 0))
            stats["jacobian_secant_skipped"] = int(getattr(self, "_secant_skipped", 0))
            stats.setdefault("score_evaluations", int(self.score_network_calls))
            self.solver_stats = stats
            log = os.environ.get("COMPASS_PROJECTION_LOG")
            if log and self.correction in self.COVARIANCE_GAUSSIAN_CORRECTIONS | {"arruda_schur_gauss"}:
                import json
                diagnostics = {k: v for k, v in self.covariance_diagnostics.items() if k != "last_repair"}
                with open(log, "a") as handle:
                    handle.write(json.dumps({
                        "label": os.environ.get("COMPASS_RUN_LABEL", ""), "correction": self.correction,
                        "observations": int(data.shape[0]), "num_samples": int(num_samples),
                        "churn_eta": self.churn_eta, "corrector_steps": getattr(self, "corrector_steps", None),
                        "corrector_steps_interval": getattr(self, "corrector_steps_interval", None),
                        "sampler_override": self.sampler_override, "score_network_calls": self.score_network_calls,
                        **diagnostics}) + "\n")
            if self.terminal_corrector_counts:
                return {count: torch.cat(values, dim=0) for count, values in sweep_samples.items()}
            return torch.cat(all_samples, dim=0)

    #############################################
    # ----- Prior & Precision Setup -----
    #############################################

    def _resolve_prior(self, prior):
        """Resolve the prior to mean, marginal std, and covariance tensors.

        Accepts either the historical ``prior=(mean, scale)`` tuple or a prior
        *object* exposing ``score(theta_y, t, lambda_t)``, ``.mean`` and
        ``.covariance`` -- see :mod:`compass.priors`.  The tuple path below is
        untouched, so existing callers are bit-for-bit unaffected; an object
        prior is additionally stored on ``self.prior_object`` and its exact
        diffused score replaces the Gaussian one in
        :meth:`_diffused_gaussian_prior_score`.

        A non-Gaussian prior matters here because the composition subtracts
        ``(n-1)`` copies of the prior score: a prior that is wrong about where
        its mass sits is amplified n-fold, which for a multimodal prior means
        the composed field inherits a mode the target does not have.
        """
        n_hierarchy = len(self.hierarchy)
        from .priors import resolve_prior_object
        prior_object = resolve_prior_object(prior, n_hierarchy)
        self.prior_object = prior_object
        if prior_object is not None:
            mean = prior_object.mean.to(torch.float32).flatten()
            covariance = prior_object.covariance.to(torch.float32)
            std = torch.diagonal(covariance).sqrt()
            return mean, std, covariance
        if prior is None:
            print("WARNING: No prior provided for compositional score modeling. "
                  "Assuming a standard normal prior N(0, 1) over the shared parameters. "
                  "Pass prior=(mean, std) if this does not match your model.")
            mean = torch.zeros(n_hierarchy)
            std = torch.ones(n_hierarchy)
            return mean, std, torch.eye(n_hierarchy)

        mean, scale = prior
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        if mean.numel() == 1:
            mean = mean.repeat(n_hierarchy)
        if mean.numel() != n_hierarchy:
            raise ValueError(
                f"Prior mean must have length {n_hierarchy} (= len(hierarchy)), "
                f"got {mean.numel()}."
            )

        scale = torch.as_tensor(scale, dtype=torch.float32)
        if scale.dim() == 2:
            covariance = self._validate_covariance(
                scale, num_observations=1, name="prior covariance"
            )[0].to(torch.float32)
            std = torch.diagonal(covariance).sqrt().to(torch.float32)
        else:
            std = scale.flatten()
            if std.numel() == 1:
                std = std.repeat(n_hierarchy)
            if std.numel() != n_hierarchy:
                raise ValueError(
                    f"Prior std must have length {n_hierarchy} (= len(hierarchy)), "
                    f"got {std.numel()}."
                )
            if not torch.all(torch.isfinite(std)) or torch.any(std <= 0):
                raise ValueError("Prior standard deviations must be finite and positive.")
            covariance = torch.diag(std.square())
        return mean, std, covariance

    def _resolve_local_prior(self, local_prior):
        """Resolve an optional Gaussian prior over the row-local latent coordinates."""
        if local_prior is None:
            return None, None
        width = len(self.local_latent_indices)
        if width == 0:
            raise ValueError(
                "local_prior was supplied but every latent coordinate is shared."
            )
        mean, std = local_prior
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(std, dtype=torch.float32).flatten()
        if mean.numel() == 1:
            mean = mean.repeat(width)
        if std.numel() == 1:
            std = std.repeat(width)
        if mean.numel() != width or std.numel() != width:
            raise ValueError(
                f"local_prior mean and std must have length {width} "
                "(= the number of local latent coordinates)."
            )
        if not torch.all(torch.isfinite(mean)):
            raise ValueError("local_prior mean must be finite.")
        if not torch.all(torch.isfinite(std)) or torch.any(std <= 0):
            raise ValueError(
                "local_prior standard deviations must be finite and positive."
            )
        return mean, std

    def _validate_covariance(
            self, covariance, num_observations, name, dimension=None):
        """Validate and normalize covariance input to shape (1 or N, H, H)."""
        covariance = torch.as_tensor(covariance, dtype=torch.float64)
        h = len(self.hierarchy) if dimension is None else int(dimension)
        if covariance.dim() == 2:
            covariance = covariance.unsqueeze(0)
        if (
            covariance.dim() != 3
            or covariance.shape[-2:] != (h, h)
            or covariance.shape[0] not in (1, int(num_observations))
        ):
            raise ValueError(
                f"{name} must have shape ({h}, {h}) or "
                f"({num_observations}, {h}, {h}), got {tuple(covariance.shape)}."
            )
        if not torch.all(torch.isfinite(covariance)):
            raise ValueError(f"{name} must contain only finite values.")
        if not torch.allclose(
            covariance, covariance.mT, rtol=1e-5, atol=1e-6
        ):
            raise ValueError(f"{name} must be symmetric.")
        _, info = torch.linalg.cholesky_ex(covariance)
        if torch.any(info != 0):
            raise ValueError(
                f"{name} must be positive definite. For an automatically "
                "estimated covariance, increase precision_est_samples so the "
                "empirical covariance is nonsingular."
            )
        return covariance

    def _validate_posterior_mean(
            self, mean, num_observations, dimension):
        """Validate optional Gaussian means to shape (1 or N, H)."""
        if mean is None:
            return None
        mean = torch.as_tensor(mean, dtype=torch.float64)
        h = int(dimension)
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
        if (mean.dim() != 2 or mean.shape[1] != h
                or mean.shape[0] not in (1, int(num_observations))):
            raise ValueError(
                f"posterior_mean must have shape ({h},) or "
                f"({num_observations}, {h}), got {tuple(mean.shape)}."
            )
        if not torch.all(torch.isfinite(mean)):
            raise ValueError("posterior_mean must contain only finite values.")
        return mean

    def _moment_projected_scores(self, scores, x, var_t):
        """Replace selected learned scores by scores of the supplied Gaussian moments."""
        mean = getattr(self, "posterior_mean", None)
        if mean is None:
            return scores
        features = self.covariance_features
        covariance = self.posterior_covariance.to(
            device=scores.device, dtype=torch.float64
        )
        mean = mean.to(device=scores.device, dtype=torch.float64)
        rows = scores.shape[0]
        if covariance.shape[0] == 1:
            covariance = covariance.expand(rows, -1, -1)
        if mean.shape[0] == 1:
            mean = mean.expand(rows, -1)
        identity = torch.eye(
            len(features), dtype=torch.float64, device=scores.device
        )
        covariance_t = covariance + torch.as_tensor(
            var_t, dtype=torch.float64, device=scores.device
        ).reshape(()) * identity
        delta = x[:, :, features].to(torch.float64) - mean[:, None, :]
        projected = -torch.linalg.solve(
            covariance_t[:, None, :, :], delta.unsqueeze(-1)
        ).squeeze(-1)
        scores = scores.clone()
        scores[:, :, features] = projected.to(scores.dtype)
        return scores

    def _reset_covariance_diagnostics(self):
        """Reset precision-repair counters without changing the sample return API."""
        self._pd_evaluation_count = 0
        self._pd_repair_count = 0
        self._pd_min_before = []
        self._pd_min_after = []
        self._pd_condition_numbers = []
        self._pd_relative_repairs = []
        self._last_pd_repair = None
        self._pd_composed_evaluations = 0
        self._pd_adaptation_count = 0
        self._pd_adaptation_relative = []
        self._pd_information_eigenvalues = 0
        self._pd_negative_information = 0
        self._pd_information_min = []
        self._pd_adaptation_relative_mean = []
        self._curv_eigenvalues = 0
        self._curv_negative = 0
        self._lams_calls = 0
        self._lams_samples = 0
        self._lams_bad = 0
        self._lams_bad_calls = 0
        self._lams_min_relative = float("inf")
        self._lams_trace = []
        self._clamp_calls = 0
        self._clamp_active = 0
        self._clamp_elements = 0

    @property
    def covariance_diagnostics(self):
        """Diagnostics for covariance conditioning and composed-precision repairs."""
        count = int(getattr(self, "_pd_evaluation_count", 0))
        repairs = int(getattr(self, "_pd_repair_count", 0))
        values = lambda name: list(getattr(self, name, []))
        conditions = values("_pd_condition_numbers")
        relatives = values("_pd_relative_repairs")
        composed_count = int(getattr(self, "_pd_composed_evaluations", 0))
        adaptations = int(getattr(self, "_pd_adaptation_count", 0))
        information_minima = values("_pd_information_min")
        adaptation_relatives = values("_pd_adaptation_relative")
        eigenvalue_count = int(getattr(self, "_pd_information_eigenvalues", 0))
        negative_count = int(getattr(self, "_pd_negative_information", 0))
        return {
            "score_evaluations": count,
            "repair_count": repairs,
            "repair_fraction": repairs / count if count else 0.0,
            # Projection of each observation's information onto the positive
            # semi-definite cone. A nonzero negative_information_fraction means
            # the estimated single-observation posteriors claimed to be wider
            # than the prior in that share of directions -- outside the model
            # class, and exactly what makes plain GAUSS invalid here
            # (Linhart et al. 2024, Eq. 21). It should fall towards zero as the
            # pilot covariance estimate improves.
            "composed_precision_evaluations": composed_count,
            "adaptation_count": adaptations,
            "adaptation_fraction": (
                adaptations / composed_count if composed_count else 0.0
            ),
            "information_eigenvalues": eigenvalue_count,
            "negative_information_count": negative_count,
            "negative_information_fraction": (
                negative_count / eigenvalue_count if eigenvalue_count else 0.0
            ),
            "minimum_information_eigenvalue": (
                min(information_minima) if information_minima else None
            ),
            "maximum_relative_adaptation": (
                max(adaptation_relatives) if adaptation_relatives else 0.0
            ),
            "information_projection": bool(getattr(self, "information_projection", True)),
            # Raw composed precision Lambda_S = Lambda_prior(t) + sum_j I_j (before any
            # PD strategy acts): how often a sample's Lambda_S was not positive definite.
            "pd_strategy": getattr(self, "pd_strategy", "row_projection"),
            "pd_fallback": getattr(self, "pd_fallback", "repair"),
            "pd_nugget": getattr(self, "pd_nugget", 0.1),
            "lambda_s_checks": int(getattr(self, "_lams_calls", 0)),
            "lambda_s_nonpd_samples": int(getattr(self, "_lams_bad", 0)),
            "lambda_s_nonpd_fraction": (
                getattr(self, "_lams_bad", 0) / getattr(self, "_lams_samples", 0)
                if getattr(self, "_lams_samples", 0) else 0.0),
            "lambda_s_nonpd_call_fraction": (
                getattr(self, "_lams_bad_calls", 0) / getattr(self, "_lams_calls", 0)
                if getattr(self, "_lams_calls", 0) else 0.0),
            # min eigenvalue of Lambda_S times lambda^2 (1 = the pure-noise precision)
            "lambda_s_min_relative": (
                None if getattr(self, "_lams_min_relative", float("inf")) == float("inf")
                else getattr(self, "_lams_min_relative")),
            # (lambda, share of samples with non-PD Lambda_S, min eigenvalue * lambda^2) per call
            "lambda_s_trace": list(getattr(self, "_lams_trace", [])),
            "clamp_mode": getattr(self, "clamp_mode", "coordinate"),
            "clamp_calls": int(getattr(self, "_clamp_calls", 0)),
            "clamp_active_fraction": (
                getattr(self, "_clamp_active", 0) / getattr(self, "_clamp_elements", 0)
                if getattr(self, "_clamp_elements", 0) else 0.0),
            "curvature_lower_clip": bool(getattr(self, "curvature_lower_clip", True)),
            # share of gauss_jacobian curvature eigenvalues below 0 (what the lower clip zeroes)
            "negative_curvature_fraction": (
                getattr(self, "_curv_negative", 0) / getattr(self, "_curv_eigenvalues", 0)
                if getattr(self, "_curv_eigenvalues", 0) else 0.0
            ),
            # mean over activations of ||I^+ - I|| / ||I||, averaged over rows and samples
            "mean_relative_adaptation": (
                sum(values("_pd_adaptation_relative_mean"))
                / len(values("_pd_adaptation_relative_mean"))
                if values("_pd_adaptation_relative_mean") else 0.0
            ),
            "minimum_eigenvalue_before": (
                min(values("_pd_min_before")) if count else None
            ),
            "minimum_eigenvalue_after": (
                min(values("_pd_min_after")) if count else None
            ),
            "maximum_condition_number": max(conditions) if conditions else None,
            "maximum_relative_repair": max(relatives) if relatives else 0.0,
            "last_repair": getattr(self, "_last_pd_repair", None),
        }

    def _regularize_covariance(self, covariance):
        """Regularize empirical covariance in float64 with shrinkage and a nugget."""
        covariance = torch.as_tensor(covariance, dtype=torch.float64)
        covariance = 0.5 * (covariance + covariance.mT)
        diagonal = torch.diag_embed(torch.diagonal(covariance, dim1=-2, dim2=-1))
        regularized = (
            (1.0 - self.covariance_shrinkage) * covariance
            + self.covariance_shrinkage * diagonal
        )
        dimension = regularized.shape[-1]
        scale = (
            torch.diagonal(regularized, dim1=-2, dim2=-1).sum(-1) / dimension
        ).clamp_min(1.0)
        identity = torch.eye(
            dimension, dtype=regularized.dtype, device=regularized.device
        )
        return regularized + (
            self.covariance_nugget * scale
        )[..., None, None] * identity

    @staticmethod
    def _precision_from_covariance(covariance):
        """Return batched precision matrices using Cholesky solves."""
        covariance = torch.as_tensor(covariance)
        factors, info = torch.linalg.cholesky_ex(covariance)
        if torch.any(info != 0):
            raise ValueError("covariance must be positive definite.")
        identity = torch.eye(
            covariance.shape[-1], dtype=covariance.dtype,
            device=covariance.device,
        ).expand(covariance.shape[:-2] + covariance.shape[-2:])
        return torch.cholesky_solve(identity, factors)

    def _configure_jacobian_schedule(self, fd_step, refresh_low_noise,
                                     refresh_lambda, refresh_high_noise=None,
                                     refresh_high_lambda=None, refresh_band=None,
                                     refresh_outside_band=None, secant=False):
        """Validate and store the finite-difference step and refresh schedule."""
        if (refresh_high_noise is None) != (refresh_high_lambda is None):
            raise ValueError("jacobian_refresh_high_noise and "
                             "jacobian_refresh_high_lambda go together.")
        self.jacobian_refresh_high_noise = (
            None if refresh_high_noise is None else int(refresh_high_noise))
        self.jacobian_refresh_high_lambda = (
            None if refresh_high_lambda is None else float(refresh_high_lambda))
        if self.jacobian_refresh_high_noise is not None and (
                self.jacobian_refresh_high_noise < 1
                or not self.jacobian_refresh_high_lambda > 0.0):
            raise ValueError("jacobian_refresh_high_noise must be >= 1 and "
                             "jacobian_refresh_high_lambda positive.")
        if refresh_band is not None:
            low, high = (float(v) for v in refresh_band)
            if not (0.0 <= low < high):
                raise ValueError("jacobian_refresh_band must be (r_low, r_high) "
                                 "with 0 <= r_low < r_high.")
            refresh_band = (low, high)
        self.jacobian_refresh_band = refresh_band
        self.jacobian_refresh_outside_band = (
            None if refresh_outside_band is None else int(refresh_outside_band))
        if (self.jacobian_refresh_outside_band is not None
                and self.jacobian_refresh_outside_band < 1):
            raise ValueError("jacobian_refresh_outside_band must be at least 1.")
        self.jacobian_secant = bool(secant)
        self.jacobian_fd_step = float(fd_step)
        if not (math.isfinite(self.jacobian_fd_step)
                and self.jacobian_fd_step > 0.0):
            raise ValueError("jacobian_fd_step must be finite and positive.")
        self.jacobian_refresh_low_noise = (
            None if refresh_low_noise is None else int(refresh_low_noise)
        )
        if (self.jacobian_refresh_low_noise is not None
                and self.jacobian_refresh_low_noise < 1):
            raise ValueError("jacobian_refresh_low_noise must be at least 1.")
        self.jacobian_refresh_lambda = float(refresh_lambda)
        if not (math.isfinite(self.jacobian_refresh_lambda)
                and self.jacobian_refresh_lambda > 0.0):
            raise ValueError("jacobian_refresh_lambda must be finite and positive.")

    # Period that stands for "never again": the modulo test below then only
    # fires at call index 0.
    NEVER_REFRESH = 10 ** 12

    def _jacobian_refresh_at(self, inverse_variance):
        """Refresh period at noise level lambda = inverse_variance ** -0.5."""
        band = getattr(self, "jacobian_refresh_band", None)
        stats = getattr(self, "_jacobian_info_scale", None)
        if band is not None:
            if stats is None:
                return 1
            k_min, _ = stats
            in_band = band[0] <= k_min / inverse_variance <= band[1]
            if in_band:
                return max(1, int(self.jacobian_refresh))
            outside = getattr(self, "jacobian_refresh_outside_band", None)
            return self.NEVER_REFRESH if outside is None else max(1, int(outside))
        high = getattr(self, "jacobian_refresh_high_noise", None)
        if high is not None and inverse_variance < self.jacobian_refresh_high_lambda ** -2:
            return max(1, int(high))
        low = getattr(self, "jacobian_refresh_low_noise", None)
        if low is not None and inverse_variance > self.jacobian_refresh_lambda ** -2:
            return max(1, int(low))
        return max(1, int(self.jacobian_refresh))

    def _reset_jacobian_state(self):
        """Clear the Jacobian-curvature refresh cache and its counters."""
        self._jacobian_call_index = 0
        self._jacobian_cached_information = None
        self._jacobian_evaluations = 0
        self._bp_jacobian_cached = None
        self._jacobian_info_scale = None
        self._jacobian_refreshes = 0
        self._secant_previous = None
        self._secant_updates = 0
        self._secant_skipped = 0

    @staticmethod
    def _batched_eigh(matrices, max_batch=2048):
        """``torch.linalg.eigh``, chunked over the leading batch dimension.

        eigh is elementwise over a batch, so this is mathematically identical
        to calling it once on the whole tensor. It exists because PyTorch's
        batched CUDA ``eigh`` kernel allocates workspace that scales with the
        batch size even for tiny (e.g. 2x2) matrices: at problem sizes such as
        30 observations x 3000 posterior samples (a batch of 90,000 matrices,
        under 6 MB of actual data at 2x2/float64) it requests tens of GB and
        reliably OOMs an 11 GB GPU. Chunking bounds peak memory independent of
        how many matrices are diagonalized, at the cost of extra kernel
        launches for very large batches.
        """
        shape = matrices.shape
        flat = matrices.reshape(-1, shape[-2], shape[-1])
        if flat.shape[0] <= max_batch:
            eigenvalues, eigenvectors = torch.linalg.eigh(flat)
        else:
            eigenvalue_chunks, eigenvector_chunks = [], []
            for start in range(0, flat.shape[0], max_batch):
                chunk_eigenvalues, chunk_eigenvectors = torch.linalg.eigh(
                    flat[start:start + max_batch]
                )
                eigenvalue_chunks.append(chunk_eigenvalues)
                eigenvector_chunks.append(chunk_eigenvectors)
            eigenvalues = torch.cat(eigenvalue_chunks, dim=0)
            eigenvectors = torch.cat(eigenvector_chunks, dim=0)
        return eigenvalues.reshape(shape[:-1]), eigenvectors.reshape(shape)

    def _jacobian_joint_precision(self, state, t, condition_mask, var_t):
        """Per-observation backward precision from the network's own Jacobian.

        Tweedie's identities are exact for the lambda-smoothed density, in the
        sampler's y = x/alpha coordinates and with lambda^2 = var_t:

            s_j(y)      = ( E[theta_0 | y, x_j] - y ) / lambda^2
            grad s_j(y) = ( Cov(theta_0 | y, x_j) - lambda^2 I ) / lambda^4

        so the backward-kernel covariance that every GAUSS-style composition
        weights by is available in closed form from the score Jacobian:

            Sigma_t,j(y) = lambda^2 ( I + lambda^2 grad s_j(y) )
                         = lambda^2 ( I - lambda^2 H_j(y) ),  H_j = -grad s_j.

        No Gaussian assumption on p(theta_0 | x_j) is made anywhere. If that
        posterior *is* N(., Sigma_0,j) then grad s_j = -(Sigma_0,j + lambda^2
        I)^-1 and the expression collapses to

            Sigma_t,j = lambda^2 Sigma_0,j (Sigma_0,j + lambda^2 I)^-1
            Lambda_j  = Sigma_t,j^-1 = Sigma_0,j^-1 + lambda^-2 I,

        which is exactly the pilot-covariance form the other covariance-aware
        corrections build. This is therefore a strict generalization of GAUSS,
        not an alternative approximation to it.

        Conditioning. Every lambda-smoothed density obeys 0 <= H_j <=
        lambda^-2 I exactly, so the eigenvalues of H_j are clamped into
        [0, (1 - floor) / lambda^2]. That bounds Lambda_j into
        [lambda^-2 I, (floor * lambda^2)^-1 I]: it can never be *wider* than
        the pure-noise kernel (the lower bound, which is what keeps the
        composed precision well posed) and never collapse to a point mass (the
        upper bound, set by `jacobian_curvature_floor`). Where the network's
        Jacobian is indefinite the smoothed density is genuinely non-log-concave
        there, and this clamp is the same projection `_map_curvature_blocks`
        already applies on the Newton path.
        """
        variance = float(torch.as_tensor(var_t, dtype=torch.float64).reshape(()))
        feature_order = list(self.full_gaussian_features)
        # torch.func.jvp propagates tangents only with grad mode enabled, and
        # the composition runs inside `_get_score`'s no_grad block.
        with torch.enable_grad():
            jacobian = self._row_score_jacobian(
                state, t, condition_mask, feature_order
            )
        self._jacobian_evaluations += 1

        curvature = -jacobian
        curvature = 0.5 * (curvature + curvature.mT)
        floor = float(self.jacobian_curvature_floor)
        if getattr(self, "cholesky_screen", False):
            return self._screened_jacobian_precision(curvature, variance, floor)
        eigenvalues, eigenvectors = self._row_eigh(curvature)
        self._curv_eigenvalues = getattr(self, "_curv_eigenvalues", 0) + int(eigenvalues.numel())
        self._curv_negative = getattr(self, "_curv_negative", 0) + int((eigenvalues < 0).sum())
        lower = 0.0 if getattr(self, "curvature_lower_clip", True) else None
        eigenvalues = eigenvalues.clamp(min=lower, max=(1.0 - floor) / variance)
        precision = 1.0 / (variance * (1.0 - variance * eigenvalues))
        return (eigenvectors * precision.unsqueeze(-2)) @ eigenvectors.mT

    def set_cholesky_screen(self, enabled):
        """Switch the Cholesky screen of the gauss_jacobian eigendecompositions on or off.

        Both per-row eigendecompositions on the gauss_jacobian path exist only to clamp
        eigenvalues: the curvature H_j into [0, (1 - floor) / lambda^2] at a Jacobian refresh,
        and the information I_j(t) onto the PSD cone at every evaluation. A Cholesky
        factorisation proves a matrix already inside those bounds (it succeeds iff the
        matrix is numerically positive definite), and then the clamp is the identity: the
        precision follows from a Cholesky solve and the PSD lift is zero. Only the rows that
        fail are eigendecomposed. The result is the unscreened one up to float rounding;
        batched Cholesky of small matrices is far cheaper than batched eigh on CUDA, and
        its workspace does not scale with the batch the way eigh's does. Off by default.
        """
        self.cholesky_screen = bool(enabled)

    def set_eigh_precision(self, dtype=torch.float64, chunk=2048):
        """dtype and chunk size of the per-row eigendecompositions on the gauss_jacobian path
        (curvature clamp at a refresh, information projection at every evaluation).

        On consumer GPUs fp64 runs at 1/32 rate: on an RTX 2080 Ti, 20,000 7x7 matrices take
        17 ms in float64 and 4 ms in float32. The matrices are formed in float64 (the
        information I_j = Lambda_j - Lambda_prior is a cancellation), only the
        eigendecomposition runs in ``dtype``, and its output is cast back. The CUDA eigh
        workspace is ~0.5 MB per matrix whatever their size (1.1 GB per 2,048-chunk), so
        ``chunk`` bounds peak memory. Defaults reproduce the previous behaviour exactly."""
        self.eigh_dtype = dtype
        self.eigh_chunk = int(chunk)

    def _row_eigh(self, matrices):
        dtype = getattr(self, "eigh_dtype", torch.float64)
        values, vectors = self._batched_eigh(matrices.to(dtype), getattr(self, "eigh_chunk", 2048))
        return values.to(matrices.dtype), vectors.to(matrices.dtype)

    def _screened_jacobian_precision(self, curvature, variance, floor):
        """`_jacobian_joint_precision`'s clamp + inverse with the Cholesky screen.

        Rows with 0 < H_j < (1 - floor) / lambda^2 (two Cholesky tests; the lower one only
        when the lower clip is on) need no clamp: Lambda_j = (lambda^2 (I - lambda^2 H_j))^-1
        directly. The others take the eigendecomposition path unchanged."""
        dimension = curvature.shape[-1]
        identity = torch.eye(dimension, dtype=curvature.dtype, device=curvature.device)
        flat = curvature.reshape(-1, dimension, dimension)
        upper = (1.0 - floor) / variance
        _, info_upper = torch.linalg.cholesky_ex(upper * identity - flat)
        inside = info_upper == 0
        lower_clip = getattr(self, "curvature_lower_clip", True)
        if lower_clip:
            _, info_lower = torch.linalg.cholesky_ex(flat)
            inside = inside & (info_lower == 0)
        kernel = variance * (identity - variance * flat)          # Sigma_t,j, PD on `inside`
        factor, _ = torch.linalg.cholesky_ex(kernel)
        precision = torch.cholesky_inverse(factor)
        outside = (~inside).nonzero(as_tuple=True)[0]
        self._screen_rows = getattr(self, "_screen_rows", 0) + int(flat.shape[0])
        self._screen_eigh_rows = getattr(self, "_screen_eigh_rows", 0) + int(outside.numel())
        if outside.numel():
            eigenvalues, eigenvectors = self._row_eigh(flat[outside])
            self._curv_eigenvalues = getattr(self, "_curv_eigenvalues", 0) + int(eigenvalues.numel())
            self._curv_negative = getattr(self, "_curv_negative", 0) + int((eigenvalues < 0).sum())
            eigenvalues = eigenvalues.clamp(min=0.0 if lower_clip else None, max=upper)
            clamped = 1.0 / (variance * (1.0 - variance * eigenvalues))
            precision[outside] = (eigenvectors * clamped.unsqueeze(-2)) @ eigenvectors.mT
        return precision.reshape(curvature.shape)

    def _bp_jacobian_score(self, scores, state, t, condition_mask, var_t):
        """Pool scores/Jacobians by block, then form the BP-JAC precision ratio."""
        refresh = max(1, int(self.jacobian_refresh))
        index = int(getattr(self, "_jacobian_call_index", 0))
        self._jacobian_call_index = index + 1
        jacobian = getattr(self, "_bp_jacobian_cached", None)
        if jacobian is None or index % refresh == 0:
            with torch.enable_grad():
                jacobian = self._row_score_jacobian(
                    state, t, condition_mask, list(self.hierarchy)
                )
            self._bp_jacobian_cached = jacobian
            self._jacobian_evaluations += 1

        global_scores = scores[:, :, self.hierarchy].to(torch.float64)
        jacobian = jacobian.to(torch.float64)
        n, samples, width = global_scores.shape
        identity = torch.eye(width, dtype=torch.float64, device=state.device)
        inverse_variance = torch.as_tensor(
            1.0 / var_t, dtype=torch.float64, device=state.device
        ).reshape(())
        precision_sum = torch.zeros(
            samples, width, width, dtype=torch.float64, device=state.device
        )
        weighted_sum = torch.zeros(
            samples, width, dtype=torch.float64, device=state.device
        )
        ordering = (global_scores.mean(dim=(1, 2)).argsort()
                    if self.jacobian_pooling == "score_sorted"
                    else torch.arange(n, device=state.device))
        for start in range(0, n, self.jacobian_block_size):
            stop = min(start + self.jacobian_block_size, n)
            size = stop - start
            members = ordering[start:stop]
            mean_score = global_scores[members].mean(dim=0)
            mean_jacobian = jacobian[members].mean(dim=0)
            block_precision = inverse_variance * torch.linalg.inv(
                identity + var_t * mean_jacobian
            )
            precision_sum += size * block_precision
            weighted_sum += size * torch.einsum(
                "sij,sj->si", block_precision, mean_score
            )

        theta_h = state[:1, :, self.hierarchy]
        prior_score = self._diffused_gaussian_prior_score(theta_h, t).to(torch.float64)
        precision_prior = self.prior_precision_matrix.to(
            device=state.device, dtype=torch.float64
        ) + inverse_variance * identity
        weighted_prior = torch.einsum(
            "ij,bsj->bsi", precision_prior, prior_score
        ).squeeze(0)
        numerator = weighted_sum + (1 - n) * weighted_prior
        composed_precision = precision_sum + (1 - n) * precision_prior
        return torch.linalg.solve(
            composed_precision, numerator.unsqueeze(-1)
        ).squeeze(-1).unsqueeze(0).to(scores.dtype)

    def _schur_factors(self, joint_precision_t, h):
        """Split a joint backward precision into global and cross-regression blocks.

        Accepts any leading batch shape, so a constant per-observation
        precision ``(n, D, D)`` and a state-dependent one ``(n, samples, D, D)``
        follow the same path.
        """
        dimension = joint_precision_t.shape[-1]
        identity = torch.eye(
            dimension, dtype=joint_precision_t.dtype,
            device=joint_precision_t.device,
        )
        if self.correction in self.FULL_GAUSSIAN_CORRECTIONS:
            return joint_precision_t, None
        factor, info = torch.linalg.cholesky_ex(joint_precision_t)
        if torch.any(info != 0):
            raise RuntimeError("Diffusion-time joint covariance is not positive definite.")
        expanded_identity = identity.expand(
            joint_precision_t.shape[:-2] + (dimension, dimension)
        )
        covariance_t = torch.cholesky_solve(expanded_identity, factor)
        global_covariance = covariance_t[..., :h, :h]
        effective_precision = self._precision_from_covariance(global_covariance)
        cross_covariance = covariance_t[..., h:, :h]
        cross_factor = torch.matmul(cross_covariance, effective_precision)
        return effective_precision, cross_factor

    def _effective_global_factors(self, var_t, num_observations=None,
                                  state=None, t=None, condition_mask=None,
                                  row_scores=None):
        """Return per-observation effective global precisions and R blocks.

        The returned precisions are projected onto the set of statistically
        admissible ones, which makes the composed precision positive definite
        by construction rather than by repair. See `_project_information`.

        The projection acts on the *global block* of every observation's
        precision. Because a Schur complement is affine in that block, adding
        `delta` there raises the effective global precision by exactly `delta`,
        and the cross-factor -A_ll^-1 A_lg does not involve that block at all,
        so the local regression coefficient is left exactly undisturbed.

        For the pilot-covariance corrections the joint precision is constant in
        the state, so the result is cached on ``(1/var_t, n)``. The
        ``gauss_jacobian`` correction builds it from the network's Jacobian at
        the current ``state`` instead, which is the whole point of that rule --
        that result is state-dependent and carries a leading sample axis, so it
        is refreshed on the `jacobian_refresh` schedule rather than cached by
        time.
        """
        n = int(
            self.num_observations if num_observations is None
            else num_observations
        )
        inverse_variance = float(
            torch.as_tensor(1.0 / var_t, dtype=torch.float64).reshape(())
        )
        h = len(self.hierarchy)

        if self.correction in self.JACOBIAN_GAUSSIAN_CORRECTIONS:
            if state is None or t is None or condition_mask is None:
                raise ValueError(
                    "correction='gauss_jacobian' needs the current state, time "
                    "and condition mask to evaluate the score Jacobian."
                )
            refresh = self._jacobian_refresh_at(inverse_variance)
            index = int(getattr(self, "_jacobian_call_index", 0))
            self._jacobian_call_index = index + 1
            information = getattr(self, "_jacobian_cached_information", None)
            if information is None or index % refresh == 0:
                # Cache the *lambda-free* information Lambda_j(t) - lambda^-2 I,
                # never Lambda_j(t) itself. Lambda_j carries an exact lambda^-2 I
                # term, and the composed precision subtracts (n-1) copies of
                # Lambda_prior(t) = Sigma_prior^-1 + lambda^-2 I built at the
                # *current* time; reusing a stale lambda there leaves the two
                # lambda^-2 terms mismatched, and since they diverge as lambda
                # shrinks the composition goes indefinite for n > 1. Separating
                # them means a lagged refresh only ever lags the network's
                # opinion about Sigma_0,j, which is exactly the quantity the
                # pilot-covariance rules hold fixed for the whole trajectory.
                # This is PSD by construction: `_jacobian_joint_precision`
                # bounds Lambda_j below by lambda^-2 I.
                joint_precision_t = self._jacobian_joint_precision(
                    state, t, condition_mask, 1.0 / inverse_variance,
                )
                identity = torch.eye(
                    joint_precision_t.shape[-1], dtype=torch.float64,
                    device=joint_precision_t.device,
                )
                information = joint_precision_t - inverse_variance * identity
                self._jacobian_cached_information = information
                self._jacobian_refreshes = getattr(self, "_jacobian_refreshes", 0) + 1
                if getattr(self, "jacobian_refresh_band", None) is not None:
                    # Chunked like every other batched eigh here: the CUDA
                    # kernel's workspace scales with the batch size.
                    eigenvalues, _ = self._batched_eigh(
                        0.5 * (information + information.mT)
                    )
                    eigenvalues = eigenvalues.clamp_min(0.0)
                    self._jacobian_info_scale = (
                        float(eigenvalues[..., 0].median()),
                        float(eigenvalues[..., -1].median()),
                    )
            elif getattr(self, "jacobian_secant", False) and row_scores is not None:
                information = self._secant_update(
                    information, state, row_scores, 1.0 / inverse_variance,
                )
                self._jacobian_cached_information = information
            if getattr(self, "jacobian_secant", False) and row_scores is not None:
                order = list(self.full_gaussian_features)
                self._secant_previous = (
                    state[..., order].to(torch.float64),
                    row_scores[..., order].to(torch.float64),
                    1.0 / inverse_variance,
                )

            identity = torch.eye(
                information.shape[-1], dtype=torch.float64,
                device=information.device,
            )
            joint_precision_t = information + inverse_variance * identity
            # metric of the conservative clamp (set_clamp_mode("metric"))
            self._last_joint_precision = joint_precision_t
            joint_precision_t = joint_precision_t + self._global_block_adaptation(
                joint_precision_t, inverse_variance, n, h,
            )
            return self._schur_factors(joint_precision_t, h)

        cache = getattr(self, "_covariance_time_cache", {})
        cache_key = (inverse_variance, n)
        if cache_key in cache:
            self._last_row_delta = getattr(self, "_row_delta_cache", {}).get(cache_key)
            return cache[cache_key]

        covariance_precision = self.posterior_precision_matrix.to(torch.float64)
        dimension = covariance_precision.shape[-1]
        identity = torch.eye(
            dimension, dtype=torch.float64, device=covariance_precision.device
        )
        joint_precision_t = covariance_precision + inverse_variance * identity
        joint_precision_t = joint_precision_t + self._global_block_adaptation(
            joint_precision_t, inverse_variance, n, h,
        )
        result = self._schur_factors(joint_precision_t, h)

        self._row_delta_cache = getattr(self, "_row_delta_cache", {})
        self._row_delta_cache[cache_key] = getattr(self, "_last_row_delta", None)
        cache[cache_key] = result
        self._covariance_time_cache = cache
        return result

    # Relative threshold below which a secant pair carries no usable curvature.
    SECANT_TOLERANCE = 1e-8

    def _secant_update(self, information, state, row_scores, var_t):
        """Damped BFGS update of the cached lambda-free information per row.

        For a Gaussian single-observation posterior N(mu, C) the Tweedie
        denoiser is D(y) = y + lambda^2 s(y) = mu - C s(y) at every lambda, so
        two evaluations (y1, lambda1) and (y2, lambda2) of the same row obey

            K v = u,   u = s2 - s1,   v = D1 - D2,   K = C^-1,

        whatever the two noise levels. At a single level the same equation is
        the exact local linearisation for any posterior, with K = Lambda_j -
        lambda^-2 I. Powell damping keeps K symmetric positive definite; pairs
        with no curvature information (|v| or u.v ~ 0) leave a row unchanged.
        """
        previous = getattr(self, "_secant_previous", None)
        order = list(self.full_gaussian_features)
        if previous is None:
            return information
        y1, s1, var1 = previous
        y2 = state[..., order].to(torch.float64)
        s2 = row_scores[..., order].to(torch.float64)
        if y1.shape != y2.shape or information.shape[:-2] != y2.shape[:-1]:
            return information
        u = s2 - s1
        v = (y1 + var1 * s1) - (y2 + float(var_t) * s2)
        Bv = torch.matmul(information, v.unsqueeze(-1)).squeeze(-1)
        vBv = (v * Bv).sum(-1)
        vu = (v * u).sum(-1)
        scale = v.square().sum(-1).sqrt() * u.square().sum(-1).sqrt()
        # Powell damping: blend u towards B v when the pair has too little
        # positive curvature, so the update stays positive definite.
        theta = torch.where(
            vu >= 0.2 * vBv, torch.ones_like(vu),
            0.8 * vBv / (vBv - vu).clamp_min(torch.finfo(torch.float64).tiny),
        )
        r = theta.unsqueeze(-1) * u + (1.0 - theta).unsqueeze(-1) * Bv
        vr = (v * r).sum(-1)
        tol = self.SECANT_TOLERANCE * scale
        has_b = vBv > tol
        ok = (vr > tol) & torch.isfinite(vr) & torch.isfinite(vBv)
        # Where K has no curvature along v yet, add the rank-one term only.
        r = torch.where(has_b.unsqueeze(-1), r, u)
        vr = torch.where(has_b, vr, vu)
        ok = ok & (vr > tol)
        safe_vBv = torch.where(has_b, vBv, torch.ones_like(vBv))
        safe_vr = torch.where(ok, vr, torch.ones_like(vr))
        update = (r.unsqueeze(-1) * r.unsqueeze(-2)) / safe_vr[..., None, None]
        update = update - torch.where(
            has_b[..., None, None],
            (Bv.unsqueeze(-1) * Bv.unsqueeze(-2)) / safe_vBv[..., None, None],
            torch.zeros_like(update),
        )
        updated = information + torch.where(ok[..., None, None], update,
                                             torch.zeros_like(update))
        updated = 0.5 * (updated + updated.mT)
        self._secant_updates = getattr(self, "_secant_updates", 0) + int(ok.sum())
        self._secant_skipped = getattr(self, "_secant_skipped", 0) + int((~ok).sum())
        return updated

    @staticmethod
    def _schur_global_precision(joint_precision_t, h):
        """Marginal global precision [A^-1]_gg^-1 of a joint backward precision.

        Any leading batch shape is accepted, so a constant per-observation
        precision and a per-sample one share this path.
        """
        if joint_precision_t.shape[-1] == h:
            return joint_precision_t
        block_gg = joint_precision_t[..., :h, :h]
        block_gl = joint_precision_t[..., :h, h:]
        block_ll = joint_precision_t[..., h:, h:]
        return block_gg - torch.matmul(
            block_gl, torch.linalg.solve(block_ll, joint_precision_t[..., h:, :h])
        )

    def _global_block_adaptation(self, joint_precision_t, inverse_variance, n, h):
        """Minimal per-observation lift that makes the composed Lambda(t) SPD.

        Returns a matrix shaped like `joint_precision_t` that is zero outside
        the global block. Zero everywhere when the composition is already
        well-conditioned, which is the case whenever the single-observation
        posteriors really are narrower than the prior.
        """
        effective = self._schur_global_precision(joint_precision_t, h)
        # Kept for _apply_pd_strategy: the lift, whether or not it is applied here. The
        # other strategies act on the composed level and leave the row precisions raw.
        delta = self._project_information(effective, inverse_variance, h, apply=True)
        self._last_row_delta = delta
        if (delta is None or self.pd_strategy != "row_projection"
                or not self.information_projection):
            return torch.zeros_like(joint_precision_t)
        adaptation = torch.zeros_like(joint_precision_t)
        adaptation[..., :h, :h] = delta
        return adaptation

    def set_curvature_lower_clip(self, enabled):
        """Switch the lower clip at 0 of the gauss_jacobian curvature eigenvalues on or off."""
        self.curvature_lower_clip = bool(enabled)

    def set_information_projection(self, enabled):
        """Switch the PSD projection of each observation's information on or off.

        On (the default) it is "projected hierarchical GAUSS": negative eigenvalues of
        I_j = K_j - P_0 are clipped to zero, so the composed precision is P_0 + sum_j I_j^+.
        Off, the raw I_j are composed; an indefinite composed precision then falls through
        to the composed-level spectral repair in _solve_composed_global (counted in
        ``covariance_diagnostics['repair_count']``)."""
        self.information_projection = bool(enabled)

    def set_training_prior(self, prior):
        """Declare the Gaussian prior ``(mean, std)`` over the shared parameters
        that the network was *trained* under, when it differs from the target
        prior passed to :meth:`sample`. Each factor p~(g | x_j) then divides out
        p~(g) instead of p(g): p(g | X) ~ p(g) prod_j p~(g | x_j) / p~(g), which is
        exact because the likelihood does not depend on the prior. ``None``
        (the default) restores the usual (1-n) p(g) composition bit for bit.
        Only the covariance-Gaussian corrections support it."""
        if prior is None:
            self.training_prior = None
            return
        mean, std = (torch.as_tensor(v, dtype=torch.float32).flatten() for v in prior)
        self.training_prior = (mean, std)

    def _training_prior_terms(self, theta_h, var_t, inverse_variance, identity):
        """(Lambda~(t), s~_t(theta_h)) of the diffused diagonal training prior."""
        mean, std = (v.to(device=theta_h.device, dtype=torch.float64) for v in self.training_prior)
        precision_t = torch.diag(1.0 / std**2) + inverse_variance * identity
        score = -(theta_h.to(torch.float64) - mean) / (std**2 + var_t)
        return precision_t, score

    def _prior_precision_at(self, inverse_variance, h, reference):
        """Lambda_prior(t): the diffused Gaussian prior's backward precision."""
        return self.prior_precision_matrix.to(
            device=reference.device, dtype=torch.float64
        ) + inverse_variance * torch.eye(
            h, dtype=torch.float64, device=reference.device
        )

    def _project_information(self, effective, inverse_variance, h, apply=None):
        """Project each observation's information onto the admissible cone.

        Writing Lambda_j(t) = Lambda_prior(t) + I_j(t) defines I_j(t), the
        information observation j carries about the shared parameters. The
        composed precision is then an identity, not an approximation:

            Lambda(t) = sum_j Lambda_j(t) + (1-n) Lambda_prior(t)
                      = Lambda_prior(t) + sum_j I_j(t).

        So Lambda(t) is positive definite for every t and every n as soon as
        each I_j(t) is positive *semi*-definite -- and I_j(t) >= 0 is not a
        numerical convenience, it is a property every real posterior has: under
        the Gaussian approximation GAUSS already assumes, conditioning on
        observation j cannot leave you less certain about the shared parameters
        than the prior alone. An estimated I_j with a negative eigenvalue is
        outside the model class, not a hard case.

        We therefore project each I_j(t) onto the positive semi-definite cone
        (its nearest admissible point in Frobenius norm) and rebuild
        Lambda_j(t) = Lambda_prior(t) + I_j(t)^+. This costs nothing when the
        estimates are already admissible, in which case the composition is
        bit-for-bit ordinary GAUSS.

        Two consequences worth stating. Lambda(t) >= Lambda_prior(t) makes the
        solve unconditionally well conditioned -- the composed score can never
        be amplified beyond the prior covariance -- so no eigenvalue floor,
        denoise clamp or spectral repair is load-bearing for validity. And the
        correction is per observation, so unlike a repair of the composed
        matrix it does not grow with n.

        Returns the per-observation delta to add to the effective global
        precision, or None when every I_j(t) is already admissible.
        """
        prior_t = self._prior_precision_at(inverse_variance, h, effective)
        if self.training_prior is not None:
            # Each factor's information is relative to the prior it was trained under.
            prior_t = torch.diag(1.0 / self.training_prior[1].to(
                device=effective.device, dtype=torch.float64) ** 2) + inverse_variance * torch.eye(
                h, dtype=torch.float64, device=effective.device)
        information = effective - prior_t
        information = 0.5 * (information + information.mT)
        if getattr(self, "cholesky_screen", False):
            return self._screened_projection(information, apply)
        # Chunked: with a per-sample precision this batch is
        # observations x samples matrices (90,000 at 30 x 3000), which the CUDA
        # eigh kernel cannot allocate workspace for in one call -- see
        # _batched_eigh.
        eigenvalues, eigenvectors = self._row_eigh(information)
        deficits = (-eigenvalues).clamp_min(0.0)

        self._pd_composed_evaluations += 1
        self._pd_information_eigenvalues += int(eigenvalues.numel())
        negative = int((eigenvalues < 0).sum())
        self._pd_negative_information += negative
        self._pd_information_min.append(float(eigenvalues.min()))
        if negative == 0:
            return None

        # Counted whether or not the projection is applied: with it off, these
        # record the lift it *would* have made.
        self._pd_adaptation_count += 1
        # I^+ - I = sum of the removed negative eigencomponents.
        delta = torch.matmul(
            eigenvectors * deficits.unsqueeze(-2), eigenvectors.mT
        )
        relative = (torch.linalg.matrix_norm(delta)
                    / torch.linalg.matrix_norm(information).clamp_min(
                        torch.finfo(torch.float64).eps))
        self._pd_adaptation_relative.append(float(relative.max()))
        self._pd_adaptation_relative_mean = getattr(self, "_pd_adaptation_relative_mean", [])
        self._pd_adaptation_relative_mean.append(float(relative.mean()))
        if not (self.information_projection if apply is None else apply):
            return None
        return delta

    def _screened_projection(self, information, apply):
        """`_project_information` with the Cholesky screen: rows whose I_j(t) factorises are
        positive definite and get a zero lift; only the others are eigendecomposed. The
        diagnostics then cover the eigendecomposed rows only (``_pd_information_min`` records
        their minimum, or 0.0 -- a valid lower bound -- when every row passed)."""
        h = information.shape[-1]
        flat = information.reshape(-1, h, h)
        _, info = torch.linalg.cholesky_ex(flat)
        failed = (info != 0).nonzero(as_tuple=True)[0]
        self._pd_composed_evaluations += 1
        self._screen_rows = getattr(self, "_screen_rows", 0) + int(flat.shape[0])
        self._screen_eigh_rows = getattr(self, "_screen_eigh_rows", 0) + int(failed.numel())
        if failed.numel() == 0:
            self._pd_information_min.append(0.0)
            return None
        eigenvalues, eigenvectors = self._row_eigh(flat[failed])
        self._pd_information_eigenvalues += int(eigenvalues.numel())
        negative = int((eigenvalues < 0).sum())
        self._pd_negative_information += negative
        self._pd_information_min.append(float(eigenvalues.min()))
        if negative == 0:
            return None
        self._pd_adaptation_count += 1
        deficits = (-eigenvalues).clamp_min(0.0)
        delta_failed = torch.matmul(eigenvectors * deficits.unsqueeze(-2), eigenvectors.mT)
        relative = (torch.linalg.matrix_norm(delta_failed)
                    / torch.linalg.matrix_norm(flat[failed]).clamp_min(torch.finfo(torch.float64).eps))
        self._pd_adaptation_relative.append(float(relative.max()))
        self._pd_adaptation_relative_mean = getattr(self, "_pd_adaptation_relative_mean", [])
        self._pd_adaptation_relative_mean.append(float(relative.mean()))
        if not (self.information_projection if apply is None else apply):
            return None
        delta = torch.zeros_like(flat)
        delta[failed] = delta_failed
        return delta.reshape(information.shape)

    PD_STRATEGIES = ("row_projection", "composed_check", "gloeckler",
                     "row_projection_post", "row_projection_consistent",
                     "row_truncation", "row_projection_blend", "gloeckler_consistent")
    PD_FALLBACKS = ("repair", "row_projection", "gloeckler")

    def set_pd_strategy(self, strategy, fallback=None, nugget=None):
        """Choose how the composed precision Lambda_S is kept positive definite.

        - "row_projection" (default): project every row's information I_j onto the
          PSD cone and weight the row's own score by the lifted precision
          (`_project_information`). Unchanged, bit for bit.
        - "composed_check": compose the raw I_j and only check Lambda_S =
          Lambda_prior(t) + sum_j I_j. Samples whose Lambda_S is not PD are counted
          (covariance_diagnostics['lambda_s_*']) and handled by `fallback`:
          "repair" (the composed-level spectral repair of _solve_composed_global),
          "row_projection" (the default rule, on the failing samples only) or
          "gloeckler" (the rule below, on the failing samples only).
        - "gloeckler": Gloeckler et al. 2025 (ICLR, "Compositional SBI for time
          series", App. B.3 and markovsbi CorrectedScoreFn): every factor gets
          V max(-A, 0) V^T / (n-1) + nugget * I, with Lambda_S = V A V^T, on every
          call with n > 1; the scores are weighted by the adjusted precisions.
        - "row_projection_post": the default rule applied after the Schur
          reduction instead of before it. A consistency check: must equal the
          default to float rounding.
        - "row_projection_consistent": the same lifted precision K_j + Delta_j, but
          the row's score is replaced by the score of the *same* Gaussian with that
          precision (same clean mean mu_j = y + (K_j^-1 + lambda^2) s_j), i.e. the
          weighted term becomes (K_j + Delta_j) K_j^-1 Lambda_j s_j instead of
          (Lambda_j + Delta_j) s_j. The default rule lifts the precision but keeps
          the score of the unlifted Gaussian.
        - "row_truncation": drop each row's message in the directions where I_j < 0.
          Precision Lambda_prior + I_j^+ (as the default), numerator Lambda_ref s_ref +
          P_j^+ (Lambda_j s_j - Lambda_ref s_ref): an exact Gaussian message with
          precision I_j^+ and the row's own mean in the kept directions, and no
          information in the others. Needs no K_j^-1.
        - "row_projection_blend": the consistent term with K_j^-1 replaced by the lifted
          (K_j + Delta_j)^-1 = (P_0 + I_j^+)^-1: Delta_j (K_j + Delta_j)^-1
          (Lambda_j + Delta_j) s_j. Equals the consistent rule where K_j >> Delta_j and
          stays bounded (by the prior covariance) where K_j -> 0.
        - "gloeckler_consistent": Gloeckler's lift L, but each row's term is
          L K_j^-1 Lambda_j s_j (the same row Gaussian, lifted) instead of L s_j.
        """
        if strategy not in self.PD_STRATEGIES:
            raise ValueError(f"pd strategy must be one of {self.PD_STRATEGIES}")
        self.pd_strategy = strategy
        if fallback is not None:
            if fallback not in self.PD_FALLBACKS:
                raise ValueError(f"pd fallback must be one of {self.PD_FALLBACKS}")
            self.pd_fallback = fallback
        if nugget is not None:
            self.pd_nugget = float(nugget)
        self._covariance_time_cache = {}

    def _gloeckler_lift(self, eigenvalues, eigenvectors, n):
        """Per-factor precision lift of Gloeckler et al.: V max(-A,0) V^T/(n-1) + eps I."""
        deficits = (-eigenvalues).clamp_min(0.0) / max(n - 1, 1)
        lift = (eigenvectors * deficits.unsqueeze(-2)) @ eigenvectors.mT
        identity = torch.eye(lift.shape[-1], dtype=lift.dtype, device=lift.device)
        return lift + self.pd_nugget * identity

    def _apply_pd_strategy(self, precision, numerator, effective, global_scores,
                           inverse_variance, n, reference=None, state=None):
        """Check Lambda_S and apply the selected PD strategy at the composed level.

        ``precision`` is the composed Lambda_S, (H, H) or (samples, H, H);
        ``numerator`` (1, samples, H); ``effective`` the per-row precisions the
        numerator was weighted with, (rows, H, H) or (rows, samples, H, H);
        ``global_scores`` (rows, samples, H); ``reference`` the (precision,
        precision @ score) of the prior the I_j are measured against, and ``state``
        the current globals (1, samples, H), both optional. Returns the (precision,
        numerator) to solve. The default strategy only records diagnostics.
        """
        strategy = self.pd_strategy
        delta = getattr(self, "_last_row_delta", None)
        self._last_row_delta = None
        rows = global_scores.shape[0]
        if delta is not None and delta.dim() == 3 and delta.shape[0] == 1 and rows > 1:
            delta = delta.expand(rows, -1, -1)
        samples = numerator.shape[1]
        raw = precision
        if strategy == "row_projection" and delta is not None and self.information_projection:
            # The rows arrive projected already; check what the raw Lambda_S was.
            raw = precision - delta.sum(dim=0)
        raw = 0.5 * (raw + raw.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(raw)
        minimum = eigenvalues[..., 0]
        bad = minimum <= 0
        if bad.dim() == 0:
            bad = bad.expand(samples)
            minimum = minimum.expand(samples)
        variance = float(1.0 / inverse_variance)
        self._lams_calls = getattr(self, "_lams_calls", 0) + 1
        self._lams_samples = getattr(self, "_lams_samples", 0) + int(bad.numel())
        self._lams_bad = getattr(self, "_lams_bad", 0) + int(bad.sum())
        self._lams_bad_calls = getattr(self, "_lams_bad_calls", 0) + int(bool(bad.any()))
        # min eigenvalue in units of lambda^-2 (the pure-noise precision)
        relative = float(minimum.min()) * variance
        self._lams_min_relative = min(getattr(self, "_lams_min_relative", float("inf")), relative)
        trace = getattr(self, "_lams_trace", [])
        trace.append((variance ** 0.5, float(bad.float().mean()), relative))
        self._lams_trace = trace
        if self.pd_probe and delta is not None:
            projected = strategy == "row_projection" and self.information_projection
            self._probe_row_lift(raw, numerator, delta, effective, global_scores,
                                 inverse_variance, state, projected)
        if strategy in ("row_projection", "composed_check") and not (
                strategy == "composed_check" and bool(bad.any()) and self.pd_fallback != "repair"):
            return precision, numerator

        if effective.dim() == 3 and effective.shape[0] == 1 and rows > 1:
            effective = effective.expand(rows, -1, -1)

        def weighted(matrices, vectors):
            # (rows[, samples], H, H) x (rows, samples, H) -> (1, samples, H)
            if matrices.dim() == 4:
                return torch.einsum("nsij,nsj->si", matrices, vectors).unsqueeze(0)
            return torch.einsum("nij,nsj->si", matrices, vectors).unsqueeze(0)

        def per_sample(matrix):
            # broadcast a (H, H) composed matrix to (samples, H, H)
            return matrix if matrix.dim() == 3 else matrix.expand(samples, -1, -1)

        if strategy == "composed_check":
            mask = bad.reshape(samples, 1, 1).to(precision.dtype)
            if self.pd_fallback == "row_projection":
                if delta is None:
                    return precision, numerator
                lift = delta.sum(dim=0)
                extra = weighted(delta, global_scores)
            else:  # gloeckler on the failing samples
                lift_one = self._gloeckler_lift(eigenvalues, eigenvectors, n)
                lift = n * lift_one
                extra = torch.einsum(
                    "...ij,bsj->bsi" if lift_one.dim() == 2 else "sij,bsj->bsi",
                    lift_one, global_scores.sum(dim=0, keepdim=True))
            precision = per_sample(precision) + mask * per_sample(lift)
            numerator = numerator + mask.squeeze(-1).unsqueeze(0) * extra
            return precision, numerator

        if strategy in ("gloeckler", "gloeckler_consistent"):
            if n <= 1:
                return precision, numerator
            lift_one = self._gloeckler_lift(eigenvalues, eigenvectors, n)
            if strategy == "gloeckler":
                extra = torch.einsum(
                    "ij,bsj->bsi" if lift_one.dim() == 2 else "sij,bsj->bsi",
                    lift_one, global_scores.sum(dim=0, keepdim=True))
            else:
                # the same Gaussian, lifted: L K_j^-1 Lambda_j s_j per row
                k_inverse = self._clean_precision_inverse(effective, inverse_variance)
                means = self._apply_rows(k_inverse, self._weighted_rows(effective, global_scores))
                lift_rows = lift_one if lift_one.dim() == 2 else lift_one.unsqueeze(0)
                extra = (lift_rows @ means.unsqueeze(-1)).squeeze(-1).sum(dim=0, keepdim=True)
            return precision + n * lift_one, numerator + extra

        if strategy == "row_truncation":
            # Drop each row's message in its negative-information directions:
            # precision Lambda_prior + I_j^+ (as the default) and numerator
            # Lambda_ref s_ref + P_j^+ (Lambda_j s_j - Lambda_ref s_ref).
            if delta is None:
                return precision, numerator
            if reference is None:
                raise ValueError("row_truncation needs the reference prior terms")
            reference_precision, reference_weighted = reference
            information = effective - reference_precision
            information = 0.5 * (information + information.mT)
            values, vectors = self._row_eigh(information)
            negative = (values < 0).to(values.dtype)
            projector = (vectors * negative.unsqueeze(-2)) @ vectors.mT        # P_j^-
            message = self._weighted_rows(effective, global_scores) - reference_weighted
            if projector.dim() == 3:
                removed = torch.einsum("nij,nsj->si", projector, message)
            else:
                removed = (projector @ message.unsqueeze(-1)).squeeze(-1).sum(dim=0)
            return precision + delta.sum(dim=0), numerator - removed.unsqueeze(0)

        if delta is None:
            return precision, numerator
        lift = delta.sum(dim=0)
        if strategy == "row_projection_post":
            return precision + lift, numerator + weighted(delta, global_scores)
        if strategy == "row_projection_blend":
            # Delta_j (K_j + Delta_j)^-1 (Lambda_j + Delta_j) s_j: the consistent term with
            # K_j^-1 replaced by the lifted (P_0 + I_j^+)^-1, i.e. weighted by K/(K + Delta).
            identity = torch.eye(effective.shape[-1], dtype=effective.dtype, device=effective.device)
            lifted = self._broadcast_rows(effective, delta) + delta
            weighted_lifted = (lifted @ self._broadcast_scores(global_scores, delta)).squeeze(-1) \
                if delta.dim() == 4 else torch.einsum("nij,nsj->nsi", lifted, global_scores)
            clean = lifted - inverse_variance * identity
            clean = 0.5 * (clean + clean.mT)
            if delta.dim() == 4:
                means = torch.linalg.solve(clean, weighted_lifted.unsqueeze(-1))
                extra = (delta @ means).squeeze(-1).sum(dim=0, keepdim=True)
            else:
                means = torch.linalg.solve(clean.unsqueeze(1), weighted_lifted.unsqueeze(-1)).squeeze(-1)
                extra = torch.einsum("nij,nsj->si", delta, means).unsqueeze(0)
            return precision + lift, numerator + extra
        # row_projection_consistent: Delta_j K_j^-1 Lambda_j s_j, K_j = Lambda_j - lambda^-2 I
        return precision + lift, numerator + self._consistent_extra(
            delta, effective, global_scores, inverse_variance)

    @staticmethod
    def _broadcast_rows(effective, delta):
        """Row precisions shaped like ``delta`` ((rows, H, H) or (rows, samples, H, H))."""
        if delta.dim() == 4 and effective.dim() == 3:
            return effective.unsqueeze(1).expand_as(delta)
        return effective

    @staticmethod
    def _broadcast_scores(global_scores, delta):
        return global_scores.unsqueeze(-1) if delta.dim() == 4 else global_scores

    @staticmethod
    def _apply_rows(matrices, vectors):
        """(rows[, samples], H, H) applied to (rows, samples, H) -> (rows, samples, H)."""
        if matrices.dim() == 3:
            return torch.einsum("nij,nsj->nsi", matrices, vectors)
        return (matrices @ vectors.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _weighted_rows(effective, global_scores):
        """Lambda_j s_j per row: (rows, samples, H)."""
        if effective.dim() == 4:
            return (effective @ global_scores.unsqueeze(-1)).squeeze(-1)
        return torch.einsum("nij,nsj->nsi", effective, global_scores)

    def _clean_precision_inverse(self, effective, inverse_variance):
        """Floored K_j^-1, K_j = Lambda_j - lambda^-2 I (shape of ``effective``)."""
        identity = torch.eye(effective.shape[-1], dtype=effective.dtype, device=effective.device)
        clean = effective - inverse_variance * identity
        clean = 0.5 * (clean + clean.mT)
        k_values, k_vectors = self._batched_eigh(clean)
        # A row with (numerically) no clean precision has no finite mean; floor it.
        floor = 1e-6 * k_values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        return (k_vectors / k_values.clamp_min(0.0).maximum(floor).unsqueeze(-2)) @ k_vectors.mT

    def _consistent_extra(self, delta, effective, global_scores, inverse_variance):
        """sum_j Delta_j K_j^-1 Lambda_j s_j as (1, samples, H)."""
        effective_full = self._broadcast_rows(effective, delta)
        k_inverse = self._clean_precision_inverse(effective_full, inverse_variance)
        weighted_scores = self._weighted_rows(effective_full, global_scores).unsqueeze(-1)
        if delta.dim() == 4:
            return (delta @ (k_inverse @ weighted_scores)).squeeze(-1).sum(dim=0, keepdim=True)
        return torch.einsum("nij,nsj->si", delta @ k_inverse,
                            weighted_scores.squeeze(-1)).unsqueeze(0)

    def _probe_row_lift(self, raw, numerator, delta, effective, global_scores, inverse_variance,
                        state, projected):
        """COMPASS_PD_PROBE: where the row lift Delta_j acts, how well determined is the
        row's clean mean, and how far do the consistent and default rules disagree?

        Appends one json line per call to ``self.pd_probe``. Per (row, sample) with an
        active lift, along each negative-information direction v of I_j (|v| = 1):
        ``k_rel`` = v^T K_j v / v^T P_0 v (clean row precision relative to the prior),
        ``shift`` = |v^T (mu_j - y)| / prior sd along v (the implied mean's distance).
        Per sample: ``dscore`` = |s_cons - s_def| / |s_def| of the composed global score,
        ``toward_origin`` = cos(s_cons - s_def, -y)."""
        import json
        h = delta.shape[-1]
        identity = torch.eye(h, dtype=delta.dtype, device=delta.device)
        if effective.dim() == 3 and effective.shape[0] == 1 and delta.shape[0] > 1:
            effective = effective.expand(delta.shape[0], -1, -1)
        effective_full = self._broadcast_rows(effective, delta)
        extra_default = self._apply_rows(delta, global_scores).sum(dim=0)
        if projected:
            # the rows (and numerator) arrive lifted already; the raw Lambda_j are needed here
            effective_full = effective_full - delta
            effective = effective_full
            numerator = numerator - extra_default.unsqueeze(0)
        values, vectors = self._row_eigh(0.5 * (delta + delta.mT))
        active = values > 1e-12 * values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-300)
        clean = effective_full - inverse_variance * identity
        prior = self.prior_precision_matrix.to(device=delta.device, dtype=delta.dtype)
        k_dir = torch.einsum("...ia,...ij,...ja->...a", vectors, clean, vectors)
        p_dir = torch.einsum("...ia,ij,...ja->...a", vectors, prior, vectors)
        k_inverse = self._clean_precision_inverse(effective_full, inverse_variance)
        weighted = self._weighted_rows(effective_full, global_scores)          # (rows, samples, H)
        offset = -self._apply_rows(k_inverse, weighted) / inverse_variance          # y - mu_j
        if vectors.dim() == 3:
            shift = torch.einsum("nia,nsi->nsa", vectors, offset).abs()
            k_rel = (k_dir / p_dir).unsqueeze(1).expand_as(shift)
            act = active.unsqueeze(1).expand_as(shift)
            sd = p_dir.rsqrt().unsqueeze(1).expand_as(shift)
        else:
            shift = torch.einsum("nsia,nsi->nsa", vectors, offset).abs()
            k_rel, act, sd = k_dir / p_dir, active, p_dir.rsqrt()
        shift = shift / sd
        composed = raw + delta.sum(dim=0)
        composed = composed if composed.dim() == 3 else composed.expand(global_scores.shape[1], -1, -1)
        extra_consistent = self._consistent_extra(delta, effective, global_scores, inverse_variance)[0]
        s_def = torch.linalg.solve(composed, (numerator[0] + extra_default).unsqueeze(-1)).squeeze(-1)
        diff = torch.linalg.solve(composed, (extra_consistent - extra_default).unsqueeze(-1)).squeeze(-1)
        ratio = diff.norm(dim=-1) / s_def.norm(dim=-1).clamp_min(1e-300)
        record = {"lambda": float(inverse_variance) ** -0.5,
                  "active_fraction": float(act.any(dim=-1).float().mean())}
        quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        if bool(act.any()):
            record["k_rel"] = torch.quantile(k_rel[act].double().cpu(), quantiles).tolist()
            record["shift"] = torch.quantile(shift[act].double().cpu(), quantiles).tolist()
        record["dscore"] = torch.quantile(ratio.double().cpu(), quantiles).tolist()
        if state is not None:
            y = state[0].to(diff.dtype)
            cos = (diff * -y).sum(-1) / (diff.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-300)
            big = ratio > 0.01
            record["toward_origin"] = float(cos[big].mean()) if bool(big.any()) else None
        with open(self.pd_probe, "a") as handle:
            handle.write(json.dumps(record) + "\n")

    def _marginal_global_factors(self, var_t):
        """Return clean-moment GAUSS precisions for p(g | x_j)."""
        covariance = self.global_posterior_covariance.to(torch.float64)
        precision = self.global_posterior_precision_matrix.to(torch.float64)
        inverse_variance = torch.as_tensor(
            1.0 / var_t, dtype=torch.float64, device=covariance.device
        ).reshape(())
        identity = torch.eye(
            len(self.hierarchy), dtype=torch.float64, device=covariance.device
        )
        return precision + inverse_variance * identity, covariance + var_t * identity

    def _marginal_global_scores(self, theta_h, var_t, rows):
        """Evaluate each pilot Gaussian p(g_t | x_j), independent of locals.

        ``theta_h`` is in the sampler's y=x/alpha coordinates, so its
        diffusion variance is lambda(t)^2 and no alpha factor is needed here.
        """
        _, covariance_t = self._marginal_global_factors(var_t)
        mean = self.global_posterior_mean.to(
            device=theta_h.device, dtype=torch.float64
        )
        delta = theta_h.to(torch.float64).expand(rows, -1, -1) - mean[:, None, :]
        return -torch.linalg.solve(
            covariance_t[:, None, :, :], delta.unsqueeze(-1)
        ).squeeze(-1)

    def _solve_composed_global(self, precision, numerator, subject_scores):
        """Repair and solve a composed global precision in float64.

        This is a last-resort guard: `_effective_global_factors` already adapts
        the per-observation precisions so the composed matrix arrives positive
        definite, and this path should not fire for the covariance-aware
        corrections. It survives for residual float error and for callers that
        build a composed precision directly.

        The deficient eigenvalues are lifted to `pd_floor * scale`, not to
        `pd_epsilon * scale`. Lifting to a near-zero epsilon is what makes an
        after-the-fact repair dangerous: the solve divides by the lifted
        eigenvalue, so the deviation from the subject-score mean is amplified
        by |lambda_min| / threshold -- of order 1e7 at pd_epsilon=1e-8. The
        repair stays minimal in matrix norm but is now bounded in its effect.

        `precision` is either a single ``(H, H)`` matrix shared by every sample
        (the pilot-covariance corrections) or one ``(samples, H, H)`` matrix per
        sample (`gauss_jacobian`, whose curvature is state-dependent). Both take
        the same code path; the shared case is unchanged bit for bit.
        """
        precision = torch.as_tensor(precision, dtype=torch.float64)
        batched = precision.dim() > 2
        numerator = torch.as_tensor(
            numerator, dtype=torch.float64, device=precision.device
        )
        subject_scores = torch.as_tensor(
            subject_scores, dtype=torch.float64, device=precision.device
        )
        precision = 0.5 * (precision + precision.mT)
        factor, info = torch.linalg.cholesky_ex(precision)
        eigenvalues = torch.linalg.eigvalsh(precision)
        scale = eigenvalues.abs().mean().clamp_min(1.0)
        threshold = max(self.pd_floor, self.pd_epsilon) * scale
        needs_repair = bool(torch.any(info != 0) or eigenvalues.min() < threshold)
        if needs_repair:
            eigenvalues, eigenvectors = torch.linalg.eigh(precision)
            deficits = (threshold - eigenvalues).clamp_min(0.0)
            adjustment = (
                eigenvectors * deficits.unsqueeze(-2)
            ) @ eigenvectors.mT
        else:
            adjustment = torch.zeros_like(precision)

        repaired = precision + adjustment
        score_mean = subject_scores.mean(dim=0, keepdim=True)
        repaired_numerator = numerator + torch.einsum(
            "sij,bsj->bsi" if batched else "ij,bsj->bsi", adjustment, score_mean
        )
        repaired_factor, repaired_info = torch.linalg.cholesky_ex(repaired)
        if torch.any(repaired_info != 0):
            raise RuntimeError("Minimal spectral repair did not produce a positive precision.")
        if batched:
            solved = torch.cholesky_solve(
                repaired_numerator.squeeze(0).unsqueeze(-1), repaired_factor
            ).squeeze(-1).unsqueeze(0)
        else:
            solved = torch.cholesky_solve(
                repaired_numerator.squeeze(0).mT, repaired_factor
            ).mT.unsqueeze(0)

        after = torch.linalg.eigvalsh(repaired)
        condition = float(after.max() / after.min())
        # A batched (per-sample) precision yields one norm per sample; report
        # the worst case so the diagnostic stays a single comparable number.
        relative = float((
            torch.linalg.matrix_norm(adjustment)
            / (torch.linalg.matrix_norm(precision) + torch.finfo(torch.float64).eps)
        ).max())
        self._pd_evaluation_count += 1
        self._pd_repair_count += int(needs_repair)
        self._pd_min_before.append(float(eigenvalues.min()))
        self._pd_min_after.append(float(after.min()))
        self._pd_condition_numbers.append(condition)
        self._pd_relative_repairs.append(relative)
        self._last_pd_repair = {
            "minimum_eigenvalue_before": float(eigenvalues.min()),
            "minimum_eigenvalue_after": float(after.min()),
            "condition_number": condition,
            "relative_repair": relative,
            "repaired": needs_repair,
        }
        return solved, repaired, repaired_numerator, adjustment

    def estimate_posterior_moments(
            self, data, condition_mask, num_samples, timesteps, eps,
            batch_size, device, feature_indices=None):
        """Estimate per-observation means and covariances from ordinary draws.

        All subjects are drawn in one batched reverse-diffusion trajectory
        (chunked only over ``num_samples``), not one trajectory per subject:
        the single-observation ``Sampler`` already vectorizes over observation
        rows exactly like the main compositional sampler does, so looping over
        subjects in Python here serialized what should be one GPU-saturating
        batched computation into dozens of small, mostly-idle ones.
        """
        feature_indices = (
            self.hierarchy if feature_indices is None else list(feature_indices)
        )
        num_samples = int(num_samples)
        batch_size = int(batch_size)
        if num_samples < 2:
            raise ValueError(
                "precision_est_samples must be at least 2 to estimate covariance."
            )
        if batch_size < 1:
            raise ValueError("precision_est_batch_size must be at least 1.")

        data = torch.as_tensor(data)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        condition_mask = torch.as_tensor(condition_mask)
        n_subjects = data.shape[0]

        sample_chunks = []
        for start in range(0, num_samples, batch_size):
            count = min(batch_size, num_samples - start)
            samples = self.SBIm.sampler.sample(
                world_size=1,
                data=data,
                condition_mask=condition_mask,
                timesteps=timesteps,
                eps=eps,
                num_samples=count,
                device=device,
                verbose=getattr(self, "verbose", False),
                method="dpm",
                capture_attention=False,
            )
            sample_chunks.append(samples.detach().cpu())

        all_samples = torch.cat(sample_chunks, dim=1)
        latent_samples = all_samples[:, :, feature_indices].to(torch.float64)
        mean = latent_samples.mean(dim=1)
        covariance = torch.stack([
            self._regularize_covariance(
                torch.atleast_2d(torch.cov(latent_samples[subject].mT))
            )
            for subject in range(n_subjects)
        ])

        mean = self._validate_posterior_mean(
            mean, num_observations=n_subjects,
            dimension=len(feature_indices),
        )
        covariance = self._validate_covariance(
            covariance,
            num_observations=n_subjects, dimension=len(feature_indices),
            name="automatically estimated posterior covariance",
        )
        return mean, covariance

    def _estimate_posterior_covariance(
            self, data, condition_mask, num_samples, timesteps, eps,
            batch_size, device, feature_indices=None):
        """Backward-compatible covariance-only wrapper."""
        _, covariance = self.estimate_posterior_moments(
            data, condition_mask, num_samples, timesteps, eps,
            batch_size, device, feature_indices=feature_indices,
        )
        return covariance

    def _estimate_posterior_precision(
            self, data, condition_mask, num_samples, timesteps, eps,
            batch_size, device):
        """Estimate per-observation precision in memory-bounded draw batches.

        As in :meth:`estimate_posterior_moments`, every subject is drawn in one
        batched trajectory (chunked only over ``num_samples``) rather than one
        trajectory per subject, since the single-observation sampler already
        vectorizes over observation rows.
        """
        num_samples = int(num_samples)
        batch_size = int(batch_size)
        if num_samples < 1:
            raise ValueError("precision_est_samples must be at least 1.")
        if batch_size < 1:
            raise ValueError("precision_est_batch_size must be at least 1.")

        data = torch.as_tensor(data)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        condition_mask = torch.as_tensor(condition_mask)

        sample_chunks = []
        for start in range(0, num_samples, batch_size):
            count = min(batch_size, num_samples - start)
            samples = self.SBIm.sampler.sample(
                world_size=1,
                data=data,
                condition_mask=condition_mask,
                timesteps=timesteps,
                eps=eps,
                num_samples=count,
                device=device,
                verbose=self.verbose,
                method="dpm",
                capture_attention=False,
            )
            sample_chunks.append(samples.detach().cpu())

        # Preserve every requested draw and moment-match only after all
        # chunks have completed.
        all_samples = torch.cat(sample_chunks, dim=1)
        var = all_samples[:, :, self.hierarchy].var(dim=1)
        return 1.0 / torch.clamp(var, min=1e-8)

    def _validate_precision(self, precision):
        """
        Bring the precision estimates to shape (num_hierarchy,) or (n_obs, num_hierarchy) and
        make sure the composed precision (1-n)*Lambda_prior + sum_j Lambda_j stays positive
        (Gloeckler et al. 2024). Where it does not, the deficit is distributed over the
        individual precisions.
        """
        precision = torch.atleast_2d(precision.float())  # (n_obs or 1, H)
        n = precision.shape[0]
        if n > 1:
            prior_precision = 1.0 / self.prior_std**2
            composed = (1 - n) * prior_precision + precision.sum(dim=0)
            deficit = torch.clamp(-composed, min=0.0)
            nudge = deficit / (n - 1) + torch.where(deficit > 0, torch.full_like(deficit, 0.1), torch.zeros_like(deficit))
            precision = precision + nudge
        return precision

    #############################################
    # ----- Composition diagnostics -----
    #############################################

    @staticmethod
    def _gaussian_log_density(points, mean, covariance):
        """log N(points; mean, covariance) for points shaped (..., D)."""
        points = points.to(torch.float64)
        mean = mean.to(torch.float64)
        covariance = covariance.to(torch.float64)
        dimension = points.shape[-1]
        factor = torch.linalg.cholesky(covariance)
        delta = (points - mean).unsqueeze(-1)
        solved = torch.cholesky_solve(delta, factor).squeeze(-1)
        quadratic = (delta.squeeze(-1) * solved).sum(-1)
        log_determinant = 2.0 * torch.log(torch.diagonal(factor)).sum()
        return -0.5 * (
            quadratic + log_determinant + dimension * math.log(2.0 * math.pi)
        )

    @staticmethod
    def _joint_points(samples, data, condition_mask):
        """Assemble full (observations, samples, nodes) node vectors.

        Accepts either full node vectors or just the latent/observed columns,
        matching what :meth:`ScoreBasedInferenceModel.sample` returns and what
        it was given, so a certification call can be written directly against a
        sampling call's inputs and outputs.
        """
        mask = torch.as_tensor(condition_mask, dtype=torch.float32)
        nodes = int(mask.shape[-1])
        samples = torch.as_tensor(samples, dtype=torch.float32)
        if samples.dim() != 3:
            raise ValueError(
                "samples must have shape (observations, samples, features), got "
                f"{tuple(samples.shape)}."
            )
        n_obs = int(samples.shape[0])
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).repeat(n_obs, 1)
        if mask.shape[0] != n_obs:
            raise ValueError(
                "condition_mask must have one row per observation "
                f"({n_obs}), got {mask.shape[0]}."
            )

        data = torch.as_tensor(data, dtype=torch.float32)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.shape[-1] != nodes:
            filled = torch.zeros(n_obs, nodes)
            filled[mask == 1] = data.flatten()
            data = filled

        if samples.shape[-1] != nodes:
            latent_columns = torch.where(mask[0] == 0)[0]
            if samples.shape[-1] != latent_columns.numel():
                raise ValueError(
                    "samples must span either every node "
                    f"({nodes}) or every latent node ({latent_columns.numel()}), "
                    f"got {samples.shape[-1]}."
                )
            filled = torch.zeros(n_obs, samples.shape[1], nodes)
            filled[:, :, latent_columns] = samples
            samples = filled

        mask_expanded = mask.unsqueeze(1)
        points = (
            samples * (1 - mask_expanded) + data.unsqueeze(1) * mask_expanded
        )
        return points, mask

    def certify_composition(
            self, samples, data, condition_mask, hierarchy=None, prior=None,
            timesteps=100, eps=1e-3, divergence="exact", hutchinson_samples=32,
            proposal="gaussian", log_proposal=None, resample=False,
            device="cpu", batch_size=4096, verbose=False):
        """Score composed samples against the *exact* tall-data target.

        Nothing else in this class measures composition error: every correction
        is an approximation to the multi-observation score, and until now the
        only way to judge one was to compare posteriors on a problem whose truth
        was already known. The probability-flow ODE removes that limitation,
        because the tall-data target factorizes into quantities the trained
        network can evaluate exactly:

            p(g, l_1..l_n | x_1..x_n)
                proportional to  p(g)^(1-n) prod_j p(g, l_j | x_j)

        (each l_j's own prior appears in exactly one factor, so only the shared
        prior is over-counted). Every factor on the right is a *single*
        observation conditional density, which :meth:`PFODE.log_prob` computes
        without a KDE, so

            log q(theta) = (1 - n) log p(g) + sum_j log p(g, l_j | x_j) + const

        is available at any point, for any n, at the cost of ``n * num_samples``
        PF-ODE evaluations -- once, on the returned samples, not once per
        sampler step.

        What this does and does not certify. The densities come from the same
        trained network that produced the samples, so this isolates *composition*
        error and is blind to the network's own error: a systematically biased
        score yields a biased target and biased samples, and they will agree.
        It is exactly the right instrument for choosing between corrections,
        tuning ``jacobian_refresh``, or deciding whether Langevin correctors
        helped, and the wrong one for asking whether the network is trained
        well enough. The target is also smoothed by ``sigma(eps)``, like every
        PF-ODE density here.

        Reweighting. The composed sampler's own density is not tractable, so the
        weights are self-normalized against an explicit proposal fitted to the
        returned samples (``proposal="gaussian"``: one moment-matched Gaussian
        over the whole latent vector; ``"factorized"``: a shared block times one
        block per observation, better conditioned when ``n`` is large;
        ``None`` or an explicit ``log_proposal``: your own). The effective
        sample size then measures how far the composed samples sit from the
        exact target, and ``resample=True`` returns draws corrected towards it.
        A high ESS is evidence the composition landed on the target; a low one
        localizes the disagreement, and the returned per-observation terms say
        which observations drive it.

        Args:
            samples: Composed draws, ``(observations, samples, nodes)`` or just
                the latent columns as returned by ``ScoreBasedInferenceModel.sample``.
            data: The observations the samples were drawn from.
            condition_mask: 1 = observed, 0 = latent.
            hierarchy: Shared latent indices; defaults to the configured ones,
                then to every latent coordinate.
            prior: Gaussian prior over the hierarchy, as in :meth:`sample`.
            proposal: ``"gaussian"``, ``"factorized"``, or None.
            log_proposal: Explicit per-sample log proposal density, overriding
                ``proposal``.
            resample: Also return systematically resampled draws.

        Returns:
            dict with ``log_target`` (samples,), ``log_prior_term``,
            ``log_observation_terms`` (observations, samples),
            ``log_proposal``/``log_weights``/``weights`` (when a proposal is
            available), ``ess``, ``ess_fraction``, ``log_evidence_gap``, and
            ``resampled_samples`` when requested.
        """
        points, mask = self._joint_points(samples, data, condition_mask)
        n_obs, num_samples, nodes = points.shape

        if hierarchy is None:
            hierarchy = getattr(self, "hierarchy", None)
        latent_indices = torch.where(mask[0] == 0)[0].tolist()
        if hierarchy is None:
            hierarchy = latent_indices
        hierarchy = [int(index) for index in hierarchy]
        if not set(hierarchy) <= set(latent_indices):
            raise ValueError("hierarchy indices must refer to latent coordinates.")
        local_indices = [
            index for index in latent_indices if index not in set(hierarchy)
        ]

        shared = points[:, :, hierarchy]
        drift = (shared - shared[:1]).abs().max().item()
        if drift > 1e-4:
            raise ValueError(
                "Shared coordinates must be synchronized across observations "
                f"before certification; maximum deviation is {drift:.3e}."
            )

        saved_hierarchy = getattr(self, "hierarchy", None)
        self.hierarchy = hierarchy
        try:
            if prior is None and saved_hierarchy == hierarchy and getattr(
                    self, "prior_covariance", None) is not None:
                prior_mean = self.prior_mean.detach().cpu()
                prior_covariance = self.prior_covariance.detach().cpu()
            else:
                prior_mean, _, prior_covariance = self._resolve_prior(prior)
        finally:
            self.hierarchy = saved_hierarchy

        # Single-observation conditionals, all (observation, sample) points in
        # one batched PF-ODE pass.
        flat_points = points.reshape(n_obs * num_samples, nodes)
        flat_mask = mask.unsqueeze(1).expand(-1, num_samples, -1).reshape(
            n_obs * num_samples, nodes
        )
        log_observation_terms = self.SBIm.pfode.log_prob(
            data=flat_points, condition_mask=flat_mask, timesteps=timesteps,
            eps=eps, divergence=divergence,
            hutchinson_samples=hutchinson_samples, device=device,
            batch_size=batch_size, verbose=verbose,
        ).reshape(n_obs, num_samples).to(torch.float64)

        prior_object = (getattr(self, "prior_object", None)
                        if prior is None and saved_hierarchy == hierarchy else None)
        if prior is not None:
            from .priors import resolve_prior_object
            prior_object = resolve_prior_object(prior, len(hierarchy))
        if prior_object is not None:
            # A non-Gaussian prior's own density: the Gaussian moment match here would
            # put (n-1) copies of the wrong log-prior into the certified target.
            log_prior = prior_object.log_prob(shared[0].to(torch.float64)).to(torch.float64)
        else:
            log_prior = self._gaussian_log_density(
                shared[0].to(torch.float64), prior_mean, prior_covariance,
            )
        log_prior_term = (1 - n_obs) * log_prior
        log_target = log_prior_term + log_observation_terms.sum(dim=0)

        result = {
            "num_observations": n_obs,
            "num_samples": num_samples,
            "log_target": log_target,
            "log_prior_term": log_prior_term,
            "log_observation_terms": log_observation_terms,
        }

        if log_proposal is None and proposal is not None:
            log_proposal = self._fit_log_proposal(
                shared[0], points[:, :, local_indices], proposal,
            )
        if log_proposal is None:
            return result

        log_proposal = torch.as_tensor(log_proposal, dtype=torch.float64)
        log_weights = log_target - log_proposal
        weights = torch.softmax(log_weights, dim=0)
        ess = float(1.0 / weights.square().sum())
        result.update({
            "log_proposal": log_proposal,
            "log_weights": log_weights,
            "weights": weights,
            "ess": ess,
            "ess_fraction": ess / num_samples,
            # log of the mean unnormalized weight: the self-normalized estimate
            # of log(Z_target / Z_proposal), useful for comparing corrections.
            "log_evidence_gap": float(
                torch.logsumexp(log_weights, dim=0)
                - math.log(num_samples)
            ),
        })

        if resample:
            cumulative = torch.cumsum(weights, dim=0)
            positions = (
                torch.arange(num_samples, dtype=torch.float64) + 0.5
            ) / num_samples
            chosen = torch.searchsorted(
                cumulative.contiguous(), positions.clamp(max=float(cumulative[-1]))
            ).clamp(max=num_samples - 1)
            result["resample_indices"] = chosen
            result["resampled_samples"] = points[:, chosen, :]
        return result

    def _fit_log_proposal(self, shared, locals_, proposal):
        """Moment-matched Gaussian proposal density over the composed draws."""
        num_samples = shared.shape[0]
        shared = shared.to(torch.float64)
        locals_ = locals_.to(torch.float64)
        has_locals = locals_.shape[-1] > 0

        def block_density(block):
            mean = block.mean(dim=0)
            covariance = self._regularize_covariance(
                torch.atleast_2d(torch.cov(block.mT))
            )
            return self._gaussian_log_density(block, mean, covariance)

        if proposal == "factorized":
            density = block_density(shared)
            if has_locals:
                for observation in range(locals_.shape[0]):
                    density = density + block_density(locals_[observation])
            return density

        if proposal != "gaussian":
            raise ValueError(
                "proposal must be 'gaussian', 'factorized', None, or replaced "
                f"by an explicit log_proposal; got {proposal!r}."
            )

        blocks = [shared]
        if has_locals:
            blocks.extend(locals_[observation] for observation in range(locals_.shape[0]))
        flat = torch.cat(blocks, dim=-1)
        if flat.shape[-1] >= num_samples:
            # An unregularizable empirical covariance; the block-diagonal
            # proposal is the honest fallback rather than a singular fit.
            return self._fit_log_proposal(shared, locals_, "factorized")
        return block_density(flat)

    def composition_curl(
            self, samples, data, condition_mask, times=None, observations=None,
            step=None, cfg_alpha=None, device=None):
        """Measure how far the composed score field is from being a gradient.

        A score is by definition a gradient, so its Jacobian is symmetric. A
        *composed* score is assembled from n separate per-observation scores
        weighted by n separate precision matrices, and nothing in that
        construction forces the result to be conservative -- the denoised-
        prediction clamps break it further. This matters concretely: the
        reverse-diffusion predictor only integrates the field, but a Langevin
        corrector is an MCMC kernel whose invariant distribution is defined by
        the field *being* a gradient. Where the antisymmetry is large,
        correctors do not refine the sample, they circulate it, which is the
        signature of correctors making a composed estimate worse rather than
        better.

        This evaluates the composed field on the latent block by central
        differences and reports the relative antisymmetry
        ``||J - J^T||_F / ||J||_F`` of its Jacobian, per diffusion time. Zero
        means conservative (and a single observation must give zero, since the
        composition is then the network's own score). Finite differences rather
        than autodiff, deliberately: the composition contains eigendecompositions
        and clamps whose derivatives are ill-conditioned or undefined, while the
        field itself is perfectly well defined.

        The sampler must already be configured -- call this after
        :meth:`sample` or :meth:`map_estimate` with the same settings, so the
        measured field is the one that actually produced the samples.

        Probed coordinates are the shared block (perturbed in every observation
        row at once, since it is synchronized) plus the local block of a subset
        of observations. Cross terms between the locals of *different*
        observations vanish identically in both directions, so restricting to a
        subset loses nothing.

        Args:
            samples, data, condition_mask: As in :meth:`certify_composition`.
            times: Diffusion times to probe. Defaults to five log-spaced points
                across the sampler's own schedule.
            observations: How many observation rows to probe local coordinates
                for (default ``min(n, 4)``).
            step: Finite-difference step; defaults to ``1e-3 * lambda(t)``, i.e.
                relative to the local noise scale.

        Returns:
            dict with ``times``, and per-time ``asymmetry_mean``/``_max``,
            ``shared_asymmetry_mean``, ``max_abs_curl`` and ``jacobian``.
        """
        if getattr(self, "correction", None) is None or getattr(
                self, "hierarchy", None) is None:
            raise RuntimeError(
                "composition_curl needs a configured sampler: run sample() or "
                "map_estimate() with the settings you want to diagnose first."
            )

        device = self.device if device is None else device
        points, mask = self._joint_points(samples, data, condition_mask)
        n_obs, num_samples, _ = points.shape
        points = points.to(device)
        mask = mask.to(device)
        expanded_mask = mask.unsqueeze(1).expand(-1, num_samples, -1)
        latent = 1 - expanded_mask
        indices = torch.arange(n_obs, device=device)

        hierarchy = list(self.hierarchy)
        local_indices = list(getattr(self, "local_latent_indices", []) or [])
        probe_rows = min(
            n_obs, int(observations) if observations is not None else 4
        )

        # (column, rows) pairs: the shared block moves in every row at once.
        probes = [(column, None) for column in hierarchy]
        probes += [
            (column, row)
            for row in range(probe_rows)
            for column in local_indices
        ]
        width = len(probes)

        if times is None:
            schedule = getattr(self, "timesteps_list", None)
            if schedule is None:
                one = torch.ones(1, device=device)
                schedule = self.sde.time_of_lambda(torch.logspace(
                    torch.log10(self.sde.lambda_t(one)).item(),
                    torch.log10(self.sde.lambda_t(1e-3 * one)).item(),
                    5, device=device,
                ))
            picks = torch.linspace(
                0, len(schedule) - 1, min(5, len(schedule))
            ).round().long()
            times = schedule.to(device)[picks]
        times = torch.as_tensor(times, dtype=torch.float32, device=device).flatten()

        # A stale cached curvature would make the field piecewise constant and
        # silently zero the finite differences.
        saved_refresh = getattr(self, "jacobian_refresh", 1)
        saved_low_noise = getattr(self, "jacobian_refresh_low_noise", None)
        self.jacobian_refresh = 1
        self.jacobian_refresh_low_noise = None

        report = {
            "times": times.detach().cpu(), "probes": probes,
            "asymmetry_mean": [], "asymmetry_max": [],
            "shared_asymmetry_mean": [], "max_abs_curl": [], "jacobian": [],
        }
        try:
            for time_value in times:
                t = time_value.reshape(1, 1)
                lam = float(self.sde.lambda_t(t).reshape(()))
                delta = float(1e-3 * lam) if step is None else float(step)
                self._reset_jacobian_state()

                columns = []
                for column, row in probes:
                    tangent = torch.zeros_like(points)
                    if row is None:
                        tangent[:, :, column] = delta
                    else:
                        tangent[row, :, column] = delta
                    tangent = tangent * latent

                    plus = self._get_score(
                        points + tangent, t, expanded_mask, indices, cfg_alpha
                    )
                    minus = self._get_score(
                        points - tangent, t, expanded_mask, indices, cfg_alpha
                    )
                    derivative = (plus - minus) / (2.0 * delta)
                    columns.append(torch.stack([
                        derivative[0 if probe_row is None else probe_row,
                                   :, probe_column]
                        for probe_column, probe_row in probes
                    ], dim=-1))

                jacobian = torch.stack(columns, dim=-1).to(torch.float64)
                antisymmetric = jacobian - jacobian.mT
                norm = torch.linalg.matrix_norm(jacobian).clamp_min(
                    torch.finfo(torch.float64).eps
                )
                asymmetry = torch.linalg.matrix_norm(antisymmetric) / norm

                shared = len(hierarchy)
                shared_block = jacobian[..., :shared, :shared]
                shared_norm = torch.linalg.matrix_norm(shared_block).clamp_min(
                    torch.finfo(torch.float64).eps
                )
                shared_asymmetry = torch.linalg.matrix_norm(
                    shared_block - shared_block.mT
                ) / shared_norm

                report["asymmetry_mean"].append(float(asymmetry.mean()))
                report["asymmetry_max"].append(float(asymmetry.max()))
                report["shared_asymmetry_mean"].append(float(shared_asymmetry.mean()))
                report["max_abs_curl"].append(float(antisymmetric.abs().max()))
                report["jacobian"].append(jacobian.mean(dim=0).cpu())
        finally:
            self.jacobian_refresh = saved_refresh
            self.jacobian_refresh_low_noise = saved_low_noise
            self._reset_jacobian_state()

        report["width"] = width
        return report

    #############################################
    # ----- Multi-GPU setup -----
    #############################################

    def _ddp_setup(self, rank, world_size):
        # Setup DistributedDataParallel
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ["MASTER_PORT"] = "29500"

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=100_000_000)
        )

        self.SBIm.model.eval()
        self.device = torch.device(f'cuda:{rank}')
        self.SBIm.model.to(self.device)
        self.model = DDP(self.SBIm.model, device_ids=[rank])

    def _gather_samples(self, all_samples, indices, result_dict):
        # Gather samples from all processes
        # Convert the list of tensors to a single tensor
        samples = torch.cat(all_samples, dim=0).to(self.device)
        indices = torch.cat(indices, dim=0).to(self.device)

        # Create empty tensors to gather results across processes
        gathered_samples = [torch.zeros_like(samples) for _ in range(self.world_size)]
        gathered_idx = [torch.zeros_like(indices) for _ in range(self.world_size)]

        # Gather data from all processes
        dist.all_gather(gathered_samples, samples)
        dist.all_gather(gathered_idx, indices)

        if self.rank == 0:
            # Sort results by index
            gathered_idx = torch.cat(gathered_idx, dim=0)
            gathered_samples = torch.cat(gathered_samples, dim=0)
            unique_sort_idx = [(gathered_idx == i).nonzero()[0,0].tolist() for i in gathered_idx.unique()]
            samples = gathered_samples[unique_sort_idx]

            result_dict['samples'] = samples.cpu()

    def _gather_scores(self, score_table, indices):
        # Gather scores from all processes and deduplicate
        # (the DistributedSampler pads shards, so observations can appear twice)
        gathered_scores = [torch.zeros_like(score_table) for _ in range(self.world_size)]

        indices = indices.to(self.device)
        gathered_idx = [torch.zeros_like(indices) for _ in range(self.world_size)]

        dist.all_gather(gathered_scores, score_table)
        dist.all_gather(gathered_idx, indices)

        # Sort results by index, keeping one entry per unique observation
        gathered_idx = torch.cat(gathered_idx, dim=0)
        gathered_scores = torch.cat(gathered_scores, dim=0)
        unique_sort_idx = [(gathered_idx == i).nonzero()[0,0].tolist() for i in gathered_idx.unique()]
        gathered_scores = gathered_scores[unique_sort_idx]

        return gathered_scores

    #############################################
    # ----- Standard Functions -----
    #############################################

    def _get_score(self, x, t, condition_mask, indices, cfg_alpha=None):
        """Get the composed score estimate with optional classifier-free guidance"""
        with torch.no_grad():
            n_obs, num_samples, num_features = x.shape

            selected = None
            if (
                self.correction in self.TRUE_MINIBATCH_CORRECTIONS
                and self.composition_batch_size < n_obs
            ):
                selected = torch.randperm(
                    n_obs, device=x.device
                )[:self.composition_batch_size]
                network_x = x[selected]
                network_mask = condition_mask[selected]
            else:
                network_x = x
                network_mask = condition_mask

            evaluated_rows = network_x.shape[0]
            alpha = self.sde.alpha_t(t).to(x.device)
            network_x = network_x * (
                alpha * (1 - network_mask) + network_mask
            )

            # Flatten all selected observation/sample rows into one network call.
            # so all observations are processed in a single forward pass
            x_flat = network_x.reshape(evaluated_rows * num_samples, num_features)
            c_flat = network_mask.reshape(evaluated_rows * num_samples, num_features)

            scores_flat = self.SBIm.model(x=x_flat, t=t, c=c_flat)
            scores_flat = self.SBIm.output_scale_function(t, scores_flat)
            self.score_network_calls += 1
            self.evaluated_subject_rows += int(evaluated_rows)
            self.evaluated_row_samples += int(evaluated_rows) * int(num_samples)

            if cfg_alpha is not None:
                scores_uncond = self.SBIm.model(x=x_flat, t=t, c=torch.zeros_like(c_flat))
                scores_uncond = self.SBIm.output_scale_function(t, scores_uncond)
                scores_flat = scores_uncond + cfg_alpha * (scores_flat - scores_uncond)
                self.score_network_calls += 1
                self.evaluated_subject_rows += int(evaluated_rows)
                self.evaluated_row_samples += int(evaluated_rows) * int(num_samples)

            # Network scores are in x-space; samplers step in y=x/alpha space.
            score_table = (alpha * scores_flat).reshape(
                evaluated_rows, num_samples, num_features
            )

            if self.world_size > 1:
                dist.barrier()
                score_table = self._gather_scores(score_table, indices)

            score = self._compositional_score(
                score_table, x, t, num_observations=n_obs,
                minibatch_selected=selected is not None,
                condition_mask=condition_mask,
            )

        if self.world_size > 1:
            return score[indices]
        return score

    def _diffused_gaussian_prior_score(self, theta_y, t):
        """Return the noised prior score in y=x/alpha coordinates.

        With an object prior (``compass.priors``) this is that prior's own exact
        diffused score; the Gaussian branches below are the historical path and
        are what a ``prior=(mean, scale)`` tuple still takes.
        """
        prior_object = getattr(self, "prior_object", None)
        if prior_object is not None:
            lam = self.sde.lambda_t(t).to(theta_y.device).reshape(())
            return prior_object.score(theta_y, t, lam).to(theta_y.dtype)
        if self.correction in self.COVARIANCE_GAUSSIAN_CORRECTIONS:
            var_t = self.sde.lambda_t(t).to(theta_y.device).square()
            identity = torch.eye(
                len(self.hierarchy), dtype=theta_y.dtype, device=theta_y.device
            )
            covariance_t = self.prior_covariance.to(theta_y) + var_t.reshape(()) * identity
            delta = theta_y - self.prior_mean.to(theta_y)
            return -torch.linalg.solve(
                covariance_t, delta.squeeze(0).mT
            ).mT.unsqueeze(0)

        alpha = self.sde.alpha_t(t).to(theta_y.device)
        sigma = self.sde.sigma_t(t).to(theta_y.device)
        theta_x = alpha * theta_y
        mean_x = alpha * self.prior_mean
        variance_x = alpha**2 * self.prior_std**2 + sigma**2
        return alpha * (-(theta_x - mean_x) / variance_x)

    def _clamp_local_scores(self, scores, x, var_t):
        """Bound the denoised prediction of the row-local latents.

        The shared coordinates are always clamped to a box around their prior;
        the local latents had no analogous bound, which is what turns a bad
        local score -- or a local cross-correction fed by a badly conditioned
        composition -- from a bias into an unbounded runaway.

        Uses the *marginal* local prior rather than the hierarchy-implied
        conditional one (mean +- k exp(log_sigma) read off the current shared
        draw). The conditional scale is itself an estimated shared coordinate,
        so a bad global draw would widen or collapse the very bound meant to
        contain it; the marginal box does not inherit that failure.

        No-ops unless a local prior was supplied, so corrections that never
        touch the local coordinates keep their exact previous behavior.
        """
        indices = getattr(self, "local_latent_indices", None)
        if (self.denoise_clamp is None or not indices
                or getattr(self, "local_prior_mean", None) is None
                or scores.shape[0] != x.shape[0]):
            return scores
        mean = self.local_prior_mean.to(scores)
        std = self.local_prior_std.to(scores)
        theta_l = x[:, :, indices]
        self._count_clamp(theta_l + var_t * scores[:, :, indices],
                          mean - self.denoise_clamp * std, mean + self.denoise_clamp * std)
        x0 = torch.clamp(
            theta_l + var_t * scores[:, :, indices],
            min=mean - self.denoise_clamp * std,
            max=mean + self.denoise_clamp * std,
        )
        scores[:, :, indices] = (x0 - theta_l) / var_t
        return scores

    CLAMP_MODES = ("coordinate", "metric")

    def set_clamp_mode(self, mode):
        """How the denoised predictions m = y + lambda^2 s are clamped into their box.

        - "coordinate" (default): clip every coordinate on its own. The clamped score
          (clip(m) - y)/lambda^2 is not conservative where some coordinates are clipped
          and others are not: d s~_i/d y_j - d s~_j/d y_i = (D_ii - D_jj) H_ij / ...
        - "metric": m~ = argmin_{z in B} (z - m)^T A (z - m) with A the inverse
          Jacobian of the denoiser, i.e. the backward precision -- the row's joint
          Lambda_j(t) (globals and locals clamped together) and the composed
          Lambda(t) for the composed score. This is the conjugate construction
          psi_B = (psi* + iota_B)* with psi(y) = |y|^2/2 + lambda^2 log p_t(y) taken
          to second order, so it is exactly conservative where psi is quadratic and
          leaves m unchanged wherever m is already inside B. Corrections without a
          precision fall back to "coordinate".
        """
        if mode not in self.CLAMP_MODES:
            raise ValueError(f"clamp mode must be one of {self.CLAMP_MODES}")
        self.clamp_mode = mode

    @staticmethod
    def _box_qp(center, metric, lower, upper, max_iterations=None):
        """argmin_{lower <= z <= upper} (z - center)^T metric (z - center), batched.

        center/lower/upper (..., D) (bounds may be +-inf), metric (..., D, D) SPD.
        Primal active-set method (Nocedal & Wright, Alg. 16.3), run in lockstep over the
        batch: from the feasible clip, solve the equality QP on the free coordinates,
        step to it or to the first blocking bound, and release the bound with the most
        negative multiplier once the step is zero. Finite termination at the KKT point.
        Rows whose center lies inside the box are returned exactly unchanged.
        """
        dimension = center.shape[-1]
        max_iterations = 6 * dimension + 10 if max_iterations is None else max_iterations
        center = center.to(torch.float64)
        metric = metric.to(torch.float64).expand(center.shape[:-1] + (dimension, dimension))
        lower = lower.to(torch.float64).expand_as(center)
        upper = upper.to(torch.float64).expand_as(center)
        z = torch.maximum(torch.minimum(center, upper), lower)
        outside = (z != center).any(dim=-1)
        if not bool(outside.any()):
            return center
        at_lower = z == lower
        at_upper = (z == upper) & ~at_lower
        eye = torch.eye(dimension, dtype=torch.float64, device=center.device)
        scale = center.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        tolerance = 1e-12 * scale
        for _ in range(max_iterations):
            active = at_lower | at_upper
            free = ~active
            fixed = torch.where(active, z - center, torch.zeros_like(z))
            rhs = torch.where(free, -(metric @ fixed.unsqueeze(-1)).squeeze(-1), torch.zeros_like(z))
            pair = free.unsqueeze(-1) & free.unsqueeze(-2)
            system = torch.where(pair, metric, torch.zeros_like(metric)) + eye * active.unsqueeze(-1)
            target = torch.where(free, center + torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1), z)
            step = target - z
            moving = (step.abs() > tolerance).any(dim=-1, keepdim=True)
            # largest feasible fraction of the step
            room = torch.where(step < 0, (lower - z) / step.clamp(max=-1e-300),
                               torch.where(step > 0, (upper - z) / step.clamp(min=1e-300),
                                           torch.full_like(z, float("inf"))))
            room = torch.where(free, room.clamp_min(0.0), torch.full_like(z, float("inf")))
            alpha, blocking = room.min(dim=-1, keepdim=True)
            alpha = alpha.clamp(max=1.0)
            blocked = moving & (alpha < 1.0)
            z = torch.where(moving, z + alpha * step, z)
            hit = torch.zeros_like(free).scatter(-1, blocking, True) & blocked
            hit_lower = hit & (step < 0)
            hit_upper = hit & (step > 0)
            z = torch.where(hit_lower, lower, torch.where(hit_upper, upper, z))
            z = torch.maximum(torch.minimum(z, upper), lower)
            at_lower = at_lower | hit_lower
            at_upper = at_upper | hit_upper
            # multipliers where the step is zero: g >= 0 at a lower bound, g <= 0 at an upper one
            gradient = (metric @ (z - center).unsqueeze(-1)).squeeze(-1)
            wrong = torch.where(at_lower, -gradient, torch.where(at_upper, gradient, torch.zeros_like(z)))
            wrong = torch.where(moving, torch.zeros_like(wrong), wrong)
            worst, index = wrong.max(dim=-1, keepdim=True)
            release = torch.zeros_like(free).scatter(-1, index, True) & (worst > tolerance * metric.diagonal(
                dim1=-2, dim2=-1).abs().amax(dim=-1, keepdim=True))
            at_lower = at_lower & ~release
            at_upper = at_upper & ~release
            if not bool((moving | release.any(dim=-1, keepdim=True)).any()):
                break
        return torch.where(outside.unsqueeze(-1), z, center)

    def _count_clamp(self, center, lower, upper):
        """Diagnostics: share of (row, sample) vectors whose denoised prediction left the box."""
        out = ((center < lower) | (center > upper)).any(dim=-1)
        self._clamp_calls = getattr(self, "_clamp_calls", 0) + 1
        self._clamp_active = getattr(self, "_clamp_active", 0) + int(out.sum())
        self._clamp_elements = getattr(self, "_clamp_elements", 0) + int(out.numel())

    def _row_clamp_metric(self, var_t, rows):
        """(features, metric) for the joint per-row clamp, or None when no precision exists."""
        if self.correction in self.JACOBIAN_GAUSSIAN_CORRECTIONS:
            metric = getattr(self, "_last_joint_precision", None)
            if metric is None:
                return None
            return list(self.full_gaussian_features), metric
        if (self.correction in self.COVARIANCE_GAUSSIAN_CORRECTIONS
                and getattr(self, "posterior_precision_matrix", None) is not None):
            precision = self.posterior_precision_matrix.to(torch.float64)
            identity = torch.eye(precision.shape[-1], dtype=torch.float64, device=precision.device)
            metric = precision + identity / float(var_t)
            if metric.dim() == 2:
                metric = metric.unsqueeze(0)
            return list(self.covariance_features), metric.unsqueeze(1)
        return None

    def _metric_row_clamp(self, scores, x, var_t, global_lower, global_upper, features, metric):
        """Clamp each row's joint denoised prediction in the metric of its own precision."""
        h = list(self.hierarchy)
        locals_ = list(getattr(self, "local_latent_indices", None) or [])
        has_local_box = (self.denoise_clamp is not None and locals_
                         and getattr(self, "local_prior_mean", None) is not None
                         and scores.shape[0] == x.shape[0])
        lower = torch.full((len(features),), float("-inf"), dtype=torch.float64, device=x.device)
        upper = torch.full_like(lower, float("inf"))
        position = {f: i for i, f in enumerate(features)}
        for k, f in enumerate(h):
            lower[position[f]] = float(torch.as_tensor(global_lower).flatten()[k])
            upper[position[f]] = float(torch.as_tensor(global_upper).flatten()[k])
        if has_local_box:
            mean = self.local_prior_mean.to(device=x.device, dtype=torch.float64)
            std = self.local_prior_std.to(device=x.device, dtype=torch.float64)
            for k, f in enumerate(locals_):
                if f in position:
                    lower[position[f]] = mean[k] - self.denoise_clamp * std[k]
                    upper[position[f]] = mean[k] + self.denoise_clamp * std[k]
        y = x[:, :, features].to(torch.float64)
        center = y + float(var_t) * scores[:, :, features].to(torch.float64)
        self._count_clamp(center, lower, upper)
        clamped = self._box_qp(center, metric, lower, upper)
        scores[:, :, features] = ((clamped - y) / float(var_t)).to(scores.dtype)
        if has_local_box and any(f not in position for f in locals_):
            scores = self._clamp_local_scores(scores, x, var_t)
        return scores

    def _metric_local_clamp(self, scores, x, var_t):
        """Local clamp after the cross-correction, in the metric Lambda_j,ll (globals fixed)."""
        locals_ = list(getattr(self, "local_latent_indices", None) or [])
        metric = getattr(self, "_last_joint_precision", None)
        h = len(self.hierarchy)
        if (self.denoise_clamp is None or not locals_ or metric is None
                or getattr(self, "local_prior_mean", None) is None
                or scores.shape[0] != x.shape[0]
                or list(self.full_gaussian_features) != list(self.hierarchy) + locals_):
            return self._clamp_local_scores(scores, x, var_t)
        mean = self.local_prior_mean.to(device=x.device, dtype=torch.float64)
        std = self.local_prior_std.to(device=x.device, dtype=torch.float64)
        y = x[:, :, locals_].to(torch.float64)
        center = y + float(var_t) * scores[:, :, locals_].to(torch.float64)
        lower, upper = mean - self.denoise_clamp * std, mean + self.denoise_clamp * std
        self._count_clamp(center, lower, upper)
        clamped = self._box_qp(center, metric[..., h:, h:], lower, upper)
        scores[:, :, locals_] = ((clamped - y) / float(var_t)).to(scores.dtype)
        return scores

    @staticmethod
    def _weight_global_scores(precision, scores):
        """Contract per-observation precisions with per-observation scores.

        ``precision`` is ``(n, H, H)`` for the pilot-covariance corrections and
        ``(n, samples, H, H)`` when it is state-dependent (`gauss_jacobian`).
        """
        if precision.dim() == 4:
            return torch.einsum("nsij,nsj->nsi", precision, scores)
        return torch.einsum("nij,nsj->nsi", precision, scores)

    def _compositional_score(
        self, scores, x, t, num_observations=None, minibatch_selected=False,
        condition_mask=None,
    ):
        """
        Compose the per-observation scores on the hierarchy (shared parameter)
        dimensions. Local latent dimensions keep their per-observation score.

        The state x is kept synchronized across the observation axis on the hierarchy
        dimensions, so x[0] provides the shared parameter values.

        - "fnpe" (Geffner et al. 2023, Eq. 7):
              s = (1-n)(1-t) * prior_score + sum_j s_j
          with the undiffused prior score, valid for annealed Langevin sampling only.
        - "uncorrected":
              s = (1-n) * diffused_prior_score + sum_j s_j
        - "gauss" (Gloeckler et al. 2024):
              s = Lambda^-1 [ (1-n) * Lambda_prior * diffused_prior_score
                              + sum_j Lambda_j * s_j ]
          where Lambda_prior / Lambda_j are the *denoising* precisions of prior and
          single-observation posteriors, and Lambda = (1-n) Lambda_prior + sum_j Lambda_j.
        - "full_gaussian" (Gloeckler et al. 2024, Algorithm 2): the same
          precision weighting with full covariance matrices and a linear solve.
        - "gauss_jacobian": the hierarchical arrow-elimination composition with
          Lambda_j(t) taken from the network's own Jacobian via Tweedie's
          second-order identity, so the weighting is exact for any
          single-observation posterior and needs no pilot covariance.
        - "damping" (Arruda et al. 2026, Eq. 9):
              s = d(t) [ (1-n)(1-t) prior_score + n/m sum_(j in B) s_j ]
        - "gauss_damping": d(t) times the Gaussian-corrected score.
        - "hybrid": Gaussian correction with the prior terms weighted by (1-t),
          including the corresponding adjusted denominator.
        - "hybrid_damping": the same hybrid correction, then multiplied by d(t).
        """
        h = self.hierarchy
        n = scores.shape[0] if num_observations is None else int(num_observations)
        m = scores.shape[0]
        pending_cross_correction = None
        composed_solved_precision = None

        var_t = self.sde.lambda_t(t).to(x.device)**2
        scores = self._moment_projected_scores(scores, x, var_t)
        # The secant update needs the network's own row scores, before the
        # clamps below rewrite them in place.
        raw_row_scores = (
            scores.detach().clone()
            if getattr(self, "jacobian_secant", False)
            and self.correction in self.JACOBIAN_GAUSSIAN_CORRECTIONS
            else None
        )

        theta_h = x[:1, :, h]                                   # (1, num_samples, H), shared
        early_factors = None
        row_metric = None

        # Stabilize the individual scores by clamping their denoised prediction
        # E[theta_0 | theta_t, x_j] = theta_t + var_t * s_j to a box around the prior.
        # Far out in the tails of the reference distribution the network
        # extrapolates; without this, single bad tail scores get amplified n-fold
        # by the composition and samples can run away.
        if self.denoise_clamp is not None:
            lo = self.prior_mean - self.denoise_clamp * self.prior_std
            hi = self.prior_mean + self.denoise_clamp * self.prior_std
            # The individual scores were trained under the training prior, so
            # their box is that prior's; the composed clamp below keeps lo/hi.
            ilo, ihi = lo, hi
            if self.training_prior is not None:
                t_mean, t_std = (v.to(theta_h.device) for v in self.training_prior)
                ilo = t_mean - self.denoise_clamp * t_std
                ihi = t_mean + self.denoise_clamp * t_std
            row_metric = None
            if self.clamp_mode == "metric":
                if (early_factors is None
                        and self.correction in self.COVARIANCE_GAUSSIAN_CORRECTIONS):
                    # the metric is the row precision: build it before the clamp
                    early_factors = self._effective_global_factors(
                        var_t, n, state=x, t=t, condition_mask=condition_mask,
                        row_scores=raw_row_scores,
                    )
                row_metric = self._row_clamp_metric(var_t, scores.shape[0])
            if row_metric is not None:
                scores = self._metric_row_clamp(scores, x, var_t, ilo, ihi, *row_metric)
            else:
                self._count_clamp(theta_h + var_t * scores[:, :, h],
                                  torch.as_tensor(ilo).to(theta_h), torch.as_tensor(ihi).to(theta_h))
                x0 = torch.clamp(theta_h + var_t * scores[:, :, h], min=ilo, max=ihi)
                scores[:, :, h] = (x0 - theta_h) / var_t
        # The same tail protection for the row-local latents. Without a local
        # prior nothing here bounds them, so a bad local score -- or a local
        # cross-correction fed by a bad composed score -- is free to run away.
        if not (self.denoise_clamp is not None and row_metric is not None):
            scores = self._clamp_local_scores(scores, x, var_t)

        sum_scores = scores[:, :, h].sum(dim=0, keepdim=True)
        scaled_sum_scores = sum_scores * (n / m) if minibatch_selected else sum_scores
        prior_score = self._diffused_gaussian_prior_score(theta_h, t)

        if (self.training_prior is not None
                and self.correction not in self.COVARIANCE_GAUSSIAN_CORRECTIONS):
            raise NotImplementedError(
                f"set_training_prior is only implemented for {sorted(self.COVARIANCE_GAUSSIAN_CORRECTIONS)}")

        if self.correction == "fnpe":
            # Eq. 7 with the *diffused* prior score. With the undiffused prior of the
            # paper (which assumes a VP-type diffusion and a standard-normal prior),
            # the VESDE bridging densities become improper for large n at
            # intermediate t (negative total precision) and Langevin diverges.
            # Diffusing the prior keeps them proper and recovers Eq. 7 as t -> 0.
            composed = (1 - n) * (1 - t) * prior_score + sum_scores

        elif self.correction in {"uncorrected", "prior_corrected_sum"}:
            composed = (1 - n) * prior_score + sum_scores

        elif self.correction == "legacy_mean":
            # Preserve the historical prior-plus-mean rule for R>1, while the
            # single-observation boundary must equal the ordinary model score.
            composed = sum_scores if n == 1 else prior_score + sum_scores / n

        elif self.correction == "damped_sum":
            composed = ((1 - n) * prior_score + sum_scores) / n

        elif self.correction == "minibatch_damped":
            composed = ((1 - n) * prior_score + scaled_sum_scores) / n

        elif self.correction == "damping":
            composed = self._damping_factor(t) * (
                (1 - n) * (1 - t) * prior_score + scaled_sum_scores
            )

        elif self.correction == "bp_jac":
            composed = self._bp_jacobian_score(
                scores, x, t, condition_mask, var_t
            )

        elif self.correction in self.COVARIANCE_GAUSSIAN_CORRECTIONS:
            effective_precision, cross_factor = early_factors if early_factors is not None else (
                self._effective_global_factors(
                    var_t, n, state=x, t=t, condition_mask=condition_mask,
                    row_scores=raw_row_scores,
                ))
            if self.correction == "Gauss_global_local" and (
                    getattr(self, "global_posterior_mean", None) is not None):
                # Compose only marginal p(g_t | x_j) scores.  The joint-network
                # g score is conditional on the current local state and must not
                # enter this cross-subject update.
                effective_precision, _ = self._marginal_global_factors(var_t)
                if self.pd_strategy != "row_projection":
                    raise NotImplementedError(
                        "set_pd_strategy: only row_projection is implemented for the "
                        "marginal Gauss_global_local branch")
                self._last_row_delta = None  # projected below, on its own terms
                # Marginal pilot moments need the same information projection
                # as the Schur ones; without it this branch composes against an
                # indefinite Lambda exactly as before.
                marginal_delta = self._project_information(
                    effective_precision, float(1.0 / var_t), len(h),
                )
                if marginal_delta is not None:
                    effective_precision = effective_precision + marginal_delta
                global_scores = self._marginal_global_scores(
                    theta_h, var_t, rows=scores.shape[0]
                )
            else:
                global_scores = scores[:, :, h].to(torch.float64)
            if effective_precision.dim() == 4:
                # State-dependent curvature: one precision per observation and
                # per sample, so the composed precision keeps a sample axis.
                precision_sum = effective_precision.sum(dim=0)
                weighted_sum = self._weight_global_scores(
                    effective_precision, global_scores
                ).sum(dim=0, keepdim=True)
            elif effective_precision.shape[0] == 1:
                precision_sum = n * effective_precision[0]
                weighted_sum = torch.einsum(
                    "ij,bsj->bsi", effective_precision[0],
                    global_scores.sum(dim=0, keepdim=True),
                )
            else:
                precision_sum = effective_precision.sum(dim=0)
                weighted_sum = self._weight_global_scores(
                    effective_precision, global_scores
                ).sum(dim=0, keepdim=True)

            inverse_variance = torch.as_tensor(
                1.0 / var_t, dtype=torch.float64, device=scores.device
            ).reshape(())
            identity = torch.eye(
                len(h), dtype=torch.float64, device=scores.device
            )
            precision_prior_t = (
                self.prior_precision_matrix.to(
                    device=scores.device, dtype=torch.float64
                )
                + inverse_variance * identity
            )
            weighted_prior = torch.einsum(
                "ij,bsj->bsi", precision_prior_t,
                prior_score.to(torch.float64),
            )
            # The prior each row's information I_j is measured against (for row_truncation).
            reference = (precision_prior_t, weighted_prior)
            if self.training_prior is None:
                composed_precision = (
                    precision_sum + (1 - n) * precision_prior_t
                )
                numerator = weighted_sum + (1 - n) * weighted_prior
            else:
                # p(g | X) ~ p(g) prod_j p~(g | x_j) / p~(g): each factor divides
                # out the prior it was trained under, the target prior enters once.
                train_precision_t, train_score = self._training_prior_terms(
                    theta_h, var_t, inverse_variance, identity)
                composed_precision = (
                    precision_sum - n * train_precision_t + precision_prior_t
                )
                numerator = weighted_sum + weighted_prior - n * torch.einsum(
                    "ij,bsj->bsi", train_precision_t, train_score)
                reference = (train_precision_t, torch.einsum(
                    "ij,bsj->bsi", train_precision_t, train_score))
            composed_precision, numerator = self._apply_pd_strategy(
                composed_precision, numerator, effective_precision,
                global_scores, inverse_variance, n, reference=reference, state=theta_h,
            )
            composed64, composed_solved_precision, _, _ = self._solve_composed_global(
                composed_precision, numerator, global_scores
            )
            composed = composed64.to(scores.dtype)

            if (
                self.correction in self.LOCAL_CROSS_CORRECTIONS
                and self.local_latent_indices
            ):
                # Deferred until after the composed denoised-prediction clamp
                # below. Reading the unclamped composed score here would let
                # the local latents bypass the only bound the shared
                # coordinates have, which is exactly how a marginally
                # indefinite composition reaches the locals unattenuated.
                pending_cross_correction = (cross_factor, global_scores)

        elif self.correction in self.DIAGONAL_GAUSSIAN_CORRECTIONS:
            Lambda_prior = 1.0 / self.prior_std**2 + 1.0 / var_t                  # (1, H)
            Lambda_j = self.posterior_precision + 1.0 / var_t                     # (n or 1, H)
            if Lambda_j.shape[0] == 1:
                Lambda_sum = n * Lambda_j
                weighted_sum = Lambda_j * sum_scores
            else:
                Lambda_sum = Lambda_j.sum(dim=0, keepdim=True)
                weighted_sum = (Lambda_j.unsqueeze(1) * scores[:, :, h]).sum(dim=0, keepdim=True)
            if self.correction in {"hybrid", "hybrid_damping"}:
                # The (1-t) prior-score bridge requires its own adaptive
                # precision normalizer.  Writing a(t) explicitly keeps this
                # adjustment isolated from the standard Gaussian correction:
                #   a(t) = (1-N)(1-t)
                #   Lambda_{t,a} = sum_j P_{t,j} + a(t) P_{t,0}.
                adaptive_prior_coefficient = (1 - n) * (1 - t)
                Lambda = (
                    Lambda_sum
                    + adaptive_prior_coefficient * Lambda_prior
                )
                numerator = (
                    weighted_sum
                    + adaptive_prior_coefficient * Lambda_prior * prior_score
                )
            else:
                # Standard Gaussian precision normalizer.  In particular,
                # gauss_damping changes only the final score magnitude by d(t).
                Lambda = (1 - n) * Lambda_prior + Lambda_sum
                numerator = (
                    (1 - n) * Lambda_prior * prior_score + weighted_sum
                )
            epsilon = torch.finfo(Lambda.dtype).eps
            if torch.any(Lambda <= epsilon):
                raise RuntimeError(
                    f"The '{self.correction}' composed precision became non-positive."
                )
            composed = numerator / Lambda
            if self.correction in self.DAMPING_CORRECTIONS:
                composed = self._damping_factor(t) * composed

        # Temper only the evidence contribution to the shared score. At t=0,
        # s - s_prior is the likelihood-ratio score. At intermediate VE
        # marginals this is a calibrated bridge; beta=1 is exactly the old path.
        beta = getattr(self, "composition_temperature", 1.0)
        if beta != 1.0:
            composed = prior_score + beta * (composed - prior_score)

        # Clamp the denoised prediction of the composed score as well. Not valid for
        # Damped scores deliberately change the score magnitude, so a second
        # denoised-prediction clamp here would undo the requested d(t) formula.
        # (The individual network-score tail clamp above still applies.)
        if (self.denoise_clamp is not None
                and self.correction not in ({"fnpe"} | self.DAMPED_CORRECTIONS)):
            composed_center = theta_h + var_t * composed
            lo_t = torch.as_tensor(lo).to(composed_center)
            hi_t = torch.as_tensor(hi).to(composed_center)
            self._count_clamp(composed_center, lo_t, hi_t)
            if self.clamp_mode == "metric" and composed_solved_precision is not None:
                x0 = self._box_qp(composed_center, composed_solved_precision, lo_t, hi_t)
                composed = ((x0 - theta_h.to(torch.float64)) / float(var_t)).to(composed.dtype)
            else:
                x0 = torch.clamp(composed_center, min=lo, max=hi)
                composed = (x0 - theta_h) / var_t

        if pending_cross_correction is not None:
            # s_l = s_l,j - Lambda_ll^-1 Lambda_lg (s_composed - s_g,j): the
            # local latent follows the shared coordinate through the joint
            # backward kernel. `composed` is the clamped score, so the local
            # kick inherits the bound the shared coordinates already carry.
            cross_factor, global_scores = pending_cross_correction
            global_delta = (
                composed.to(torch.float64).expand(n, -1, -1) - global_scores
            )
            if cross_factor.dim() == 4:
                local_delta = torch.einsum(
                    "nslg,nsg->nsl", cross_factor, global_delta
                )
            else:
                if cross_factor.shape[0] == 1:
                    cross_factor = cross_factor.expand(n, -1, -1)
                local_delta = torch.einsum(
                    "nlg,nsg->nsl", cross_factor, global_delta
                )
            scores[:, :, self.local_latent_indices] = (
                scores[:, :, self.local_latent_indices]
                + local_delta.to(scores.dtype)
            )
            if self.clamp_mode == "metric" and self.correction in self.JACOBIAN_GAUSSIAN_CORRECTIONS:
                scores = self._metric_local_clamp(scores, x, var_t)
            else:
                scores = self._clamp_local_scores(scores, x, var_t)

        if not torch.all(torch.isfinite(composed)):
            name = "full-Gaussian" if (
                self.correction in self.COVARIANCE_GAUSSIAN_CORRECTIONS
            ) else self.correction
            raise RuntimeError(
                f"The stabilized {name} score became non-finite."
            )

        # Broadcast the composed score to all observation rows on the shared dims
        result = scores if scores.shape[0] == x.shape[0] else torch.zeros_like(x)
        result[:, :, h] = composed
        return result

    def _final_denoise(self, x, condition_mask, idx):
        """
        Analytic denoising (Tweedie) step from the final diffusion time t_end to t=0
        on the shared (hierarchy) dimensions.

        The reverse diffusion stops at sigma_m(eps), so the samples follow the
        posterior *convolved* with N(0, sigma_m(eps)^2) -- a visible overdispersion
        once the composed posterior gets narrower than sigma_m(eps) (it contracts
        like 1/sqrt(n)). Under the Gaussian (posterior-precision) approximation
        already used by the "gauss" correction, p(theta_0 | theta_t_end) is Gaussian
        with mean theta + sigma^2 * score(theta) and precision Lambda* + 1/sigma^2,
        where Lambda* = (1-n) Lambda_prior + sum_j Lambda_j is the composed posterior
        precision at t=0. Drawing from it removes the leftover noise exactly.
        """
        h = self.hierarchy
        t_end = self.timesteps_list[-1].reshape(-1, 1)
        var_end = self.sde.lambda_t(t_end).to(x.device)**2

        score = self._get_score(x, t_end, condition_mask, idx, self.cfg_alpha)

        n = self.num_observations
        Lambda_prior = 1.0 / self.prior_std**2
        if self.posterior_precision.shape[0] == 1:
            Lambda_star = (1 - n) * Lambda_prior + n * self.posterior_precision[0]
        else:
            Lambda_star = (1 - n) * Lambda_prior + self.posterior_precision.sum(dim=0)
        var_denoise = 1.0 / (Lambda_star + 1.0 / var_end)                     # (1, H)

        noise = self._shared_noise(x)
        x = x.clone()
        x[:, :, h] = x[:, :, h] + var_end * score[:, :, h] \
                     + torch.sqrt(var_denoise) * noise[:, :, h]
        return x

    def _shared_noise(self, x):
        """
        Noise that is shared across the observation axis on the hierarchy dimensions
        (so the shared parameters stay synchronized across observation rows) and
        independent on all other dimensions.
        """
        noise = torch.randn_like(x)
        shared = torch.randn(1, x.shape[1], x.shape[2], device=x.device)
        if self.world_size > 1:
            dist.broadcast(shared, src=0)
        noise[:, :, self.hierarchy] = shared[:, :, self.hierarchy]
        return noise

    def _hierarchy_scale(self, x):
        """
        Per-feature scale for initial noise / Langevin steps. For "fnpe" the bridging
        densities have reference N(0, sigma_max^2 / n) on the shared dimensions
        (Geffner et al. 2023), so noise and step sizes shrink by 1/n there.
        """
        scale = torch.ones(x.shape[2], device=x.device)
        if self.correction == "fnpe":
            scale[self.hierarchy] = 1.0 / self.num_observations
        return scale

    def _check_data_shape(self, data, condition_mask):
        # Check data shape
        # Required shape: (num_samples, num_features)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)

        # Check condition mask shape
        # Required shape: (num_samples, num_features)
        if len(condition_mask.shape) == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)

        return data, condition_mask

    def _check_data_structure(self, data, condition_mask):
        # Convert data to DataLoader

        data, condition_mask = self._check_data_shape(data, condition_mask)
        dataset_cond = TensorTupleDataset(data, condition_mask)
        num_observations = dataset_cond.__len__()

        # The scores of ALL observations must be composed together at every step,
        # so each (per-process) shard has to fit into a single batch.
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset_cond,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
            batch_size = (num_observations + self.world_size - 1) // self.world_size
            data_loader = DataLoader(
                dataset_cond,
                batch_size=batch_size,
                sampler=sampler,
                pin_memory=True,
                shuffle=False,
                drop_last=False
            )
        else:
            data_loader = DataLoader(dataset_cond, batch_size=num_observations, shuffle=False)

        return data_loader, num_observations

    def _prepare_data(self, batch, num_samples, device):
        # Expand data and condition mask to match num_samples
        data, condition_mask, idx = batch
        data = data.to(device)
        condition_mask = condition_mask.to(device)

        data = data.unsqueeze(1).repeat(1,num_samples,1)
        condition_mask = condition_mask.unsqueeze(1).repeat(1,num_samples,1)

        joint_data = torch.zeros_like(condition_mask)
        if torch.sum(condition_mask==1).item()!=0:
            joint_data[condition_mask==1] = data.flatten()

        return joint_data, condition_mask, idx

    def _warm_start_state(self, data, condition_mask):
        """The caller's ``init_state`` with the observed columns restored."""
        init = torch.as_tensor(self.init_state).to(device=data.device,
                                                   dtype=data.dtype)
        if init.shape != data.shape:
            raise ValueError(
                f"init_state has shape {tuple(init.shape)}, expected "
                f"{tuple(data.shape)} (observations, num_samples, nodes); "
                "warm starts need a single data batch."
            )
        state = init * (1 - condition_mask) + data * condition_mask
        state[:, :, self.hierarchy] = state[:1, :, self.hierarchy]
        return state

    def _initial_sample(self, data, condition_mask):
        # Initialize latent variables with noise from the diffusion reference
        # distribution, keeping observed variables fixed. The noise on the shared
        # (hierarchy) dimensions is identical across observation rows.
        std_max = self.sde.lambda_t(torch.ones_like(data))
        scale = torch.sqrt(self._hierarchy_scale(data))
        noise = self._shared_noise(data)

        data = data + std_max * scale * noise * (1 - condition_mask)
        return data

    #############################################
    # ----- Basic Sampling -----
    #############################################

    # Euler-Maruyama sampling
    def _basic_sampler(self, data, condition_mask, idx):
        """
        Basic Euler-Maruyama sampling method

        Args:
            data: Input data
                    Shape: (batch_size, num_samples, num_features)
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (batch_size, num_samples, num_features)
        """

        if self.save_trajectory:
            # Storage for trajectory (optional)
            self.data_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.score_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.dx_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data

        # Main sampling loop
        for i in tqdm.tqdm(range(self.timesteps-1), disable=not self.verbose, total=self.timesteps-1):

            t = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i+1].reshape(-1, 1)

            # Get score estimate
            score = self._get_score(data, t, condition_mask, idx, self.cfg_alpha)

            # Euler-Maruyama step of the reverse SDE. Integrated over one step,
            # g(t)^2 dt equals the decrease of the marginal variance sigma_m^2:
            # dx = dvar * score + sqrt(dvar) * z
            dvar = self.sde.lambda_t(t)**2 - self.sde.lambda_t(t_next)**2
            dx = dvar * score
            noise = self._shared_noise(data) * torch.sqrt(dvar)

            # Apply update respecting condition mask
            data = data + (dx + noise) * (1-condition_mask)

            if self.save_trajectory:
                # Store trajectory data
                self.data_t[:,i+1] = data
                self.dx_t[:,i] = dx
                self.score_t[:,i] = score

        return data.detach()

    def _adaptive_sampler(self, data, condition_mask, idx):
        """Adaptive stochastic Heun solver for the reverse SDE.

        Each proposal compares an Euler-Maruyama update with a stochastic
        Heun update that reuses exactly the same noise realization. The step is
        accepted when the normalized local error is within tolerance; otherwise
        only the diffusion-time step is reduced.
        """
        start_time = float(self.timesteps_list[0])
        end_time = float(self.timesteps_list[-1])
        total_span = start_time - end_time
        max_proposals = self.adaptive_max_evals // 2
        # The evaluation budget must not silently turn a rejected proposal into
        # an accepted one.  Keep the controller floor at numerical resolution;
        # an overly strict tolerance then terminates through the explicit
        # evaluation-budget error below.
        min_step = max(total_span * 1e-12, torch.finfo(data.dtype).eps)
        step = (
            total_span / 50.0
            if self.adaptive_initial_step is None
            else min(self.adaptive_initial_step, total_span)
        )
        current_time = start_time
        accepted = 0
        rejected = 0
        evaluations = 0
        latent = 1 - condition_mask
        previous_euler = data.clone()

        progress = tqdm.tqdm(
            total=max_proposals, disable=not self.verbose,
            desc="adaptive reverse SDE",
        )
        while current_time > end_time and evaluations + 2 <= self.adaptive_max_evals:
            step = min(step, current_time - end_time)
            next_time = current_time - step
            t = torch.tensor([[current_time]], dtype=data.dtype, device=data.device)
            t_next = torch.tensor([[next_time]], dtype=data.dtype, device=data.device)
            variance_drop = self.sde.lambda_t(t)**2 - self.sde.lambda_t(t_next)**2
            variance_drop = variance_drop.clamp_min(0.0)
            noise = self._shared_noise(data) * torch.sqrt(variance_drop)

            score = self._get_score(data, t, condition_mask, idx, self.cfg_alpha)
            euler = data + (variance_drop * score + noise) * latent
            endpoint_score = self._get_score(
                euler, t_next, condition_mask, idx, self.cfg_alpha
            )
            endpoint = data + (variance_drop * endpoint_score + noise) * latent
            heun = 0.5 * (euler + endpoint)
            heun = heun * latent + data * condition_mask
            heun[:, :, self.hierarchy] = heun[:1, :, self.hierarchy]
            evaluations += 2

            scale = torch.maximum(
                torch.full_like(euler, self.adaptive_abs_tol),
                self.adaptive_rel_tol
                * torch.maximum(euler.abs(), previous_euler.abs()),
            )
            normalized = ((euler - heun).abs() / scale) * latent
            error = float(normalized.max())
            if not torch.isfinite(heun).all() or not torch.isfinite(
                torch.tensor(error)
            ):
                raise RuntimeError(
                    "Adaptive reverse-SDE sampling produced a non-finite proposal."
                )

            if error <= 1.0:
                data = heun
                previous_euler = euler
                current_time = next_time
                accepted += 1
            else:
                rejected += 1

            controlled_error = max(error, 1e-10)
            candidate = (
                step * self.adaptive_safety
                * controlled_error ** (-self.adaptive_exponent)
            )
            step = max(min_step, candidate)
            progress.update(1)

        progress.close()
        self.solver_stats = {
            "accepted_steps": accepted,
            "rejected_steps": rejected,
            "score_evaluations": evaluations,
            "start_time": start_time,
            "end_time": current_time,
            "target_end_time": end_time,
        }
        if current_time > end_time + 1e-7:
            raise RuntimeError(
                "Adaptive reverse-SDE solver exhausted adaptive_max_evals="
                f"{self.adaptive_max_evals} at t={current_time:.6g}; "
                f"target t={end_time:.6g}."
            )
        return data.detach()

    # Annealed Langevin dynamics (Algorithm 1 of Geffner et al. 2023)
    def _langevin_sampler(self, data, condition_mask, idx, snr, steps_per_level):
        """
        Pure annealed Langevin dynamics: at every noise level, run `steps_per_level`
        Langevin MCMC steps targeting the bridging density. This is the sampling
        algorithm the "fnpe" composition is derived for.

        Args:
            data: Input data, shape (n_obs, num_samples, num_features)
            condition_mask: Binary mask indicating observed (1) and latent (0) values
            snr: Step size scale of the Langevin steps
            steps_per_level: Number of Langevin steps per noise level
        """
        if self.save_trajectory:
            self.data_t = torch.zeros(data.shape[0], self.timesteps, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data

        for i, t in tqdm.tqdm(enumerate(self.timesteps_list), disable=not self.verbose, total=self.timesteps):
            t = t.reshape(-1, 1)
            data = self._corrector_step(data, t, condition_mask, idx,
                                        steps_per_level, snr, self.cfg_alpha)
            if self.save_trajectory and i+1 < self.timesteps:
                self.data_t[:,i+1] = data

        return data.detach()

    #############################################
    # ----- Advanced Sampling -----
    #############################################

    # DPM sampling with Langevin corrector steps
    def _corrector_step(self, x, t, condition_mask, idx, steps, snr, cfg_alpha=None):
        """
        Corrector steps using Langevin dynamics

        Args:
            x: Input data
            t: Time step
            """
        # For "fnpe" the target bridging density contracts by 1/n on the shared
        # dimensions, so the Langevin step size shrinks accordingly.
        step_scale = self._hierarchy_scale(x)

        for _ in range(steps):
            # Get score estimate
            score = self._get_score(x, t, condition_mask, idx, cfg_alpha)

            # Langevin dynamics update; the noise on the shared dimensions is
            # identical across observation rows to keep them synchronized
            step_size = snr * self.sde.lambda_t(t)**2 * step_scale
            noise = self._shared_noise(x) * torch.sqrt(2 * step_size)

            # Update x with the score and noise, respecting the condition mask
            grad_step = step_size * score
            x = x + grad_step * (1-condition_mask) + noise * (1-condition_mask)

        return x

    def _dpm_solver_1_step(self, data_t, t, t_next, condition_mask, idx):
        """First-order solver (in noise-scale space).

        The probability-flow ODE dx = 1/2 g(t)^2 * score * dt is, in terms of the
        marginal noise scale sigma_m(t) (with g^2 = d sigma_m^2/dt), exactly
        dx = sigma_m * score * d(sigma_m) -- so the solver steps in d(sigma_m),
        not in dt.
        """
        sigma_now = self.sde.lambda_t(t)
        sigma_next = self.sde.lambda_t(t_next)
        h = sigma_now - sigma_next

        # First-order step
        score_now = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)
        data_next = data_t + h * sigma_now * score_now * (1-condition_mask)

        return data_next

    def _dpm_solver_2_step(self, data_t, t, t_next, condition_mask, idx):
        """Second-order solver (in noise-scale space)"""
        sigma_now = self.sde.lambda_t(t)
        sigma_next = self.sde.lambda_t(t_next)
        h = sigma_now - sigma_next

        # First-order step
        score_half = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)
        data_half = data_t + h * sigma_now * score_half * (1-condition_mask)

        # Second-order (Heun) step
        score_next = self._get_score(data_half, t_next, condition_mask, idx, self.cfg_alpha)
        data_next = data_t + 0.5 * h * (sigma_now * score_half + sigma_next * score_next) * (1-condition_mask)

        return data_next

    def _dpm_solver_3_step(self, data_t, t, t_next, condition_mask, idx):
        """Third-order solver (in noise-scale space)"""
        # Get sigma values at different time points
        sigma_t = self.sde.lambda_t(t)
        t_mid = (t + t_next) / 2
        sigma_mid = self.sde.lambda_t(t_mid)
        sigma_next = self.sde.lambda_t(t_next)

        # First calculate the intermediate score at time t
        score_t = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)

        # First intermediate point (Euler step)
        data_mid1 = data_t + (sigma_t - sigma_mid) * sigma_t * score_t * (1-condition_mask)

        # Get score at the first intermediate point
        score_mid1 = self._get_score(data_mid1, t_mid, condition_mask, idx, self.cfg_alpha)

        # Second intermediate point (using first intermediate)
        data_mid2 = data_t + (sigma_t - sigma_mid) * ((1/3) * sigma_t * score_t + (2/3) * sigma_mid * score_mid1) * (1-condition_mask)

        # Get score at the second intermediate point
        score_mid2 = self._get_score(data_mid2, t_mid, condition_mask, idx, self.cfg_alpha)

        # Final step using all information
        data_next = data_t + (sigma_t - sigma_next) * ((1/4) * sigma_t * score_t +
                                            (3/4) * sigma_next * score_mid2) * (1-condition_mask)

        return data_next

    def _dpm_sampler(self, data, condition_mask, idx,
                     order=2,
                     snr=0.1, corrector_steps_interval=5, corrector_steps=0,
                     final_corrector_steps=3, terminal_corrector_steps=0):
        """
        DPM-Solver predictor, stochastic by default (Karras churn), with optional
        Predictor-Corrector refinement.

        Args:
            data: Input data
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            final_corrector_steps: Extra correction steps at the end

        The churn strength is ``self.churn_eta`` (default 2.0). Each level steps
        **up** in noise by ``gamma = min(eta * rel_step, churn_gamma_cap)`` and
        then takes the ordinary deterministic step down, which costs no extra
        score evaluations. ``churn_eta=0`` recovers the stock probability-flow
        predictor. See ``compass/churn.py``.
        """

        if self.save_trajectory:
            # Storage for trajectory (optional)
            self.data_t = torch.zeros(data.shape[0], self.timesteps, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data

        # Observed coordinates are conditioning data: churn must not perturb them.
        latent = 1 - condition_mask
        ceiling = schedule_ceiling(self.sde, data.device)

        # Main sampling loop
        for i in tqdm.tqdm(range(self.timesteps-1), disable=not self.verbose, total=self.timesteps-1):

            # ------- PREDICTOR: churn up, then DPM-Solver down -------
            t_now = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i+1].reshape(-1, 1)

            # _shared_noise keeps the hierarchy coordinates identical across
            # observation rows, as everywhere else in this sampler.
            data, t_start = apply_churn(
                data, latent, self.sde, t_now, t_next, self.churn_eta,
                gamma_cap=self.churn_gamma_cap, ceiling=ceiling,
                s_noise=self.churn_s_noise, noise_fn=self._shared_noise,
                sigma_min=getattr(self, "churn_sigma_min", 0.0),
                sigma_max=getattr(self, "churn_sigma_max", float("inf")),
            )

            if order == 1:
                data = self._dpm_solver_1_step(data, t_start, t_next, condition_mask, idx)
            elif order == 2:
                data = self._dpm_solver_2_step(data, t_start, t_next, condition_mask, idx)
            elif order == 3:
                data = self._dpm_solver_3_step(data, t_start, t_next, condition_mask, idx)
            else:
                raise ValueError(f"Only orders 1, 2 or 3 are supported in the DPM-Solver.")

            # ------- CORRECTOR: Langevin MCMC steps -------
            # Only apply corrector steps occasionally to save computation
            if corrector_steps > 0 and (i % corrector_steps_interval == 0 or i >= self.timesteps - final_corrector_steps):
                steps = corrector_steps
                if i >= self.timesteps - final_corrector_steps:
                    steps = corrector_steps * 5  # More steps at the end

                data = self._corrector_step(data, t_next, condition_mask, idx,
                                            steps, snr, self.cfg_alpha)

            if self.save_trajectory:
                # Store trajectory data
                self.data_t[:,i+1] = data

        # Optional terminal-only correction. These steps are never interleaved with the DPM trajectory.
        if terminal_corrector_steps:
            terminal_time = self.timesteps_list[-1].reshape(-1, 1)
            data = self._corrector_step(
                data, terminal_time, condition_mask, idx,
                terminal_corrector_steps, snr, self.cfg_alpha,
            )
        return data.detach()
