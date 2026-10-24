import os
import sys
import pickle
import math
import tqdm

import torch
import torch.nn as nn

import numpy as np

import scipy
from scipy import optimize
from scipy.stats import norm, gaussian_kde

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import seaborn as sns

from .ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm
from . import information_criteria as ic
from .likelihood_maximization import maximize_likelihood

from itertools import compress
import warnings
warnings.filterwarnings('ignore')

#################################################################################################
# ///////////////////////////////////// Model Comparison ////////////////////////////////////////
#################################################################################################

class ModelTransfuser():
    def __init__(self, path=None):
        
        ## Check if the path exists
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
        self.path = path
        
        self.models_dict = {}
        self.data_dict = {}
        self.trained_models = False # Flag to check if models are trained

    #############################################
    # ----- Model Management -----
    #############################################

    #---------------------------
    # Add a trained model to the transfuser
    def add_model(self, model_name, model):
        """
        Add a trained model to the transfuser.

        Args:
            model_name: The name of the model.
            model: The model itself.
        """
        self.models_dict[model_name] = model
        self.trained_models = True
        print(f"Model {model_name} added to transfuser.")

    #---------------------------
    # Add multiple trained models to the transfuser
    def add_models(self, models_dict):
        """
        Add multiple trained models to the transfuser.

        Args:
            models_dict: A dictionary of models to add.
        """
        for model_name, model in models_dict.items():
            self.add_model(model_name, model)

        self.trained_models = True
        print("All models added to transfuser.")

    #---------------------------
    # Add data to a model
    def add_data(self, model_name, theta, x, val_theta=None, val_x=None):
        """
        Add training and validation data to a model.

        Args:
            model_name: The name of the model.
            train_data: The training data.
            val_data: The validation data (optional).
        """
        if val_theta is None:
            self.data_dict[model_name] = {
                "train_theta": theta,
                "train_x": x,
            }
        else:
            self.data_dict[model_name] = {
                "train_theta": theta,
                "train_x": x,
                "val_theta": val_theta,
                "val_x": val_x,
            }
        self.trained_models = False
        print(f"Data added to model {model_name}")

    #---------------------------
    # Remove a model from the transfuser
    def remove_model(self, model_name):
        """
        Remove a model from the transfuser.

        Args:
            model_name: The name of the model to remove.
        """
        if model_name in self.models_dict:
            del self.models_dict[model_name]
            print(f"Model {model_name} removed from transfuser.")
        else:
            print(f"Model {model_name} not found in transfuser.")

    #############################################
    # ----- Initialize Models -----
    #############################################

    def init_models(self, sde_type, sigma, hidden_size, depth, num_heads, mlp_ratio):
        """
        Initialize the Score-Based Inference Models with the given parameters

        Args:
            sde_type: The type of SDE
            sigma: The sigma value
            hidden_size: The size of the hidden layer
            depth: The depth of the model
            num_heads: The number of heads in the model
            mlp_ratio: The MLP ratio
        """

        if self.trained_models:
            print("Models are already trained. This will overwrite the models.")
            return
        else:
            init_models = []
            for model_name in self.data_dict.keys():
                nodes_size = self.data_dict[model_name]["train_theta"].shape[1] + self.data_dict[model_name]["train_x"].shape[1]
                self.models_dict[model_name] = SBIm(nodes_size=nodes_size,
                                                    sde_type=sde_type,
                                                    sigma=sigma,
                                                    hidden_size=hidden_size,
                                                    depth=depth,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio)
                init_models.append(model_name)

            print(f"Models initialized: {init_models}")

    #############################################
    # ----- Train Models -----
    #############################################

    def train_models(self, batch_size=128, max_epochs=500, lr=1e-3, device="cuda",
                verbose=False, path=None, early_stopping_patience=100): 
        
        """
        Train the models on the provided data

        Args:
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            lr: Learning rate
            device: Device to run training on
                    if "cuda", training will be distributed across all available GPUs
            verbose: Whether to show training progress
            path: Path to save model
            early_stopping_patience: Number of epochs to wait before early stopping
        """

        if self.trained_models:
            print("Continue training existing models.")

        if path is not None:
            self.path = path
        elif self.path is not None:
            path = self.path

        for model_name in self.models_dict.keys():
            model = self.models_dict[model_name]

            theta = self.data_dict[model_name]["train_theta"]
            x = self.data_dict[model_name]["train_x"]
            val_theta = self.data_dict[model_name].get("val_theta", None)
            val_x = self.data_dict[model_name].get("val_x", None)

            model.train(theta=theta, x=x, theta_val=val_theta, x_val=val_x,
                        batch_size=batch_size, max_epochs=max_epochs, lr=lr, device=device,
                        verbose=verbose, path=path, name=model_name ,early_stopping_patience=early_stopping_patience)

            load_path = f"{path}/{model_name}.pt"
            self.models_dict[model_name] = SBIm.load(path=load_path, device="cpu")
            print(f"Model {model_name} trained")
            torch.cuda.empty_cache()

        self.trained_models = True

    #############################################
    # ----- Model Comparison -----
    #############################################

    def compare(self, x, err=None, condition_mask=None,
               timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None, multi_obs_inference=False, hierarchy=None,
               prior=None, correction=None,
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=None, final_corrector_steps=3,
               device="cuda", verbose=False, method="dpm",
               likelihood_method="pfode", map_method="score", criterion="aic",
               log_prob_timesteps=100, map_num_starts=8, map_timesteps=100,
               map_iterations_per_level=3, map_max_iterations_per_level=None,
               map_convergence_tol=1e-6,
               point_estimate="mle", ic_convention="counted", n_per_observation=None,
               theta_bounds=None, mle_starts=4, mle_screen=32, mle_iterations=50,
               mle_timesteps=32, mle_box_width=3.0,
               jacobian_refresh=None, jacobian_fd_step=1e-2,
               waic_draws=256, waic_timesteps=50, waic_divergence="exact",
               waic_hutchinson_samples=1, waic_batch_size=4096):
        """
        Compare the models on the provided observations.
        The results are saved in the self.stats dictionary and the provided path.

        Args:
            x:              The observations to compare the models on.
                                Shape: (num_samples, num_obs_features)
            err:            (optional) The observation uncertainties. If not provided, it is assumed to be zero.
                                Shape: (num_samples, num_obs_features)
            condition_mask: (optional) Binary mask indicating observed values (1) and latent values (0).
                                Should be provided if the there are missing observations in the data.
                                If not provided, it is assumed, that
                                Shape: (num_samples, num_total_features)
            timesteps:      Number of timesteps for the diffusion process.
                                (default) - 50 timesteps
            eps:            Epsilon value for the model
            num_samples:    Number of samples to generate
            cfg_alpha:      CFG alpha value for the model
            multi_obs_inference: Whether to use multi-observation inference
            hierarchy:      Shared (global) latent dims composed across rows. A dict
                                {model_name: dims} gives every model its own; a model
                                with an empty list has no globals and its rows get
                                independent posteriors.
            prior:          Gaussian prior (mean, std) over the hierarchy dims, or a
                                dict {model_name: (mean, std)}.
            correction:     Composition rule of the multi-observation sampler.
                                None (default): "gauss", or "gauss_jacobian_FD"
                                for criterion="waic" (the only rule WAIC supports).
            jacobian_refresh: Composed-score evaluations between Jacobian refreshes
                                of the Jacobian corrections. None: 9 for WAIC
                                (odd, see docs/jacobian_speedup_results.md), else 1.
            jacobian_fd_step: Finite-difference step of "gauss_jacobian_FD".
            order:          Order of the model
            snr:            Signal-to-noise ratio for the model
            corrector_steps_interval: Corrector steps interval for the model
            corrector_steps: Corrector steps for the model. None (default) uses the
                                sampler's own default (0 with the churned "dpm"
                                predictor, 5 for "langevin"); a positive value with
                                method="dpm" also needs churn to be off.
            final_corrector_steps: Final corrector steps for the model
            device:         Device to run inference on
            verbose:        (bool) Whether to show inference progress
            method:         (string) Method used to solve the SDE during inference.
                                "dpm"   - (default) Using the DPM-Solver for infernce with order 'order'
                                "euler" - Using the Euler-Maruyama method for inference
            likelihood_method: (string) How the per-observation likelihood
                                log p(x_i | theta_MAP,i) is evaluated:
                                "pfode" - (default) directly through the probability-flow
                                          ODE of the score model (exact continuous
                                          normalizing-flow likelihood, no KDE, no
                                          likelihood sampling step needed).
                                "kde"   - legacy behavior: sample from the likelihood and
                                          evaluate a Gaussian KDE of the samples at x_i.
            map_method:     (string) How the posterior MAP is estimated:
                                "score" - (default) KDE-free annealed score ascent using
                                          the trained score network (mean-shift on the
                                          diffused posterior), initialized at the
                                          posterior sample mean.
                                "joint_score" - estimate the joint mode of the marginal
                                          posterior over all shared parameters from their
                                          synchronized samples, then hold that shared vector
                                          fixed during one local-only annealed score-ascent
                                          pass; requires multi_obs_inference.
                                "kde"   - legacy behavior: mode of a Gaussian KDE fitted
                                          to the posterior samples.
            criterion:      (string) Information criterion for model weights:
                                "aic"   - (default) corrected Akaike IC (AICc); kept
                                          under this name for backward compatibility
                                "aicc"  - the same, explicitly
                                "aic_plain" - uncorrected AIC: 2K - 2*logL
                                "bic"   - Bayesian (Schwarz) IC
                                "waic"  - widely applicable IC: S composed-posterior
                                          draws, the PF-ODE log p(x_i | theta_s)
                                          of every row at every draw, and
                                          -2 sum_i [lppd_i - p_waic_i]. No point
                                          estimate and no parameter count. Needs
                                          multi_obs_inference=True and the
                                          gauss_jacobian_FD composition.
                                "loo"   - PSIS-LOO from the same draws and pointwise
                                          likelihoods as "waic": -2 sum_i
                                          log p(x_i | x_-i) by Pareto-smoothed
                                          importance sampling, with a Pareto-k
                                          reliability check per row. With locals
                                          it predicts a new unit (the left-out
                                          row's locals revert to their prior).
                                          Same requirements as "waic"; both are
                                          always stored in stats.
                                See compass.information_criteria for the formulas.
            waic_draws:     Posterior draws per row used by WAIC (evenly thinned from
                                num_samples; each costs one PF-ODE solve per row).
            waic_timesteps: PF-ODE integration nodes for WAIC (50 is at the network
                                error floor, validation/b5_b6_pfode_accuracy).
            waic_divergence: "exact" (default) or "hutchinson". Hutchinson noise adds
                                its variance to p_waic and biases WAIC upward, so it
                                is only for high-dimensional x.
            waic_hutchinson_samples: Probe vectors if waic_divergence="hutchinson".
            waic_batch_size: Points per PF-ODE batch for WAIC.
            point_estimate: (string) Parameter value the likelihood is evaluated at:
                                "mle"  - (default) maximize the PF-ODE likelihood
                                         itself (compass.likelihood_maximization),
                                         started from the MAP and posterior draws.
                                         Information criteria are defined at the
                                         maximum likelihood; the MAP is shrunk
                                         toward the prior and under-estimates it.
                                "map"  - legacy: plug in the MAP of `map_method`.
            ic_convention:  (string) Parameter and data counting of the criterion:
                                "counted" - (default) K = k_global + N * k_local
                                            effective parameters on
                                            n_data = N * n_per_observation scalars;
                                            globals (`hierarchy` dims under
                                            multi_obs_inference) are counted once.
                                "legacy"  - the historical per-row criterion with the
                                            full latent dimension summed over rows
                                            (penalizes globals N times). Only for
                                            reproducing old numbers.
            n_per_observation: Scalar measurements per observation row (default:
                                the number of observed dims x.shape[-1]).
            theta_bounds:   Optional (lower, upper) tensors over the latent
                                (parameter) dims, e.g. the prior support. The MLE is
                                confined to this box. Default: the per-row posterior
                                sample range widened by `mle_box_width` posterior sds
                                (the network has not seen parameters far outside).
            mle_starts:     Starts of the MLE: the point estimate of `map_method`
                                plus the mle_starts - 1 best of `mle_screen`
                                posterior draws.
            mle_screen:     Posterior draws screened by one gradient-free PF-ODE
                                likelihood evaluation before the MLE starts are
                                picked; finds every well-populated basin of a
                                multimodal likelihood (validation/a2, nonconcave).
            mle_iterations: L-BFGS-B iterations per start.
            mle_timesteps:  PF-ODE grid during the MLE optimization (the final
                                likelihood is evaluated with `log_prob_timesteps`).
            mle_box_width:  Widening of the default MLE box, in posterior sds.
            log_prob_timesteps: Integration nodes of the PF-ODE likelihood solver.
            map_num_starts: Multivariate shared-KDE optimizer starts used by joint_score.
            map_timesteps: Local-only annealing levels used by joint_score.
            map_iterations_per_level: Local score-ascent iterations per annealing level.
            map_max_iterations_per_level: Maximum alternating block iterations
                                per level. Retained for backward compatibility with
                                the former hierarchical optimizer.
            map_convergence_tol: Numerical tolerance for shared-KDE mode optimization.
        """

        if not self.trained_models:
            print("Models are not trained or provided. Please train the models before comparing.")
            return

        if criterion not in ("aic", "aicc", "aic_plain", "bic", "waic", "loo"):
            raise ValueError(
                f"criterion must be 'aic', 'aicc', 'aic_plain', 'bic', 'waic' or 'loo', got '{criterion}'"
            )
        use_waic = criterion in ("waic", "loo")
        if use_waic:
            if not multi_obs_inference:
                raise ValueError("criterion='waic' is implemented for compositional "
                                 "inference only: pass multi_obs_inference=True.")
            if likelihood_method != "pfode":
                raise ValueError("criterion='waic' needs likelihood_method='pfode'.")
            if correction is None:
                correction = "gauss_jacobian_FD"
            if correction != "gauss_jacobian_FD":
                raise ValueError("criterion='waic' supports correction='gauss_jacobian_FD' "
                                 f"only, got {correction!r}.")
            if int(waic_draws) < 2:
                raise ValueError("waic_draws must be at least 2.")
            if waic_divergence not in ("exact", "hutchinson"):
                raise ValueError("waic_divergence must be 'exact' or 'hutchinson'.")
        elif correction is None:
            correction = "gauss"
        if jacobian_refresh is None:
            jacobian_refresh = 9 if use_waic else 1
        if criterion == "aic_plain" and ic_convention == "legacy":
            raise ValueError("criterion='aic_plain' is only available with ic_convention='counted'.")
        if point_estimate not in ("mle", "map"):
            raise ValueError(f"point_estimate must be 'mle' or 'map', got '{point_estimate}'")
        if ic_convention not in ("counted", "legacy"):
            raise ValueError(f"ic_convention must be 'counted' or 'legacy', got '{ic_convention}'")
        if int(mle_starts) < 1 or int(mle_iterations) < 0:
            raise ValueError("mle_starts must be >= 1 and mle_iterations >= 0.")
        if map_method not in ("score", "joint_score", "kde"):
            raise ValueError(
                f"map_method must be 'score', 'joint_score' or 'kde', got '{map_method}'"
            )
        if map_method == "joint_score" and not multi_obs_inference:
            raise ValueError("map_method='joint_score' requires multi_obs_inference=True.")
        if int(map_num_starts) < 1:
            raise ValueError("map_num_starts must be at least 1.")
        if int(map_timesteps) < 1 or int(map_iterations_per_level) < 1:
            raise ValueError("map_timesteps and map_iterations_per_level must be at least 1.")
        if (map_max_iterations_per_level is not None
                and int(map_max_iterations_per_level) < int(map_iterations_per_level)):
            raise ValueError(
                "map_max_iterations_per_level must not be smaller than "
                "map_iterations_per_level."
            )
        if float(map_convergence_tol) <= 0:
            raise ValueError("map_convergence_tol must be positive.")

        self.stats = {}
        self.model_null_log_probs = {}
        self.softmax = nn.Softmax(dim=0)
        self._aicc_warned_once = False
        self._bic_warned_once = False
        self.criterion = criterion
        self.ic_convention = ic_convention
        self.point_estimate = point_estimate
        self.n_per_observation = (x.shape[-1] if n_per_observation is None
                                  else float(n_per_observation))
        self.likelihood_method = likelihood_method
        self.map_method = map_method

        # Remember whether the user explicitly supplied a condition mask. If not,
        # it must be rebuilt per model because different models can have a
        # different number of (physical) parameters -> different nodes_size.
        user_condition_mask = condition_mask

        # Loop over all models
        for model_name, model in tqdm.tqdm(self.models_dict.items(), desc="Comparing models", unit="model"):
            self.stats[model_name] = {}
            model_hierarchy = hierarchy.get(model_name) if isinstance(hierarchy, dict) else hierarchy
            model_prior = prior.get(model_name) if isinstance(prior, dict) else prior
            # A model without shared dims has nothing to compose: its rows get
            # independent single-observation posteriors.
            model_multi_obs = bool(multi_obs_inference) and not (
                model_hierarchy is not None and len(model_hierarchy) == 0)
            if not model_multi_obs:
                model_hierarchy, model_prior = None, None
            if user_condition_mask is None:
                condition_mask = torch.cat([torch.zeros(model.nodes_size-x.shape[-1]),torch.ones(x.shape[-1])])
            else:
                condition_mask = user_condition_mask
            self.condition_mask = condition_mask

            ####################
            # Posterior sampling
            posterior_samples = model.sample(x=x, err=err, condition_mask=condition_mask,
                                            timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                            multi_obs_inference=model_multi_obs, hierarchy=model_hierarchy,
                                            prior=model_prior, correction=correction,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                            device=device, verbose=verbose, method=method,
                                            jacobian_refresh=jacobian_refresh,
                                            jacobian_fd_step=jacobian_fd_step)
            posterior_samples = posterior_samples.cpu().numpy()

            # Inference Attention weights
            self.stats[model_name]["attn_weights"] = getattr(model.sampler, "all_attn_weights", None)

            if use_waic:
                self._waic_model(model_name, model, x, condition_mask, posterior_samples,
                                 model_hierarchy if model_multi_obs else [],
                                 waic_draws, waic_timesteps, waic_divergence,
                                 waic_hutchinson_samples, waic_batch_size, eps, device,
                                 verbose)
                continue

            ####################
            # MAP estimation
            x_t = torch.as_tensor(x, dtype=torch.float32)
            c_bool = condition_mask.bool()

            if map_method == "kde":
                theta_hat = np.array([self._map_kde(posterior_samples[i]) for i in range(len(posterior_samples))])
                MAP_posterior = torch.tensor(theta_hat[:, 0], dtype=torch.float)
                std_MAP_posterior = torch.tensor(theta_hat[:, 1], dtype=torch.float)
            elif map_method == "score":
                # Annealed score ascent (mean-shift with the trained score network):
                # no KDE bandwidth bias, works in any dimension. Initialized at the
                # posterior sample mean, annealed from the posterior scale downwards.
                post_mean = torch.tensor(posterior_samples.mean(axis=1), dtype=torch.float)
                post_std = torch.tensor(posterior_samples.std(axis=1), dtype=torch.float)
                joint_init = torch.zeros(x_t.shape[0], model.nodes_size)
                joint_init[:, c_bool] = x_t
                joint_init[:, ~c_bool] = post_mean

                # With compositional (multi-obs) inference the hierarchy dims are
                # GLOBAL: their posterior samples are shared across all stars, so
                # they must not be re-optimized per observation (a per-star ascent
                # would drag them to each star's individual mode). Hold them fixed
                # at the joint posterior mean and ascend only the local dims.
                map_condition_mask = condition_mask
                if model_multi_obs:
                    map_condition_mask = condition_mask.clone()
                    hier = model_hierarchy if model_hierarchy is not None else torch.nonzero(1 - condition_mask).flatten().tolist()
                    map_condition_mask[list(hier)] = 1

                joint_map = model.map_estimate(joint_init, map_condition_mask,
                                               sigma_start=2.0 * post_std.max().item(),
                                               device=device)
                MAP_posterior = joint_map[:, ~c_bool].float()
                std_MAP_posterior = post_std
                theta_hat = np.stack([MAP_posterior.numpy(), std_MAP_posterior.numpy()], axis=1)
            elif map_method == "joint_score":
                hier = model_hierarchy
                if hier is None:
                    hier = torch.where(~c_bool)[0].tolist()
                hier = [int(index) for index in hier]
                joint_map, shared_result = self._shared_then_local_map(
                    model=model,
                    posterior_samples=posterior_samples,
                    x=x_t,
                    condition_mask=condition_mask,
                    hierarchy=hier,
                    num_starts=map_num_starts,
                    timesteps=map_timesteps,
                    eps=eps,
                    iterations_per_level=map_iterations_per_level,
                    convergence_tol=map_convergence_tol,
                    device=device,
                )
                MAP_posterior = joint_map[:, ~c_bool].float()
                std_MAP_posterior = torch.tensor(
                    posterior_samples.std(axis=1), dtype=torch.float
                )
                theta_hat = np.stack(
                    [MAP_posterior.numpy(), std_MAP_posterior.numpy()], axis=1
                )
                # Preserve historical keys for saved-result consumers; these now
                # describe multivariate shared-KDE candidates rather than full
                # hierarchical score-ascent candidates.
                self.stats[model_name]["joint_map_candidate_scores"] = shared_result[
                    "candidate_log_densities"
                ]
                self.stats[model_name]["joint_map_selected_start"] = shared_result[
                    "selected_start"
                ]
                self.stats[model_name]["joint_map_shared_map"] = shared_result[
                    "shared_map"
                ]
                self.stats[model_name]["joint_map_shared_kde_starts"] = shared_result[
                    "candidate_modes"
                ]
                self.stats[model_name]["joint_map_conditional_effective_samples"] = (
                    shared_result["effective_sample_size"]
                )
                self.stats[model_name]["joint_map_strategy"] = (
                    "shared_marginal_kde_then_fixed_shared_local_score"
                )
            else:
                raise ValueError(f"map_method must be 'score', 'joint_score' or 'kde', got '{map_method}'")

            # Storing MAP and std MAP
            self.stats[model_name]["MAP"] = theta_hat

            # Global (shared) and local parameter dims of this model
            latent_dims = torch.where(~c_bool)[0].tolist()
            if model_multi_obs:
                hier = model_hierarchy if model_hierarchy is not None else latent_dims
                global_dims = sorted(int(d) for d in hier if int(d) in latent_dims)
            else:
                global_dims = []
            k_global = len(global_dims)
            k_local = len(latent_dims) - k_global
            self.stats[model_name]["k_global"] = k_global
            self.stats[model_name]["k_local"] = k_local

            ####################
            # Likelihood evaluation
            if likelihood_method == "pfode":
                # Direct likelihood log p(x_i | theta_i) through the
                # probability-flow ODE: conditions the score model on a point
                # estimate of all latent dims and evaluates the exact model
                # density at x_i. No likelihood sampling and no KDE involved.
                likelihood_mask = (1 - condition_mask)
                joint_eval = torch.zeros(x_t.shape[0], model.nodes_size)
                joint_eval[:, c_bool] = x_t
                joint_eval[:, ~c_bool] = MAP_posterior
                log_probs = model.log_prob(joint_eval, condition_mask=likelihood_mask,
                                           timesteps=log_prob_timesteps, eps=eps,
                                           device=device, verbose=verbose).float()
                self.stats[model_name]["log_probs_at_map"] = log_probs.clone()

                if point_estimate == "mle":
                    mle = self._maximum_likelihood(
                        model, joint_eval, likelihood_mask, posterior_samples,
                        latent_dims, global_dims, theta_bounds, mle_starts,
                        mle_iterations, mle_timesteps, mle_box_width, eps, device,
                        mle_screen,
                    )
                    mle_log_probs = model.log_prob(
                        mle.theta, condition_mask=likelihood_mask,
                        timesteps=log_prob_timesteps, eps=eps,
                        device=device, verbose=verbose,
                    ).float()
                    # Keep the higher of MAP and MLE on the evaluation grid: per row
                    # when the rows are independent, on the total when they share
                    # global parameters.
                    if global_dims:
                        use_mle = torch.full_like(log_probs, float(
                            mle_log_probs.sum() >= log_probs.sum()), dtype=torch.bool)
                    else:
                        use_mle = mle_log_probs >= log_probs
                    point = torch.where(use_mle.unsqueeze(1), mle.theta, joint_eval)
                    log_probs = torch.where(use_mle, mle_log_probs, log_probs)
                    MAP_posterior = point[:, ~c_bool].float()
                    self.stats[model_name]["MLE"] = MAP_posterior.numpy()
                    self.stats[model_name]["mle_info"] = {
                        "rows_using_map": int((~use_mle).sum()),
                        "at_bound": mle.at_bound[:, ~c_bool],
                        "log_probs_optimization_grid": mle.log_likelihood,
                        "start_index": mle.start_index,
                        "evaluations": mle.evaluations,
                        "messages": mle.messages,
                    }
                    n_at_bound = int(mle.at_bound.any(dim=1).sum())
                    if n_at_bound:
                        print(f"[compass] {model_name}: the MLE of {n_at_bound}/{len(x_t)} "
                              "rows sits on the parameter box (theta_bounds / "
                              "mle_box_width); the likelihood there is extrapolated.")
            elif likelihood_method == "kde":
                if point_estimate == "mle":
                    raise ValueError("point_estimate='mle' requires likelihood_method='pfode'.")
                # Legacy: sample from the likelihood at the MAP and evaluate a KDE
                likelihood_samples = model.sample(theta=MAP_posterior, err=std_MAP_posterior, condition_mask=(1-condition_mask),
                                                timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                                multi_obs_inference=model_multi_obs, hierarchy=model_hierarchy,
                                                prior=model_prior, correction=correction,
                                                order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                                device=device, verbose=verbose, method=method)
                likelihood_samples = likelihood_samples.cpu().numpy()
                log_probs = torch.tensor([self._log_prob(likelihood_samples[i], x[i]) for i in range(len(x))])
            else:
                raise ValueError(f"likelihood_method must be 'pfode' or 'kde', got '{likelihood_method}'")

            self.stats[model_name]["log_probs"] = log_probs

            # Information criterion calculation.
            # k must be the number of PHYSICAL model parameters (the theta
            # dimension of this model), NOT the neural-network weight count.
            # The posterior samples are already sliced to the latent (theta)
            # dimensions, so their last axis gives the parameter count -- this
            # is also robust to models with different theta dimensions.
            param_count = posterior_samples.shape[-1]
            self.stats[model_name]["param_count"] = param_count
            sample_size = x.shape[0]
            if ic_convention == "legacy":
                # Historical per-observation criterion with the full latent
                # dimension, summed over observations.
                aicc_per_obs = self._ic(log_probs, param_count, sample_size)
                self.stats[model_name]["AIC"] = aicc_per_obs.sum()
                self.stats[model_name]["IC_per_obs"] = aicc_per_obs

        if use_waic:
            self._waic_summary()
        elif ic_convention == "counted":
            # Decided jointly over all models: AICc is used only if it is defined
            # for every model (compass.information_criteria.compare_criteria).
            names = list(self.stats.keys())
            results, used = ic.compare_criteria(
                [self.stats[n]["log_probs"] for n in names], self._criterion_name(),
                [self.stats[n]["k_global"] for n in names],
                [self.stats[n]["k_local"] for n in names], self.n_per_observation,
            )
            self.criterion_used = used
            for name, result in zip(names, results):
                self._report_counting(name, result, self.stats[name]["k_local"])
                self.stats[name]["AIC"] = result["total"].float()
                self.stats[name]["IC_per_obs"] = result["per_row"].float()
                self.stats[name]["K_effective"] = result["K"]
                self.stats[name]["n_data"] = result["n_data"]
                self.stats[name]["aicc_defined"] = result["aicc_defined"]
                self.stats[name]["criterion_used"] = used


        # Calculate Model Probabilitys from the criterion.
        # Akaike weights: w_i = softmax(-IC_i / 2); the model with the LOWEST
        # criterion gets the highest probability.
        aics = [self.stats[model_name]["AIC"] for model_name in self.stats.keys()]
        aics = torch.tensor(aics)
        model_probs = self.softmax(-0.5 * aics)

        # Per-observation probabilities from the per-row criterion terms (they
        # sum to the totals above).
        individual_aicc = torch.stack([self.stats[model_name]["IC_per_obs"]
                                       for model_name in self.stats.keys()])
        probs = self.softmax(-0.5 * individual_aicc)

        for i, model_name in enumerate(self.stats.keys()):
            self.stats[model_name]["model_prob"] = model_probs[i].item()
            self.stats[model_name]["obs_probs"] = probs[i]

        model_names = list(self.stats.keys())
        best_model = model_names[model_probs.argmax()]
        best_model_prob = 100*model_probs.max()

        model_print_length = len(max(model_names, key=len))
        print(f"Probabilities of the models after {len(x)} observations:")
        for model in model_names:
            print(f"{model.ljust(model_print_length)}: {100*self.stats[model]['model_prob']:6.2f} %")
        print()
        print(f"Model {best_model} fits the data best " + 
                f"with a relative support of {best_model_prob:.1f}% among the considered models.")
        
        if self.path is not None:
            with open(f"{self.path}/model_comp.pkl", "wb") as f:
                pickle.dump(self.stats, f)

    #############################################
    # ----- Maximum likelihood / counting helpers -----
    #############################################

    def _waic_model(self, model_name, model, x, condition_mask, posterior_samples,
                    hierarchy, draws, timesteps, divergence, hutchinson_samples,
                    batch_size, eps, device, verbose):
        """Pointwise PF-ODE log-likelihoods at composed-posterior draws, and WAIC.

        Draw s of the composed posterior is one joint sample (globals synchronized
        across rows), so row i is evaluated at (g_s, l_{i,s}).
        """
        x_t = torch.as_tensor(x, dtype=torch.float32)
        mask = torch.as_tensor(condition_mask, dtype=torch.float32)
        if mask.dim() != 1:
            raise ValueError("criterion='waic' needs one condition mask shared by all rows.")
        c_bool = mask.bool()
        samples = torch.as_tensor(posterior_samples, dtype=torch.float32)
        n_rows, n_total = samples.shape[0], samples.shape[1]
        picks = torch.linspace(0, n_total - 1, min(int(draws), n_total)).round().long().unique()
        theta = samples[:, picks]
        n_draws = theta.shape[1]
        joint = torch.zeros(n_rows, n_draws, model.nodes_size)
        joint[:, :, c_bool] = x_t.unsqueeze(1).expand(-1, n_draws, -1)
        joint[:, :, ~c_bool] = theta
        log_lik = model.log_prob(joint.reshape(n_rows * n_draws, -1), condition_mask=1 - mask,
                                 timesteps=timesteps, eps=eps, divergence=divergence,
                                 hutchinson_samples=hutchinson_samples, device=device,
                                 batch_size=batch_size, verbose=verbose)
        log_lik = log_lik.reshape(n_rows, n_draws).double()
        if not torch.all(torch.isfinite(log_lik)):
            bad = int((~torch.isfinite(log_lik)).any(dim=1).sum())
            raise RuntimeError(f"{model_name}: the PF-ODE likelihood is non-finite for "
                               f"{bad}/{n_rows} rows at some posterior draws.")
        result = ic.waic(log_lik)
        loo = ic.psis_loo(log_lik) if log_lik.shape[1] >= 10 else None
        use_loo = getattr(self, "criterion", "waic") == "loo"
        if use_loo and loo is None:
            raise ValueError("criterion='loo' needs waic_draws >= 10.")
        chosen = loo if use_loo else result

        latent_dims = torch.where(~c_bool)[0].tolist()
        hier = latent_dims if hierarchy is None else hierarchy
        global_dims = sorted(int(d) for d in hier if int(d) in latent_dims)
        stats = self.stats[model_name]
        stats.update(
            pointwise_log_lik=log_lik.float(),
            log_probs=result["lppd_rows"].float(),
            WAIC=result["total"].float(), AIC=chosen["total"].float(),
            IC_per_obs=chosen["per_row"].float(),
            elpd_rows=chosen["elpd_rows"], elpd_waic_rows=result["elpd_rows"],
            lppd=float(result["lppd"]),
            p_waic=float(result["p_waic"]), p_waic_rows=result["p_waic_rows"],
            waic_se=result["se"], waic_mc_se=result["mc_se"],
            n_p_high=result["n_p_high"], waic_draws=n_draws,
            k_global=len(global_dims), k_local=len(latent_dims) - len(global_dims),
            param_count=len(latent_dims), criterion_used="loo" if use_loo else "waic",
            posterior_samples=theta,
        )
        if loo is not None:
            stats.update(LOO=loo["total"].float(), elpd_loo_rows=loo["elpd_rows"],
                         p_loo=loo["p_loo"], pareto_k=loo["pareto_k"],
                         n_k_high=loo["n_k_high"], loo_se=loo["se"], loo_mc_se=loo["mc_se"])
        if use_loo and loo["n_k_high"]:
            print(f"[compass] {model_name}: Pareto k > {ic.PSIS_K_WARNING} on "
                  f"{loo['n_k_high']}/{n_rows} rows; PSIS-LOO is unreliable there.")
        if result["n_p_high"] and not use_loo:
            print(f"[compass] {model_name}: p_waic > {ic.WAIC_P_WARNING} on "
                  f"{result['n_p_high']}/{n_rows} rows; WAIC is unreliable there "
                  "(Vehtari, Gelman & Gabry 2017).")

    def _waic_summary(self):
        """Pseudo-BMA+ weights and elpd differences to the best model."""
        names = list(self.stats.keys())
        elpd = torch.stack([self.stats[n]["elpd_rows"] for n in names])
        weights = ic.pseudo_bma_weights(elpd, bootstrap=True)
        best = int(elpd.sum(dim=1).argmax())
        self.criterion_used = getattr(self, "criterion", "waic")
        for index, name in enumerate(names):
            difference, se = ic.elpd_difference(elpd[index], elpd[best])
            self.stats[name]["model_prob_pseudo_bma_plus"] = float(weights[index])
            self.stats[name]["elpd_diff"] = difference
            self.stats[name]["elpd_diff_se"] = 0.0 if index == best else se

    def _criterion_name(self):
        """Map compare()'s criterion names onto compass.information_criteria."""
        return {"aic": "aicc", "aicc": "aicc", "aic_plain": "aic",
                "bic": "bic"}[getattr(self, "criterion", "aic")]

    def _report_counting(self, model_name, result, k_local):
        if not result["aicc_defined"] and not self._aicc_warned_once:
            print("[compass] Warning: AICc small-sample correction is undefined "
                  "(n_data <= K + 1 for at least one model); falling back to plain "
                  "AIC (2K - 2*logL) for all models.")
            self._aicc_warned_once = True
        message = ic.bic_local_penalty_warning(self._criterion_name(), k_local,
                                               self.n_per_observation)
        if message and not getattr(self, "_bic_warned_once", False):
            print(f"[compass] Warning: {message}")
            self._bic_warned_once = True

    def _maximum_likelihood(self, model, joint_eval, likelihood_mask, posterior_samples,
                            latent_dims, global_dims, theta_bounds, mle_starts,
                            mle_iterations, mle_timesteps, mle_box_width, eps, device,
                            mle_screen=32):
        """MLE of every row's parameters, started from the point estimate and draws."""
        n_rows, n_draws = posterior_samples.shape[0], posterior_samples.shape[1]
        samples = torch.as_tensor(posterior_samples, dtype=torch.float32)
        post_std = samples.std(dim=1).clamp_min(1e-6)

        scale = torch.ones_like(joint_eval)
        scale[:, latent_dims] = post_std
        if theta_bounds is None:
            lower = torch.full_like(joint_eval, -np.inf)
            upper = torch.full_like(joint_eval, np.inf)
            lower[:, latent_dims] = samples.min(dim=1).values - mle_box_width * post_std
            upper[:, latent_dims] = samples.max(dim=1).values + mle_box_width * post_std
        else:
            low, high = (torch.as_tensor(b, dtype=torch.float32).reshape(-1)
                         for b in theta_bounds)
            lower = torch.full_like(joint_eval, -np.inf)
            upper = torch.full_like(joint_eval, np.inf)
            lower[:, latent_dims] = low
            upper[:, latent_dims] = high

        starts = None
        if int(mle_starts) > 1:
            n_candidates = min(n_draws, max(int(mle_screen), int(mle_starts) - 1))
            picks = torch.linspace(0, n_draws - 1, n_candidates).round().long().unique()
            starts = joint_eval.unsqueeze(0).repeat(len(picks), 1, 1)
            # Draw s of the composed posterior is one joint sample: shared dims
            # are synchronized across rows, so a start keeps them consistent.
            starts[:, :, latent_dims] = samples[:, picks].permute(1, 0, 2)

        return maximize_likelihood(
            model, joint_eval, likelihood_mask, shared_dims=global_dims,
            starts=starts, scale=scale, lower=lower, upper=upper,
            max_iterations=mle_iterations, timesteps=mle_timesteps, eps=eps,
            device=device, keep_best_starts=int(mle_starts) - 1,
        )

    #############################################
    # ----- Kernel Density Estimation -----
    #############################################

    #---------------------------
    # Estimate the log probability
    def _log_prob(self, samples, observation):
        """Compute the log probability of the samples"""
        kde = gaussian_kde(samples.T)
        log_prob = kde.logpdf(observation).item()
        return log_prob

    #---------------------------
    # Estimate the Maximum A Posteriori (MAP)
    @staticmethod
    def _shared_marginal_kde_map(posterior_samples, condition_mask,
                                 hierarchy, num_starts=8,
                                 convergence_tol=1e-6):
        """Jointly optimize a KDE over the synchronized shared sample block.

        The KDE is fitted after coordinate-wise standardization but retains a
        full covariance matrix, so correlations between shared parameters are
        part of the mode optimization. A small covariance ridge also supports
        rank-deficient shared samples. The returned kernel weights approximate
        conditioning the joint samples on the selected shared mode and are used
        to initialize the local MAP pass.
        """
        samples = torch.as_tensor(posterior_samples, dtype=torch.float64)
        mask = torch.as_tensor(condition_mask, dtype=torch.bool)
        if samples.dim() != 3:
            raise ValueError(
                "posterior_samples must have shape "
                "(observations, samples, latent_dims)."
            )
        if mask.dim() != 1:
            raise ValueError(
                "joint_score currently requires a one-dimensional condition_mask."
            )
        if int(num_starts) < 1:
            raise ValueError("map_num_starts must be at least 1.")
        if float(convergence_tol) <= 0:
            raise ValueError("map_convergence_tol must be positive.")

        n_obs, n_samples, n_latent = samples.shape
        if n_samples < 1:
            raise ValueError("posterior_samples must contain at least one sample.")
        latent_nodes = torch.where(~mask)[0].tolist()
        if len(latent_nodes) != n_latent:
            raise ValueError(
                "Posterior samples and condition_mask have incompatible latent widths."
            )
        hierarchy = [int(index) for index in hierarchy]
        if not hierarchy:
            raise ValueError("joint_score requires at least one hierarchy coordinate.")
        if len(set(hierarchy)) != len(hierarchy):
            raise ValueError("hierarchy indices must be unique.")
        if min(hierarchy) < 0 or max(hierarchy) >= mask.numel():
            raise ValueError("hierarchy contains an out-of-range coordinate.")
        if mask[hierarchy].any():
            raise ValueError("Every hierarchy coordinate must be latent.")

        latent_lookup = {node: position for position, node in enumerate(latent_nodes)}
        shared_positions = [latent_lookup[index] for index in hierarchy]
        shared = samples[:, :, shared_positions]
        sync_error = (shared - shared[:1]).abs().max().item()
        if sync_error > 1e-6:
            raise ValueError(
                "Compositional posterior samples are not synchronized on hierarchy "
                f"coordinates (maximum deviation {sync_error:.3e})."
            )

        values = shared[0].numpy()
        centre = values.mean(axis=0)
        if n_samples == 1:
            return {
                "shared_map": torch.tensor(centre, dtype=torch.float32),
                "sample_weights": torch.ones(1, dtype=torch.float32),
                "candidate_modes": torch.tensor(centre[None], dtype=torch.float32),
                "candidate_log_densities": torch.zeros(1, dtype=torch.float32),
                "selected_start": 0,
                "effective_sample_size": 1.0,
                "bandwidth_factor": 0.0,
            }

        scale = values.std(axis=0, ddof=1)
        scale_threshold = max(float(np.max(scale)) * 1e-8, 1e-10)
        active = scale > scale_threshold
        if not np.any(active):
            weights = np.full(n_samples, 1.0 / n_samples)
            return {
                "shared_map": torch.tensor(centre, dtype=torch.float32),
                "sample_weights": torch.tensor(weights, dtype=torch.float32),
                "candidate_modes": torch.tensor(centre[None], dtype=torch.float32),
                "candidate_log_densities": torch.zeros(1, dtype=torch.float32),
                "selected_start": 0,
                "effective_sample_size": float(n_samples),
                "bandwidth_factor": 0.0,
            }

        standardized = (values[:, active] - centre[active]) / scale[active]
        active_dim = standardized.shape[1]
        bandwidth_factor = n_samples ** (-1.0 / (active_dim + 4.0))
        sample_covariance = np.atleast_2d(
            np.cov(standardized, rowvar=False, ddof=1)
        )
        covariance_scale = max(
            float(np.trace(sample_covariance)) / active_dim, 1.0
        )
        kernel_covariance = (
            bandwidth_factor**2 * sample_covariance
            + 1e-6 * covariance_scale * np.eye(active_dim)
        )
        inverse_covariance = np.linalg.inv(kernel_covariance)
        _, log_determinant = np.linalg.slogdet(kernel_covariance)
        log_normalizer = 0.5 * (
            active_dim * np.log(2 * np.pi) + log_determinant
        )
        log_sample_count = np.log(n_samples)

        def log_density_and_gradient(point):
            point = np.asarray(point, dtype=np.float64)
            differences = standardized - point
            quadratic = np.einsum(
                "si,ij,sj->s", differences, inverse_covariance, differences
            )
            log_kernels = -0.5 * quadratic
            log_sum = scipy.special.logsumexp(log_kernels)
            kernel_weights = np.exp(log_kernels - log_sum)
            gradient = (kernel_weights @ differences) @ inverse_covariance
            log_density = log_sum - log_sample_count - log_normalizer
            return float(log_density), gradient, kernel_weights

        # Score a bounded, representative pool of observed shared samples. Each
        # pool point is still evaluated against every posterior draw, but the
        # number of candidate points does not grow without bound with S.
        count = min(int(num_starts), n_samples)
        pool_size = min(n_samples, max(128, 32 * count))
        pool_indices = np.unique(
            np.linspace(0, n_samples - 1, pool_size, dtype=int)
        )
        pool_log_densities = np.empty(len(pool_indices))
        chunk_size = max(1, min(128, 1_000_000 // max(n_samples * active_dim, 1)))
        for begin in range(0, len(pool_indices), chunk_size):
            indices = pool_indices[begin:begin + chunk_size]
            points = standardized[indices]
            differences = standardized[None, :, :] - points[:, None, :]
            quadratic = np.einsum(
                "bsi,ij,bsj->bs",
                differences, inverse_covariance, differences,
            )
            pool_log_densities[begin:begin + len(points)] = (
                scipy.special.logsumexp(-0.5 * quadratic, axis=1)
                - log_sample_count - log_normalizer
            )

        density_order = pool_indices[np.argsort(pool_log_densities)[::-1]]
        selected_indices = []
        for sample_index in density_order:
            if not selected_indices:
                selected_indices.append(int(sample_index))
            else:
                displacement = standardized[selected_indices] - standardized[sample_index]
                separation = np.einsum(
                    "si,ij,sj->s",
                    displacement, inverse_covariance, displacement,
                )
                if float(separation.min()) > 1.0:
                    selected_indices.append(int(sample_index))
            if len(selected_indices) == count:
                break
        if len(selected_indices) < count:
            for sample_index in density_order:
                if int(sample_index) not in selected_indices:
                    selected_indices.append(int(sample_index))
                if len(selected_indices) == count:
                    break

        bounds = list(zip(standardized.min(axis=0), standardized.max(axis=0)))
        candidate_points = []
        candidate_scores = []
        for sample_index in selected_indices:
            def objective(point):
                log_density, gradient, _ = log_density_and_gradient(point)
                return -log_density, -gradient

            optimized = optimize.minimize(
                objective,
                standardized[sample_index],
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                tol=float(convergence_tol),
                options={"maxiter": 200},
            )
            point = optimized.x if np.all(np.isfinite(optimized.x)) else standardized[sample_index]
            log_density, _, _ = log_density_and_gradient(point)
            candidate_points.append(point)
            candidate_scores.append(log_density - np.log(scale[active]).sum())

        selected_start = int(np.argmax(candidate_scores))
        selected_point = candidate_points[selected_start]
        shared_map = centre.copy()
        shared_map[active] = (
            centre[active] + scale[active] * selected_point
        )
        _, _, conditional_weights = log_density_and_gradient(selected_point)

        candidate_modes = np.repeat(centre[None, :], len(candidate_points), axis=0)
        candidate_modes[:, active] = (
            centre[active][None, :]
            + scale[active][None, :] * np.asarray(candidate_points)
        )
        effective_sample_size = float(1.0 / np.sum(conditional_weights**2))
        return {
            "shared_map": torch.tensor(shared_map, dtype=torch.float32),
            "sample_weights": torch.tensor(conditional_weights, dtype=torch.float32),
            "candidate_modes": torch.tensor(candidate_modes, dtype=torch.float32),
            "candidate_log_densities": torch.tensor(candidate_scores, dtype=torch.float32),
            "selected_start": selected_start,
            "effective_sample_size": effective_sample_size,
            "bandwidth_factor": float(bandwidth_factor),
        }

    @classmethod
    def _shared_then_local_map(cls, model, posterior_samples, x,
                               condition_mask, hierarchy, num_starts=8,
                               timesteps=100, eps=1e-3,
                               iterations_per_level=3,
                               convergence_tol=1e-6, device="cpu"):
        """Find the marginal shared MAP, freeze it, then refine local MAPs."""
        samples = torch.as_tensor(posterior_samples, dtype=torch.float32)
        observations = torch.as_tensor(x, dtype=torch.float32)
        mask = torch.as_tensor(condition_mask, dtype=torch.bool)
        if observations.dim() == 1:
            observations = observations.unsqueeze(-1)
        if observations.shape[0] != samples.shape[0]:
            raise ValueError(
                "Posterior samples and observations have incompatible row counts."
            )
        if int(mask.sum()) != observations.shape[1]:
            raise ValueError(
                "Observed width does not match the conditioned dimensions."
            )

        shared_result = cls._shared_marginal_kde_map(
            samples, mask, hierarchy, num_starts=num_starts,
            convergence_tol=convergence_tol,
        )
        weights = shared_result["sample_weights"]
        conditional_latent_mean = torch.einsum("s,nsd->nd", weights, samples)

        joint_init = torch.zeros(samples.shape[0], mask.numel(), dtype=torch.float32)
        joint_init[:, mask] = observations
        joint_init[:, ~mask] = conditional_latent_mean
        hierarchy = [int(index) for index in hierarchy]
        joint_init[:, hierarchy] = shared_result["shared_map"]

        local_nodes = [
            index for index in torch.where(~mask)[0].tolist()
            if index not in hierarchy
        ]
        if local_nodes:
            latent_nodes = torch.where(~mask)[0].tolist()
            latent_lookup = {
                node: position for position, node in enumerate(latent_nodes)
            }
            local_positions = [latent_lookup[index] for index in local_nodes]
            local_samples = samples[:, :, local_positions]
            local_mean = conditional_latent_mean[:, local_positions]
            local_variance = torch.einsum(
                "s,nsd->nd",
                weights,
                (local_samples - local_mean.unsqueeze(1)).square(),
            )
            sigma_start = max(
                2.0 * float(local_variance.sqrt().max()), 1e-3
            )
            fixed_shared_mask = mask.clone()
            fixed_shared_mask[hierarchy] = True
            joint_map = model.map_estimate(
                data=joint_init,
                condition_mask=fixed_shared_mask.float(),
                init=joint_init,
                sigma_start=sigma_start,
                timesteps=int(timesteps),
                eps=float(eps),
                iterations_per_level=int(iterations_per_level),
                device=device,
            )
        else:
            joint_map = joint_init

        joint_map = torch.as_tensor(joint_map, dtype=torch.float32).clone()
        joint_map[:, hierarchy] = shared_result["shared_map"]
        joint_map[:, mask] = observations
        return joint_map, shared_result

    @staticmethod
    def _joint_map_initializations(posterior_samples, x, condition_mask,
                                   hierarchy, num_starts):
        """Return coherent, diverse joint starts and local-neighbourhood scales."""
        samples = torch.as_tensor(posterior_samples, dtype=torch.float32)
        observations = torch.as_tensor(x, dtype=torch.float32)
        mask = torch.as_tensor(condition_mask, dtype=torch.bool)
        if samples.dim() != 3:
            raise ValueError("posterior_samples must have shape (observations, samples, latent_dims).")
        if mask.dim() != 1:
            raise ValueError("joint_score currently requires a one-dimensional condition_mask.")
        n_obs, n_samples, n_latent = samples.shape
        if observations.shape[0] != n_obs or int((~mask).sum()) != n_latent:
            raise ValueError("Posterior samples, observations and condition_mask have incompatible shapes.")
        if int(mask.sum()) != observations.shape[1]:
            raise ValueError("Observed width does not match the conditioned dimensions.")
        if int(num_starts) < 1:
            raise ValueError("map_num_starts must be at least 1.")
        if n_samples < 1:
            raise ValueError("posterior_samples must contain at least one sample.")

        hierarchy = [int(index) for index in hierarchy]
        if not hierarchy:
            raise ValueError("joint_score requires at least one hierarchy coordinate.")
        if len(set(hierarchy)) != len(hierarchy):
            raise ValueError("hierarchy indices must be unique.")
        if min(hierarchy) < 0 or max(hierarchy) >= mask.numel():
            raise ValueError("hierarchy contains an out-of-range coordinate.")
        if mask[hierarchy].any():
            raise ValueError("Every hierarchy coordinate must be latent.")

        joint = torch.zeros(n_obs, n_samples, mask.numel(), dtype=torch.float32)
        joint[:, :, mask] = observations.unsqueeze(1).expand(-1, n_samples, -1)
        joint[:, :, ~mask] = samples
        sync_error = (joint[:, :, hierarchy] - joint[:1, :, hierarchy]).abs().max().item()
        if sync_error > 1e-6:
            raise ValueError(
                "Compositional posterior samples are not synchronized on hierarchy "
                f"coordinates (maximum deviation {sync_error:.3e})."
            )

        local = [index for index in torch.where(~mask)[0].tolist() if index not in hierarchy]
        global_features = joint[0, :, hierarchy]
        feature_parts = [global_features]
        if local:
            feature_parts.append(joint[:, :, local].permute(1, 0, 2).reshape(n_samples, -1))
        features = torch.cat(feature_parts, dim=1)
        feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        block_scale = torch.full_like(feature_scale, math.sqrt(max(len(hierarchy), 1)))
        if local:
            block_scale[len(hierarchy):] = math.sqrt(n_obs * len(local))
        standardized = (features - features.mean(dim=0)) / feature_scale / block_scale

        count = min(int(num_starts), n_samples)
        candidates = [joint.mean(dim=1)]
        candidate_features = [features.mean(dim=0)]
        if count > 1:
            min_distance = (standardized**2).sum(dim=1)
            selected = set()
            for _ in range(count - 1):
                for index in selected:
                    min_distance[index] = -1
                selected_index = int(torch.argmax(min_distance))
                selected.add(selected_index)
                candidates.append(joint[:, selected_index, :])
                candidate_features.append(features[selected_index])
                distance = ((standardized - standardized[selected_index])**2).sum(dim=1)
                min_distance = torch.minimum(min_distance, distance)

        neighbour_count = min(n_samples, max(8, int(np.ceil(n_samples / count))))
        annealing_scales = []
        latent_indices = torch.where(~mask)[0]
        for feature in candidate_features:
            standardized_feature = (
                (feature - features.mean(dim=0)) / feature_scale / block_scale
            )
            distances = ((standardized - standardized_feature)**2).sum(dim=1)
            neighbours = torch.topk(distances, neighbour_count, largest=False).indices
            local_width = joint[:, neighbours][:, :, latent_indices].std(
                dim=1, unbiased=False
            ).max()
            annealing_scales.append(max(2.0 * float(local_width), 1e-3))

        return torch.stack(candidates, dim=1), annealing_scales

    @staticmethod
    def _hierarchical_candidate_scores(model, candidates, condition_mask,
                                       hierarchy, prior, timesteps, eps,
                                       device, verbose):
        """Evaluate the composed joint log posterior for each refined candidate."""
        n_obs, n_candidates, nodes_size = candidates.shape
        flat = candidates.permute(1, 0, 2).reshape(n_candidates * n_obs, nodes_size)
        single_log_prob = model.log_prob(
            flat, condition_mask=condition_mask, timesteps=timesteps, eps=eps,
            device=device, verbose=verbose,
        ).reshape(n_candidates, n_obs)
        prior_mean = torch.as_tensor(prior[0], dtype=torch.float32).flatten()
        prior_std = torch.as_tensor(prior[1], dtype=torch.float32).flatten()
        shared = candidates[0, :, list(hierarchy)]
        log_prior = (
            -torch.log(prior_std)
            -0.5 * np.log(2 * np.pi)
            -0.5 * ((shared - prior_mean) / prior_std)**2
        ).sum(dim=1)
        return single_log_prob.sum(dim=1) + (1 - n_obs) * log_prior

    def _map_kde(self, samples):
        """Find the joint mode of the multivariate distribution"""
        kde = gaussian_kde(samples.T)  # KDE expects (n_dims, n_samples)
        
        # Start optimization from the mean
        initial_guess = np.mean(samples, axis=0)
        
        # Use full minimize with multiple dimensions
        result = optimize.minimize(lambda x: -kde(x.reshape(-1, 1)), initial_guess)
        std_devs = np.sqrt(np.diag(kde.covariance))

        return result.x, std_devs

    #############################################
    # ----- Information Criterion -----
    #############################################

    #---------------------------
    # Criterion dispatcher
    def _ic(self, log_likelihood, k, n):
        """
        Per-observation information criterion, dispatching on the criterion
        selected in `compare()` ("aic" -> AICc (default), "bic" -> BIC).
        """
        if getattr(self, "criterion", "aic") == "bic":
            return self._bic(log_likelihood, k, n)
        return self._aicc(log_likelihood, k, n)

    #---------------------------
    # Bayesian (Schwarz) Information Criterion
    def _bic(self, log_likelihood, k, n):
        """
        Bayesian Information Criterion (BIC):

            BIC = k * ln(n) - 2 * logL

        with ``k`` the number of PHYSICAL model parameters and ``n`` the number
        of observations. Applied per observation (mirroring the per-observation
        AICc convention used throughout): each observation term carries the full
        k*ln(n) penalty. BIC penalizes parameters more strongly than AIC for
        n >= 8 and is consistent (selects the true model as n -> inf) when the
        true model is among the candidates.
        """
        ll = torch.as_tensor(log_likelihood, dtype=torch.float32)
        k = torch.as_tensor(k, dtype=torch.float32)
        n = torch.as_tensor(n, dtype=torch.float32)
        # Guard n < 1 (cumulative plot evaluates n = 0 terms elsewhere)
        n = torch.clamp(n, min=1.0)
        return k * torch.log(n) - 2 * ll

    #---------------------------
    # Corrected Akaike Information Criterion
    def _aicc(self, log_likelihood, k, n):
        """
        Corrected Akaike Information Criterion (AICc).

            AICc = 2k - 2*logL + (2k^2 + 2k) / (n - k - 1)

        where ``k`` is the number of PHYSICAL model parameters (the theta
        dimension of the model) and ``n`` is the number of observations.
        The neural-network weight count must NOT be used for ``k``: with
        k ~ 1e5 and n ~ tens of observations the correction denominator
        (n - k - 1) is negative and the result is meaningless.

        When ``n - k - 1 <= 0`` the small-sample correction is undefined, so
        this falls back to the plain AIC (``2k - 2*logL``) for the affected
        terms and prints a warning once per comparison/plot call.

        ``log_likelihood`` may be a scalar or a tensor; ``k`` and ``n`` may be
        scalars or tensors broadcastable against it. The return has the
        broadcast shape.
        """
        ll = torch.as_tensor(log_likelihood, dtype=torch.float32)
        k = torch.as_tensor(k, dtype=torch.float32)
        n = torch.as_tensor(n, dtype=torch.float32)

        denom = n - k - 1
        small_sample = denom <= 0
        if bool(torch.any(small_sample)) and not getattr(self, "_aicc_warned_once", False):
            print("[compass] Warning: AICc small-sample correction is undefined "
                  "(n_obs - k - 1 <= 0); falling back to plain AIC (2k - 2*logL) "
                  "for the affected terms.")
            self._aicc_warned_once = True

        safe_denom = torch.where(small_sample, torch.ones_like(denom), denom)
        correction = torch.where(small_sample,
                                 torch.zeros_like(safe_denom),
                                 (2 * k ** 2 + 2 * k) / safe_denom)
        return 2 * k - 2 * ll + correction

    ##############################################
    # ----- Plotting -----
    ##############################################
    
    #---------------------------
    # Model Comparison plotting
    def plot_comparison(self, stats_dict=None, n_models=10, sort="median", model_names=None, path=None, show=True):
        """
        Plot the results from the Model Comparison.
        Saves the Violin plots for individual model probability and the cumulative model probability of all observations.

        Args:
            stats_dict: (dict)(optional) Dictionary with the comparison results. 
                            If not provided, it uses the results from the `compare()` call.
            n_models:   (int) Number of models to plot in the comparison.
            sort:       (string) How to sort the models for the plots.
                            median - (default) median model probability of all observations
                            mean   -  mean model probability of all observations
                            none   - the order the models are defined in
            model_names: (list of strings)(optional) List with the names of the models to plot.
            path:       (string)(optional) The path the plots are saved to.
                            If not provided, the plots are not saved.
            show:       (bool) Whether to show the created plots or not.
        """

        # Check path
        if path is None:
            path = self.path
            if not os.path.exists(path):
                os.makedirs(path)

        # Check stats_dict
        if stats_dict is None:
            stats_dict = self.stats

        # Sort models by observation probabilities
        if sort == "median":
            sorted_models = sorted(stats_dict, key=lambda x: stats_dict[x]["obs_probs"].median(),reverse=True)
        elif sort == "mean":
            sorted_models = sorted(stats_dict, key=lambda x: stats_dict[x]["obs_probs"].mean(),reverse=True)
        elif type(sort) == list:
            sorted_models = sort
            # add the remaining models to the end of the list for correct probability calculation
            for model in stats_dict.keys():
                if model not in sorted_models:
                    sorted_models.append(model)
        elif sort == "none":
            sorted_models = list(stats_dict.keys())
        stats_dict = {model: stats_dict[model] for model in sorted_models}

        model_keys = list(stats_dict.keys())

        model_log_probs = torch.stack([stats_dict[model]["log_probs"] for model in model_keys])
        model_obs_probs = torch.stack([stats_dict[model]["obs_probs"] for model in model_keys])
        param_counts = torch.tensor([stats_dict[model]["param_count"] for model in model_keys])

        if model_names is None:
            model_names = model_keys

        if len(model_names) < n_models:
            n_models = len(model_names)

        legend_cols = 1 if len(model_names) < 6 else 2

        # plt.style.use('ggplot')

        #---------------------------
        # Plot violin plot of model probabilities
        plt.figure(figsize=(12, 6), dpi=500)
        model_names_violin = [name.replace(", ", "\n") for name in model_names[:n_models]]
        sns.violinplot(data=model_obs_probs.T[:,:n_models],label=model_names_violin, palette=sns.color_palette("dark"), inner_kws=dict(box_width=5, whis_width=2, color="k"))

        if model_names != "":
            plt.xticks(ticks=range(n_models), labels=model_names_violin)
            plt.tick_params(axis='x', which='major', labelsize=16)

        plt.tick_params(axis='y', which='major', labelsize=16)
        plt.ylabel(r"$P(\mathcal{M} | x_i)$", fontsize=20)
        sns.despine()
        plt.tight_layout()

        if path is not None:
            plt.savefig(f"{path}/model_probs_violin.png")
        if show:
            plt.show()
        plt.close()

        #---------------------------
        # Plot cumulative model probabilities

        # Calculate mean model probabilities for N observations
        self._aicc_warned_once = getattr(self, "_aicc_warned_once", False)
        avg_model_probs = []
        for n in range(50):
            all_N_AICc = []
            for i in range(0,model_log_probs.shape[1]+1):
                if i != 0:
                    idx = torch.randperm(model_log_probs.shape[1])[:i]
                    N_log_probs = model_log_probs[:,idx].T
                    if getattr(self, "criterion_used", None) in ("waic", "loo"):
                        # Pointwise WAIC terms of the sampled rows (the posterior is
                        # the one from all rows, so with shared globals this is the
                        # usual pointwise approximation, not a refit on i rows).
                        N_AICc = torch.stack([stats_dict[name]["IC_per_obs"][idx]
                                              for name in model_keys], dim=1)
                    elif getattr(self, "ic_convention", "legacy") == "legacy":
                        # k = physical parameter count per model; guard small samples.
                        N_AICc = self._ic(N_log_probs, param_counts, i)
                    else:
                        subset, _ = ic.compare_criteria(
                            [N_log_probs[:, m] for m in range(len(model_keys))],
                            self._criterion_name(),
                            [stats_dict[name]["k_global"] for name in model_keys],
                            [stats_dict[name]["k_local"] for name in model_keys],
                            self.n_per_observation,
                        )
                        N_AICc = torch.stack([r["per_row"].float() for r in subset], dim=1)
                elif i == 0:
                    N_AICc = torch.zeros_like(model_log_probs[:,0]).unsqueeze(0)

                all_N_AICc.append(torch.nn.functional.softmax(-0.5 * N_AICc.sum(0),0).T)
            all_N_AICc = torch.stack(all_N_AICc)
            avg_model_probs.append(all_N_AICc)

        avg_model_probs = torch.stack(avg_model_probs)
        avg_mean = avg_model_probs.mean(0)
        avg_std = avg_model_probs.std(0)/torch.sqrt(torch.tensor(avg_model_probs.shape[0]))

        plt.figure(figsize=(12, 6), dpi=500)
        palette = sns.color_palette("dark", n_colors=n_models)
        for n in range(n_models):
            plt.errorbar(
            torch.arange(0, model_log_probs.shape[1]+1).T,
            avg_mean[:, n], yerr=avg_std[:, n],
            label=model_names[n], marker='o', markersize=6, linewidth=3, elinewidth=1, capsize=2,
            color=palette[n]
            )
        if model_names != "":
            plt.legend(title="Models", loc='right', fontsize=15, title_fontsize=16, frameon=True, ncol=legend_cols)
    
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel("# Observations", fontsize=20)
        plt.ylabel(r"$P(\mathcal{M} | x_0,..., x_i)$", fontsize=20)
        # plt.grid(True)
        sns.despine()
        plt.tight_layout()
        if path is not None:
            plt.savefig(f"{path}/model_probs_cumulative.png")
        if show:
            plt.show()
        plt.close()

    #---------------------------
    # Attention Heatmap plotting
    def plot_attention(self, stats_dict=None, labels=None, path=None, show=True):
        """
        Plot the attention weights for the best performing model for interpretability.

        Args:
            stats_dict: (dict)(optional) Dictionary with the comparison results. 
                            If not provided, it uses the results from the `compare()` call.
            labels:     (list of strings)(optional) List with the names of the parameters and data points.
            path:       (string)(optional) The path the plots are saved to.
                            If not provided, the plots are not saved.
            show:       (bool) Whether to show the created plots or not.
        """

        # Check path
        if path is None:
            path = self.path
            if not os.path.exists(path):
                os.makedirs(path)

        # Check stats_dict
        if stats_dict is None:
            stats_dict = self.stats

        # Get the best performing model (lowest AICc).
        best_model = sorted(stats_dict, key=lambda x: stats_dict[x]["AIC"])[0]

        def _plot_heatmap(data, xlabels, ylabels, name, show):
            # Set annotations in the attention blocks
            annotation_mask = data > 0.0
            annot = np.where(annotation_mask, data.round(2), np.nan)  # Use NaN to hide annotations below threshold
            annotations = annot.astype(str)
            annotations[np.isnan(annot)] = ""

            # Set up colours
            vmin, vmax = 0.0, 1.0
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax) 

            # Create figure
            fig = plt.figure(figsize=(12,6), dpi=500)
            ax = sns.heatmap(
                data,
                xticklabels=xlabels,
                yticklabels=ylabels,
                cmap='magma',
                cbar=False,
                linewidths=.5,
                square=False,
                vmin=vmin,
                vmax=vmax,
                annot=annotations,
                norm=norm,
                fmt='',
                annot_kws={"size": 35 / np.sqrt(len(data))}
            ) 

            ax.set_xlabel("Keys", fontsize=25)
            ax.set_ylabel("Queries", fontsize=25)
            plt.xticks(rotation=0, ha='center', fontsize=20)
            plt.yticks(rotation=0, fontsize=20)

            # Add the single, shared color bar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]

            # Create the color bar using a "dummy" mappable object
            sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
            sm._A = [] # Dummy empty array
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Attention Weight', fontsize=20)

            fig.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust rect to make space for suptitle and cbar

            
            #plt.tight_layout()
            if path is not None:
                plt.savefig(f"{path}/{name}.png")
            if show:
                plt.show()
            plt.close()
        
        ####################
        # Avg Attention between informative Tokens

        # attn_weights is (layers, nodes, nodes+1); the sampler may prepend a
        # batch dimension (num_batches, layers, nodes, nodes+1) -> average it out.
        attn = stats_dict[best_model]["attn_weights"]
        if attn.dim() == 4:
            attn = attn.mean(0)

        data = attn.mean(0).numpy()
        data = data[np.ix_(~self.condition_mask.bool(),torch.cat((self.condition_mask.bool(), torch.tensor([True]))))]

        xlabels = list(compress(labels, self.condition_mask)) + ["Bias KV"]
        ylabels = list(compress(labels, 1-self.condition_mask))
        _plot_heatmap(data, xlabels, ylabels, "selected_attention_map", show)

        ####################
        # Layer by Layer Attention

        data = attn.numpy()

        # Create a list to hold the data for each layer
        plot_data = []
        for layer_weights in data:
            avg_attention_map = layer_weights
            
            param_attention_subset = avg_attention_map[np.ix_(~self.condition_mask.bool(),torch.cat((self.condition_mask.bool(), torch.tensor([True]))))]
            plot_data.append(param_attention_subset)

        # Set up Figure: one row per transformer layer
        nrows = attn.shape[0]
        fig, axes = plt.subplots(
            nrows=nrows, 
            ncols=1, 
            figsize=(12, 3*nrows),
            sharex=True,
            dpi=500
        )

        # Set up colours
        vmin, vmax = 0.0, 1.0
        norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax) 

        # Loop through each layer and plot the heatmap
        for i, ax in enumerate(axes):
            # Get the data for the current layer
            data_to_plot = plot_data[i]

            # Create a boolean mask for annotations. Only show values > 0.0
            annotation_mask = data_to_plot > 0.0
            annot = np.where(annotation_mask, data_to_plot.round(2), np.nan)  # Use NaN to hide annotations below threshold
            annotations = annot.astype(str)
            annotations[np.isnan(annot)] = ""
            
            # Create the heatmap on the current subplot axis `ax`
            sns.heatmap(
                data_to_plot,
                xticklabels=xlabels,
                yticklabels=ylabels,
                cmap='magma',
                linewidths=.5,
                ax=ax,
                cbar=False,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                annot=annotations,
                fmt="",
                annot_kws={"size": 35 / np.sqrt(len(data_to_plot))}
            )
            
            # Set titles and labels for each subplot
            ax.set_title(f"Layer {i+1}", fontsize=25, loc='left')
            ax.tick_params(axis='y', rotation=0, labelsize=20) # Rotate y-axis labels for better readability

            # Only show x-axis labels on the very last plot
            if i == len(axes) - 1:
                #ax.set_xlabel("Information Source (Observations and Bias)", fontsize=14)
                ax.tick_params(axis='x', rotation=0, labelsize=20)
            else:
                ax.set_xlabel('')

        # Add the single, shared color bar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]

        # Create the color bar using a "dummy" mappable object
        sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
        sm._A = [] # Dummy empty array
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Attention Weight', fontsize=20)

        fig.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust rect to make space for suptitle and cbar

        if path is not None:
            plt.savefig(f"{path}/layer_attention.png")
        if show:
            plt.show()
        plt.close()
