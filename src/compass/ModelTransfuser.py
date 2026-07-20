import os
import sys
import pickle
import tqdm

import torch
import torch.nn as nn

import numpy as np

import scipy
from scipy import optimize
from scipy.stats import norm, gaussian_kde

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D

import seaborn as sns

from .ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm

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
                verbose=False, path=None, early_stopping_patience=20): 
        
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
               prior=None, correction="gauss",
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3,
               device="cuda", verbose=False, method=None, equation="reverse_sde",
               likelihood_method="pfode", map_method="score", criterion="aic",
               log_prob_timesteps=100, log_prob_divergence="exact"):
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
            hierarchy:      Hierarchy for the model
            order:          Order of the model
            snr:            Signal-to-noise ratio for the model
            corrector_steps_interval: Corrector steps interval for the model
            corrector_steps: Corrector steps for the model
            final_corrector_steps: Final corrector steps for the model
            device:         Device to run inference on
            verbose:        (bool) Whether to show inference progress
            method:         Solver method. Defaults to "dpm" for reverse_sde
                                and "heun" for probability_flow_ode.
            equation:       Equation used for posterior and legacy likelihood sampling:
                                "reverse_sde" (default) or "probability_flow_ode".
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
                                "kde"   - legacy behavior: mode of a Gaussian KDE fitted
                                          to the posterior samples.
            criterion:      (string) Information criterion for model weights:
                                "aic"   - (default) corrected Akaike IC (AICc)
                                "bic"   - Bayesian (Schwarz) IC: k*ln(n) - 2*logL
            log_prob_timesteps: Integration nodes of the PF-ODE likelihood solver.
            log_prob_divergence: Divergence evaluator for PF-ODE likelihoods:
                                "exact" (default), "hutchinson", or "learned".
        """

        if not self.trained_models:
            print("Models are not trained or provided. Please train the models before comparing.")
            return

        if criterion not in ("aic", "bic"):
            raise ValueError(f"criterion must be 'aic' or 'bic', got '{criterion}'")

        self.stats = {}
        self.model_null_log_probs = {}
        self.softmax = nn.Softmax(dim=0)
        self._aicc_warned_once = False
        self.criterion = criterion
        self.likelihood_method = likelihood_method
        self.map_method = map_method

        # Remember whether the user explicitly supplied a condition mask. If not,
        # it must be rebuilt per model because different models can have a
        # different number of (physical) parameters -> different nodes_size.
        user_condition_mask = condition_mask

        # Loop over all models
        for model_name, model in tqdm.tqdm(self.models_dict.items(), desc="Comparing models", unit="model"):
            self.stats[model_name] = {}
            if user_condition_mask is None:
                condition_mask = torch.cat([torch.zeros(model.nodes_size-x.shape[-1]),torch.ones(x.shape[-1])])
            else:
                condition_mask = user_condition_mask
            self.condition_mask = condition_mask

            ####################
            # Posterior sampling
            posterior_samples = model.sample(x=x, err=err, condition_mask=condition_mask,
                                            timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                            multi_obs_inference=multi_obs_inference, hierarchy=hierarchy,
                                            prior=prior, correction=correction,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                            device=device, verbose=verbose, method=method,
                                            equation=equation)
            posterior_samples = posterior_samples.cpu().numpy()

            # Inference Attention weights
            active_sampler = (
                model.multi_obs_sampler if multi_obs_inference else model.sampler
            )
            self.stats[model_name]["attn_weights"] = getattr(
                active_sampler, "all_attn_weights", None,
            )

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
                if multi_obs_inference:
                    map_condition_mask = condition_mask.clone()
                    hier = hierarchy if hierarchy is not None else torch.nonzero(1 - condition_mask).flatten().tolist()
                    map_condition_mask[list(hier)] = 1

                joint_map = model.map_estimate(joint_init, map_condition_mask,
                                               sigma_start=2.0 * post_std.max().item(),
                                               device=device)
                MAP_posterior = joint_map[:, ~c_bool].float()
                std_MAP_posterior = post_std
                theta_hat = np.stack([MAP_posterior.numpy(), std_MAP_posterior.numpy()], axis=1)
            else:
                raise ValueError(f"map_method must be 'score' or 'kde', got '{map_method}'")

            # Storing MAP and std MAP
            self.stats[model_name]["MAP"] = theta_hat

            ####################
            # Likelihood evaluation
            if likelihood_method == "pfode":
                # Direct likelihood log p(x_i | theta_MAP,i) through the
                # probability-flow ODE: conditions the score model on the MAP of
                # all latent dims and evaluates the exact model density at x_i.
                # No likelihood sampling and no KDE involved.
                joint_eval = torch.zeros(x_t.shape[0], model.nodes_size)
                joint_eval[:, c_bool] = x_t
                joint_eval[:, ~c_bool] = MAP_posterior
                log_probs = model.log_prob(joint_eval, condition_mask=(1 - condition_mask),
                                           timesteps=log_prob_timesteps, eps=eps,
                                           divergence=log_prob_divergence,
                                           device=device, verbose=verbose).float()
            elif likelihood_method == "kde":
                # Legacy: sample from the likelihood at the MAP and evaluate a KDE
                likelihood_samples = model.sample(theta=MAP_posterior, err=std_MAP_posterior, condition_mask=(1-condition_mask),
                                                timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                                multi_obs_inference=multi_obs_inference, hierarchy=hierarchy,
                                                prior=prior, correction=correction,
                                                order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                                device=device, verbose=verbose, method=method,
                                                equation=equation)
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
            # Store the unpenalized fit and both information criteria. The
            # legacy "AIC" entry remains the selected criterion used for model
            # weights, so criterion="bic" keeps returning BIC through that key.
            neg2_log_likelihood = (-2 * log_probs).sum()
            aicc_per_obs = self._aicc(log_probs, param_count, sample_size)
            bic_per_obs = self._bic(log_probs, param_count, sample_size)
            selected_ic_per_obs = self._ic(log_probs, param_count, sample_size)
            self.stats[model_name]["neg2_log_likelihood"] = neg2_log_likelihood
            self.stats[model_name]["AICc"] = aicc_per_obs.sum()
            self.stats[model_name]["BIC"] = bic_per_obs.sum()
            self.stats[model_name]["AIC"] = selected_ic_per_obs.sum()


        # Calculate Model Probabilitys from AICs.
        # Akaike weights: w_i = softmax(-AICc_i / 2); the model with the LOWEST
        # AICc gets the highest probability.
        aics = [self.stats[model_name]["AIC"] for model_name in self.stats.keys()]
        aics = torch.tensor(aics)
        delta_aics = aics - aics.min()
        model_probs = self.softmax(-0.5 * delta_aics)

        # Calculate Probability of each observation
        param_counts = torch.tensor([self.stats[model_name]["param_count"] for model_name in self.stats.keys()])
        log_probs = torch.stack([self.stats[model_name]["log_probs"] for model_name in self.stats.keys()])
        individual_aicc = self._ic(log_probs, param_counts.unsqueeze(1), x.shape[0])
        individual_delta_aicc = individual_aicc - individual_aicc.min(dim=0, keepdim=True).values
        probs = self.softmax(-0.5 * individual_delta_aicc)

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
    def _map_kde(self, samples):
        """Find the joint mode of the multivariate distribution"""
        samples = np.asarray(samples, dtype=np.float32)
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
    # Akaike Information Criterion
    def _aic(self, log_likelihood, k):
        """Plain Akaike Information Criterion: AIC = 2k - 2*logL."""
        ll = torch.as_tensor(log_likelihood, dtype=torch.float32)
        k = torch.as_tensor(k, dtype=torch.float32)
        return 2 * k - 2 * ll

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

        # plt.style.use('ggplot')

        #---------------------------
        sample_size = model_log_probs.shape[1]
        # AICc is valid only when n - k - 1 > 0 for every candidate model.
        aicc_first_valid_observation = int(param_counts.max().item()) + 2
        palette = sns.color_palette("dark", n_colors=n_models)
        legend_handles = [
            Line2D([0], [0], color=palette[index], linewidth=3, label=model_names[index])
            for index in range(n_models)
        ]
        legend_handles.append(
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=2,
                label=(
                    f"AICc valid from $n={aicc_first_valid_observation}$ "
                    r"($n > k + 1$)"
                ),
            )
        )
        # Independent baseline: the original single-panel violin layout using
        # likelihood-only probabilities, with no information-criterion penalty.
        no_penalty_probabilities = torch.nn.functional.softmax(model_log_probs, dim=0)
        baseline_model_labels = [name.replace(", ", "\n") for name in model_names[:n_models]]
        baseline_fig, baseline_axis = plt.subplots(figsize=(12, 6), dpi=500)
        sns.violinplot(
            data=no_penalty_probabilities.T[:, :n_models],
            ax=baseline_axis,
            palette=palette,
            cut=2,
            inner_kws=dict(box_width=5, whis_width=2, color="k"),
        )
        baseline_axis.set_xticks(range(n_models), baseline_model_labels)
        baseline_axis.tick_params(axis="x", which="major", labelsize=16)
        baseline_axis.tick_params(axis="y", which="major", labelsize=16)
        baseline_axis.set_ylabel(r"$P(\mathcal{M} | x_i)$", fontsize=20)
        baseline_axis.set_ylim(0.0, 1.0)
        baseline_axis.set_title("No parameter penalty", fontsize=18, pad=12)
        sns.despine(ax=baseline_axis)
        baseline_fig.tight_layout()
        if path is not None:
            baseline_fig.savefig(f"{path}/model_probs_violin_no_penalty.png")
        if show:
            plt.show()
        plt.close(baseline_fig)

        # Exact binned distributions in the same categorical layout as the
        # violin plot.  Bin widths encode observation counts; no KDE is used.
        histogram_fig, histogram_axis = plt.subplots(figsize=(12, 6), dpi=500)
        bin_edges = np.linspace(0.0, 1.0, 31)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_height = 0.95 * (bin_edges[1] - bin_edges[0])
        histogram_counts = [
            np.histogram(
                no_penalty_probabilities[model_index].detach().cpu().numpy(),
                bins=bin_edges,
            )[0]
            for model_index in range(n_models)
        ]
        max_count = max(counts.max() for counts in histogram_counts)
        for model_index, counts in enumerate(histogram_counts):
            widths = 0.8 * counts / max_count if max_count else counts
            histogram_axis.barh(
                bin_centers, widths, height=bin_height,
                left=model_index - widths / 2,
                color=palette[model_index], edgecolor="white", linewidth=0.3,
            )
        histogram_axis.set_xticks(range(n_models), baseline_model_labels)
        histogram_axis.set_xlim(-0.6, n_models - 0.4)
        histogram_axis.set_ylim(0.0, 1.0)
        histogram_axis.set_ylabel(r"$P(\mathcal{M} | x_i)$", fontsize=20)
        histogram_axis.set_title("No parameter penalty", fontsize=18, pad=12)
        histogram_axis.tick_params(axis="x", which="major", labelsize=16)
        histogram_axis.tick_params(axis="y", which="major", labelsize=16)
        sns.despine(ax=histogram_axis)
        histogram_fig.tight_layout()
        if path is not None:
            histogram_fig.savefig(f"{path}/model_probs_histogram_no_penalty.png")
        if show:
            plt.show()
        plt.close(histogram_fig)

        per_observation_criteria = {
            "No parameter penalty": -2 * model_log_probs,
            "AICc penalty": self._aicc(model_log_probs, param_counts.unsqueeze(1), sample_size),
            "BIC penalty": self._bic(model_log_probs, param_counts.unsqueeze(1), sample_size),
        }
        fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=500, sharey=True)
        for axis, (title, criterion_values) in zip(axes, per_observation_criteria.items()):
            delta_criterion = criterion_values - criterion_values.min(dim=0, keepdim=True).values
            probabilities = torch.nn.functional.softmax(-0.5 * delta_criterion, dim=0)
            sns.violinplot(
                data=probabilities.T[:, :n_models],
                ax=axis,
                palette=palette,
                cut=2,
                inner_kws=dict(box_width=5, whis_width=2, color="k"),
            )
            # Retain the full KDE tail inside the probability domain and clip
            # only any extrapolation beyond its valid [0, 1] boundaries.
            axis.set_ylim(0.0, 1.0)
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
            axis.tick_params(axis='both', which='major', labelsize=14)
            axis.set_xlabel("")
            axis.set_title(title, fontsize=18, pad=12)
        axes[0].set_ylabel(r"$P(\mathcal{M} | x_i)$", fontsize=20)
        fig.legend(handles=legend_handles, loc="lower center", fontsize=14,
                   frameon=False, ncol=len(legend_handles))
        fig.tight_layout(rect=(0, 0.08, 1, 1))

        if path is not None:
            fig.savefig(f"{path}/model_probs_violin.png")
        if show:
            plt.show()
        plt.close(fig)

        #---------------------------
        # Plot cumulative model probabilities for all penalty regimes.
        self._aicc_warned_once = getattr(self, "_aicc_warned_once", False)
        cumulative_probabilities = {
            "No parameter penalty": [],
            "AIC penalty": [],
            "AICc penalty": [],
            "BIC penalty": [],
        }
        for _ in range(50):
            all_criterion_probabilities = {name: [] for name in cumulative_probabilities}
            for i in range(model_log_probs.shape[1] + 1):
                if i == 0:
                    N_log_probs = torch.zeros_like(model_log_probs[:, 0]).unsqueeze(0)
                else:
                    idx = torch.randperm(model_log_probs.shape[1])[:i]
                    N_log_probs = model_log_probs[:, idx].T

                total_log_probs = N_log_probs.sum(0)
                criterion_values = {
                    "No parameter penalty": -2 * total_log_probs,
                    "AIC penalty": self._aic(total_log_probs, param_counts),
                    "AICc penalty": self._aicc(total_log_probs, param_counts, i),
                    "BIC penalty": self._bic(total_log_probs, param_counts, i),
                }
                for name, values in criterion_values.items():
                    delta_criterion = values - values.min()
                    all_criterion_probabilities[name].append(
                        torch.nn.functional.softmax(-0.5 * delta_criterion, dim=0)
                    )

            for name, probabilities in all_criterion_probabilities.items():
                cumulative_probabilities[name].append(torch.stack(probabilities))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=500, sharex=True, sharey=True)
        observation_counts = torch.arange(model_log_probs.shape[1] + 1)
        for axis, (title, repetitions) in zip(axes.flat, cumulative_probabilities.items()):
            if title == "AICc penalty":
                axis.axvspan(0, aicc_first_valid_observation, color="0.5", alpha=0.12, zorder=0)
                axis.axvline(aicc_first_valid_observation, color="black", linestyle="--", linewidth=2, zorder=1)
            mean_probabilities = torch.stack(repetitions).mean(0)
            for model_index in range(n_models):
                axis.plot(
                    observation_counts,
                    mean_probabilities[:, model_index],
                    linewidth=3,
                    color=palette[model_index],
                )
            axis.set_title(title, fontsize=18)
            axis.tick_params(axis='both', which='major', labelsize=14)
            axis.set_xlabel("# Observations", fontsize=16)
            axis.set_ylabel(r"$P(\mathcal{M} | x_0,..., x_i)$", fontsize=16)
            sns.despine(ax=axis)

        fig.legend(handles=legend_handles, loc="lower center", fontsize=14,
                   frameon=False, ncol=len(legend_handles))
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        if path is not None:
            fig.savefig(f"{path}/model_probs_cumulative.png")
        if show:
            plt.show()
        plt.close(fig)

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
