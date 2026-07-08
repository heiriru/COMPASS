import os
import sys
import pickle
import tqdm
import time

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

    # helpers for parameter correction
    def _model_parameter_count(self, model, x):
        return model.nodes_size - x.shape[-1]

    def _condition_mask_for_model(self, model, x, condition_mask):
        observation_dim = x.shape[-1]
        parameter_count = model.nodes_size - observation_dim

        if parameter_count < 0:
            raise ValueError(
                f"Observation dimension ({observation_dim}) is larger than model "
                f"joint dimension ({model.nodes_size})."
            )

        if condition_mask is None:
            return torch.cat([
                torch.zeros(parameter_count),
                torch.ones(observation_dim),
            ])

        condition_mask = torch.as_tensor(condition_mask).float()
        mask_dim = condition_mask.shape[-1]

        if mask_dim == model.nodes_size:
            return condition_mask

        raise ValueError(
            "condition_mask must either be None or have one entry per joint feature "
            f"for the current model ({model.nodes_size}); got {mask_dim}."
        )
    def _akaike_weights(self, aics):
        aics = torch.as_tensor(aics, dtype=torch.float)
        delta_aic = aics - aics.min(dim=0, keepdim=True).values
        return torch.softmax(-0.5 * delta_aic, dim=0)

    def _information_criterion(self, log_likelihood, param_count, n_obs):
        log_likelihood = torch.as_tensor(log_likelihood, dtype=torch.float)
        param_count = torch.as_tensor(param_count, dtype=torch.float, device=log_likelihood.device)
        n_obs = torch.as_tensor(n_obs, dtype=torch.float, device=log_likelihood.device)

        aic = 2 * param_count - 2 * log_likelihood
        criterion = aic.clone()
        use_aicc = n_obs > param_count + 1
        if torch.any(use_aicc):
            criterion[use_aicc] = (
                aic[use_aicc]
                + (2 * param_count[use_aicc] * (param_count[use_aicc] + 1))
                / (n_obs - param_count[use_aicc] - 1)
            )
        return criterion

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
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3,
               device="cuda", verbose=False, method="dpm", equation="reverse_sde",
               likelihood_backend="kde", divergence_timesteps=100,
               divergence_batch_size=128, divergence_method="hutchinson",
               hutchinson_samples=1, exact_divergence_max_dim=16):
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
            method:         (string) Numerical solver used during inference.
                                "dpm"   - (default) DPM-style solver with order 'order'
                                "euler" - Euler integration
            equation:       (string) Equation family to solve during inference.
                                "reverse_sde" - (default) Current reverse-SDE behavior
                                "probability_flow_ode" - Deterministic probability-flow ODE
        """

        if not self.trained_models:
            print("Models are not trained or provided. Please train the models before comparing.")
            return
        if likelihood_backend not in {"kde", "divergence_integration"}:
            raise ValueError("likelihood_backend must be 'kde' or 'divergence_integration'.")
        
        self.stats = {}
        self.model_null_log_probs = {}
        self.softmax = nn.Softmax(dim=0)

        provided_condition_mask = condition_mask
        
        # Loop over all models
        for model_name, model in tqdm.tqdm(self.models_dict.items(), desc="Comparing models", unit="model"):
            self.stats[model_name] = {}
            model_condition_mask = self._condition_mask_for_model(
                model, x, provided_condition_mask
            )
            model_parameter_count = self._model_parameter_count(model, x)
            self.condition_mask = model_condition_mask

            ####################
            # Posterior sampling
            posterior_samples = model.sample(x=x, err=err, condition_mask=model_condition_mask,
                                            timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                            multi_obs_inference=multi_obs_inference, hierarchy=hierarchy,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                            device=device, verbose=verbose, method=method, equation=equation)
            posterior_samples = posterior_samples.cpu().numpy()

            # Inference Attention weights
            self.stats[model_name]["attn_weights"] = model.sampler.all_attn_weights

            # MAP estimation
            theta_hat = np.array([self._map_kde(posterior_samples[i]) for i in range(len(posterior_samples))])
            MAP_posterior, std_MAP_posterior = torch.tensor(theta_hat[:,0], dtype=torch.float), torch.tensor(theta_hat[:,1], dtype=torch.float)

            # Storing MAP and std MAP
            self.stats[model_name]["MAP"] = theta_hat

            likelihood_started = time.perf_counter()
            if likelihood_backend == "kde":
                # Preserve the existing stochastic likelihood/KDE path.
                likelihood_samples = model.sample(theta=MAP_posterior, err=std_MAP_posterior, condition_mask=(1-model_condition_mask),
                                                timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                                multi_obs_inference=False, hierarchy=None,
                                                order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                                device=device, verbose=verbose, method=method, equation=equation)
                likelihood_samples = likelihood_samples.cpu().numpy()
                log_probs = torch.tensor([self._log_prob(likelihood_samples[i], x[i]) for i in range(len(x))])
            else:
                log_probs = model.log_prob_probability_flow(
                    theta=MAP_posterior, x=x, condition_mask=1-model_condition_mask,
                    timesteps=divergence_timesteps, eps=eps,
                    divergence_method=divergence_method, hutchinson_samples=hutchinson_samples,
                    exact_divergence_max_dim=exact_divergence_max_dim,
                    divergence_batch_size=divergence_batch_size, device=device,
                    verbose=verbose, cfg_alpha=cfg_alpha)
            if verbose:
                print(f"{model_name} {likelihood_backend} likelihood: {time.perf_counter()-likelihood_started:.3f}s")
            self.stats[model_name]["condition_mask"] = model_condition_mask.cpu()
            self.stats[model_name]["likelihood_backend"] = likelihood_backend
            self.stats[model_name]["log_probs"] = log_probs
            self.stats[model_name]["param_count"] = model_parameter_count
            self.stats[model_name]["log_likelihood"] = log_probs.sum()
            n_obs = len(x)
            k = model_parameter_count
            log_likelihood = log_probs.sum()
            aic = 2 * k - 2 * log_likelihood
            bic = k * torch.log(torch.tensor(float(n_obs))) - 2 * log_likelihood
            self.stats[model_name]["AIC"] = aic
            self.stats[model_name]["BIC"] = bic
            if n_obs > k + 1:
                aicc = aic + (2 * k * (k + 1)) / (n_obs - k - 1)
                self.stats[model_name]["AICc"] = aicc
                self.stats[model_name]["IC"] = aicc
                self.stats[model_name]["IC_name"] = "AICc"
            else:
                self.stats[model_name]["AICc"] = None
                self.stats[model_name]["IC"] = aic
                self.stats[model_name]["IC_name"] = "AIC"


        # Calculate model probabilities from four parallel criteria:
        # - IC: AICc where defined, otherwise AIC. This remains the default.
        # - AIC: the uncorrected Akaike criterion.
        # - BIC: Schwarz/Bayesian information criterion.
        # - no penalty: raw likelihood weights with no parameter penalty.
        model_names = list(self.stats.keys())
        information_criteria = torch.stack([self.stats[name]["IC"] for name in model_names])
        aics = torch.stack([self.stats[name]["AIC"] for name in model_names])
        bics = torch.stack([self.stats[name]["BIC"] for name in model_names])
        log_likelihoods = torch.stack([self.stats[name]["log_likelihood"] for name in model_names])
        model_probs = self._akaike_weights(information_criteria)
        model_probs_aic = self._akaike_weights(aics)
        model_probs_bic = self._akaike_weights(bics)
        model_probs_no_penalty = torch.softmax(log_likelihoods - log_likelihoods.max(), dim=0)
        param_counts = torch.tensor([self.stats[name]["param_count"] for name in model_names], dtype=torch.float,)

        # Calculate per-observation probabilities for the violin plot using AIC.
        log_probs = torch.stack([self.stats[model_name]["log_probs"] for model_name in self.stats.keys()])
        individual_aics = 2 * param_counts.unsqueeze(1) - 2 * log_probs
        probs = self._akaike_weights(individual_aics)

        for i, model_name in enumerate(self.stats.keys()):
            self.stats[model_name]["model_prob"] = model_probs[i].item()
            self.stats[model_name]["model_prob_ic"] = model_probs[i].item()
            self.stats[model_name]["model_prob_aic"] = model_probs_aic[i].item()
            self.stats[model_name]["model_prob_bic"] = model_probs_bic[i].item()
            self.stats[model_name]["model_prob_no_penalty"] = model_probs_no_penalty[i].item()
            self.stats[model_name]["obs_probs"] = probs[i]

        model_names = list(self.stats.keys())
        best_model = model_names[model_probs.argmax()]
        best_model_prob = 100*model_probs.max()

        model_print_length = len(max(model_names, key=len))
        print(f"Probabilities of the models after {len(x)} observations:")
        for model in model_names:
            ic_name = self.stats[model]["IC_name"]
            print(f"{model.ljust(model_print_length)}: {100*self.stats[model]['model_prob']:6.2f} % ({ic_name})")
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
        samples = np.asarray(samples, dtype=np.float64)
        kde = gaussian_kde(samples.T)  # KDE expects (n_dims, n_samples)
        
        # Start optimization from the mean
        initial_guess = np.mean(samples, axis=0).astype(np.float64)
        
        # Use full minimize with multiple dimensions
        result = optimize.minimize(lambda x: -kde(x.reshape(-1, 1)), initial_guess)
        std_devs = np.sqrt(np.diag(kde.covariance))

        return result.x, std_devs
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

        # Sort models by the AIC-based per-observation probabilities used in the violin plot.
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
        param_counts = torch.tensor([stats_dict[model]["param_count"] for model in model_keys], dtype=torch.float)
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

        # Calculate mean model probabilities for N observations. The log likelihoods
        # are summed over the selected observations, but the AIC/AICc penalty is
        # applied once for the whole selected subset.
        avg_model_probs = []
        avg_model_probs_no_penalty = []
        avg_model_probs_bic = []
        for n in range(50):
            all_N_model_probs = []
            all_N_model_probs_no_penalty = []
            all_N_model_probs_bic = []
            for i in range(0,model_log_probs.shape[1]+1):
                if i != 0:
                    idx = torch.randperm(model_log_probs.shape[1])[:i]
                    N_log_likelihood = model_log_probs[:,idx].sum(dim=1)
                    N_criterion = self._information_criterion(N_log_likelihood, param_counts, i)
                    N_model_probs = self._akaike_weights(N_criterion)
                    N_model_probs_no_penalty = torch.softmax(N_log_likelihood - N_log_likelihood.max(), dim=0)
                    N_bic = param_counts * torch.log(torch.tensor(float(i))) - 2 * N_log_likelihood
                    N_model_probs_bic = self._akaike_weights(N_bic)
                elif i == 0:
                    N_model_probs = torch.ones(len(model_keys), dtype=torch.float) / len(model_keys)
                    N_model_probs_no_penalty = N_model_probs.clone()
                    N_model_probs_bic = N_model_probs.clone()

                all_N_model_probs.append(N_model_probs)
                all_N_model_probs_no_penalty.append(N_model_probs_no_penalty)
                all_N_model_probs_bic.append(N_model_probs_bic)
            all_N_model_probs = torch.stack(all_N_model_probs)
            avg_model_probs.append(all_N_model_probs)
            avg_model_probs_no_penalty.append(torch.stack(all_N_model_probs_no_penalty))
            avg_model_probs_bic.append(torch.stack(all_N_model_probs_bic))

        avg_model_probs = torch.stack(avg_model_probs)
        avg_model_probs_no_penalty = torch.stack(avg_model_probs_no_penalty)
        avg_model_probs_bic = torch.stack(avg_model_probs_bic)
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
            plt.legend(
                title="Models",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=15,
                title_fontsize=16,
                frameon=True,
                ncol=legend_cols,
            )

        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel("# Observations", fontsize=20)
        plt.ylabel(r"$P(\mathcal{M} | x_0,..., x_i)$", fontsize=20)
        # plt.grid(True)
        sns.despine()
        plt.tight_layout(rect=(0, 0, 0.78, 1))
        if path is not None:
            plt.savefig(f"{path}/model_probs_cumulative.png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        # Compare cumulative probabilities under no penalty, AICc/AIC, and BIC.
        cumulative_variants = [
            ("No parameter penalty", avg_model_probs_no_penalty),
            ("AICc/AIC penalty", avg_model_probs),
            ("BIC penalty", avg_model_probs_bic),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(24, 6), dpi=500, sharex=True, sharey=True)
        observation_counts = torch.arange(0, model_log_probs.shape[1] + 1)
        for ax, (title, probabilities) in zip(axes, cumulative_variants):
            probability_mean = probabilities.mean(0)
            probability_se = probabilities.std(0) / torch.sqrt(torch.tensor(probabilities.shape[0]))
            for n in range(n_models):
                ax.errorbar(
                    observation_counts, probability_mean[:, n], yerr=probability_se[:, n],
                    label=model_names[n], marker="o", markersize=4, linewidth=2,
                    elinewidth=1, capsize=2, color=palette[n],
                )
            ax.set_title(title, fontsize=17)
            ax.set_xlabel("# Observations", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=13)
            sns.despine(ax=ax)
            if title == "AICc/AIC penalty":
                first_all_valid_n = int(param_counts[:n_models].max().item()) + 2
                ax.axvline(
                    first_all_valid_n,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label=rf"AICc valid from $n={first_all_valid_n}$ ($n > k + 1$)",
                )
        axes[0].set_ylabel(r"$P(\mathcal{M} | x_0,..., x_i)$", fontsize=16)
        model_handles, model_labels = axes[0].get_legend_handles_labels()
        _, aicc_labels = axes[1].get_legend_handles_labels()
        aicc_handle = axes[1].lines[-1]
        fig.legend(
            model_handles[:n_models] + [aicc_handle],
            model_labels[:n_models] + [aicc_labels[-1]],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=n_models + 1,
            fontsize=11,
            frameon=False,
            handlelength=2.8,
            handletextpad=0.7,
            columnspacing=2.0,
            borderaxespad=0.8,
        )
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        if path is not None:
            fig.savefig(f"{path}/model_probs_cumulative_comparison.png")
        if show:
            plt.show()
        plt.close(fig)

    def plot_comparison_across_runs(self, stats_dicts, path=None, show=True, filename="model_probs_cumulative_comparison_across_runs.png"):
        """Plot cumulative model probabilities summarized across comparison runs."""
        if len(stats_dicts) < 2:
            return

        model_names = list(stats_dicts[0].keys())
        n_models = len(model_names)
        param_counts = torch.tensor(
            [stats_dicts[0][name]["param_count"] for name in model_names], dtype=torch.float
        )
        n_observations = len(stats_dicts[0][model_names[0]]["log_probs"])
        run_probabilities = {
            "No parameter penalty": [],
            "AIC penalty": [],
            "AICc penalty": [],
            "BIC penalty": [],
        }

        for stats in stats_dicts:
            log_probs = torch.stack([stats[name]["log_probs"] for name in model_names])
            if log_probs.shape[1] != n_observations:
                raise ValueError("All comparison runs must contain the same number of observations.")

            cumulative_log_likelihood = log_probs.cumsum(dim=1)
            variants = {name: [] for name in run_probabilities}
            uniform = torch.ones(n_models, dtype=torch.float) / n_models
            for variant in variants.values():
                variant.append(uniform)

            for n_obs in range(1, n_observations + 1):
                log_likelihood = cumulative_log_likelihood[:, n_obs - 1]
                variants["No parameter penalty"].append(
                    torch.softmax(log_likelihood - log_likelihood.max(), dim=0)
                )
                aic = 2 * param_counts - 2 * log_likelihood
                variants["AIC penalty"].append(self._akaike_weights(aic))
                aicc = self._information_criterion(log_likelihood, param_counts, n_obs)
                variants["AICc penalty"].append(self._akaike_weights(aicc))
                bic = param_counts * torch.log(torch.tensor(float(n_obs))) - 2 * log_likelihood
                variants["BIC penalty"].append(self._akaike_weights(bic))

            for title, probabilities in variants.items():
                run_probabilities[title].append(torch.stack(probabilities))

        run_probabilities = {
            title: torch.stack(probabilities) for title, probabilities in run_probabilities.items()
        }
        observation_counts = torch.arange(n_observations + 1)
        palette = sns.color_palette("dark", n_colors=n_models)
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=500, sharex=True, sharey=True)
        axes = axes.ravel()

        for ax, (title, probabilities) in zip(axes, run_probabilities.items()):
            mean = probabilities.mean(dim=0)
            lower = torch.quantile(probabilities, 0.05, dim=0)
            upper = torch.quantile(probabilities, 0.95, dim=0)
            if title == "AICc penalty":
                first_all_valid_n = int(param_counts.max().item()) + 2
                invalid_mask = observation_counts < first_all_valid_n
                mean = mean.clone()
                lower = lower.clone()
                upper = upper.clone()
                mean[invalid_mask, :] = float("nan")
                lower[invalid_mask, :] = float("nan")
                upper[invalid_mask, :] = float("nan")
                ax.axvspan(
                    observation_counts[0].item(),
                    first_all_valid_n,
                    color="0.85",
                    alpha=0.35,
                    linewidth=0,
                    zorder=0,
                )
            for model_index, model_name in enumerate(model_names):
                ax.plot(
                    observation_counts,
                    mean[:, model_index],
                    color=palette[model_index],
                    linewidth=2.5,
                    label=model_name,
                )
                ax.fill_between(
                    observation_counts,
                    lower[:, model_index],
                    upper[:, model_index],
                    color=palette[model_index],
                    alpha=0.2,
                    linewidth=0,
                )
            ax.set_title(title, fontsize=17)
            ax.set_xlabel("# Observations", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=13)
            sns.despine(ax=ax)
            if title == "AICc penalty":
                ax.axvline(
                    first_all_valid_n,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label=rf"AICc valid from $n={first_all_valid_n}$ ($n > k + 1$)",
                )

        axes[0].set_ylabel(r"$P(\mathcal{M}_k | x_0, ..., x_i)$", fontsize=16)
        axes[2].set_ylabel(r"$P(\mathcal{M}_k | x_0, ..., x_i)$", fontsize=16)
        model_handles, model_labels = axes[0].get_legend_handles_labels()
        _, aicc_labels = axes[2].get_legend_handles_labels()
        fig.legend(
            model_handles + [axes[2].lines[-1]],
            model_labels + [aicc_labels[-1]],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=n_models + 1,
            fontsize=11,
            frameon=False,
            handlelength=2.8,
            handletextpad=0.7,
            columnspacing=2.0,
            borderaxespad=0.8,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        if path is not None:
            os.makedirs(path, exist_ok=True)
            fig.savefig(f"{path}/{filename}")
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

        # Get the best performing model. Lower AIC/AICc is better.
        best_model = sorted(stats_dict, key=lambda x: stats_dict[x].get("IC", stats_dict[x]["AIC"]))[0]

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

        data = stats_dict[best_model]["attn_weights"].mean(0).numpy()
        data = data[np.ix_(~self.condition_mask.bool(),torch.cat((self.condition_mask.bool(), torch.tensor([True]))))]

        xlabels = list(compress(labels, self.condition_mask)) + ["Bias KV"]
        ylabels = list(compress(labels, 1-self.condition_mask))
        _plot_heatmap(data, xlabels, ylabels, "selected_attention_map", show)

        ####################
        # Layer by Layer Attention

        data = stats_dict[best_model]["attn_weights"].numpy()

        # Create a list to hold the data for each layer
        plot_data = []
        for layer_weights in data:
            avg_attention_map = layer_weights
            
            param_attention_subset = avg_attention_map[np.ix_(~self.condition_mask.bool(),torch.cat((self.condition_mask.bool(), torch.tensor([True]))))]
            plot_data.append(param_attention_subset)

        # Set up Figure   
        nrows = stats_dict[best_model]["attn_weights"].shape[2]
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

            # Create a boolean mask for annotations. Only show values > 0.1
            annotation_mask = data_to_plot > 0.1
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
                # annot=annotations,
                fmt="",
                # annot_kws={"size": 35 / np.sqrt(len(data_to_plot))}
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
