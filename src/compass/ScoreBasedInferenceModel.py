import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import time

from .ConditionTransformer import ConditionTransformer
from .SDE import VESDE, VPSDE
from .Sampler import Sampler
from .Trainer import Trainer
from .MultiObsSampler import MultiObsSampler

#################################################################################################
# ///////////////////////////////////// Diffusion Model /////////////////////////////////////////
#################################################################################################

class ScoreBasedInferenceModel(nn.Module):
    def __init__(
            self,
            nodes_size,         # Number of features in joint (theta, x)
            sde_type="vesde",   # Type of Stochastic Differential Equation (SDE)
            sigma=25.0,         # Variance for VESDE
            hidden_size=128,    # Hidden size of the transformer
            depth=6,            # Number of transformer blocks
            num_heads=16,       # Number of attention heads
            mlp_ratio=4,        # Ratio of MLP hidden size to embedding size
            device="cpu"        # Device to initialize the model on
            ):
        
        super(ScoreBasedInferenceModel, self).__init__()

        self.nodes_size = nodes_size
        self.sde_type = sde_type
        self.sigma = sigma
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Init SDE
        self.sigma = sigma
        if sde_type == "vesde":
            self.sde = VESDE(sigma=self.sigma)
        elif sde_type == "vpsde":
            self.sde = VPSDE()
        else:
            raise ValueError("Invalid SDE type")
        
        # Init Model
        self.model = ConditionTransformer(nodes_size=self.nodes_size, hidden_size=hidden_size, 
                                       depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.model.to(device)
        
        # Init Trainer
        self.trainer = Trainer(self)

        # Init Sampler
        self.sampler = Sampler(self)
        self.multi_obs_sampler = MultiObsSampler(self)
        
    #############################################
    # ----- Forward Diffusion -----
    #############################################

    def forward_diffusion_sample(self, x_0, t, x_1=None, condition_mask=None):
        # Diffusion process for time t with defined SDE
        
        if condition_mask is None:
            condition_mask = torch.zeros_like(x_0)

        if x_1 is None:
            x_1 = torch.randn_like(x_0)*(1-condition_mask)+(condition_mask)*x_0

        std = self.sde.marginal_prob_std(t).reshape(-1, 1).to(x_0.device)
        x_t = x_0 + std * x_1 * (1-condition_mask)
        return x_t
    
    #############################################
    # ----- Output Scaling -----
    #############################################

    def output_scale_function(self, t, x):
        scale = self.sde.marginal_prob_std(t).to(x.device)
        return x / scale
    
    #############################################
    # ----- Probability-flow log probability -----
    #############################################

    def _probability_flow_score(self, joint, t, mask, cfg_alpha=None):
        score = self.output_scale_function(t, self.model(x=joint, t=t, c=mask))
        if cfg_alpha is not None:
            uncond = self.output_scale_function(t, self.model(x=joint, t=t, c=torch.zeros_like(mask)))
            score = uncond + cfg_alpha * (score - uncond)
        return score

    def _probability_flow_vector_field(self, joint, t, mask, cfg_alpha=None):
        if self.sde_type != "vesde":
            raise NotImplementedError("Probability-flow likelihood currently supports only VESDE.")
        sigma = torch.as_tensor(self.sde.sigma, device=joint.device, dtype=joint.dtype)
        return -0.5 * sigma.pow(2 * t) * self._probability_flow_score(joint, t, mask, cfg_alpha) * (1-mask)

    def _probability_flow_divergence(self, field, joint, latent_mask, method, samples):
        divergence = torch.zeros(joint.shape[0], device=joint.device, dtype=joint.dtype)
        if method == "exact":
            for index in torch.where(latent_mask[0])[0]:
                gradient = torch.autograd.grad(field[:, index].sum(), joint, retain_graph=True)[0]
                divergence += gradient[:, index]
            return divergence
        for sample in range(samples):
            probe = torch.empty_like(joint).bernoulli_(0.5).mul_(2).sub_(1) * latent_mask
            vjp = torch.autograd.grad((field*probe).sum(), joint, retain_graph=sample < samples-1)[0]
            divergence += (vjp*probe).sum(1)
        return divergence / samples

    def _integrate_probability_flow_batch(self, joint, mask, timesteps, eps,
                                          divergence_method, hutchinson_samples,
                                          cfg_alpha=None, save_trajectory=False):
        conditioned = joint * mask
        correction = torch.zeros(joint.shape[0], device=joint.device, dtype=joint.dtype)
        trajectory = [joint.detach().cpu()] if save_trajectory else None
        times = torch.linspace(eps, 1., timesteps+1, device=joint.device, dtype=joint.dtype)
        for step in range(timesteps):
            dt, t = times[step+1]-times[step], times[step].reshape(1, 1)
            state = joint.detach().requires_grad_(True)
            field = self._probability_flow_vector_field(state, t, mask, cfg_alpha)
            divergence = self._probability_flow_divergence(field, state, (1-mask).bool(), divergence_method, hutchinson_samples)
            joint = (state + dt*field).detach()
            joint = joint*(1-mask) + conditioned
            correction += dt*divergence.detach()
            if save_trajectory:
                trajectory.append(joint.cpu())
        return joint, correction, trajectory

    def log_prob_probability_flow(self, theta=None, x=None, condition_mask=None,
                                  timesteps=100, eps=1e-3,
                                  divergence_method="hutchinson", hutchinson_samples=1,
                                  exact_divergence_max_dim=16,
                                  divergence_batch_size=128, device="cuda",
                                  verbose=False, cfg_alpha=None):
        """Return one deterministic PF-ODE log probability per paired theta/x row."""
        if theta is None or x is None:
            raise ValueError("theta and x must both be provided.")
        theta, x = torch.as_tensor(theta), torch.as_tensor(x)
        if theta.ndim != 2 or x.ndim != 2:
            raise ValueError("theta and x must have shape (n_obs, dimension).")
        if theta.shape[0] != x.shape[0]:
            raise ValueError("theta and x must have the same number of observations.")
        if theta.shape[0] == 0:
            raise ValueError("theta and x must contain at least one observation.")
        if theta.shape[1] + x.shape[1] != self.nodes_size:
            raise ValueError("theta_dim + x_dim must equal nodes_size.")
        if divergence_method not in {"exact", "hutchinson"}:
            raise ValueError("divergence_method must be 'exact' or 'hutchinson'.")
        if not isinstance(timesteps, int) or timesteps <= 0:
            raise ValueError("timesteps must be a positive integer.")
        if not 0 < eps < 1:
            raise ValueError("eps must lie strictly between 0 and 1.")
        if not isinstance(hutchinson_samples, int) or hutchinson_samples <= 0:
            raise ValueError("hutchinson_samples must be a positive integer.")
        if not isinstance(divergence_batch_size, int) or divergence_batch_size <= 0:
            raise ValueError("divergence_batch_size must be a positive integer.")
        joint = torch.cat((theta, x), 1)
        if not joint.is_floating_point(): joint = joint.float()
        expected = torch.cat((torch.ones(theta.shape[1]), torch.zeros(x.shape[1])))
        mask = torch.as_tensor(expected if condition_mask is None else condition_mask, dtype=joint.dtype)
        if mask.ndim == 1:
            if mask.shape[0] != self.nodes_size: raise ValueError("condition_mask has the wrong size.")
            mask = mask.unsqueeze(0).expand(joint.shape[0], -1)
        elif mask.shape != joint.shape:
            raise ValueError("condition_mask must have shape (nodes_size,) or (n_obs, nodes_size).")
        if not torch.all((mask == 0) | (mask == 1)): raise ValueError("condition_mask must be binary.")
        if not torch.all(mask == expected.to(mask)):
            raise ValueError("Likelihood integration requires theta conditioned and x latent.")
        latent_dim = int((mask[0] == 0).sum())
        if divergence_method == "exact" and latent_dim > exact_divergence_max_dim:
            raise ValueError(f"Exact divergence latent dimension {latent_dim} exceeds exact_divergence_max_dim={exact_divergence_max_dim}.")
        if self.sde_type != "vesde": raise NotImplementedError("Probability-flow likelihood currently supports only VESDE.")
        target, original = torch.device(device), next(self.model.parameters()).device
        self.model.to(target).eval()
        loader = DataLoader(TensorDataset(joint, mask), batch_size=divergence_batch_size, shuffle=False)
        result, started = [], time.perf_counter()
        try:
            for batch_index, (state, batch_mask) in enumerate(loader):
                batch_started = time.perf_counter()
                state, batch_mask = state.to(target), batch_mask.to(target)
                terminal, correction, _ = self._integrate_probability_flow_batch(state, batch_mask, timesteps, eps, divergence_method, hutchinson_samples, cfg_alpha)
                latent = terminal[batch_mask == 0].reshape(terminal.shape[0], -1)
                std = self.sde.marginal_prob_std(torch.ones((), device=target, dtype=terminal.dtype)).to(target)
                prior = (-.5*(latent/std).square()-torch.log(std)-.5*math.log(2*math.pi)).sum(1)
                result.append((prior+correction).detach().cpu())
                if verbose: print(f"PF likelihood batch {batch_index+1}/{len(loader)}: {len(state)} observations in {time.perf_counter()-batch_started:.3f}s")
        finally:
            self.model.to(original)
        if verbose: print(f"PF likelihood total: {len(joint)} observations in {time.perf_counter()-started:.3f}s")
        return torch.cat(result)

    #############################################
    # ----- Training -----
    #############################################
    
    def train(self, theta, x, theta_val=None, x_val=None, 
                batch_size=128, max_epochs=500, lr=1e-3, device="cpu", 
                verbose=True, path=None, name="Model", early_stopping_patience=20):
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
        """

        # Combine theta and x into a single tensor
        # train_data: (num_samples, node_size)
        train_data = torch.cat([theta, x], dim=1)
        if theta_val is not None and x_val is not None:
            val_data = torch.cat([theta_val, x_val], dim=1)

        if device == "cuda":
            world_size = torch.cuda.device_count()
        else :
            world_size = 1

        self.trainer.train(world_size=world_size, train_data=train_data, val_data=val_data,
                            max_epochs=max_epochs, early_stopping_patience=early_stopping_patience, batch_size=batch_size, lr=lr,
                            path=path, name=name, device=device, verbose=verbose)

    #############################################
    # ----- Sample -----
    #############################################
        
    def sample(self, theta=None, x=None, err=None, condition_mask=None,
               timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None, multi_obs_inference=False, hierarchy=None,
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3,
               device="cpu", verbose=True, method="dpm", equation="reverse_sde", time_grid_type="linear_t", save_trajectory=False):
        """
        Sample from the model using the specified method

        Args:
            data: Input data : Tensor of shape (num_samples, num_observed_features)
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (num_samples, num_total_features)
            timesteps: Number of diffusion steps
            eps: End time for diffusion process
            num_samples: Number of samples to generate
            cfg_alpha: Classifier-free guidance strength

            - DPM-Solver parameters -
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            final_corrector_steps: Extra correction steps at the end

            - Other parameters -
            device: Device to run sampling on
            verbose: Whether to show progress bar
            method: Sampling method to use (euler, dpm)
            equation: Equation to solve (reverse_sde, probability_flow_ode)
            save_trajectory: Whether to save the intermediate denoising trajectory
        """

        # Combine data and create condition mask
        # 0 for latent values, 1 for observed values
        data = None
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
                                    verbose=verbose, method=method, equation=equation, time_grid_type=time_grid_type, save_trajectory=save_trajectory)
            
        elif multi_obs_inference == True:
            # Hierarchical Compositional Score Modeling
            samples = self.multi_obs_sampler.sample(world_size=world_size, data=data, condition_mask=condition_mask, timesteps=timesteps, eps=eps, num_samples=num_samples, device=device, cfg_alpha=cfg_alpha, hierarchy=hierarchy,
                                      order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                      verbose=verbose, method=method, equation=equation, time_grid_type=time_grid_type, save_trajectory=save_trajectory)

        # just return the sampled values
        samples = samples[:,:,(1-condition_mask).bool()] 
        
        return samples
    
    #############################################
    # ----- Save & Load -----
    #############################################
    
    def save(self, path, name="Model"):
        state_dict = {
                'model_state_dict' : self.model.state_dict(),
                'nodes_size': self.nodes_size,
                'sde_type': self.sde_type,
                'sigma': self.sigma,
                'hidden_size': self.hidden_size,
                'depth': self.depth,
                'num_heads': self.num_heads,
                'mlp_ratio': self.mlp_ratio
            }
        
        torch.save(state_dict, f"{path}/{name}.pt")
        
    @staticmethod
    def load(path, device=None):

        if device is None:
            # Automatically detect device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(path, map_location=device)

        model = ScoreBasedInferenceModel(
            nodes_size=checkpoint['nodes_size'],
            sde_type=checkpoint['sde_type'],
            sigma=checkpoint['sigma'],
            hidden_size=checkpoint['hidden_size'],
            depth=checkpoint['depth'],
            num_heads=checkpoint['num_heads'],
            mlp_ratio=checkpoint['mlp_ratio']
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])
        return model
    