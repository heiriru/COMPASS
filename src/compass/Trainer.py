import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import schedulefree

from .schema import ConditionalMenu, GroupedBernoulli

import tqdm
import datetime
import time

class TensorTupleDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        
    def __len__(self):
        return len(self.tensor1)
    
    def __getitem__(self, idx):
        data = self.tensor1[idx]

        # Either a per-example mask tensor, or anything that can draw one:
        # a torch Distribution, or the schema-aware GroupedBernoulli.
        if isinstance(self.tensor2, torch.distributions.Distribution) or hasattr(
            self.tensor2, "sample"
        ):
            cond_mask = self.tensor2.sample(data.shape)
        else:
            cond_mask = self.tensor2[idx]

        return data, cond_mask, idx
    
#################################################################################################
# ////////////////////////////////////////// Training //////////////////////////////////////////
#################################################################################################
class Trainer():
    def __init__(self, SBIm):
        self.SBIm = SBIm
        #self.model_copy = copy.deepcopy(self.SBIm.model)
        # Get SDE from model for calculations
        self.sde = self.SBIm.sde

    #############################################
    # ----- Training loop -----
    #############################################
    def train(self, world_size, train_data, val_data=None,
              max_epochs=500, early_stopping_patience=20, batch_size=128, lr=1e-3,
              path=None, name="Model", device="cpu", verbose=True,
              time_sampling="mixture", mask_loss_weights=None,
              mask_sampler=None, validation_mask_seed=20250718,
              global_target_fn=None, global_target_indices=None, tsm_lambda_c=1.0,
              tsm_variance_fn=None, time_loss_weight=None, sigma_data=1.0):

        """
        Training function for the score prediction task

        Args:
            rank: Rank of the current process
            world_size: Number of processes
            train_data: Training data
            val_data: Validation data
            max_epochs: Maximum number of epochs
            early_stopping_patience: Number of epochs to wait before early stopping
            batch_size: Batch size
            lr: Learning rate
            path: Path to save the model
            name: Name of the model
            device: Device to use
            verbose: Verbosity
            time_sampling: How diffusion times are drawn during training:
                    "uniform": t ~ U(eps, 1). Under a VESDE this severely
                        undersamples small noise scales (only ~1% of draws reach
                        sigma_m < 0.1 for sigma=25), leaving the score network
                        inaccurate exactly where posteriors are resolved.
                    "log_sigma": noise scales log-uniform between sigma_m(eps)
                        and sigma_m(1).
                    "mixture" (default): 50/50 mix of both.
            global_target_fn: Optional target score matching (TSM) for the shared
                    (global) coordinates of a hierarchical model. A callable mapping a
                    batch of *clean* node vectors (B, nodes) to the clean joint score
                    of the globals,
                        T = grad_g log p(g) + grad_g log p(l | g)     (B, len(indices)),
                    in network coordinates. Because g reaches the data only through l,
                    E[T | noisy g, x] / alpha_t is exactly the diffused score of the
                    globals (Tweedie / target score matching + Fisher's identity), for
                    every mask in which they are latent, and its variance stays bounded
                    as the noise vanishes, unlike the DSM target -eps / sigma. ``None``
                    (default) is plain DSM, unchanged.
            global_target_indices: Node indices of the globals T refers to.
            tsm_lambda_c: Blend scale. The global target is
                    w T / alpha_t + (1 - w) (-eps / sigma_t), w = 1 / (1 + (lambda/lambda_c)^2):
                    TSM at low noise, where DSM is noisiest, DSM at high noise. Any w
                    keeps the target unbiased.
            tsm_variance_fn: Optional per-coordinate blend instead of ``tsm_lambda_c``.
                    Maps the *noisy* globals x_t[:, indices] and lambda (B, 1) to V (B, k), a
                    variance (bound) of T; the blend is then the inverse-variance weight
                    w = 1 / (1 + lambda^2 V), the DSM target having variance 1/lambda^2.
                    V may depend only on network inputs (here x_t), never on the clean
                    values, or the blend is biased.
        """
        start_time = time.time()

        # Create Checkpoint directory
        if path is None:
            path = "data/models/Model_test/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Set Parameters
        self.world_size = world_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.lr = lr
        self.path = path
        self.name = name
        self.name_checkpoint = f"{self.name}_checkpoint"
        self.verbose = verbose
        self.eps = 1e-3 # Epsilon for numerical stability and endpoint in diffusion process
        self.time_sampling = time_sampling
        # Condition-mask sampler. None keeps the historical Bernoulli(0.33)
        # draw; a ConditionalMenu concentrates the draws on the conditionals the
        # samplers actually evaluate.
        self.mask_sampler = mask_sampler
        self.validation_mask_seed = validation_mask_seed
        if (global_target_fn is None) != (global_target_indices is None):
            raise ValueError("global_target_fn and global_target_indices go together.")
        self.global_target_fn = global_target_fn
        self.global_target_indices = (None if global_target_indices is None
                                      else torch.as_tensor(global_target_indices, dtype=torch.long))
        self.tsm_lambda_c = float(tsm_lambda_c)
        self.tsm_variance_fn = tsm_variance_fn
        if time_loss_weight not in (None, "edm"):
            raise ValueError(f"time_loss_weight must be None or 'edm', got {time_loss_weight!r}")
        # Optional noise-level weighting of the per-example epsilon loss. "edm"
        # multiplies by (1 + sigma^2 / sigma_data^2), i.e. the EDM weighting written
        # for an epsilon target: the epsilon loss weights denoiser errors by 1/sigma^2,
        # which leaves high-noise accuracy (where the PF-ODE log-density is most
        # sensitive) nearly untrained. Normalised per batch like the mask weights.
        self.time_loss_weight = time_loss_weight
        self.sigma_data = float(sigma_data)
        self.mask_loss_weights = []
        for mask, weight in (mask_loss_weights or []):
            mask = torch.as_tensor(mask, dtype=torch.float32)
            if mask.shape != (self.SBIm.nodes_size,):
                raise ValueError(
                    "Each mask_loss_weights mask must have shape "
                    f"({self.SBIm.nodes_size},); got {tuple(mask.shape)}."
                )
            if float(weight) <= 0.0:
                raise ValueError("Mask loss weights must be positive.")
            self.mask_loss_weights.append((mask, float(weight)))

        if self.world_size > 1:
            mp.spawn(self._train_loop, args=(train_data, val_data), nprocs=self.world_size)
        else:
            rank = 0
            self.device = device
            self._train_loop(rank, train_data, val_data)

        end_time = time.time()
        training_time = (end_time-start_time) / 60
        if self.verbose:
            print(f"Training took {training_time:.1f} minutes")

    def _train_loop(self, rank, train_data, val_data):

        # Set device distribution
        if self.world_size > 1:
            self._ddp_setup(rank, self.world_size)
        else:
            self.model = self.SBIm.model.to(self.device)

        self.verbose = self.verbose if rank == 0 else False

        # Check data structure
        data_loader = self._prepare_data(train_data, batch_size=self.batch_size, rank=rank)
        if val_data is not None:
            # Validation draws its masks once, from a dedicated seed, so every
            # epoch scores the model on the *same* task. Redrawing them each
            # epoch turns validation into a moving target: for a ConditionalMenu
            # the six conditionals differ ~50x in magnitude, so the epoch-to-
            # epoch swing swamps real progress and both early stopping and
            # best-checkpoint selection end up firing on a lucky draw.
            val_loader = self._prepare_data(
                val_data, batch_size=self.batch_size, rank=rank,
                fixed_masks=self._fixed_validation_masks(val_data),
            )

        # Init tracking variables
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_loss = []
        self.val_loss = []

        # Set optimizer
        optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.lr)

        for epoch in range(self.max_epochs):
            # Train
            if self.world_size > 1: dist.barrier()

            train_loss_process = self._run_epoch(epoch, data_loader, optimizer, is_train=True)
            
            if self.world_size > 1:
                # Wait for all processes to finish training
                dist.barrier()

                # Gather losses from all processes
                epoch_train_loss = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(epoch_train_loss, torch.tensor(train_loss_process).to(self.device))
                train_loss_all = torch.mean(torch.stack(epoch_train_loss)).item()
            else:
                train_loss_all = train_loss_process

            self.train_loss.append(train_loss_all)

            # Early stopping on training loss if no validation data is provided
            if val_data is None and train_loss_all < best_val_loss:
                best_val_loss = train_loss_all
                patience_counter = 0
                if rank == 0:
                    self._save_checkpoint(name=self.name_checkpoint)
            elif val_data is None and train_loss_all >= best_val_loss:
                patience_counter += 1


            # Validate
            if val_data is not None:
                val_loss_process = self._run_epoch(epoch, val_loader, optimizer, is_train=False)

                if self.world_size > 1:
                    # Wait for all processes to finish validation
                    dist.barrier()

                    # Gather losses from all processes
                    epoch_val_loss = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
                    dist.all_gather(epoch_val_loss, torch.tensor(val_loss_process).to(self.device))
                    val_loss_all = torch.mean(torch.stack(epoch_val_loss)).item()
                else:
                    val_loss_all = val_loss_process

                self.val_loss.append(val_loss_all)

                # Early stopping
                if val_loss_all < best_val_loss:
                    best_val_loss = val_loss_all
                    patience_counter = 0
                    if rank == 0:
                        self._save_checkpoint(name=self.name_checkpoint)
                elif val_loss_all >= best_val_loss:
                    patience_counter += 1

            # Wait for all processes to finish validation
            if self.world_size > 1: dist.barrier()

            if self.verbose:
                if val_data is not None:
                    print(f'--- Epoch: {epoch+1:3d} --- Training Loss: {train_loss_all:8.3f} --- Validation Loss: {val_loss_all:8.3f} ---')
                else:
                    print(f'--- Epoch: {epoch+1:3d} --- Training Loss: {train_loss_all:8.3f} ---')
                print()
                time.sleep(0.2)

            if patience_counter == self.early_stopping_patience:
                    break
            
            
        # Schedule-free AdamW keeps its evaluation weights (the averaged sequence)
        # behind optimizer.eval(). With validation data the last pass already was an
        # eval pass; without it the model would otherwise end on the raw training
        # weights, both in memory and in the saved final model.
        optimizer.eval()
        self.model.eval()

        if self.world_size > 1:
            dist.barrier()
            if rank == 0:
                self._save_checkpoint(name=self.name)
            dist.destroy_process_group()
        elif getattr(self, "path", None) is not None:
            # Single GPU/CPU: write the final-epoch model as ``{name}.pt`` too, as the
            # multi-GPU branch does. ModelTransfuser.train_models reloads exactly that
            # file, and it used not to exist here. The final model is also the better
            # one: the best-validation checkpoint had 30-70% larger log-likelihood
            # error in every family and budget (validation/c9_network_seeds, E-C9.2).
            self._save_checkpoint(name=self.name)
        
    def _run_epoch(self, epoch, data_loader, optimizer, is_train):
        if self.world_size > 1:
            data_loader.sampler.set_epoch(epoch)

        if is_train:
            self.model.train()
            optimizer.train()
        else:
            self.model.eval()
            optimizer.eval()

        total_loss = 0
        batch_count = 0

        show_progress = self.verbose if is_train else False
        for batch in tqdm.tqdm(data_loader, disable=not show_progress):
            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                loss = self._run_batch(batch)
            total_loss += loss.item() / batch[0].shape[0]
            batch_count += 1

            if is_train:
                loss.backward()
                if self.world_size > 1: dist.barrier()
                optimizer.step()

        return total_loss / batch_count

    def _run_batch(self, batch):
        # Get data
        data, condition_mask, idx = self._prepare_batch(batch, self.device)

        # Get timesteps
        timesteps = self._sample_timesteps(data.shape[0])

        # Sample x_1 from noise distribution
        x_1 = torch.randn_like(data)*(1-condition_mask) + data*condition_mask
        # Calculate x at time t in diffusion process
        x_t = self.SBIm.forward_diffusion_sample(data, timesteps, x_1, condition_mask)
        # Get score
        score = self._get_score(x_t, timesteps, condition_mask)
        # Calculate loss (DSM target -x_1/sigma, optionally blended with TSM on the globals)
        target_noise = self._blend_global_target(data, timesteps, x_1, condition_mask, x_t)
        loss = self.loss_fn(score, timesteps, target_noise, condition_mask)

        return loss

    def _blend_global_target(self, data, timesteps, x_1, condition_mask, x_t=None):
        """The loss compares x_1 + sigma_t * score with 0, i.e. score with -x_1 / sigma_t.
        With a global target, x_1 on the latent globals is replaced by
        w * (-sigma_t * T / alpha_t) + (1 - w) * x_1, i.e. score with
        w T / alpha_t + (1 - w)(-x_1 / sigma_t). Conditioned globals are untouched (their
        loss term is masked anyway). x_t itself was built from the real x_1."""
        if getattr(self, "global_target_fn", None) is None:
            return x_1
        idx = self.global_target_indices.to(data.device)
        with torch.no_grad():
            target = self.global_target_fn(data).to(x_1.dtype)              # (B, k)
            sigma = self.sde.marginal_prob_std(timesteps).reshape(-1, 1).to(x_1.dtype)
            alpha = self.sde.alpha_t(timesteps).reshape(-1, 1).to(x_1.dtype)
            lam = sigma / alpha
            if getattr(self, "tsm_variance_fn", None) is not None:
                variance = self.tsm_variance_fn(x_t[:, idx], lam).to(x_1.dtype)
                w = 1.0 / (1.0 + lam ** 2 * variance)
            else:
                w = 1.0 / (1.0 + (lam / self.tsm_lambda_c) ** 2)
            blended = w * (-sigma * target / alpha) + (1.0 - w) * x_1[:, idx]
            latent = 1.0 - condition_mask[:, idx]
            out = x_1.clone()
            out[:, idx] = latent * blended + (1.0 - latent) * x_1[:, idx]
        return out

    def _fixed_validation_masks(self, val_data):
        """One mask per validation row, drawn once from a dedicated seed.

        The global RNG state is saved and restored, so adding this does not
        shift any other random draw in training.
        """
        sampler = getattr(self, "mask_sampler", None)
        if sampler is None:
            sampler = GroupedBernoulli(self.SBIm.schema, 0.33)
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(getattr(self, "validation_mask_seed", 20250718))
            return sampler.sample(
                (val_data.shape[0], self.SBIm.nodes_size)
            ).to(val_data.dtype)
        finally:
            torch.random.set_rng_state(state)

    def _sample_timesteps(self, batch_size):
        """Draw diffusion times according to the configured time_sampling scheme."""
        t_uniform = torch.rand(batch_size, 1, device=self.device) * (1. - self.eps) + self.eps
        if self.time_sampling == "uniform":
            return t_uniform

        # Log-uniform in the noise scale lambda(t) = sigma(t)/alpha(t)
        # (equals sigma_m(t) for the VESDE)
        lam_max = self.sde.lambda_t(torch.ones(1, device=self.device))
        lam_min = self.sde.lambda_t(torch.full((1,), self.eps, device=self.device))
        u = torch.rand(batch_size, 1, device=self.device)
        t_log_sigma = self.sde.time_of_lambda(lam_min * (lam_max / lam_min)**u)
        if self.time_sampling == "log_sigma":
            return t_log_sigma

        # 50/50 mixture of both
        pick = (torch.rand(batch_size, 1, device=self.device) < 0.5).float()
        return pick * t_uniform + (1 - pick) * t_log_sigma

    #############################################
    # ----- Loss Function -----
    #############################################
    def loss_fn(self, score, timestep, x_1, condition_mask):
        '''
        Loss function for the score prediction task

        Args:
            score: Predicted score
                    
        The target is the noise added to the data at a specific timestep 
        Meaning the prediction is the approximation of the noise added to the data
        '''
        raw_condition_mask = condition_mask.to(score.device)
        sigma_t = self.sde.marginal_prob_std(timestep).unsqueeze(1).to(score.device)
        x_1 = x_1.unsqueeze(2).to(score.device)
        condition_mask = condition_mask.unsqueeze(2).to(score.device)
        score = score.unsqueeze(2)

        # Per-example residual sum, shape (batch,).
        per_example = torch.sum(
            (1-condition_mask) * (x_1 + sigma_t*score)**2, dim=(1, 2),
        )

        # Optional per-mask reweighting, normalized to mean 1 so that all-ones
        # weights reproduce the unweighted loss exactly.
        weights = torch.ones_like(per_example)
        for mask, weight in getattr(self, "mask_loss_weights", []):
            match = torch.all(
                raw_condition_mask == mask.to(raw_condition_mask.device),
                dim=1,
            )
            weights = torch.where(match, weight, weights)
        if getattr(self, "time_loss_weight", None) == "edm":
            weights = weights * (1.0 + sigma_t.reshape(-1) ** 2 / self.sigma_data ** 2)
        weights = weights / weights.mean()

        # The sigma^2 factor is applied as the batch mean, not per example. That
        # is the form every existing checkpoint was trained under, so it is kept
        # deliberately rather than "corrected" to sum_j sigma_j^2 S_j: changing
        # it would silently redefine the objective and make new arms
        # incomparable with the frozen baselines.
        #
        # The result is a *sum* over the batch, which is what makes it
        # batch-size invariant after Trainer._run_epoch divides by the batch
        # size. Returning a batch mean here instead gives the final short batch
        # of an epoch (8 rows of a 5,000-row validation split) sixteen times the
        # weight of a full one, which makes the reported validation loss swing
        # by tens of percent and drives early stopping off a handful of rows.
        return torch.mean(sigma_t**2) * torch.sum(weights * per_example)
    
    #############################################
    # ----- Multi-GPU setup -----
    #############################################
    def _ddp_setup(self, rank, world_size):

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

        self.device = torch.device(f'cuda:{rank}')
        self.SBIm.model.to(self.device)
        self.model = DDP(self.SBIm.model, device_ids=[rank], output_device=rank)

    #############################################
    # ----- Standard Functions -----
    #############################################

    def _get_score(self, x, t, condition_mask):
        """Get score estimate from model"""
        # Get conditional score
        out = self.model(x=x, t=t, c=condition_mask)
        score = self.SBIm.output_scale_function(t, out)
                
        return score

    def _prepare_batch(self, batch, device):
        # Expand data and condition mask to match num_samples
        data, condition_mask, idx = batch
        data = data.to(device)
        condition_mask = condition_mask.to(device)

        return data, condition_mask, idx

    def _prepare_data(self, data, batch_size, rank, fixed_masks=None):
        # Grouped by the model's node schema: an IID item is observed or latent
        # as a whole, so a trial's features never split across the conditioning
        # boundary. The marginal rate of observed *scalars* stays 0.33, which is
        # what keeps a grouped model comparable with the legacy one; for an
        # all-atomic schema this is the same Bernoulli(0.33) draw as before.
        # A tensor of masks makes the conditioning a fixed property of each row;
        # a sampler redraws it every time the row is visited.
        condition_mask = fixed_masks
        if condition_mask is None:
            condition_mask = getattr(self, "mask_sampler", None)
        if condition_mask is None:
            condition_mask = GroupedBernoulli(self.SBIm.schema, 0.33)

        dataset = TensorTupleDataset(data, condition_mask)

        if self.world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=True)
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,
                sampler=sampler
            )
        else:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader

    def _save_checkpoint(self, name="Model_checkpoint"):
        # Save checkpoint
        if self.world_size > 1:
            # Save torch model in case of multi-GPU training
            state_dict = {
                'model_state_dict' : self.model.module.state_dict(),
                'nodes_size': self.SBIm.nodes_size,
                'sde_type': self.SBIm.sde_type,
                'sigma': self.SBIm.sigma,
                'beta_min': self.SBIm.beta_min,
                'beta_max': self.SBIm.beta_max,
                'hidden_size': self.SBIm.hidden_size,
                'depth': self.SBIm.depth,
                'num_heads': self.SBIm.num_heads,
                'mlp_ratio': self.SBIm.mlp_ratio,
                **self.SBIm.architecture_payload(),
                **self.SBIm.time_payload(),
            }
            torch.save(state_dict, f"{self.path}/{name}.pt")
        else:
            self.SBIm.save(path=self.path ,name=name)
