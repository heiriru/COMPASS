import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import schedulefree

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

        if isinstance(self.tensor2, torch.distributions.Distribution):
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
              path=None, name="Model", device="cpu", verbose=True, time_sampling="mixture",
              train_divergence=False, divergence_loss_weight=1.0,
              divergence_target="exact", hutchinson_samples=1,
              divergence_warmup_epochs=5):

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
            train_divergence: Jointly train the instantaneous log-density drift head.
            divergence_loss_weight: Final weight of the divergence regression loss.
            divergence_target: Trace estimator used for supervision ("exact" or
                    "hutchinson").
            hutchinson_samples: Probe vectors per batch for Hutchinson supervision.
            divergence_warmup_epochs: Epochs over which to linearly ramp the
                    divergence loss weight.
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
        self.train_divergence = bool(train_divergence)
        self.divergence_loss_weight = float(divergence_loss_weight)
        self.divergence_target = divergence_target
        self.hutchinson_samples = int(hutchinson_samples)
        self.divergence_warmup_epochs = int(divergence_warmup_epochs)
        if self.divergence_target not in ("exact", "hutchinson"):
            raise ValueError("divergence_target must be 'exact' or 'hutchinson'.")
        if self.hutchinson_samples < 1:
            raise ValueError("hutchinson_samples must be at least 1.")
        if self.divergence_warmup_epochs < 0:
            raise ValueError("divergence_warmup_epochs must be non-negative.")

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

        # The new head is completely opt-in. Freeze it for legacy score-only
        # training so optimizers and DDP see exactly the original trainable path.
        for parameter in self.SBIm.model.divergence_head.parameters():
            parameter.requires_grad_(self.train_divergence)

        # Set device distribution
        if self.world_size > 1:
            self._ddp_setup(rank, self.world_size)
        else:
            self.model = self.SBIm.model.to(self.device)

        self.verbose = self.verbose if rank == 0 else False

        # Check data structure
        data_loader = self._prepare_data(train_data, batch_size=self.batch_size, rank=rank)
        if val_data is not None:
            val_loader = self._prepare_data(val_data, batch_size=1_000, rank=rank)

        # Init tracking variables
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_loss = []
        self.val_loss = []
        self.train_score_loss = []
        self.train_divergence_loss = []
        self.val_score_loss = []
        self.val_divergence_loss = []

        # Set optimizer
        optimizer = schedulefree.AdamWScheduleFree(
            (parameter for parameter in self.model.parameters()
             if parameter.requires_grad),
            lr=self.lr)

        for epoch in range(self.max_epochs):
            # Train
            if self.world_size > 1: dist.barrier()

            train_result_process = self._run_epoch(
                epoch, data_loader, optimizer, is_train=True,
                return_components=self.train_divergence)
            
            if self.world_size > 1:
                # Wait for all processes to finish training
                dist.barrier()

                if self.train_divergence:
                    train_metrics_all = []
                    for metric in train_result_process:
                        gathered = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
                        dist.all_gather(gathered, torch.tensor(metric).to(self.device))
                        train_metrics_all.append(torch.mean(torch.stack(gathered)).item())
                else:
                    # Preserve the legacy single-loss collective exactly.
                    gathered = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
                    dist.all_gather(
                        gathered, torch.tensor(train_result_process).to(self.device))
                    train_loss = torch.mean(torch.stack(gathered)).item()
                    train_metrics_all = (train_loss, train_loss, 0.0)
            else:
                train_metrics_all = (train_result_process
                                     if self.train_divergence
                                     else (train_result_process,
                                           train_result_process, 0.0))

            train_loss_all, train_score_loss_all, train_divergence_loss_all = train_metrics_all

            self.train_loss.append(train_loss_all)
            self.train_score_loss.append(train_score_loss_all)
            self.train_divergence_loss.append(train_divergence_loss_all)

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
                val_result_process = self._run_epoch(
                    epoch, val_loader, optimizer, is_train=False,
                    return_components=self.train_divergence)

                if self.world_size > 1:
                    # Wait for all processes to finish validation
                    dist.barrier()

                    if self.train_divergence:
                        val_metrics_all = []
                        for metric in val_result_process:
                            gathered = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
                            dist.all_gather(gathered, torch.tensor(metric).to(self.device))
                            val_metrics_all.append(torch.mean(torch.stack(gathered)).item())
                    else:
                        gathered = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
                        dist.all_gather(
                            gathered, torch.tensor(val_result_process).to(self.device))
                        val_loss = torch.mean(torch.stack(gathered)).item()
                        val_metrics_all = (val_loss, val_loss, 0.0)
                else:
                    val_metrics_all = (val_result_process
                                       if self.train_divergence
                                       else (val_result_process,
                                             val_result_process, 0.0))

                val_loss_all, val_score_loss_all, val_divergence_loss_all = val_metrics_all

                self.val_loss.append(val_loss_all)
                self.val_score_loss.append(val_score_loss_all)
                self.val_divergence_loss.append(val_divergence_loss_all)

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
                if self.train_divergence:
                    print(f'    Score: {train_score_loss_all:8.3f} --- Divergence: {train_divergence_loss_all:8.3f}')
                print()
                time.sleep(0.2)

            if patience_counter == self.early_stopping_patience:
                    break
            
            
        if self.world_size > 1:
            dist.barrier()
            if rank == 0:
                self._save_checkpoint(name=self.name)
            dist.destroy_process_group()
        
    def _run_epoch(self, epoch, data_loader, optimizer, is_train,
                   return_components=False):
        if self.world_size > 1:
            data_loader.sampler.set_epoch(epoch)

        if is_train:
            self.model.train()
            optimizer.train()
        else:
            self.model.eval()
            optimizer.eval()

        total_loss = 0
        total_score_loss = 0
        total_divergence_loss = 0
        batch_count = 0

        show_progress = self.verbose if is_train else False
        for batch in tqdm.tqdm(data_loader, disable=not show_progress):
            if is_train:
                optimizer.zero_grad()

            if return_components:
                loss, score_loss, divergence_loss = self._run_batch(
                    batch, epoch=epoch, return_components=True)
            else:
                loss = self._run_batch(batch, epoch=epoch)
                score_loss = loss
                divergence_loss = loss.new_zeros(())
            total_loss += loss.item() / batch[0].shape[0]
            if return_components:
                total_score_loss += score_loss.item() / batch[0].shape[0]
                total_divergence_loss += divergence_loss.item() / batch[0].shape[0]
            batch_count += 1

            if is_train:
                loss.backward()
                if self.world_size > 1: dist.barrier()
                optimizer.step()
                if self.train_divergence:
                    self.SBIm.divergence_head_trained = True

        if not return_components:
            return total_loss / batch_count
        return (total_loss / batch_count,
                total_score_loss / batch_count,
                total_divergence_loss / batch_count)

    def _run_batch(self, batch, epoch=0, return_components=False):
        # Get data
        data, condition_mask, idx = self._prepare_batch(batch, self.device)

        # Get timesteps
        timesteps = self._sample_timesteps(data.shape[0])

        # Sample x_1 from noise distribution
        x_1 = torch.randn_like(data)*(1-condition_mask) + data*condition_mask
        # Calculate x at time t in diffusion process
        x_t = self.SBIm.forward_diffusion_sample(data, timesteps, x_1, condition_mask)
        if self.train_divergence:
            x_t = x_t.detach().requires_grad_(True)
            raw_score, divergence = self.model(
                x=x_t, t=timesteps, c=condition_mask,
                return_divergence=True)
            score = self.SBIm.output_scale_function(timesteps, raw_score)
            divergence_target = self.instantaneous_divergence_target(
                raw_score, x_t, timesteps, condition_mask,
                estimator=self.divergence_target,
                hutchinson_samples=self.hutchinson_samples)
            divergence_loss = self.divergence_loss_fn(
                divergence, divergence_target, condition_mask)
        else:
            score = self._get_score(x_t, timesteps, condition_mask)
            divergence_loss = score.new_zeros(())

        score_loss = self.loss_fn(score, timesteps, x_1, condition_mask)
        loss = score_loss + self._divergence_weight(epoch) * divergence_loss

        components = (loss, score_loss, divergence_loss)
        return components if return_components else loss

    def _divergence_weight(self, epoch):
        if not self.train_divergence:
            return 0.0
        if self.divergence_warmup_epochs == 0:
            return self.divergence_loss_weight
        warmup = min((epoch + 1) / self.divergence_warmup_epochs, 1.0)
        return self.divergence_loss_weight * warmup

    def instantaneous_divergence_target(self, raw_score, x, timestep,
                                        condition_mask, estimator="exact",
                                        hutchinson_samples=1):
        """Return D_lambda = alpha * trace(d raw_score / dx) on latent nodes.

        The returned target is detached: divergence regression therefore does
        not introduce second derivatives or update the score through its target.
        """
        latent = 1 - condition_mask
        trace = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        if estimator == "exact":
            latent_dims = torch.nonzero(latent.any(dim=0)).flatten().tolist()
            for j in latent_dims:
                grad_j = torch.autograd.grad(
                    raw_score[:, j].sum(), x, retain_graph=True,
                    create_graph=False)[0]
                trace = trace + grad_j[:, j] * latent[:, j]
        elif estimator == "hutchinson":
            for _ in range(int(hutchinson_samples)):
                probe = (torch.randint(0, 2, x.shape, device=x.device,
                                       dtype=torch.int64).to(x.dtype) * 2 - 1) * latent
                grad_v = torch.autograd.grad(
                    (raw_score * probe).sum(), x, retain_graph=True,
                    create_graph=False)[0]
                trace = trace + (grad_v * probe).sum(dim=-1)
            trace = trace / int(hutchinson_samples)
        else:
            raise ValueError(f"Unknown divergence target estimator '{estimator}'.")

        alpha = self.sde.alpha_t(timestep).to(x.device).reshape(-1)
        return (alpha * trace).detach()

    @staticmethod
    def divergence_loss_fn(prediction, target, condition_mask):
        """Dimension-normalized MSE for scalar latent divergence targets."""
        latent_count = (1 - condition_mask).sum(dim=-1)
        valid = latent_count > 0
        if not valid.any():
            return prediction.sum() * 0
        residual = (prediction[valid] - target[valid]) / latent_count[valid]
        return torch.mean(residual**2)

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
        sigma_t = self.sde.marginal_prob_std(timestep).unsqueeze(1).to(score.device)
        x_1 = x_1.unsqueeze(2).to(score.device)
        condition_mask = condition_mask.unsqueeze(2).to(score.device)
        score = score.unsqueeze(2)

        loss = torch.mean(sigma_t**2 * torch.sum((1-condition_mask)*(x_1+sigma_t*score)**2))

        return loss
    
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

    def _prepare_data(self, data, batch_size, rank):
        condition_mask = torch.distributions.bernoulli.Bernoulli(0.33)

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
                'divergence_head_trained': self.SBIm.divergence_head_trained,
            }
            torch.save(state_dict, f"{self.path}/{name}.pt")
        else:
            self.SBIm.save(path=self.path ,name=name)
