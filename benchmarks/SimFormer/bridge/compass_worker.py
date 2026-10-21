"""COMPASS side of the bridge: a persistent worker process in the ``.COMPASS`` env.

The Simformer harness (``scoresbibm``, JAX, ``.simformer`` env) starts this
script once per run and talks to it over stdin/stdout with length-prefixed
pickles of plain numpy arrays and dicts. Nothing about COMPASS is modified: the
worker only calls the public ``ScoreBasedInferenceModel`` API.

Commands
--------
``train``  -- fit one model on the harness's own training simulations.
``sample`` -- draw from an arbitrary conditional; returns only the latent
              coordinates, in the order they appear in the node vector, which is
              the contract ``scoresbibm`` models follow.
``quit``   -- exit.

All human-readable output goes to stderr; stdout carries the protocol only.
"""

import math
import os
import pickle
import struct
import sys
import time

CPU_THREAD_LIMIT = 3


def _cap_cpu(max_threads=CPU_THREAD_LIMIT):
    """Repository rule (COMPASS/CLAUDE.md): at most 3 CPUs, set before numeric imports."""
    available = sorted(os.sched_getaffinity(0))
    permitted = min(max_threads, len(available))
    start = (os.getpid() * permitted) % len(available)
    os.sched_setaffinity(0, {available[(start + i) % len(available)] for i in range(permitted)})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(permitted)


_cap_cpu()

# COMPASS_SRC: import `compass` from another checkout instead of the installed
# one (the editable install is a .pth path entry, so sys.path[0] wins).
if os.environ.get("COMPASS_SRC"):
    sys.path.insert(0, os.environ["COMPASS_SRC"])

import numpy as np  # noqa: E402
import torch  # noqa: E402

import compass  # noqa: E402
from compass import ScoreBasedInferenceModel as SBIm  # noqa: E402

# The upstream package has no schemas, no arch=, no mask_sampler=, no churn.
UPSTREAM = not hasattr(compass, "Atomic")

PROTO_OUT = sys.stdout.buffer
sys.stdout = sys.stderr          # anything COMPASS prints must not corrupt the pipe


def log(*args):
    print("[compass-worker]", *args, file=sys.stderr, flush=True)


def read_message(stream=sys.stdin.buffer):
    header = stream.read(8)
    if len(header) < 8:
        return None
    (size,) = struct.unpack("<Q", header)
    return pickle.loads(stream.read(size))


def write_message(obj, stream=PROTO_OUT):
    blob = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<Q", len(blob)))
    stream.write(blob)
    stream.flush()


class SimformerMaskLaw:
    """The Simformer training condition-mask law, exactly as their code draws it.

    ``scoresbibm.utils.condition_masks.sample_strutured_conditional_mask``: every
    *row* picks one of {joint, posterior, likelihood, rnd(0.3), rnd(0.7)} with
    probability 0.2 each; the two Bernoulli patterns are drawn *once per call*
    (i.e. once per training batch) and shared by all rows that pick them; an
    all-ones mask is replaced by all-zeros. COMPASS's trainer asks for the
    validation masks in a single call, so the shared draws are refreshed every
    ``block`` rows to keep validation from collapsing onto one random pattern.
    """

    def __init__(self, theta_dim, x_dim, block=1000, probs=(0.2,) * 5, rates=(0.3, 0.7)):
        self.theta_dim, self.x_dim = int(theta_dim), int(x_dim)
        self.block = int(block)
        self.probs = torch.tensor(probs, dtype=torch.float)
        self.rates = rates

    def _draw(self, n, d):
        fixed = torch.zeros(5, d)
        fixed[1, self.theta_dim:] = 1.0                 # posterior
        fixed[2, : self.theta_dim] = 1.0                # likelihood
        fixed[3] = (torch.rand(d) < self.rates[0]).float()
        fixed[4] = (torch.rand(d) < self.rates[1]).float()
        pick = torch.multinomial(self.probs, n, replacement=True)
        mask = fixed[pick]
        all_ones = mask.bool().all(-1)
        mask[all_ones] = 0.0
        return mask

    def sample(self, shape):
        shape = torch.Size(shape)
        n, d = int(np.prod(shape[:-1])), shape[-1]
        parts = [self._draw(min(self.block, n - s), d) for s in range(0, n, self.block)]
        return torch.cat(parts).reshape(shape)


class BatchLoader:
    """Drop-in for the Trainer's DataLoader: whole batches, on the device.

    compass.Trainer wraps the data in a TensorTupleDataset whose
    __getitem__ calls the mask sampler once per *row*, and the DataLoader
    then collates 1000 Python items per batch. At batch 1000 that per-row work,
    not the network, bounded the step rate (GPU at ~12%). This loader draws the
    masks for a whole batch in one sample((B, D)) call -- which is also what
    makes the Simformer law's shared per-batch Bernoulli patterns per-batch --
    and keeps the data resident on the GPU. Model, loss and optimiser are
    untouched.
    """

    def __init__(self, data, masks, batch_size, device, shuffle=True):
        self.data = data.to(device)
        self.masks = masks.to(device) if torch.is_tensor(masks) else masks
        self.batch_size, self.device, self.shuffle = int(batch_size), device, shuffle

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __iter__(self):
        n, d = self.data.shape
        order = torch.randperm(n, device=self.device) if self.shuffle else torch.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            if torch.is_tensor(self.masks):
                mask = self.masks[idx]
            else:
                mask = self.masks.sample((len(idx), d)).to(self.device)
            yield self.data[idx], mask, idx


TRAIN_EPS = None   # set per run from cfg["train_eps"]


def _install_batch_loader():
    import importlib
    trainer_module = importlib.import_module("compass.Trainer")   # the module, not the class

    def _prepare_data(self, data, batch_size, rank, fixed_masks=None):
        # Trainer.train hard-codes self.eps = 1e-3 just before building the
        # loaders; this is the first hook after that, so an override lands here.
        if TRAIN_EPS is not None:
            self.eps = TRAIN_EPS
        if fixed_masks is not None:
            masks = fixed_masks
        else:
            # Upstream has no mask_sampler: it draws Bernoulli(0.33) per element
            # (Trainer._prepare_data); same law, drawn per batch here.
            masks = getattr(self, "mask_sampler", None) or torch.distributions.Bernoulli(0.33)
        return BatchLoader(data, masks, batch_size, self.device, shuffle=fixed_masks is None)

    trainer_module.Trainer._prepare_data = _prepare_data


class PosteriorOnlyLaw:
    def __init__(self, theta_dim, x_dim):
        self.mask = torch.zeros(theta_dim + x_dim)
        self.mask[theta_dim:] = 1.0

    def sample(self, shape):
        return self.mask.expand(torch.Size(shape)).clone()


def six_mixed_flat_law(schema, theta_dim, x_dim, bernoulli_fraction=0.25):
    """COMPASS's ``mask6mix`` (uniform SIX_CONDITIONALS + 25% Bernoulli(0.33)) on a flat task.

    A flat task has no subject level: theta is the global role ``g`` and ``l`` is
    empty (as in comparisons/simformer/common/mask_allocation.BENCHMARKS). Dropping
    ``l`` collapses the six to p(g|x) twice and p(x|g) twice; p(l|g,x) conditions
    everything and p(x,g|l) nothing, and ConditionalMenu refuses both (no loss /
    "no sampler evaluates the joint"), so they are left out exactly as the library
    would. Net: 37.5% posterior, 37.5% likelihood, 25% Bernoulli(0.33).
    """
    roles = {"g": list(range(theta_dim)), "x": list(range(theta_dim, theta_dim + x_dim))}
    patterns = []
    for name, cond in compass.schema.SIX_CONDITIONALS:
        kept = [r for r in cond if r != "l"]
        n_cond = sum(len(roles[r]) for r in kept)
        if 0 < n_cond < theta_dim + x_dim:
            patterns.append((name, kept))
    return compass.schema.ConditionalMenu(schema, roles, patterns,
                                          bernoulli_fraction=bernoulli_fraction,
                                          bernoulli_probability=0.33)


class Worker:
    def __init__(self):
        self.model = None
        self.cfg = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log("device", self.device, "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"),
            "compass", compass.__file__, "upstream" if UPSTREAM else "")

    # ---------------------------------------------------------------- train
    def train(self, msg):
        cfg = dict(msg["cfg"])
        self.cfg = cfg
        theta = np.asarray(msg["theta"], dtype=np.float32)
        x = np.asarray(msg["x"], dtype=np.float32)
        theta_dim, x_dim = theta.shape[1], x.shape[1]
        self.theta_dim, self.x_dim = theta_dim, x_dim
        seed = int(msg.get("seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)

        if cfg.get("time_cond", "per_node") == "shared":
            # Must precede both construction and SBIm.load so shapes match.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import shared_time_transformer
            shared_time_transformer.install(int(cfg.get("time_embedding_size", 256)))
            log("legacy trunk with shared time conditioning, time_embedding_size",
                shared_time_transformer.SharedTimeConditionTransformer.TIME_EMBEDDING_SIZE)

        if cfg.get("arch_variant") == "cond_embed":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import cond_embed_transformer
            cond_embed_transformer.install()
            log("legacy trunk + node-ID and condition-state token embeddings")

        if cfg.get("arch_variant") == "lowrank_time":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import lowrank_time_transformer
            lowrank_time_transformer.install(int(cfg.get("time_rank", 8)), int(cfg.get("time_embedding_size", 256)))
            log("legacy trunk with shared + rank", cfg.get("time_rank", 8), "per-node time conditioning")
        if cfg.get("arch_variant") == "no_time_mod":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import notime_transformer
            notime_transformer.install(cfg.get("time_input", "input"), int(cfg.get("time_embedding_size", 256)))
            log("legacy trunk without adaLN time modulation, time input:", cfg.get("time_input", "input"))
        if cfg.get("arch_variant") == "full_attn":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import fullattn_transformer
            fullattn_transformer.install(cfg.get("full_attn_base", "legacy"), int(cfg.get("time_embedding_size", 256)))
            log("unmasked attention over all tokens + condition-state embedding, base:",
                cfg.get("full_attn_base", "legacy"))
        if cfg.get("input_scaling"):
            # Wraps the trunk chosen above (legacy or a variant), so it must come last.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import input_scaling
            input_scaling.install(float(cfg.get("sigma", 25.0)))
            log("EDM input scaling: latent inputs x c_in = 1/sqrt(1+sigma(t)^2)")
        if cfg.get("weight_decay"):
            # Trainer builds schedulefree.AdamWScheduleFree(params, lr=lr); give it weight decay.
            import functools
            import importlib
            import types
            import schedulefree
            tm = importlib.import_module("compass.Trainer")
            tm.schedulefree = types.SimpleNamespace(AdamWScheduleFree=functools.partial(
                schedulefree.AdamWScheduleFree, weight_decay=float(cfg["weight_decay"])))
            log("AdamWScheduleFree weight_decay", cfg["weight_decay"])

        if cfg.get("load_from"):
            # Sampler ablation: reuse a trained checkpoint, skip training.
            src = cfg["load_from"]
            z = np.load(os.path.join(src, "zscore.npz"))
            self.mean, self.std = z["mean"], z["std"]
            self.model = SBIm.load(os.path.join(src, "Model_checkpoint.pt"), device=self.device)
            self.model.model.eval()
            log("loaded", src)
            return {"loaded_from": src}

        data = np.concatenate([theta, x], axis=1)
        # Split first, then fit the z-score on the training part only.
        perm = np.random.permutation(len(data))
        # validation_fraction 0: train on every simulation; the Trainer then
        # checkpoints on the training loss (i.e. close to the final weights).
        n_val = int(round(cfg.get("validation_fraction", 0.1) * len(data)))
        val, train = data[perm[:n_val]], data[perm[n_val:]]
        if cfg.get("z_score", True):
            self.mean = train.mean(0)
            self.std = np.clip(train.std(0), 1e-3, None)
        else:
            self.mean = np.zeros(data.shape[1], np.float32)
            self.std = np.ones(data.shape[1], np.float32)
        z = lambda a: torch.tensor((a - self.mean) / self.std, dtype=torch.float32)
        if cfg.get("checkpoint") == "final":
            # Simformer keeps its final weights. Train on every simulation; a 5%
            # subset *of the training data* is still passed as "validation" only
            # so the schedule-free optimizer ends in eval mode (averaged weights).
            train = data
            val = data[perm[:max(1, int(0.05 * len(data)))]]
            n_val = len(val)
        train_t, val_t = z(train), z(val)

        global TRAIN_EPS
        TRAIN_EPS = float(cfg["train_eps"]) if cfg.get("train_eps") is not None else None
        log("train_eps", TRAIN_EPS)

        arch = cfg.get("arch", "legacy")
        common = dict(
            sde_type=cfg.get("sde", "vesde"), sigma=float(cfg.get("sigma", 25.0)),
            hidden_size=int(cfg.get("hidden_size", 64)), depth=int(cfg.get("depth", 6)),
            num_heads=int(cfg.get("num_heads", 4)), mlp_ratio=cfg.get("mlp_ratio", 3),
            device=self.device,
        )
        if UPSTREAM:
            model = SBIm(nodes_size=theta_dim + x_dim, **common)
        elif arch == "legacy":
            model = SBIm(nodes_size=theta_dim + x_dim, arch="legacy", **common)
        else:
            model = SBIm(theta_size=theta_dim, x_spec=compass.Atomic("x", x_dim), arch="tokenized",
                         time_embedding_size=int(cfg.get("time_embedding_size", 128)),
                         token_offset=bool(cfg.get("token_offset", True)), **common)
        n_params = int(sum(p.numel() for p in model.parameters()))

        law = cfg.get("mask_law", "simformer")
        if law == "simformer":
            sampler = SimformerMaskLaw(theta_dim, x_dim, block=int(cfg.get("batch_size", 1000)))
        elif law == "six_mixed":
            sampler = six_mixed_flat_law(model.schema, theta_dim, x_dim,
                                         float(cfg.get("mask_bernoulli_fraction", 0.25)))
            log("mask law six_mixed:", sampler.describe().replace("\n", "; "))
        else:
            sampler = PosteriorOnlyLaw(theta_dim, x_dim)

        # Same gradient-step budget as the Simformer run: their
        # total_number_steps = clip(3 * N, 5k, 100k) at batch 1000.
        batch = int(cfg.get("batch_size", 1000))
        steps_per_epoch = max(1, math.ceil(len(train_t) / batch))
        max_steps = int(cfg["max_steps"])
        max_epochs = max(1, math.ceil(max_steps / steps_per_epoch))
        patience = max(int(cfg.get("min_patience_epochs", 20)),
                       math.ceil(int(cfg.get("patience_steps", 5000)) / steps_per_epoch))

        out = msg["out_dir"]
        os.makedirs(out, exist_ok=True)
        if cfg.get("optimizer") == "simformer":
            # Adam, constant lr then linear decay to 1e-6 over the second half of
            # the steps actually run, adaptive grad clip 10 (bridge/simformer_optim.py).
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import simformer_optim
            simformer_optim.install(max_epochs * steps_per_epoch, float(cfg.get("min_lr", 1e-6)),
                                    float(cfg.get("clip", 10.0)))
            log("Simformer optimizer: Adam, linear decay to", cfg.get("min_lr", 1e-6),
                "over the second half of", max_epochs * steps_per_epoch, "steps, AGC", cfg.get("clip", 10.0))
        log(f"train arch={arch} params={n_params} rows={len(train_t)} val={len(val_t)} "
            f"steps/epoch={steps_per_epoch} max_epochs={max_epochs} patience={patience}")
        started = time.time()
        model.train(
            train_t[:, :theta_dim], train_t[:, theta_dim:],
            theta_val=val_t[:, :theta_dim] if n_val else None,
            x_val=val_t[:, theta_dim:] if n_val else None,
            batch_size=batch, max_epochs=max_epochs, lr=float(cfg.get("lr", 1e-3)),
            device=self.device, verbose=False, path=out, name="Model",
            early_stopping_patience=patience,
            time_sampling=cfg.get("time_sampling", "log_sigma"),
            **({} if UPSTREAM else {"mask_sampler": sampler}),
        )
        seconds = time.time() - started
        epochs = len(model.trainer.train_loss)
        best = os.path.join(out, "Model_checkpoint.pt")
        if cfg.get("checkpoint") == "final":
            model.save(out, "Model_checkpoint")        # overwrite: resampling loads the final weights
        self.model = SBIm.load(best, device=self.device)
        self.model.model.eval()
        np.savez(os.path.join(out, "zscore.npz"), mean=self.mean, std=self.std)
        info = {"n_params": n_params, "train_seconds": seconds, "epochs": epochs,
                "steps": epochs * steps_per_epoch, "best_val": float(min(model.trainer.val_loss)) if model.trainer.val_loss else None,
                "max_epochs": max_epochs, "patience": patience}
        log("trained", info)
        return info

    # --------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(self, msg):
        cfg = self.cfg
        mask = np.asarray(msg["condition_mask"]).astype(bool).reshape(-1)
        n = int(msg["num_samples"])
        seed = int(msg.get("seed", 0))
        torch.manual_seed(seed)
        D = mask.size
        x_o = np.asarray(msg["x_o"], dtype=np.float32).reshape(-1)
        if x_o.size != mask.sum():
            raise ValueError(f"x_o has {x_o.size} values for {mask.sum()} conditioned nodes")
        cond = ((x_o - self.mean[mask]) / self.std[mask]).reshape(1, -1)
        # COMPASS wants a 1-D mask and the conditioned values only; it returns
        # (1, num_samples, n_latent) -- the latent coordinates, in node order.
        cmask = torch.tensor(mask, dtype=torch.float32)
        kwargs = dict(method=cfg.get("sampler", "dpm"), timesteps=int(cfg.get("timesteps", 50)),
                      eps=float(cfg.get("eps", 1e-3)))
        if kwargs["method"] == "dpm":
            kwargs["order"] = int(cfg.get("order", 2))
            if not UPSTREAM:      # upstream: DPM + Langevin correctors at its defaults
                kwargs["churn_eta"] = float(cfg.get("churn_eta", 2.0))
                kwargs.update(_corrector_kwargs(cfg))
        if cfg.get("forward_eps", False) and not getattr(self, "_eps_forwarded", False):
            # ScoreBasedInferenceModel.sample accepts eps but never passes it on
            # to Sampler.sample (ScoreBasedInferenceModel.py:543), so every
            # sample stops at the Sampler's default eps=1e-3 with sigma_z~0.03 of
            # noise still on it. Forward it explicitly.
            import functools
            self.model.sampler.sample = functools.partial(self.model.sampler.sample, eps=kwargs["eps"])
            self._eps_forwarded = True
            log("forwarding eps", kwargs["eps"], "to Sampler.sample")
        chunk = int(cfg.get("sample_chunk", 10000))
        outs, done = [], 0
        while done < n:
            m = min(chunk, n - done)
            s = self.model.sample(
                x=torch.tensor(cond) if mask.any() else None, condition_mask=cmask,
                num_samples=m, device=self.device, verbose=False, **kwargs)
            s = torch.as_tensor(s).detach().float().cpu().numpy().reshape(m, -1)
            outs.append(s)
            done += m
        latent = np.concatenate(outs)[:n] * self.std[~mask] + self.mean[~mask]
        bad = ~np.isfinite(latent).all(1)
        return {"samples": latent.astype(np.float32), "n_nonfinite": int(bad.sum())}


def _corrector_kwargs(cfg):
    """Langevin corrector settings (churn_eta 0 + correctors = predictor-corrector SDE sampler)."""
    out = {k: int(cfg[k]) for k in ("corrector_steps", "corrector_steps_interval", "final_corrector_steps")
           if cfg.get(k) is not None}
    if cfg.get("snr") is not None:
        out["snr"] = float(cfg["snr"])
    return out


def _sample_kwargs(worker):
    cfg = worker.cfg
    kwargs = dict(method=cfg.get("sampler", "dpm"), timesteps=int(cfg.get("timesteps", 50)),
                  eps=float(cfg.get("eps", 1e-3)))
    if kwargs["method"] == "dpm":
        kwargs["order"] = int(cfg.get("order", 2))
        if not UPSTREAM:
            kwargs["churn_eta"] = float(cfg.get("churn_eta", 2.0))
            kwargs.update(_corrector_kwargs(cfg))
    if cfg.get("forward_eps", False) and not getattr(worker, "_eps_forwarded", False):
        import functools
        worker.model.sampler.sample = functools.partial(worker.model.sampler.sample, eps=kwargs["eps"])
        worker._eps_forwarded = True
        log("forwarding eps", kwargs["eps"], "to Sampler.sample")
    return kwargs


@torch.no_grad()
def sample_batched(worker, msg):
    """num_samples draws for each of B conditioning rows -> (B, num_samples, n_latent)."""
    mask = np.asarray(msg["condition_mask"]).astype(bool).reshape(-1)
    x_o = np.asarray(msg["x_o"], dtype=np.float32).reshape(-1, int(mask.sum()))
    n = int(msg["num_samples"])
    torch.manual_seed(int(msg.get("seed", 0)))
    cond = (x_o - worker.mean[mask]) / worker.std[mask]
    cmask = torch.tensor(mask, dtype=torch.float32)
    rows = max(1, int(worker.cfg.get("sample_chunk", 10000)) // n)
    outs = []
    for s in range(0, len(cond), rows):
        c = torch.tensor(cond[s:s + rows])
        out = worker.model.sample(x=c, condition_mask=cmask, num_samples=n, device=worker.device,
                                  verbose=False, **_sample_kwargs(worker))
        outs.append(torch.as_tensor(out).detach().float().cpu().numpy().reshape(len(c), n, -1))
    latent = np.concatenate(outs) * worker.std[~mask] + worker.mean[~mask]
    return {"samples": latent.astype(np.float32), "n_nonfinite": int((~np.isfinite(latent)).any(-1).sum())}


def log_prob_batched(worker, msg):
    """log q(latent | x_o) in data units via the probability-flow ODE.

    latent: (B, d_lat) or (B, K, d_lat), x_o: (B, d_obs) -> (B,) or (B, K).
    The z-score is undone with the Jacobian: log q_data = log q_z - sum log std_latent.
    """
    cfg = worker.cfg
    mask = np.asarray(msg["condition_mask"]).astype(bool).reshape(-1)
    lat = np.asarray(msg["latent"], dtype=np.float32)
    x_o = np.asarray(msg["x_o"], dtype=np.float32).reshape(lat.shape[0], -1)
    lead = lat.shape[:-1]
    lat = lat.reshape(-1, lat.shape[-1])
    if len(lead) == 2:
        x_o = np.repeat(x_o, lead[1], axis=0)
    joint = np.zeros((len(lat), mask.size), np.float32)
    joint[:, ~mask] = lat
    joint[:, mask] = x_o
    z = (joint - worker.mean) / worker.std
    lp = worker.model.log_prob(torch.tensor(z, dtype=torch.float32),
                               torch.tensor(mask, dtype=torch.float32),
                               timesteps=int(cfg.get("log_prob_steps", 200)),
                               eps=float(cfg.get("log_prob_eps", cfg.get("eps", 1e-3))),
                               device=worker.device,
                               batch_size=int(cfg.get("log_prob_batch", 10000)))
    lp = torch.as_tensor(lp).double().cpu().numpy() - np.log(worker.std[~mask]).sum()
    return {"log_prob": lp.reshape(lead).astype(np.float64)}


def main():
    _install_batch_loader()
    worker = Worker()
    while True:
        msg = read_message()
        if msg is None or msg.get("cmd") == "quit":
            break
        try:
            if msg["cmd"] == "train":
                reply = {"ok": True, **worker.train(msg)}
            elif msg["cmd"] == "sample":
                reply = {"ok": True, **worker.sample(msg)}
            elif msg["cmd"] == "sample_batched":
                reply = {"ok": True, **sample_batched(worker, msg)}
            elif msg["cmd"] == "log_prob_batched":
                reply = {"ok": True, **log_prob_batched(worker, msg)}
            elif msg["cmd"] == "ping":
                reply = {"ok": True}
            else:
                reply = {"ok": False, "error": f"unknown command {msg['cmd']!r}"}
        except Exception as error:  # report, do not die silently
            import traceback
            traceback.print_exc(file=sys.stderr)
            reply = {"ok": False, "error": repr(error)}
        write_message(reply)


if __name__ == "__main__":
    main()
