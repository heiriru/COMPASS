"""Harness side of the bridge: COMPASS as a ``scoresbibm`` method.

Imported by ``run_scoresbi.py`` inside the ``.simformer`` env. It registers a
``compass`` method with the official Simformer harness, so COMPASS is trained on
the harness's own ``task.get_data`` simulations and scored by the harness's own
evaluation loop and C2ST -- the same code path their Simformer goes through.

The model object is a proxy: training and sampling happen in a persistent
``compass_worker.py`` subprocess running in the repository's ``.COMPASS`` env.
"""

import atexit
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from scoresbibm.methods import method_base
from scoresbibm.methods.models import AllConditionalModel
from scoresbibm.scripts import hydra_script

HERE = Path(__file__).resolve().parent
COMPASS_ROOT = HERE.parents[2]
COMPASS_PYTHON = os.environ.get("COMPASS_PYTHON", str(COMPASS_ROOT / ".COMPASS" / "bin" / "python"))


class CompassWorker:
    def __init__(self, compass_src=None):
        env = dict(os.environ)
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        if compass_src:   # import `compass` from another checkout (e.g. upstream v1.1-integration)
            env["COMPASS_SRC"] = str((HERE.parent / compass_src).resolve())
        self.proc = subprocess.Popen(
            [COMPASS_PYTHON, "-u", str(HERE / "compass_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            env=env, cwd=str(COMPASS_ROOT),
        )
        atexit.register(self.close)

    def call(self, **msg):
        blob = pickle.dumps(msg, protocol=4)
        self.proc.stdin.write(struct.pack("<Q", len(blob)))
        self.proc.stdin.write(blob)
        self.proc.stdin.flush()
        header = self.proc.stdout.read(8)
        if len(header) < 8:
            raise RuntimeError("COMPASS worker died (see stderr above).")
        (size,) = struct.unpack("<Q", header)
        reply = pickle.loads(self.proc.stdout.read(size))
        if not reply.pop("ok", False):
            raise RuntimeError(f"COMPASS worker error: {reply.get('error')}")
        return reply

    def close(self):
        if self.proc.poll() is None:
            try:
                blob = pickle.dumps({"cmd": "quit"})
                self.proc.stdin.write(struct.pack("<Q", len(blob)) + blob)
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()


class CompassModel(AllConditionalModel):
    """Proxy with the ``AllConditionalModel`` sampling contract."""

    def __init__(self, worker, theta_dim, x_dim, train_info):
        super().__init__("compass", backend="torch")
        self.worker = worker
        self.train_info = train_info
        self.set_default_node_id(jnp.arange(theta_dim + x_dim))
        self.set_default_condition_mask(
            jnp.array([False] * theta_dim + [True] * x_dim, dtype=jnp.bool_))
        self.sampled_nonfinite = 0

    def _sample(self, num_samples, x_o, rng=None, node_id=None, condition_mask=None, **kwargs):
        seed = 0 if rng is None else int(jax.random.randint(rng, (), 0, 2**31 - 1))
        mask = np.asarray(condition_mask).astype(bool).reshape(-1)
        x_o = np.asarray(x_o, dtype=np.float32).reshape(-1)
        reply = self.worker.call(cmd="sample", x_o=x_o, condition_mask=mask,
                                 num_samples=int(num_samples), seed=seed)
        self.sampled_nonfinite += reply["n_nonfinite"]
        if reply["n_nonfinite"]:
            print(f"[compass] WARNING {reply['n_nonfinite']} non-finite samples", flush=True)
        return jnp.asarray(reply["samples"])

    # NLL and coverage (eval_negative_log_likelihood / eval_coverage) take the
    # harness's torch path: jax.jit cannot trace through the worker pipe. Only
    # those two metrics read `backend`; C2ST goes through `sample`.
    def sample_batched(self, num_samples, x_o, condition_mask=None, rng=None, **kwargs):
        seed = 0 if rng is None else int(jax.random.randint(rng, (), 0, 2**31 - 1))
        reply = self.worker.call(cmd="sample_batched", x_o=np.asarray(x_o, dtype=np.float32),
                                 condition_mask=np.asarray(condition_mask).astype(bool),
                                 num_samples=int(num_samples), seed=seed)
        self.sampled_nonfinite += reply["n_nonfinite"]
        return torch.as_tensor(reply["samples"])

    def log_prob_batched(self, theta, x_o=None, condition_mask=None, **kwargs):
        reply = self.worker.call(cmd="log_prob_batched", latent=np.asarray(theta, dtype=np.float32),
                                 x_o=np.asarray(x_o, dtype=np.float32),
                                 condition_mask=np.asarray(condition_mask).astype(bool))
        return torch.as_tensor(reply["log_prob"])

    def _log_prob(self, theta, x_o, **kwargs):
        raise NotImplementedError("use log_prob_batched")

    def __getstate__(self):   # the harness tries to pickle models; ours holds a pipe
        raise TypeError("CompassModel proxies a live worker and cannot be pickled.")


def run_compass(task, data, method_cfg, rng=None):
    theta = np.asarray(data["theta"], dtype=np.float32)
    x = np.asarray(data["x"], dtype=np.float32)
    n = theta.shape[0]
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(method_cfg, resolve=True)
    cfg.update(cfg.pop("compass", None) or {})
    # Their gradient-step budget: clip(3 N, 5k, 100k) at batch 1000.
    cfg.setdefault("max_steps", int(max(min(3 * n, 100_000), 5_000)))
    try:  # each run's own Hydra output dir (Hydra does not chdir here)
        from hydra.core.hydra_config import HydraConfig
        run_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        run_dir = os.getcwd()
    out_dir = os.path.join(run_dir, "compass_model")
    seed = 0 if rng is None else int(jax.random.randint(rng, (), 0, 2**31 - 1))
    worker = CompassWorker(cfg.get("compass_src"))
    info = worker.call(cmd="train", theta=theta, x=x, cfg=cfg, out_dir=out_dir, seed=seed)
    print("[compass] train info:", info, flush=True)
    with open(os.path.join(run_dir, "compass_train_info.txt"), "w") as fh:
        fh.write(repr({"cfg": cfg, **info}))
    return CompassModel(worker, task.get_theta_dim(), task.get_x_dim(), info)


_original_get_method = method_base.get_method


def get_method(name):
    if name.startswith("compass"):
        return run_compass
    return _original_get_method(name)


method_base.get_method = get_method
hydra_script.get_method = get_method
