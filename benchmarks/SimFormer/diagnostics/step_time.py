"""Training seconds per step at batch 1000: upstream v1.1 vs our COMPASS.

The benchmark trains every COMPASS arm through the bridge's BatchLoader, so its
wall-clock numbers do not show what the upstream package costs out of the box
(its Trainer draws a condition mask per *row* inside a DataLoader). This times
a fixed number of optimizer steps, three ways, on random data of the task's
width:

  upstream_stock   upstream package, its own Trainer data path
  upstream_batch   upstream package, bridge BatchLoader (same mask law)
  ours_batch       current COMPASS legacy arm as benchmarked (hidden 64)

Usage:  step_time.py <upstream|ours> <stock|batch> <nodes> [steps]
Run one configuration per process (the two packages cannot share one).
Prints one line: "<config> nodes=<n> params=<p> s_per_step=<t>".
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIMFORMER = os.path.dirname(HERE)
flavour, loader, nodes = sys.argv[1], sys.argv[2], int(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 200
if flavour == "upstream":
    os.environ["COMPASS_SRC"] = os.path.join(SIMFORMER, "compass_upstream_v11", "src")
sys.path.insert(0, os.path.join(SIMFORMER, "bridge"))

from autocvd import autocvd  # noqa: E402

autocvd(num_gpus=1, interval=1)    # waits for a free GPU, sets CUDA_VISIBLE_DEVICES

import compass_worker as cw  # noqa: E402  (caps CPUs, honours COMPASS_SRC)
import torch  # noqa: E402

if loader == "batch":
    cw._install_batch_loader()

batch = 1000
rows = batch * steps // 2          # two epochs of `steps / 2` steps
data = torch.randn(rows, nodes)
theta_dim = max(1, nodes // 2)
common = dict(sde_type="vesde", sigma=25.0, depth=6, device="cuda")
if flavour == "upstream":
    model = cw.SBIm(nodes_size=nodes, hidden_size=128, num_heads=16, mlp_ratio=4, **common)
    kw = dict(time_sampling="mixture")
else:
    model = cw.SBIm(nodes_size=nodes, arch="legacy", hidden_size=64, num_heads=4, mlp_ratio=3, **common)
    kw = dict(time_sampling="log_sigma", mask_sampler=cw.SimformerMaskLaw(theta_dim, nodes - theta_dim))
n_params = sum(p.numel() for p in model.parameters())

out = os.path.join(SIMFORMER, "diagnostics", "out", "step_time_ckpt")
torch.cuda.synchronize()
start = time.time()
model.train(data[:, :theta_dim], data[:, theta_dim:], batch_size=batch, max_epochs=2, lr=1e-3,
            device="cuda", verbose=False, path=out, name="M", early_stopping_patience=10**9, **kw)
torch.cuda.synchronize()
seconds = time.time() - start
print(f"{flavour}_{loader} nodes={nodes} params={n_params} s_per_step={seconds / steps:.4f}", flush=True)
