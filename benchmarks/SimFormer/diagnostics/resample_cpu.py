"""Resample a trained COMPASS checkpoint on CPU with different terminal handling.

variants: base (as benchmarked: eps silently 1e-3, noise left on),
          denoise (terminal_denoise=True), eps5 (eps=1e-5 forwarded to the Sampler,
          working around SBIm.sample dropping its eps argument).
"""
import os, sys, functools
os.environ["CUDA_VISIBLE_DEVICES"] = ""
av = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, set(av[(os.getpid()*3+i) % len(av)] for i in range(3)))
for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[n] = "3"
import numpy as np, torch
from compass import ScoreBasedInferenceModel as SBIm
ckpt, task_name, variant, out = sys.argv[1:5]
z = np.load(os.path.join(ckpt, "zscore.npz")); mean, std = z["mean"], z["std"]
model = SBIm.load(os.path.join(ckpt, "Model_checkpoint.pt"), device="cpu"); model.model.eval()
refs = np.load(f"diagnostics/out/{task_name}_refs.npz")
d = refs["ref"].shape[-1]
kw = dict(method="dpm", order=2, churn_eta=2.0, timesteps=50, device="cpu", verbose=False)
if variant == "denoise":
    kw["terminal_denoise"] = True
if variant == "eps5":
    model.sampler.sample = functools.partial(model.sampler.sample, eps=1e-5)
    kw["timesteps"] = 100
torch.manual_seed(0)
res = []
for x in refs["x"]:
    cond = torch.tensor(((x - mean[d:]) / std[d:])[None], dtype=torch.float32)
    mask = torch.tensor([0.0] * d + [1.0] * len(x))
    with torch.no_grad():
        s = model.sample(x=cond, condition_mask=mask, num_samples=2000, **kw)
    res.append(np.asarray(s).reshape(2000, d) * std[:d] + mean[:d])
np.save(out, np.stack(res))
print("saved", out)
