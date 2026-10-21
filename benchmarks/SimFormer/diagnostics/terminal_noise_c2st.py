"""How much C2ST does COMPASS's un-removed terminal noise cost by itself?

Takes the sbibm reference posterior samples, adds the Gaussian noise COMPASS
leaves on its samples when it stops at diffusion time eps (sigma_z(eps) times
the per-dimension z-score std, in theta units), and scores C2ST against clean
reference samples with the harness's own metric (sbi c2st, random forest).
CPU only (.simformer env; JAX is not imported).
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
av = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, set(av[(os.getpid()*3+i) % len(av)] for i in range(3)))
for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[n] = "3"
import math, numpy as np, torch, sbibm
from sbi.analysis.sbc import c2st

def sigma_vesde(t, s=25.0):
    return math.sqrt((s ** (2 * t) - 1) / (2 * math.log(s)))

torch.manual_seed(0)
print(f"{'task':>17} {'eps':>7} {'sigma_theta':>18} | C2ST (10 obs mean)")
for task_name, theta_std in (("two_moons", None), ("gaussian_mixture", None), ("gaussian_linear", None), ("slcp", None)):
    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    std = prior(num_samples=100000).std(0)          # the z-score std COMPASS uses (prior-dominated)
    for eps in (1e-3, 1e-5):
        sig = sigma_vesde(eps) * std
        vals = []
        for i in range(1, 11):
            ref = task.get_reference_posterior_samples(num_observation=i)
            perm = torch.randperm(len(ref)); a, b = ref[perm[:5000]], ref[perm[5000:10000]]
            vals.append(float(c2st(a, b + sig * torch.randn_like(b), classifier="rf")))
        print(f"{task_name:>17} {eps:7.0e} {np.array2string(sig.numpy()[:2], precision=4):>18} | {np.mean(vals):.3f} ± {np.std(vals, ddof=1)/math.sqrt(10):.3f}")
