"""Dump sbibm observations + reference posterior samples (.simformer env, CPU)."""
import os, sys; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, sbibm
task_name, n_obs = sys.argv[1], int(sys.argv[2])
t = sbibm.get_task(task_name)
np.savez(f"diagnostics/out/{task_name}_refs.npz",
         x=np.stack([t.get_observation(i).numpy()[0] for i in range(1, n_obs + 1)]),
         ref=np.stack([t.get_reference_posterior_samples(i).numpy()[:2000] for i in range(1, n_obs + 1)]))
