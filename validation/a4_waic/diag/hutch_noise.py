"""Exp 1 check: is the inflated p_waic of the true model Hutchinson noise?
Same composed draws, log p by exact divergence and by two independent Hutchinson-1 runs."""
import validation.common.runtime_gpu  # noqa: F401
import json
import numpy as np
import torch
from validation.a4_waic import common
from validation.a4_waic.families import family
from validation.a4_waic.scaling import Scaler
from validation.a4_waic.train import load

out = {}
for true in ("LV", "LP"):
    model = family("tutorial")[true]
    scaler = Scaler.for_model("tutorial", model)
    net = load("tutorial", true, 131072, 0, device="cuda"); net.model.to("cuda")
    x = common.dataset("tutorial", true, 0)["x"]
    n, s = x.shape[0], 100
    mask = torch.tensor(model.condition_mask, dtype=torch.float32)
    xt = torch.tensor(scaler.x_forward(x), dtype=torch.float32)
    torch.manual_seed(0)
    th = net.sample(x=xt, condition_mask=mask, timesteps=50, num_samples=s, multi_obs_inference=False,
                    device="cuda", verbose=False).cpu()            # (N, S, 4)
    joint = torch.cat([th, xt[:, None].expand(-1, s, -1)], -1).reshape(n * s, -1)
    res = {}
    for name, div in (("exact", "exact"), ("hutch_a", "hutchinson"), ("hutch_b", "hutchinson")):
        lp = net.log_prob(joint, condition_mask=1 - mask, timesteps=50, divergence=div,
                          hutchinson_samples=1, device="cuda", batch_size=512)
        res[name] = lp.double().numpy().reshape(n, s)
    th_raw = scaler.theta_backward(th.double().numpy())
    exact_ll = model.loglik(None, th_raw, x[:, None, :]) + np.log(scaler.x_sd).sum()   # net coords
    d = res["hutch_a"] - res["hutch_b"]
    out[true] = dict(
        hutch_noise_sd_per_row=float(d.std() / np.sqrt(2)),
        var_exact_pfode_per_row=float(res["exact"].var(1, ddof=1).mean()),
        var_hutch_per_row=float(res["hutch_a"].var(1, ddof=1).mean()),
        var_true_ll_per_row=float(exact_ll.var(1, ddof=1).mean()),
        bias_exact_pfode_vs_true=float((res["exact"] - exact_ll).mean()),
        sd_exact_pfode_minus_true=float((res["exact"] - exact_ll).std()),
        post_sd_net=th.std(1).mean(0).tolist())
    print(true, json.dumps(out[true]), flush=True)
json.dump(out, open("validation/a4_waic/results/diag_hutch_noise_tutorial.json", "w"), indent=1)
