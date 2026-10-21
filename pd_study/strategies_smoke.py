import sys
from golden import model, torch

def run(n, strategy, fallback="repair", corr="gauss_jacobian_FD"):
    m = model()
    m.multi_obs_sampler.set_pd_strategy(strategy, fallback)
    torch.manual_seed(1); data = torch.randn(n, 3) * 0.5
    torch.manual_seed(2)
    d = m.sample(x=data, multi_obs_inference=True, hierarchy=[0, 1], correction=corr,
                 timesteps=12, num_samples=64, verbose=False)
    return d, m.multi_obs_sampler.covariance_diagnostics

ref = torch.load("golden.pt", weights_only=False)
for n in (4, 16):
    base = ref[("gauss_jacobian_FD", n)][0]
    for strat, fb in [("row_projection", "repair"), ("row_projection_post", "repair"),
                      ("composed_check", "repair"), ("composed_check", "row_projection"),
                      ("composed_check", "gloeckler"), ("gloeckler", "repair"),
                      ("row_projection_consistent", "repair")]:
        d, diag = run(n, strat, fb)
        print(f"n={n:2d} {strat:26s} {fb:15s} max|d-default| {float((d-base).abs().max()):.2e} "
              f"mean {d[:, :2].mean(0).tolist()} nonPD {diag['lambda_s_nonpd_fraction']:.3f} "
              f"minrel {diag['lambda_s_min_relative']:.3g} repairs {diag['repair_count']}")
