"""
Analytic verification of the compositional (multi-observation) score modeling.

A mock "score network" returns the EXACT score of the diffused single-observation
posterior for a linear-Gaussian toy model, so any error in the multi-observation
samples is attributable to the composition / sampling machinery, not to training.
The sampled posterior is compared against the analytic multi-observation posterior.

Runs on a CUDA GPU in a few minutes:
    python tests/test_multiobs_analytic.py
or with pytest:
    pytest tests/test_multiobs_analytic.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

# Keep this standalone executable from consuming more than 3 host CPU threads.
_CPU_THREAD_LIMIT = 3
_cpu_limit = min(_CPU_THREAD_LIMIT, os.cpu_count())
if _cpu_limit < 1:
    raise RuntimeError("The 3-thread CPU cap permits fewer than one logical CPU.")
_available_cpus = sorted(os.sched_getaffinity(0))
_cpu_limit = min(_cpu_limit, len(_available_cpus))
os.sched_setaffinity(0, set(_available_cpus[:_cpu_limit]))
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = str(_cpu_limit)
print(f"Using {_cpu_limit} logical CPU(s) (limit of {_CPU_THREAD_LIMIT} threads).")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from compass.MultiObsSampler import MultiObsSampler
from compass.Sampler import Sampler
from compass.SDE import VESDE

if not torch.cuda.is_available():
    raise RuntimeError("test_multiobs_analytic.py requires a CUDA-capable GPU.")

DEVICE = torch.device("cuda")
MU0 = torch.tensor([-2.3, -2.89], device=DEVICE)
SIG0 = torch.tensor([0.3, 0.3], device=DEVICE)
SIGX = torch.tensor([0.5, 0.5], device=DEVICE)


class MockSBIm:
    """Minimal stand-in for ScoreBasedInferenceModel."""
    def __init__(self, model_cls, nodes):
        self.sde = VESDE(sigma=25.0)
        self.sde.sigma = self.sde.sigma.to(DEVICE)
        self.model = model_cls(self.sde)
        self.sampler = Sampler(self)
        self.nodes_size = nodes

    def output_scale_function(self, t, x):
        return x / self.sde.marginal_prob_std(t).to(x.device)


class LinearGaussianModel(torch.nn.Module):
    """Nodes [th1, th2, x1, x2]; returns std_t * true diffused posterior score
    (matching the training convention model_out ~ -noise)."""
    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    def forward(self, x, t, c, return_attn_weights=False):
        std_t = self.sde.marginal_prob_std(t).to(x.device)
        theta_t, x_obs = x[:, :2], x[:, 2:]
        lam = 1 / SIG0**2 + 1 / SIGX**2
        m = (MU0 / SIG0**2 + x_obs / SIGX**2) / lam
        score = -(theta_t - m) / (1 / lam + std_t**2)
        out = torch.zeros_like(x)
        out[:, :2] = std_t * score
        if return_attn_weights:
            return out, torch.zeros(1)
        return out


def analytic_posterior(x_all):
    n = x_all.shape[0]
    lam_star = 1 / SIG0**2 + n / SIGX**2
    m_star = (MU0 / SIG0**2 + x_all.sum(0) / SIGX**2) / lam_star
    return m_star, torch.sqrt(1 / lam_star)


def _sample(n_obs, correction, method="dpm", timesteps=50, use_est_precision=False, **kw):
    torch.manual_seed(42)
    theta_true = MU0 + SIG0 * torch.randn(2, device=DEVICE)
    x_all = theta_true + SIGX * torch.randn(n_obs, 2, device=DEVICE)
    m_star, s_star = analytic_posterior(x_all)

    sbim = MockSBIm(LinearGaussianModel, 4)
    mos = MultiObsSampler(sbim)
    mask = torch.tensor([0., 0., 1., 1.], device=DEVICE)
    post_prec = None
    if correction == "gauss" and not use_est_precision:
        post_prec = (1 / SIG0**2 + 1 / SIGX**2).repeat(n_obs, 1)

    samples = mos.sample(
        world_size=1, data=x_all, condition_mask=mask,
        timesteps=timesteps, num_samples=2000,
        hierarchy=[0, 1], prior=(MU0, SIG0),
        correction=correction, posterior_precision=post_prec,
        precision_est_samples=500,
        method=method, device=DEVICE, verbose=False, **kw)

    th = samples[0, :, :2]   # shared dims are synchronized across observation rows
    mean_err_sigma = (th.mean(0) - m_star).abs().max().item() / s_star.max().item()
    std_ratio = th.std(0) / s_star
    return mean_err_sigma, std_ratio


def test_gauss_correction_analytic_precision():
    for n in [1, 5, 50, 200]:
        mean_err, std_ratio = _sample(n, "gauss")
        print(f"gauss n={n}: mean err {mean_err:.2f} sigma, std ratio {std_ratio.tolist()}")
        assert mean_err < 0.20, f"n={n}: composed posterior mean off by {mean_err:.2f} sigma"
        assert ((std_ratio > 0.85) & (std_ratio < 1.15)).all(), \
            f"n={n}: composed posterior width off ({std_ratio.tolist()})"


def test_gauss_correction_estimated_precision():
    for n in [5, 50]:
        mean_err, std_ratio = _sample(n, "gauss", use_est_precision=True)
        print(f"gauss(est) n={n}: mean err {mean_err:.2f} sigma, std ratio {std_ratio.tolist()}")
        assert mean_err < 0.25
        assert ((std_ratio > 0.8) & (std_ratio < 1.2)).all()


def test_pfode_gauss_correction():
    for n in [5, 50]:
        mean_err, std_ratio = _sample(
            n, "gauss", method="heun", equation="probability_flow_ode")
        print(f"PF-ODE gauss n={n}: mean err {mean_err:.2f} sigma, "
              f"std ratio {std_ratio.tolist()}")
        assert mean_err < 0.25
        assert ((std_ratio > 0.8) & (std_ratio < 1.2)).all()


def test_pfode_gauss_estimated_precision():
    mean_err, std_ratio = _sample(
        5, "gauss", method="heun", equation="probability_flow_ode",
        use_est_precision=True)
    print(f"PF-ODE gauss(est): mean err {mean_err:.2f} sigma, "
          f"std ratio {std_ratio.tolist()}")
    assert mean_err < 0.25
    assert ((std_ratio > 0.8) & (std_ratio < 1.2)).all()


def test_pfode_uncorrected_and_fnpe_validation():
    mean_err, std_ratio = _sample(
        5, "uncorrected", method="heun", equation="probability_flow_ode")
    assert mean_err < 1.0
    assert torch.isfinite(std_ratio).all()

    sbim = MockSBIm(LinearGaussianModel, 4)
    mos = MultiObsSampler(sbim)
    try:
        mos.sample(
            world_size=1, data=torch.zeros(2, 2, device=DEVICE),
            condition_mask=torch.tensor([0., 0., 1., 1.], device=DEVICE),
            hierarchy=[0, 1], prior=(MU0, SIG0), correction="fnpe",
            method="heun", equation="probability_flow_ode",
            num_samples=2, device=DEVICE, verbose=False)
    except ValueError as exc:
        assert "not compatible" in str(exc)
    else:
        raise AssertionError("PF-ODE must reject correction='fnpe'")


def test_fnpe_langevin():
    # F-NPSE (Eq. 7) with annealed Langevin: means must be right; the width is
    # limited by Langevin mixing and allowed a generous margin.
    for n in [5, 50]:
        mean_err, std_ratio = _sample(n, "fnpe", method="langevin",
                                      timesteps=100, corrector_steps=5)
        print(f"fnpe n={n}: mean err {mean_err:.2f} sigma, std ratio {std_ratio.tolist()}")
        assert mean_err < 0.5
        assert ((std_ratio > 0.8) & (std_ratio < 2.2)).all()


def test_multimodal_two_modes():
    """Paper Sec. 5.1: p(theta)=N(0,I), p(x|theta)=0.5 N(theta, I/2)+0.5 N(-theta, I/2).
    The multi-observation posterior must keep both symmetric modes."""

    class MixtureModel(torch.nn.Module):
        def __init__(self, sde):
            super().__init__()
            self.sde = sde

        def forward(self, x, t, c, return_attn_weights=False):
            std_t = self.sde.marginal_prob_std(t).to(x.device)
            theta, xo = x[:, :2], x[:, 2:]
            s2 = 1/3 + std_t**2
            m1, m2 = 2*xo/3, -2*xo/3
            lw1 = -0.5*((theta-m1)**2).sum(1, keepdim=True)/s2
            lw2 = -0.5*((theta-m2)**2).sum(1, keepdim=True)/s2
            w1 = torch.sigmoid(lw1 - lw2)
            score = (w1*(m1-theta) + (1-w1)*(m2-theta))/s2
            out = torch.zeros_like(x)
            out[:, :2] = std_t*score
            if return_attn_weights:
                return out, torch.zeros(1)
            return out

    torch.manual_seed(3)
    theta_true = torch.randn(2, device=DEVICE)
    n = 5
    comp = (torch.rand(n, 1, device=DEVICE) < 0.5).float()
    x_all = (comp*theta_true + (1-comp)*(-theta_true)) + (0.5**0.5)*torch.randn(n, 2, device=DEVICE)

    sbim = MockSBIm(MixtureModel, 4)
    mos = MultiObsSampler(sbim)
    mask = torch.tensor([0., 0., 1., 1.], device=DEVICE)
    samples = mos.sample(world_size=1, data=x_all, condition_mask=mask,
                         timesteps=100, num_samples=4000, hierarchy=[0, 1],
                         prior=(0.0, 1.0), correction="gauss",
                         precision_est_samples=500,
                         method="dpm", device=DEVICE, verbose=False)
    th = samples[0, :, :2]
    d_plus = ((th - theta_true)**2).sum(1)
    d_minus = ((th + theta_true)**2).sum(1)
    frac_plus = (d_plus < d_minus).float().mean().item()
    print(f"multimodal: mode fractions {frac_plus:.2f}/{1-frac_plus:.2f}")
    assert 0.4 < frac_plus < 0.6, f"mode weights unbalanced: {frac_plus:.2f}"


def hierarchical_shared_and_local_metrics():
    """Run the exact-score global/local benchmark and return plot-ready metrics."""
    S0G, S0L, SX_, MUG = 0.3, 0.4, 0.2, -2.5

    class HierModel(torch.nn.Module):
        def __init__(self, sde):
            super().__init__()
            self.sde = sde
            P = torch.tensor([[1/S0G**2 + 1/SX_**2, 1/SX_**2],
                              [1/SX_**2, 1/S0L**2 + 1/SX_**2]])
            self.register_buffer("Sigma_post", torch.linalg.inv(P))

        def forward(self, x, t, c, return_attn_weights=False):
            std_t = self.sde.marginal_prob_std(t).to(x.device)
            theta, xo = x[:, :2], x[:, 2:3]
            b = torch.zeros_like(theta)
            b[:, 0] = MUG / S0G**2 + xo[:, 0] / SX_**2
            b[:, 1] = xo[:, 0] / SX_**2
            m = b @ self.Sigma_post.T
            Sig_t = self.Sigma_post + std_t**2 * torch.eye(2, device=theta.device)
            score = torch.linalg.solve(Sig_t, (m - theta).unsqueeze(-1)).squeeze(-1)
            out = torch.zeros_like(x)
            out[:, :2] = std_t * score
            if return_attn_weights:
                return out, torch.zeros(1)
            return out

    # Hierarchical models (strong global-local coupling the diagonal Gaussian
    # correction cannot capture at intermediate t) need dense Langevin correction:
    # corrector steps at every level.
    rows = []
    for n, csi, cs, snr in [(5, 5, 5, 0.1), (50, 1, 10, 0.2), (200, 1, 10, 0.2)]:
        torch.manual_seed(7)
        tg = MUG + S0G*torch.randn(1, device=DEVICE)
        x_all = (tg + S0L*torch.randn(n, device=DEVICE) + SX_*torch.randn(n, device=DEVICE)).unsqueeze(1)
        lam = 1/S0G**2 + n/(S0L**2 + SX_**2)
        m_star = (MUG/S0G**2 + x_all.sum()/(S0L**2 + SX_**2)) / lam
        s_star = (1/lam)**0.5

        sbim = MockSBIm(HierModel, 3)
        mos = MultiObsSampler(sbim)
        samples = mos.sample(world_size=1, data=x_all,
                             condition_mask=torch.tensor([0., 0., 1.], device=DEVICE),
                             timesteps=100, num_samples=2000, hierarchy=[0],
                             prior=([MUG], [S0G]), correction="gauss",
                             posterior_precision=torch.full((n, 1), 1/S0G**2 + 1/(S0L**2 + SX_**2)),
                             corrector_steps_interval=csi, corrector_steps=cs, snr=snr,
                             method="dpm", device=DEVICE, verbose=False)
        th_g = samples[0, :, 0]
        mean_err = abs(th_g.mean() - m_star).item() / s_star
        std_ratio = th_g.std().item() / s_star
        print(f"hier n={n}: mean err {mean_err:.2f} sigma, std ratio {std_ratio:.2f}")
        # Width is conservative (overdispersed) for strongly coupled hierarchical
        # models at large n: the diagonal Gaussian correction cannot represent the
        # global-local coupling at intermediate t. Means stay accurate.
        assert mean_err < 0.35
        assert 0.8 < std_ratio < 1.7
        rows.append({
            "n_observations": n,
            "mean_error_in_analytic_std": mean_err,
            "posterior_std_ratio": std_ratio,
            "mean_error_limit": 0.35,
            "std_ratio_lower_limit": 0.8,
            "std_ratio_upper_limit": 1.7,
        })
    return rows


def test_hierarchical_shared_and_local():
    hierarchical_shared_and_local_metrics()


def write_metric_rows(path, rows):
    """Write standalone metrics while keeping pytest side-effect free."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchical-only", action="store_true",
                        help="run only the exact-score global/local validation")
    parser.add_argument("--output-csv", type=Path,
                        help="optionally save hierarchical metrics as CSV")
    args = parser.parse_args()
    if args.hierarchical_only:
        metric_rows = hierarchical_shared_and_local_metrics()
        if args.output_csv:
            write_metric_rows(args.output_csv, metric_rows)
            print(f"Wrote {args.output_csv}")
        raise SystemExit(0)
    test_gauss_correction_analytic_precision()
    test_gauss_correction_estimated_precision()
    test_pfode_gauss_correction()
    test_pfode_gauss_estimated_precision()
    test_pfode_uncorrected_and_fnpe_validation()
    test_fnpe_langevin()
    test_multimodal_two_modes()
    metric_rows = hierarchical_shared_and_local_metrics()
    if args.output_csv:
        write_metric_rows(args.output_csv, metric_rows)
        print(f"Wrote {args.output_csv}")
    print("All compositional score modeling tests passed.")
