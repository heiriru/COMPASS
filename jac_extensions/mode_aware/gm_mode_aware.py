#!/usr/bin/env python3
"""Mode-aware JAC on the GM benchmark (bimodal tall posterior, exact branch masses).

Networks: ``oracle`` (exact score), ``hierarchical`` / ``none`` (production GM nets).
Arms: ``jac`` (gauss_jacobian_FD, run_scaling settings), ``mode`` (mode-aware JAC with
the moment-matched Gaussian prior inside modes), ``mode_comp`` (mode-aware JAC composed
within the prior component each mode sits in).

    python gm_mode_aware.py run --net oracle --observations 8 16 32 64
    python gm_mode_aware.py report
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path("/export/home/rheinric/COMPASS")
sys.path.insert(0, str(HERE.parent))
import gpu  # noqa: E402

if __name__ == "__main__" and "report" not in sys.argv:
    gpu.claim(int(os.environ.get("NEED_MB", "2000")))
sys.path.insert(0, str(MAIN / "experiments_arruda_schur"))
sys.path.insert(0, str(MAIN / "Gaussian_mixtures_compositional_score_modelling"))
sys.path.insert(0, str(HERE.parent / "covhead"))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import compass  # noqa: E402  (worktree first; see covhead/gm_head.py)
import compass.MultiObsSampler  # noqa: E402,F401
import compass.experimental.mode_aware  # noqa: E402,F401
assert "COMPASS-jacext" in compass.__file__, compass.__file__

torch.set_num_threads(3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = HERE / os.environ.get("ART", "artifacts")
SETTINGS = dict(correction="gauss_jacobian_FD", method="dpm", order=2, timesteps=100,
                churn_eta=4.0, denoise_clamp=5.0, jacobian_refresh=5, corrector_steps=0,
                final_corrector_steps=0, terminal_corrector_steps=0)


def network(name):
    if name == "oracle":
        import model as M
        from oracle import OracleScore
        from compass import ScoreBasedInferenceModel as SBIm
        net = SBIm(nodes_size=M.NODES, sde_type="vesde", sigma=15.0, hidden_size=8, depth=1,
                   num_heads=1, mlp_ratio=1)
        net.model = OracleScore(M.DEFAULT, net.sde).to(DEVICE)
        return net
    from gm_head import load_network
    return load_network(name)


def log_prior_fn(mixture):
    centres = torch.as_tensor(mixture.components, dtype=torch.float64)
    log_w = torch.log(torch.as_tensor(mixture.weights, dtype=torch.float64))
    tau = mixture.tau_g

    def f(g):
        g = g.reshape(-1, 1)
        comp = -0.5 * ((g - centres) / tau) ** 2 - np.log(tau) - 0.5 * np.log(2 * np.pi)
        return torch.logsumexp(comp + log_w, 1)
    return f


def component_prior(mixture):
    """Within-component composition: the Gaussian prior component nearest a mode centre."""
    centres = torch.as_tensor(mixture.components, dtype=torch.float32)

    def f(centre):
        c = centres[int((centres - float(centre.reshape(-1)[0])).abs().argmin())]
        return (c.reshape(1), torch.tensor([float(mixture.tau_g)]))
    return f


def run(net_name, observations, regimes, replicates, draws):
    import datasets
    import methods
    import metrics
    import model as M
    from compass.MultiObsSampler import MultiObsSampler
    from compass.experimental.mode_aware import mode_aware_sample
    mixture = M.DEFAULT
    net = network(net_name)
    path = OUT / f"{net_name}.jsonl"
    done = set()
    if path.exists():
        done = {(r["regime"], r["n"], r["rep"], r["arm"])
                for r in map(json.loads, path.read_text().splitlines())}
    prior, local_prior = methods._priors(mixture)
    mask = torch.tensor([0.0, 0.0, 1.0])
    for regime in regimes:
        for n in observations:
            for rep in range(replicates):
                rng = np.random.default_rng([2026, n, rep, regimes.index(regime)])
                x = datasets.make_dataset(regime, n, rng, mixture)["x"]
                data = torch.as_tensor(np.asarray(x).reshape(-1, 1), dtype=torch.float32)
                for arm in ("jac", "mode", "mode_comp"):
                    if (regime, n, rep, arm) in done:
                        continue
                    sampler = MultiObsSampler(net)
                    torch.manual_seed(rep)
                    common = dict(local_prior=local_prior, device=DEVICE, **SETTINGS)
                    started = time.perf_counter()
                    info = {}
                    with methods.mixture_prior_score(mixture, enabled=(arm != "mode_comp")):
                        if arm == "jac":
                            out = sampler.sample(world_size=1, data=data, condition_mask=mask,
                                                 hierarchy=[0], prior=prior, num_samples=draws,
                                                 verbose=False, **common)
                        else:
                            out, info = mode_aware_sample(
                                sampler, data, mask, [0], draws, prior=prior,
                                discovery_samples=1000,
                                draws_per_mode=1000, log_prior_fn=log_prior_fn(mixture),
                                mode_prior_fn=(component_prior(mixture) if arm == "mode_comp"
                                               else None),
                                discovery_context=((lambda: methods.mixture_prior_score(mixture))
                                                   if arm == "mode_comp" else None),
                                **common)
                    seconds = time.perf_counter() - started
                    samples = out.detach().cpu().numpy()
                    samples[:, :, 2] = np.asarray(x).reshape(-1, 1)
                    row = metrics.evaluate(samples, mixture, x, seconds)
                    row.update(net=net_name, regime=regime, n=n, rep=rep, arm=arm,
                               modes=int(info.get("modes", 0)),
                               masses=[float(v) for v in info.get("masses", [])],
                               ess=[float(v) for v in info.get("ess_fraction", [])])
                    print(json.dumps(row), flush=True)
                    with path.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")


def report():
    import collections
    lines = ["| net | regime | n | arm | reps | W1/sd | branch mass err | width ratio | local W1 | modes | s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(OUT.glob("*.jsonl")):
        groups = collections.defaultdict(list)
        for r in map(json.loads, path.read_text().splitlines()):
            groups[(r["regime"], r["n"], r["arm"])].append(r)
        for (regime, n, arm), rs in sorted(groups.items()):
            m = lambda k: np.median([r[k] for r in rs])  # noqa: E731
            lines.append(f"| {path.stem} | {regime} | {n} | {arm} | {len(rs)} | {m('w1_over_sigma'):.3f} | "
                         f"{m('branch_mass_error'):+.3f} | {m('width_ratio'):.3f} | "
                         f"{m('local_w1_over_sigma'):.3f} | {m('modes'):.0f} | {m('seconds'):.0f} |")
    text = "\n".join(lines)
    (OUT / "TABLE.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--net", default="oracle")
    parser.add_argument("--observations", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--regimes", nargs="+", default=["asymmetric", "symmetric"])
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        run(args.net, args.observations, args.regimes, args.replicates, args.draws)
    report()
