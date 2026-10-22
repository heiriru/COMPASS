#!/usr/bin/env python3
"""Conservative (s = grad E) vs plain score network on the GM benchmark.

Same recipe as the production GM net (tokenized h32d3wide, 200k sims, six_weighted,
log_sigma, best-validation checkpoint), trained here for both arms and two seeds.

    python gm_conservative.py train --arm cons --seed 7
    python gm_conservative.py accuracy --arm cons --seed 7      # vs the exact oracle
    python gm_conservative.py compose --arm cons --seed 7       # run_scaling, gauss_jacobian_FD
    python gm_conservative.py report
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path("/export/home/rheinric/COMPASS")
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "covhead"))
import gpu  # noqa: E402

if __name__ == "__main__" and "report" not in sys.argv:
    gpu.claim(int(os.environ.get("NEED_MB", "1500")))
GM = MAIN / "Gaussian_mixtures_compositional_score_modelling"
sys.path.insert(0, str(GM))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import argparse  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402

import compass  # noqa: E402  worktree copy first: experiments_arruda_schur/common.py
import compass.MultiObsSampler  # noqa: E402,F401  prepends the main checkout's src
import compass.experimental.conservative  # noqa: E402,F401
assert "COMPASS-jacext" in compass.__file__, compass.__file__

torch.set_num_threads(3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RECIPE = dict(config="h32d3wide", arch="tokenized", simulations=200_000,
              mask_menu="six_weighted", time_sampling="log_sigma")
LAMBDAS = [0.032, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def root(arm, seed):
    return HERE / os.environ.get("ART", "artifacts") / f"gm_{arm}_s{seed}"


def patch_train(arm):
    import train
    from compass.experimental.conservative import enable_conservative, load_conservative
    if arm != "cons":
        return train
    original_build = train.build

    def build(*args, **kwargs):
        return enable_conservative(original_build(*args, **kwargs))

    def load_to(checkpoint, device):
        import model as M  # noqa: F401
        net = original_build(RECIPE["config"], RECIPE["arch"], device=device)
        return load_conservative(net, checkpoint, device)

    train.build = build
    train.load_to = load_to
    return train


def directory(train, arm, seed):
    return train.model_directory(root(arm, seed), RECIPE["config"], RECIPE["arch"],
                                 RECIPE["simulations"], RECIPE["mask_menu"],
                                 RECIPE["time_sampling"])


def load(arm, seed):
    train = patch_train(arm)
    net, seconds, params = train.train_or_load(directory(train, arm, seed), RECIPE["config"],
                                               RECIPE["arch"], RECIPE["simulations"], DEVICE,
                                               seed=seed, mask_menu=RECIPE["mask_menu"],
                                               time_sampling=RECIPE["time_sampling"],
                                               max_epochs=(int(os.environ["MAX_EPOCHS"])
                                                           if os.environ.get("MAX_EPOCHS") else None))
    return net, seconds


def do_train(arm, seed):
    net, seconds = load(arm, seed)
    (root(arm, seed) / "train_seconds.json").write_text(json.dumps({"seconds": seconds}))


def accuracy(arm, seed, count=4096):
    import model as M
    from oracle import OracleScore
    from compass.experimental.multilevel import finite_difference_jacobian, jacobian_precision
    from gm_head import score_fn, simulate
    net, _ = load(arm, seed)
    for p in net.model.parameters():
        p.requires_grad_(False)
    net.model.eval()
    sde = net.sde
    oracle = OracleScore(M.DEFAULT, sde).to(DEVICE)
    net_score, exact_score = score_fn(net.model, sde), score_fn(oracle, sde)
    generator = torch.Generator().manual_seed(123)
    theta, x = simulate(count, generator)
    theta, x = theta.to(DEVICE), x.to(DEVICE)
    eye = torch.eye(2, dtype=torch.float64, device=DEVICE)
    rows = []
    for lam in LAMBDAS:
        y = theta + lam * torch.randn(theta.shape, generator=generator).to(DEVICE)
        with torch.no_grad():
            s_net, s_ex = net_score(y, x, lam), exact_score(y, x, lam)
            J_fd = finite_difference_jacobian(lambda p, l: net_score(p.float(), x.repeat(3, 1), l),
                                              y.double(), lam)
            J_ex = finite_difference_jacobian(lambda p, l: exact_score(p.float(), x.repeat(3, 1), l),
                                              y.double(), lam, step=1e-3)

        def rel(a, b):
            return (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp_min(1e-12)
        K_fd = jacobian_precision(J_fd, lam) - eye / lam ** 2
        K_ex = jacobian_precision(J_ex, lam) - eye / lam ** 2
        rows.append({
            "lam": lam,
            "score_rel_err": float(rel(s_net, s_ex).median()),
            "jac_rel_err": float(rel(J_fd, J_ex).median()),
            "jac_antisym": float(((J_fd - J_fd.mT).flatten(1).norm(dim=1)
                                  / J_fd.flatten(1).norm(dim=1)).median()),
            "info_rel_err": float(rel(K_fd, K_ex).median()),
        })
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(root(arm, seed) / "accuracy.json", "w"), indent=1)


def compose(arm, seed, extra, tag="jac", head=None):
    import run_scaling
    sys.path.insert(0, str(MAIN / "experiments_arruda_schur"))
    import common
    import gm
    common.install_configured_sampler()
    gm.register_methods()
    artifacts = root(arm, seed)
    if arm.startswith("prod_"):                 # production GM nets (gm.py modes)
        import train
        mode = arm[len("prod_"):]
        artifacts = gm.artifacts(mode)
        if mode == "none":
            gm.link_production_checkpoint(gm.DEFAULT_RECIPE)
        train.load_to = common.load_with_latent_attention(train.load_to)
    else:
        patch_train(arm)
    if head is not None:
        os.environ["COMPASS_COVARIANCE_HEAD"] = str(head)
    argv = ["--artifacts", str(artifacts), "--tag", tag,
            "--config", RECIPE["config"], "--arch", RECIPE["arch"],
            "--simulations", str(RECIPE["simulations"]), "--mask-menu", RECIPE["mask_menu"],
            "--time-sampling", RECIPE["time_sampling"], "--device", DEVICE,
            "--observations", "8", "16", "32", "64", "--regimes", "asymmetric", "symmetric",
            "--replicates", "1", "--draws", "2000", "--methods", "gauss_jacobian_FD", *extra]
    from runtime import configure_runtime
    run_scaling.configure_runtime = lambda max_threads=3, gpu=True: configure_runtime(max_threads, gpu=False)
    return run_scaling.main(argv)


def report():
    out = []
    base = HERE / os.environ.get("ART", "artifacts")
    base.mkdir(exist_ok=True)
    for path in sorted(base.glob("gm_*_s*")):
        acc = path / "accuracy.json"
        summ = path / "scaling" / "jac" / "summary.json"
        if acc.exists():
            for r in json.load(open(acc)):
                out.append(f"{path.name} lam={r['lam']:<6} score {r['score_rel_err']:.3f} "
                           f"jac {r['jac_rel_err']:.3f} antisym {r['jac_antisym']:.3f} info {r['info_rel_err']:.3f}")
        if summ.exists():
            d = json.load(open(summ))
            for k, v in sorted(d.items()):
                if "gauss_jacobian" in k:
                    out.append(f"{path.name} {k:34s} W1/sd {v['w1_over_sigma']['median']:.3f} "
                               f"branch err {v['branch_mass_error']['median']:+.3f} "
                               f"local W1 {v['local_w1_over_sigma']['median']:.3f}")
    text = "\n".join(out)
    (base / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "accuracy", "compose", "report"])
    parser.add_argument("--arm", default="cons",
                        help="plain | cons | prod_none | prod_hierarchical")
    parser.add_argument("--tag", default="jac")
    parser.add_argument("--head", default=None, help="covariance head .pt (compose only)")
    parser.add_argument("--seed", type=int, default=7)
    args, extra = parser.parse_known_args()
    if args.command == "train":
        do_train(args.arm, args.seed)
    elif args.command == "accuracy":
        accuracy(args.arm, args.seed)
    elif args.command == "compose":
        compose(args.arm, args.seed, extra, tag=args.tag, head=args.head)
    else:
        report()
