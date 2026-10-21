#!/usr/bin/env python3
"""Cost/accuracy of the gauss_jacobian_FD speed-ups on the |g| toys (abs, lean, ...).

Reuses abs_hierarchy's datasets, networks and exact-score oracles. Every arm is a
variant spec (see ``parse``); each is run per n and seed and graded against

- the exact global posterior (closed form on a grid): W1/sd, P(g>0), sd ratio;
- the refresh-every-call baseline ``r1`` with the same seed (W1/sd on g, mean
  local W1/sd), next to the baseline's own reseed spread (the noise floor).

Cost is counted in network row-evaluations (observation rows x draws, FD copies
included) from the sampler's counters, plus wall-clock.

    python bench_toy.py --model lean --mode oracle_joint --variants r1 legacy band0.05-20 ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPASS = HERE.parent
ABS = COMPASS / "abs_hierarchy"
GM = COMPASS / "Gaussian_mixtures_compositional_score_modelling"
sys.path[:0] = [str(ABS), str(GM), str(COMPASS / "experiments_arruda_schur"), str(COMPASS / "src")]

from runtime import configure_runtime  # noqa: E402
sys.path.insert(0, str(HERE))
from variants import parse  # noqa: E402

OUT = HERE / "artifacts" / "toy"


def w1_exact(samples, grid, density):
    """W1 between draws and a gridded density, plus that density's sd."""
    import numpy as np
    dx = grid[1] - grid[0]
    cdf = np.cumsum(density) * dx
    cdf /= cdf[-1]
    empirical = np.searchsorted(np.sort(samples), grid, side="right") / len(samples)
    mean = np.trapezoid(grid * density, grid) / np.trapezoid(density, grid)
    sd = np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid) / np.trapezoid(density, grid))
    return float(np.sum(np.abs(empirical - cdf)) * dx), float(sd)


def load_network(run, mode, device):
    import common
    import methods
    import train as gm_train
    if mode.startswith("oracle_"):
        import oracle_abs
        return oracle_abs.oracle_model(mode.removeprefix("oracle_"), device,
                                       model=run.EXPERIMENT["model"])
    load = common.load_with_latent_attention(gm_train.load_to)
    directory = gm_train.model_directory(run.EXPERIMENT["root"] / mode, "h32d3wide",
                                         "tokenized", 200_000, "six_weighted", "log_sigma")
    network = load(directory / "Model_checkpoint.pt", device)
    import precondition
    config = precondition.load(network, directory, run.EXPERIMENT["model"])
    if config is not None:
        methods.COMMON["eps"] = config["eps"]
    return network


def draw(network, x, spec, seed, draws, device, run):
    import numpy as np
    import torch
    import methods
    from compass.MultiObsSampler import MultiObsSampler

    model = run.EXPERIMENT["model"]
    kwargs, warm = parse(spec)
    data = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(-1, 1), dtype=torch.float32)
    import model as M  # the GM node layout the toys share
    mask = torch.tensor([0.0] * len(M.LATENT_INDICES) + [1.0] * len(M.OBSERVED_INDICES))
    prior, local_prior = methods._priors(model)
    sampler = MultiObsSampler(network)
    common_kwargs = dict(method="dpm", order=2, correction="gauss_jacobian_FD", corrector_steps=0,
                         final_corrector_steps=0, terminal_corrector_steps=0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    extra, warm_rows = {}, 0
    if warm is not None:
        generator = torch.Generator(device=device).manual_seed(seed + 7919)
        init = sampler.laplace_initial_state(
            data, mask, draws, warm, hierarchy=list(M.HIERARCHY), prior=prior,
            local_prior=local_prior, denoise_clamp=5.0, generator=generator, device=device)
        warm_rows = sampler.laplace_diagnostics["row_samples"]  # already includes FD copies
        extra = {"start_lambda": warm, "init_state": init}
    samples = sampler.sample(
        world_size=1, data=data, condition_mask=mask, num_samples=draws,
        hierarchy=list(M.HIERARCHY), prior=prior, local_prior=local_prior,
        churn_eta=4.0, denoise_clamp=5.0, device=device, verbose=False,
        **methods.COMMON, **common_kwargs, **kwargs, **extra)
    seconds = time.perf_counter() - started
    stats = sampler.solver_stats
    cost = {"score_rows": stats["evaluated_row_samples"] - stats["jacobian_row_samples"],
            "jacobian_rows": stats["jacobian_row_samples"], "warm_rows": warm_rows,
            "refreshes": stats["jacobian_refreshes"],
            "secant_updates": stats["jacobian_secant_updates"],
            "secant_skipped": stats["jacobian_secant_skipped"],
            "levels": int(sampler.timesteps), "seconds": seconds}
    cost["total_rows"] = cost["score_rows"] + cost["jacobian_rows"] + warm_rows
    samples = samples.detach().cpu().numpy()
    return samples[0, :, 0], samples[:, :, 1].T, cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lean")
    parser.add_argument("--mode", default="oracle_joint")
    parser.add_argument("--variants", nargs="+", default=["r1"])
    parser.add_argument("--observations", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    import numpy as np
    import run
    run.select(arguments.model)
    configure_runtime(gpu=arguments.device.startswith("cuda")
                      and not os.environ.get("CUDA_VISIBLE_DEVICES"))
    network = load_network(run, arguments.mode, arguments.device)
    for parameter in network.model.parameters():
        parameter.requires_grad_(False)
    network.model.eval()

    out = OUT / arguments.model / (arguments.mode + arguments.tag)
    out.mkdir(parents=True, exist_ok=True)
    model = run.EXPERIMENT["model"]
    for n in arguments.observations:
        _, x = run.dataset(n)
        grid = np.linspace(-8, 8, 16001) * model.global_prior_std
        density = model.global_density(grid, x)
        exact_positive = float(np.trapezoid(density * (grid > 0), grid))
        for spec in arguments.variants:
            for seed in arguments.seeds:
                path = out / f"{spec.replace('/', 'o')}_n{n}_s{seed}.npz"
                if path.exists() and not arguments.force:
                    continue
                try:
                    g, l, cost = draw(network, x, spec, seed, arguments.draws, arguments.device, run)
                except Exception as error:  # noqa: BLE001
                    print(f"!! {spec} n={n} seed={seed}: {error!r}", flush=True)
                    continue
                w1, sd = w1_exact(g, grid, density)
                np.savez(path, g=g, l=l, cost=json.dumps(cost))
                record = {"model": arguments.model, "mode": arguments.mode, "variant": spec, "n": n,
                          "seed": seed, "w1_sd": w1 / sd, "p_pos": float(np.mean(g > 0)),
                          "p_pos_exact": exact_positive, "sd_ratio": float(g.std() / sd), **cost}
                with open(out / "records.jsonl", "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(f"{spec:<22} n={n:<3} s={seed} W1/sd {w1 / sd:.3f} P+ {record['p_pos']:.3f} "
                      f"(exact {exact_positive:.3f}) rows {cost['total_rows'] / 1e6:7.2f}M "
                      f"(jac {cost['jacobian_rows'] / 1e6:.2f}M, warm {cost['warm_rows'] / 1e6:.3f}M) "
                      f"refresh {cost['refreshes']} {cost['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
