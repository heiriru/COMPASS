#!/usr/bin/env python3
"""Per-conditional validation loss for a trained arm.

Every arm optimizes the denoising score-matching loss under *its own* condition
mask distribution, so validation losses from different mask menus are not
comparable as printed: a Bernoulli(0.33) arm and a six-conditional arm are being
scored on different objectives. This script removes that confound by evaluating
each checkpoint on a fixed set of conditionals, one at a time:

    p(g,l|x)  p(g|l,x)  p(l|g,x)  p(x|g,l)  p(x,l|g)  p(x,g|l)

``p(g,l|x)`` is the mask every posterior sampler runs at, and ``p(l|g,x)`` is
the one the hierarchical score ascent and ``newton_map_estimate`` run at, so
those two columns are the ones that predict recovery. The same validation rows,
the same diffusion times and the same noise draws are used for every arm and
every conditional, so the numbers differ only by the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_runtime  # noqa: E402  (before numeric imports)

import argparse  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "src"))

from compass.schema import SIX_CONDITIONALS, hierarchical_roles  # noqa: E402
from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.data_pipeline import load_split  # noqa: E402
from Partial_Pooling.model_pipeline import (  # noqa: E402
    GLOBAL_THETA_DIM, LOCAL_THETA_DIM, _raw_pair, load_model, load_normalizers,
    model_specs,
)
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402

NAME = "sde_joint"


def menu_masks(schema):
    """The six conditional masks as a ``(6, nodes_size)`` tensor."""
    roles = hierarchical_roles(schema, GLOBAL_THETA_DIM, LOCAL_THETA_DIM)
    masks = []
    for _, conditioned in SIX_CONDITIONALS:
        mask = torch.zeros(schema.nodes_size)
        for role in conditioned:
            mask[roles[role]] = 1.0
        masks.append(mask)
    return torch.stack(masks)


def conditional_losses(model, data, masks, repeats=16, seed=20250718,
                       device="cpu"):
    """Mean DSM loss per conditional, with shared times and noise across arms."""
    trainer = model.trainer
    trainer.sde = model.sde
    trainer.device = device
    trainer.model = model.model
    trainer.eps = 1e-3
    trainer.time_sampling = "mixture"
    trainer.mask_loss_weights = []

    data = data.to(device)
    rows = data.shape[0]
    results = []
    for mask in masks:
        mask = mask.to(device).expand(rows, -1)
        total = 0.0
        # Same seed for every arm and every conditional: the diffusion times and
        # the noise are a fixed evaluation grid, not a source of arm-to-arm noise.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(repeats):
            torch.manual_seed(int(torch.randint(
                0, 2**31 - 1, (1,), generator=generator).item()))
            with torch.no_grad():
                t = trainer._sample_timesteps(rows).to(device)
                x_1 = torch.randn_like(data) * (1 - mask) + data * mask
                x_t = model.forward_diffusion_sample(data, t, x_1, mask)
                score = trainer._get_score(x_t, t, mask)
                total += float(trainer.loss_fn(score, t, x_1, mask))
        results.append(total / repeats)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="LABEL:flag=value,flag=value overrides for get_config")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.device == "cuda":
        configure_runtime(3)
        device = "cuda"
    else:
        from runtime import configure_cpu_limit
        configure_cpu_limit(3)
        device = "cpu"

    paths = BenchmarkPaths.default().ensure()
    names = [name for name, _ in SIX_CONDITIONALS]
    records = []

    for entry in args.arm:
        label, _, overrides = entry.partition(":")
        kwargs = {}
        for item in filter(None, overrides.split(",")):
            key, _, value = item.partition("=")
            if value in ("True", "False"):
                value = value == "True"
            elif value.replace(".", "", 1).isdigit():
                value = float(value) if "." in value else int(value)
            kwargs[key] = value
        config = get_config(args.preset, **kwargs)
        spec = model_specs(config)[NAME]
        model = load_model(config, NAME, paths, device=device)
        model.model.to(device).eval()

        validation = load_split(config, "validation", paths)
        theta, x = _raw_pair(NAME, validation)
        normalizers = load_normalizers(config, paths)
        theta = normalizers[spec.theta_key].transform(theta)[:args.rows]
        x = normalizers[spec.x_key].transform(x)[:args.rows]
        data = torch.cat([theta, x], dim=1).float()

        losses = conditional_losses(
            model, data, menu_masks(model.schema), repeats=args.repeats,
            device=device,
        )
        parameters = sum(p.numel() for p in model.model.parameters())
        records.append({"label": label, "parameters": parameters,
                        "losses": dict(zip(names, losses))})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    width = max(len(r["label"]) for r in records) + 2
    print(f"{'arm':{width}s} {'params':>9s} " + " ".join(f"{n:>10s}" for n in names))
    for record in records:
        print(f"{record['label']:{width}s} {record['parameters']:9,d} " + " ".join(
            f"{record['losses'][n]:10.3f}" for n in names))

    if args.output:
        args.output.write_text(json.dumps(
            {"rows": args.rows, "repeats": args.repeats, "arms": records},
            indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
