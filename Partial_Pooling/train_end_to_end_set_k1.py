#!/usr/bin/env python3
"""Train a K=1 SetEncoder jointly with the posterior score network.

Each training row contains *one* subject's 30 trials.  The encoder is called
inside the denoising-score objective and is therefore optimized for posterior
inference, rather than frozen after a predictive pretext task.  The encoder
never receives a group of subjects; cross-subject information is introduced
only later by the existing compositional samplers.

Unlike ``train_encoder_summary.py``, the learned summary is always conditioned:
the model trains only p(g,l|z), p(g|l,z), and p(l|g,z).  It never diffuses or
generates z, so all encoder gradients serve the posterior task we deploy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from compass import Atomic, ConditionalMenu, ScoreBasedInferenceModel
from compass.SetEncoder import SetEncoder
from Partial_Pooling.config import get_config
from Partial_Pooling.data_pipeline import load_split
from Partial_Pooling.model_pipeline import load_normalizers
from Partial_Pooling.normalization import Normalizer
from Partial_Pooling.paths import BenchmarkPaths


TAG = "set-k1-e2e-d64-w64x2s4-h48d2-log_sigma"
GLOBAL, LOCAL, TRIAL_WIDTH, TRIALS = 7, 3, 3, 30
POSTERIOR_PATTERNS = (
    ("p(g,l|z)", "x"),
    ("p(g|l,z)", "lx"),
    ("p(l|g,z)", "gx"),
)


def sample_times(model, batch_size, device):
    """The existing log-sigma distribution, without instantiating Trainer."""
    eps = 1e-3
    lam_max = model.sde.lambda_t(torch.ones(1, device=device))
    lam_min = model.sde.lambda_t(torch.full((1,), eps, device=device))
    u = torch.rand(batch_size, 1, device=device)
    return model.sde.time_of_lambda(lam_min * (lam_max / lam_min) ** u)


def posterior_mask(menu, batch_size, device, dtype):
    return menu.sample((batch_size, menu.schema.nodes_size)).to(device=device, dtype=dtype)


def encode(encoder, items, item_normalizer):
    mean = item_normalizer.mean.to(device=items.device, dtype=items.dtype)
    scale = item_normalizer.scale.to(device=items.device, dtype=items.dtype)
    trials = items.reshape(-1, TRIALS, TRIAL_WIDTH)
    return encoder((trials - mean) / scale)


def score_loss(model, encoder, theta, items, item_normalizer, menu):
    z = encode(encoder, items, item_normalizer)
    data = torch.cat((theta, z), dim=1)
    mask = posterior_mask(menu, len(data), data.device, data.dtype)
    noise = torch.randn_like(data) * (1.0 - mask)
    time_ = sample_times(model, len(data), data.device)
    state = model.forward_diffusion_sample(data, time_, noise, mask)
    score = model.model(state, time_, mask)
    # The standard loss ignores conditioned summary coordinates exactly; its
    # gradient therefore reaches E_phi only through the parameter score.
    return model.trainer.loss_fn(score, time_, noise, mask)


def evaluate(model, encoder, theta, items, item_normalizer, masks, batch_size):
    model.model.eval(); encoder.eval()
    total, rows = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(theta), batch_size):
            stop = min(start + batch_size, len(theta))
            z = encode(encoder, items[start:stop], item_normalizer)
            data = torch.cat((theta[start:stop], z), dim=1)
            mask = masks[start:stop].to(device=data.device, dtype=data.dtype)
            noise = torch.randn_like(data) * (1.0 - mask)
            time_ = sample_times(model, len(data), data.device)
            state = model.forward_diffusion_sample(data, time_, noise, mask)
            score = model.model(state, time_, mask)
            total += float(model.trainer.loss_fn(score, time_, noise, mask))
            rows += len(data)
    return total / max(rows, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-subset", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--summary-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20250909)
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    paths = BenchmarkPaths.default().ensure()
    config = get_config("small", train_subset=args.train_subset, arch="tokenized",
                        tokenization="trial", hidden_size=48, depth=2,
                        num_heads=4, mlp_ratio=4, time_sampling="log_sigma")
    train = load_split(config, "train", paths)["sde"]
    validation = load_split(config, "validation", paths)["sde"]
    theta_normalizer = load_normalizers(config, paths)["sde_joint_theta"]
    train_theta = theta_normalizer.transform(torch.cat((train["globals"], train["locals"][:, 0]), 1))[:args.train_subset]
    train_items_raw = train["observations"][:args.train_subset, 0]
    val_rows = max(1, round(config.validation_size * args.train_subset / config.train_size))
    val_theta = theta_normalizer.transform(torch.cat((validation["globals"], validation["locals"][:, 0]), 1))[:val_rows]
    val_items_raw = validation["observations"][:val_rows, 0]
    item_normalizer = Normalizer.fit(train_items_raw.reshape(-1, TRIAL_WIDTH))

    encoder = SetEncoder(item_dim=TRIAL_WIDTH, summary_dim=args.summary_dim,
                         width=64, depth=2, num_heads=4, num_seeds=4).to(device)
    model = ScoreBasedInferenceModel(
        theta_size=GLOBAL + LOCAL, x_spec=[Atomic("summary", args.summary_dim)],
        arch="tokenized", trunk="shared", time_embedding_size=256,
        sde_type=config.sde_type, sigma=config.sigma, hidden_size=48, depth=2,
        num_heads=4, mlp_ratio=4, device=str(device),
    )
    roles = {
        "g": range(GLOBAL),
        "l": range(GLOBAL, GLOBAL + LOCAL),
        "x": range(GLOBAL + LOCAL, model.nodes_size),
    }
    menu = ConditionalMenu(model.schema, roles, POSTERIOR_PATTERNS,
                           weights=[6.0, 6.0, 6.0])
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(model.model.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 1)
    # Fixed validation masks make early stopping compare the same three tasks.
    pick = torch.arange(len(val_theta)) % len(POSTERIOR_PATTERNS)
    val_masks = menu.masks[pick]
    train_theta, train_items_raw = train_theta.to(device), train_items_raw.to(device)
    val_theta, val_items_raw = val_theta.to(device), val_items_raw.to(device)

    out = paths.data.parent / "end_to_end_set_k1" / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"{TAG}_checkpoint.pt"
    manifest = out / f"{TAG}.json"
    best, waited, history = float("inf"), 0, []
    started = time.perf_counter()
    print(f"K=1 end-to-end model: encoder={sum(p.numel() for p in encoder.parameters()):,}, "
          f"score={sum(p.numel() for p in model.model.parameters()):,}, nodes={model.nodes_size}", flush=True)
    for epoch in range(1, args.epochs + 1):
        encoder.train(); model.model.train()
        order = torch.randperm(len(train_theta), generator=generator)
        total, rows = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = score_loss(model, encoder, train_theta[index], train_items_raw[index], item_normalizer, menu)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(model.model.parameters()), 1.0)
            optimizer.step()
            total += float(loss); rows += len(index)
        train_loss = total / max(rows, 1)
        val_loss = evaluate(model, encoder, val_theta, val_items_raw, item_normalizer,
                            val_masks, args.batch_size)
        history.append({"epoch": epoch, "train": train_loss, "validation": val_loss})
        print(f"epoch {epoch:3d}  train {train_loss:9.5f}  validation {val_loss:9.5f}", flush=True)
        if val_loss < best:
            best, waited = val_loss, 0
            torch.save({
                "representation": "end_to_end_k1_conditional_set_encoder",
                "encoder_config": {"item_dim": TRIAL_WIDTH, "summary_dim": args.summary_dim,
                                   "width": 64, "depth": 2, "num_heads": 4, "num_seeds": 4},
                "encoder_state_dict": encoder.state_dict(),
                "score_state_dict": model.model.state_dict(),
                "score_architecture": model.architecture_payload(),
                "item_normalizer": item_normalizer.state_dict(),
                "theta_normalizer": theta_normalizer.state_dict(),
                "best_validation": best, "epoch": epoch, "seed": args.seed,
            }, checkpoint)
        else:
            waited += 1
            if waited >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break
    payload = {
        "representation": "end_to_end_k1_conditional_set_encoder",
        "objective": "posterior_score_only", "summary_is_diffused": False,
        "k1_only": True, "posterior_patterns": [name for name, _ in POSTERIOR_PATTERNS],
        "checkpoint": str(checkpoint), "best_validation": best,
        "epochs": len(history), "history": history, "train_subset": args.train_subset,
        "parameters": {"encoder": sum(p.numel() for p in encoder.parameters()),
                       "score": sum(p.numel() for p in model.model.parameters())},
        "elapsed_minutes": (time.perf_counter() - started) / 60,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {checkpoint} and {manifest}", flush=True)


if __name__ == "__main__":
    main()
