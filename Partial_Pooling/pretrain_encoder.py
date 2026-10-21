#!/usr/bin/env python3
"""Pretrain the frozen set encoder on a held-out-trial predictive objective.

The encoder compresses one subject's trials to a fixed vector ``z``. COMPASS
then models ``p(theta, z)`` over a 26-node vector instead of ``p(theta, x)`` over
100, and the encoder is **frozen** -- which is the whole point: a single shared
representation that does not have to be retrained when a new simulation model is
added.

**Why a predictive objective and not an autoencoder.** An autoencoder is free to
memorise the particular trials it was shown; nothing pushes it to represent the
*distribution* those trials came from. Here the trials of one subject are split
into a context set and a held-out target set, ``z`` is computed from the context
alone, and the loss is the negative log-likelihood of the *target* trials under a
decoder that sees only ``z``. To score held-out trials well, ``z`` has to carry
the subject's trial distribution -- which is exactly the sufficient statistic the
posterior needs.

The context size is drawn per batch rather than fixed, so ``z`` stays meaningful
for a subject with fewer trials than the 30 this benchmark happens to use, and so
the encoder's ``log n`` feature is actually exercised.

**Decoder.** Matched to what the simulator produces rather than generic:
``choice`` is Bernoulli, ``log RT`` is a Gaussian mixture *conditioned on the
choice* (a DDM's two boundaries have genuinely different RT distributions, and a
single Gaussian cannot represent the skew), and ``censored`` is dropped -- it is
identically zero in this benchmark, so modelling it would only add a constant.
The decoder is thrown away after pretraining; only the encoder is kept.

Run:  python Partial_Pooling/pretrain_encoder.py [--epochs N] [--summary-dim D]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_runtime  # noqa: E402  (before numeric imports)

CPU_LIMIT = configure_runtime(max_threads=3)   # caps CPUs and reserves one free GPU

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from compass.SetEncoder import SetEncoder  # noqa: E402

from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.data_pipeline import load_split  # noqa: E402
from Partial_Pooling.io_utils import atomic_json  # noqa: E402
from Partial_Pooling.normalization import Normalizer  # noqa: E402
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402

ITEM_DIM = 3          # (choice, log RT, censored)
CHOICE, LOG_RT = 0, 1
MIXTURE = 3           # Gaussian components per choice branch


class TrialDecoder(nn.Module):
    """``z -> p(choice) and p(log RT | choice)``, a mixture per branch."""

    def __init__(self, summary_dim, width=128, mixture=MIXTURE):
        super().__init__()
        self.mixture = mixture
        self.trunk = nn.Sequential(
            nn.Linear(summary_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.choice_logit = nn.Linear(width, 1)
        # Two branches (choice 0 / choice 1), each a mixture over log RT.
        self.rt_head = nn.Linear(width, 2 * mixture * 3)

    def forward(self, z):
        h = self.trunk(z)
        logits = self.rt_head(h).reshape(z.shape[0], 2, self.mixture, 3)
        return (
            self.choice_logit(h).squeeze(-1),
            logits[..., 0],                                   # mixture logits
            logits[..., 1],                                   # means
            # exp(-6) = 0.0025 lets one component collapse onto a single
            # trial and take the log-density with it; -3 (sd 0.05) is well
            # below the data's own 0.45 spread and stays finite.
            logits[..., 2].clamp(-3.0, 3.0),                  # log sds
        )


def target_nll(decoder, z, targets):
    """Negative log-likelihood of held-out trials, per subject.

    ``targets`` is ``(batch, n_target, ITEM_DIM)`` with the *standardized*
    log RT in column 1 and the raw 0/1 choice in column 0.
    """
    choice_logit, weight_logit, mean, log_sd = decoder(z)
    choice = targets[..., CHOICE]
    log_rt = targets[..., LOG_RT]
    # Cheap and worth it: an out-of-range choice makes the BCE term negative and
    # the whole objective unbounded below, which looks like spectacular training
    # progress rather than an error.
    if not bool(((choice == 0) | (choice == 1)).all()):
        raise ValueError(
            "choice targets must be raw 0/1; got values outside {0, 1}. "
            "Standardized items are for the encoder, not the decoder targets."
        )

    choice_ll = -nn.functional.binary_cross_entropy_with_logits(
        choice_logit.unsqueeze(1).expand_as(choice), choice, reduction="none",
    )
    # Both choice branches are evaluated and the matching one selected. A gather
    # would save half the mixture arithmetic and cost far more clarity than the
    # three components are worth.
    # (B, 1, 2, M) against (B, T, 1, 1) broadcasts to (B, T, 2, M): every target
    # trial scored under every branch and component.
    log_weight = torch.log_softmax(weight_logit, dim=-1)[:, None]
    normal = torch.distributions.Normal(mean[:, None], log_sd[:, None].exp())
    component = normal.log_prob(log_rt[..., None, None]) + log_weight
    rt_ll_both = torch.logsumexp(component, dim=-1)                 # (B, T, 2)
    rt_ll = torch.where(choice > 0.5, rt_ll_both[..., 1], rt_ll_both[..., 0])
    return -(choice_ll + rt_ll).sum(dim=-1).mean()


def split_trials(encoder_items, target_items, generator, low=5, high=25):
    """Random context/target split, applied identically to both views.

    Two views of the same trials are needed, and conflating them is a silent
    disaster. The **encoder** sees standardized items, because that is what
    keeps its inputs well scaled. The **decoder targets** must keep ``choice``
    as raw 0/1: ``binary_cross_entropy_with_logits`` is only non-negative for a
    target in [0, 1], and feeding it a standardized -1.25/+0.80 makes the loss
    unbounded below -- measured, it dove to -5e9 while the model got worse.
    """
    trials = encoder_items.shape[1]
    n_context = int(torch.randint(low, min(high, trials - 1) + 1, (1,),
                                  generator=generator).item())
    # Drawn on CPU so the split is reproducible from a CPU generator regardless
    # of where the batch lives; indexing moves it.
    order = torch.randperm(trials, generator=generator).to(encoder_items.device)
    context, target = order[:n_context], order[n_context:]
    return encoder_items[:, context], target_items[:, target]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pretrain the frozen set encoder")
    parser.add_argument("--summary-dim", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-subset", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20250718)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    paths = BenchmarkPaths.default().ensure()
    config = get_config("small", train_subset=args.train_subset, arch="tokenized",
                        tokenization="trial", hidden_size=64, depth=2,
                        num_heads=8, mlp_ratio=4)

    def items_of(split, limit=None):
        blob = load_split(config, split, paths)
        raw = blob["sde"]["observations"][:, 0]
        if limit:
            raw = raw[:limit]
        return raw.reshape(raw.shape[0], -1, ITEM_DIM)

    train_raw = items_of("train", args.train_subset)
    val_raw = items_of("validation", max(1, args.train_subset // 20))
    print(f"train {tuple(train_raw.shape)}  validation {tuple(val_raw.shape)}",
          flush=True)

    # Standardize the items the encoder sees. Fitted on the training split only,
    # and saved, because inference has to apply exactly these constants.
    flat = train_raw.reshape(-1, ITEM_DIM)
    item_norm = Normalizer.fit(flat)
    train_items = item_norm.transform(flat).reshape(train_raw.shape)
    val_items = item_norm.transform(
        val_raw.reshape(-1, ITEM_DIM)).reshape(val_raw.shape)
    # Decoder targets: raw choice (Bernoulli needs 0/1) with the standardized
    # log RT (the Gaussian mixture is fitted in standardized units).
    def targets_of(raw, standardized):
        out = raw.clone()
        out[..., LOG_RT] = standardized[..., LOG_RT]
        return out
    train_targets = targets_of(train_raw, train_items)
    val_targets = targets_of(val_raw, val_items)
    print(f"item normalization  mean {item_norm.mean.tolist()}  "
          f"scale {item_norm.scale.tolist()}  degenerate {item_norm.degenerate.tolist()}",
          flush=True)

    encoder = SetEncoder(
        item_dim=ITEM_DIM, summary_dim=args.summary_dim, width=args.width,
        depth=args.depth, num_heads=args.num_heads, num_seeds=args.num_seeds,
    ).to(device)
    decoder = TrialDecoder(args.summary_dim).to(device)
    parameters = sum(p.numel() for p in encoder.parameters())
    print(f"encoder {parameters:,} parameters, decoder "
          f"{sum(p.numel() for p in decoder.parameters()):,} (discarded after)",
          flush=True)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best, best_state, waited = float("inf"), None, 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        encoder.train(); decoder.train()
        order = torch.randperm(train_items.shape[0], generator=generator)
        total, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            pick = order[start:start + args.batch_size]
            context, target = split_trials(
                train_items[pick].to(device), train_targets[pick].to(device),
                generator,
            )
            loss = target_nll(decoder, encoder(context), target)
            optimizer.zero_grad(); loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    args.grad_clip,
                )
            optimizer.step()
            total += float(loss); batches += 1
        train_loss = total / max(batches, 1)

        # Validation on a *fixed* split, so the score is the same task each epoch
        # -- the trap docs/group_block_progress.md documents for the diffusion
        # trainer, not repeated here.
        encoder.eval(); decoder.eval()
        fixed = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        with torch.no_grad():
            total, batches = 0.0, 0
            for start in range(0, val_items.shape[0], args.batch_size):
                stop = start + args.batch_size
                context, target = split_trials(
                    val_items[start:stop].to(device),
                    val_targets[start:stop].to(device), fixed,
                )
                total += float(target_nll(decoder, encoder(context), target))
                batches += 1
            val_loss = total / max(batches, 1)

        history.append({"epoch": epoch, "train": train_loss, "validation": val_loss})
        print(f"epoch {epoch:3d}  train {train_loss:9.4f}  validation {val_loss:9.4f}",
              flush=True)
        if val_loss < best:
            best, waited = val_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in encoder.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    encoder.load_state_dict(best_state)
    encoder.eval()

    # z-normalization over the training split, with the frozen encoder, so the
    # COMPASS node vector sees a standardized summary.
    with torch.no_grad():
        chunks = []
        for start in range(0, train_items.shape[0], 1024):
            chunks.append(encoder(train_items[start:start + 1024].to(device)).cpu())
        z = torch.cat(chunks)
    z_norm = Normalizer.fit(z)
    print(f"z over the training split: mean |z| {z.abs().mean():.3f}  "
          f"sd range {z.std(0).min():.3f}-{z.std(0).max():.3f}", flush=True)

    out = paths.data.parent / "encoders"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"setenc-d{args.summary_dim}-w{args.width}x{args.depth}s{args.num_seeds}"
    torch.save({
        "encoder_state_dict": best_state,
        "config": {
            "item_dim": ITEM_DIM, "summary_dim": args.summary_dim,
            "width": args.width, "depth": args.depth,
            "num_heads": args.num_heads, "num_seeds": args.num_seeds,
        },
        "item_normalizer": item_norm.state_dict(),
        "z_normalizer": z_norm.state_dict(),
        "objective": "held_out_trial_predictive",
        "train_subset": args.train_subset,
        "best_validation": best,
        "seed": args.seed,
    }, out / f"{tag}.pt")
    atomic_json(out / f"{tag}.json", {
        "tag": tag, "parameters": parameters, "best_validation": best,
        "epochs": len(history), "history": history,
        "objective": "held_out_trial_predictive",
        "train_subset": args.train_subset, "seed": args.seed,
        "cpu_limit": CPU_LIMIT,
    })
    print(f"\nWrote {out / (tag + '.pt')}  ({(time.perf_counter()-started)/60:.1f} min)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
