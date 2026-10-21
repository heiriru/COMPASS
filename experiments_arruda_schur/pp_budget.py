"""The 0.928 run (hierarchical + Rg + 1.5x-wide training prior) at the upstream simulation budget.

Upstream (BayesFlow) trains on 65,536 simulations in total; the 0.928 run used 100k for the
encoder plus 100k for the score net. Both arms here use ONE shared set of 65,536 wide-prior
simulations (plus 3,277 validation sets, the stock 1/20 ratio):

    data      simulate the shared set once (both arms read it; run before launching them)
    pretrain  arm A, step 1: the SetEncoder (setenc-d16-w64x2s4) on the held-out-trial NLL,
              on the shared set; then ``pp.py train --encoder <it> --train-subset 65536``
    joint     arm B: a fresh SetEncoder of the same architecture trained end to end with the
              C2 score net on the denoising loss alone (no predictive objective)

Joint training reproduces ``Trainer`` exactly (schedule-free AdamW at 1e-3, batch 256, the
six_mixed menu, log_sigma times, fixed validation masks, patience 20) except that ``z`` is
computed inside the step. Where ``z`` is diffused (the three p(x, ...) patterns and Bernoulli
rows) it is a regression target, so it is detached there: the encoder only learns from
coordinates where ``z`` conditions the parameters, and cannot shrink its own target.
The checkpoint is saved where ``pp.py infer --variant <variant> --encoder <its encoder>``
finds it; the saved encoder carries the z normalizer the net saw (identity from scratch).

Joint variants (``--variant``), all on the same shared set:
    joint_n65536    from scratch, Trainer's settings (lr 1e-3, no clipping, six_mixed menu)
    jointft_n65536  warm start: arm A's encoder (with its frozen z normalizer) AND score net,
                    then joint fine-tuning at lr 3e-4, clip 1.0, six_mixed menu. Starts at A,
                    so any change is what joint training adds.
    jointbf_n65536  from scratch, BayesFlow-style: lr 3e-4, clip 1.0, posterior-only menu
                    (p(g,l|z), p(g|l,z), p(l|g,z)) so every row sends gradient to the encoder.
                    Safe for inference: the samplers query only those patterns.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pp  # noqa: E402  (sets up paths and the CPU cap)

sys.modules.setdefault("runtime", pp.pp_runtime)  # pretrain_encoder's bare import: no autocvd search

import torch  # noqa: E402

from compass.SetEncoder import SetEncoder  # noqa: E402
from Partial_Pooling import pretrain_encoder as pe  # noqa: E402
from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.model_pipeline import load_normalizers  # noqa: E402
from Partial_Pooling.normalization import Normalizer  # noqa: E402
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402

N = 65_536
SCALE = 1.5
ENCODER = dict(item_dim=3, summary_dim=16, width=64, depth=2, num_heads=4, num_seeds=4)
OUT = HERE / "artifacts" / "pp" / "budget"
A_ENCODER = OUT / "encoders" / f"{pp.TAG}-wide{SCALE:g}-n{N}.pt"


def a_run(precondition):
    """Arm A's score-net checkpoint folder (the warm start) for the same precondition."""
    return pp.artifacts("hierarchical", precondition, SCALE, f"n{N}") / "checkpoints"
VARIANTS = {  # lr, gradient clip, menu, warm start from arm A
    f"joint_n{N}": dict(lr=1e-3, clip=0.0, menu="six_mixed", warm=False),
    f"jointft_n{N}": dict(lr=3e-4, clip=1.0, menu="six_mixed", warm=True),
    f"jointbf_n{N}": dict(lr=3e-4, clip=1.0, menu="posterior", warm=False),
}


def shared_sets():
    config = get_config("small", train_subset=N)
    rows = max(1, round(config.validation_size * N / config.train_size))
    return (pp.wide_split(config, SCALE, N, "train"), pp.wide_split(config, SCALE, rows, "validation"))


def save_encoder(path, encoder, item_norm, z_norm, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "config": ENCODER, "item_normalizer": item_norm.state_dict(),
                "z_normalizer": z_norm.state_dict(), **extra}, path)


def pretrain(arguments):
    """pretrain_encoder.main's loop on the shared set (same objective, defaults and selection)."""
    train, validation = shared_sets()
    train_raw = train["observations"][:, 0].reshape(N, -1, 3)
    val_raw = validation["observations"][:, 0].reshape(len(validation["observations"]), -1, 3)
    torch.manual_seed(arguments.seed)
    flat = train_raw.reshape(-1, 3)
    item_norm = Normalizer.fit(flat)
    train_items = item_norm.transform(flat).reshape(train_raw.shape)
    val_items = item_norm.transform(val_raw.reshape(-1, 3)).reshape(val_raw.shape)

    def targets_of(raw, standardized):
        out = raw.clone()
        out[..., pe.LOG_RT] = standardized[..., pe.LOG_RT]
        return out

    train_targets, val_targets = targets_of(train_raw, train_items), targets_of(val_raw, val_items)
    encoder = SetEncoder(**ENCODER).to("cuda")
    decoder = pe.TrialDecoder(ENCODER["summary_dim"]).to("cuda")
    print(f"pretrain: train {tuple(train_raw.shape)} validation {tuple(val_raw.shape)}; "
          f"encoder {sum(p.numel() for p in encoder.parameters()):,} parameters", flush=True)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    generator = torch.Generator(device="cpu").manual_seed(arguments.seed)
    best, best_state, waited, history = float("inf"), None, 0, []
    for epoch in range(1, 61):
        encoder.train(); decoder.train()
        order = torch.randperm(N, generator=generator)
        total, batches = 0.0, 0
        for start in range(0, N, 256):
            pick = order[start:start + 256]
            context, target = pe.split_trials(train_items[pick].to("cuda"), train_targets[pick].to("cuda"), generator)
            loss = pe.target_nll(decoder, encoder(context), target)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += float(loss); batches += 1
        encoder.eval(); decoder.eval()
        fixed = torch.Generator(device="cpu").manual_seed(arguments.seed + 1)
        with torch.no_grad():
            vtotal, vbatches = 0.0, 0
            for start in range(0, len(val_items), 256):
                context, target = pe.split_trials(val_items[start:start + 256].to("cuda"),
                                                  val_targets[start:start + 256].to("cuda"), fixed)
                vtotal += float(pe.target_nll(decoder, encoder(context), target)); vbatches += 1
        train_loss, val_loss = total / batches, vtotal / vbatches
        history.append({"epoch": epoch, "train": train_loss, "validation": val_loss})
        print(f"epoch {epoch:3d}  train {train_loss:9.4f}  validation {val_loss:9.4f}", flush=True)
        if val_loss < best:
            best, waited = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            waited += 1
            if waited >= 10:
                print(f"early stopping at epoch {epoch}", flush=True)
                break
    encoder.load_state_dict(best_state); encoder.eval()
    with torch.no_grad():
        z = torch.cat([encoder(train_items[s:s + 1024].to("cuda")).cpu() for s in range(0, N, 1024)])
    path = OUT / "encoders" / f"{pp.TAG}-wide{SCALE:g}-n{N}.pt"
    save_encoder(path, encoder, item_norm, Normalizer.fit(z),
                 {"objective": "held_out_trial_predictive", "train_subset": N, "train_prior_scale": SCALE,
                  "best_validation": best, "seed": arguments.seed})
    (path.with_suffix(".json")).write_text(json.dumps({"best_validation": best, "history": history}, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def joint(arguments):
    paths = BenchmarkPaths.default().ensure()
    config = get_config("small", train_subset=N)
    train, validation = shared_sets()
    tn = load_normalizers(config, paths)["sde_joint_theta"]
    theta = tn.transform(torch.cat((train["globals"], train["locals"][:, 0]), 1)).to("cuda")
    theta_val = tn.transform(torch.cat((validation["globals"], validation["locals"][:, 0]), 1)).to("cuda")
    raw, raw_val = train["observations"][:, 0], validation["observations"][:, 0]
    setting = VARIANTS[arguments.variant]
    torch.manual_seed(arguments.seed)
    encoder = SetEncoder(**ENCODER).to("cuda")
    if setting["warm"]:
        state = torch.load(A_ENCODER, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["encoder_state_dict"])
        item_norm = Normalizer.from_state_dict(state["item_normalizer"])
        z_norm = Normalizer.from_state_dict(state["z_normalizer"])
    else:
        item_norm = Normalizer.fit(raw.reshape(-1, 3))
        z_norm = Normalizer(torch.zeros(16), torch.ones(16), torch.zeros(16, dtype=torch.bool), "identity")
    z_mean, z_scale = z_norm.mean.float().to("cuda"), z_norm.scale.float().to("cuda")
    items = item_norm.transform(raw.reshape(-1, 3)).reshape(len(raw), -1, 3).to("cuda")
    items_val = item_norm.transform(raw_val.reshape(-1, 3)).reshape(len(raw_val), -1, 3).to("cuda")

    model, menu, out, name = pp.build_model(arguments, config, theta.cpu(), tn, ENCODER["summary_dim"])
    if setting["warm"]:
        model.model.load_state_dict(torch.load(a_run(arguments.precondition) / f"{name}_checkpoint.pt", map_location="cpu",
                                               weights_only=False)["model_state_dict"])
    if setting["menu"] == "posterior":
        from compass import ConditionalMenu
        menu = ConditionalMenu(model.schema, pp.ROLES, (("p(g,l|z)", "x"), ("p(g|l,z)", "lx"), ("p(l|g,z)", "gx")))
    trainer, network = model.trainer, model.model
    trainer.device, trainer.eps, trainer.time_sampling = "cuda", 1e-3, pp.C2["time_sampling"]
    params = list(encoder.parameters()) + list(network.parameters())
    import schedulefree
    optimizer = schedulefree.AdamWScheduleFree(params, lr=setting["lr"])
    x_nodes = slice(10, 10 + ENCODER["summary_dim"])
    print(f"joint {name}: score {sum(p.numel() for p in network.parameters()):,} + encoder "
          f"{sum(p.numel() for p in encoder.parameters()):,} parameters; train {len(theta)} validation "
          f"{len(theta_val)}; {arguments.variant} {setting}", flush=True)

    state = torch.random.get_rng_state()  # Trainer._fixed_validation_masks
    torch.manual_seed(20250718)
    val_masks = menu.sample((len(theta_val), model.nodes_size)).float().to("cuda")
    torch.random.set_rng_state(state)

    def batch_loss(th, it, c):
        z = (encoder(it) - z_mean) / z_scale
        cx = c[:, x_nodes]
        data = torch.cat((th, cx * z + (1 - cx) * z.detach()), 1)
        t = trainer._sample_timesteps(len(data))
        x_1 = torch.randn_like(data) * (1 - c) + data * c
        x_t = model.forward_diffusion_sample(data, t, x_1, c)
        score = model.output_scale_function(t, network(x=x_t, t=t, c=c))
        return trainer.loss_fn(score, t, x_1, c)

    checkpoint_name = f"{name}_checkpoint"
    best, waited, history, started = float("inf"), 0, [], time.time()
    for epoch in range(1, pp.C2["epochs"] + 1):
        encoder.train(); network.train(); optimizer.train()
        order = torch.randperm(len(theta), device="cuda")
        total, batches = 0.0, 0
        for start in range(0, len(order), pp.C2["batch_size"]):
            pick = order[start:start + pp.C2["batch_size"]]
            c = menu.sample((len(pick), model.nodes_size)).float().to("cuda")
            optimizer.zero_grad()
            loss = batch_loss(theta[pick], items[pick], c)
            loss.backward()
            if setting["clip"]:
                torch.nn.utils.clip_grad_norm_(params, setting["clip"])
            optimizer.step()
            total += loss.item() / len(pick); batches += 1
        encoder.eval(); network.eval(); optimizer.eval()
        with torch.no_grad():
            vtotal, vbatches = 0.0, 0
            for start in range(0, len(theta_val), pp.C2["batch_size"]):
                s = slice(start, start + pp.C2["batch_size"])
                vtotal += batch_loss(theta_val[s], items_val[s], val_masks[s]).item() / len(theta_val[s])
                vbatches += 1
        train_loss, val_loss = total / batches, vtotal / vbatches
        history.append({"epoch": epoch, "train": train_loss, "validation": val_loss})
        print(f"--- Epoch: {epoch:3d} --- Training Loss: {train_loss:8.3f} --- Validation Loss: {val_loss:8.3f} ---",
              flush=True)
        if val_loss < best:  # saved in optimizer.eval() mode: the schedule-free averaged weights
            best, waited = val_loss, 0
            model.save(path=str(out), name=checkpoint_name)
            save_encoder(out / "encoder.pt", encoder, item_norm, z_norm,
                         {"objective": "joint_denoising_score", "train_subset": N, "train_prior_scale": SCALE,
                          "epoch": epoch, "best_validation": best, "seed": arguments.seed})
        else:
            waited += 1
            if waited >= pp.C2["patience"]:
                print(f"early stopping at epoch {epoch}", flush=True)
                break
    (out / f"{name}.json").write_text(json.dumps({
        "recipe": {**pp.C2, "train_subset": N}, "mode": arguments.mode, "variant": arguments.variant,
        "precondition": arguments.precondition, "train_prior_scale": SCALE, "encoder": str(out / "encoder.pt"),
        "objective": "joint end-to-end; z detached where diffused", "setting": setting, "seed": arguments.seed,
        "history": history, "best_validation": best, "minutes": (time.time() - started) / 60,
    }, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("data", "pretrain", "joint"))
    parser.add_argument("--seed", type=int, default=None, help="default: 20250718 pretrain, 7 joint")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=f"joint_n{N}", help="joint only")
    parser.add_argument("--precondition", choices=("Rg", ""), default="Rg",
                        help="joint only: '' = the plain net, no residual on the globals")
    arguments = parser.parse_args()
    if arguments.action == "data":
        train, validation = shared_sets()
        print(f"shared set: train {tuple(train['observations'].shape)} validation "
              f"{tuple(validation['observations'].shape)}", flush=True)
    elif arguments.action == "pretrain":
        arguments.seed = 20250718 if arguments.seed is None else arguments.seed
        pretrain(arguments)
    else:
        arguments.seed = 7 if arguments.seed is None else arguments.seed
        arguments.mode, arguments.train_prior_scale = "hierarchical", SCALE
        joint(arguments)


if __name__ == "__main__":
    main()
