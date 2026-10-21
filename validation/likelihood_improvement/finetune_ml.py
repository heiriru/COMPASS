"""Maximum-likelihood fine-tuning of the ONE COMPASS network, through the PF-ODE.

    .COMPASS/bin/python -m validation.likelihood_improvement.finetune_ml \
        --family bimodal --seed 0 [--source menu_edm] [--beta 1.0]

A score network's probability-flow ODE is a continuous normalising flow, and its log
density is exactly what COMPASS evaluates. Denoising score matching (DSM) fits the
score but not its divergence, while the PF-ODE density depends on both (Lu et al.
2022). The network is therefore fine-tuned on

    L = DSM(theta, x)                              # unchanged training objective
      + beta * ( -log p_PF(x | theta)              # likelihood direction
                 -log p_PF(theta | x) )            # posterior direction

on the *same network* that also samples the posterior:

- The DSM term is the exact training objective (log_sigma times, inference mask menu,
  EDM weights, Trainer.loss_fn). It keeps the score itself right for sampling.
- The two NLL terms use simulations only, no exact likelihood.
- Model selection uses held-out simulations from a separate stream: the held-out NLL
  of both directions (a proper scoring rule; better means a closer density), subject
  to the held-out DSM loss not degrading by more than `--dsm-tolerance`.
- Posterior sampling quality is graded separately by posterior_check.py.

Output: MODEL_ROOT/<family>_nu0.5/n65536_s<seed>_<source>_ml/final.pt + finetune.json
"""

import validation.common.runtime_gpu  # noqa: F401

import argparse
import copy
import json
import os
import time

import torch

from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm
from compass.schema import ConditionalMenu
from validation.b5_b6_pfode_accuracy.families import Family, THETA_DIM, X_DIM
from validation.b5_b6_pfode_accuracy.train import model_dir

LIK_MASK = torch.tensor([1.0] * THETA_DIM + [0.0] * X_DIM)      # theta given -> x latent
POST_MASK = 1 - LIK_MASK                                        # x given -> theta latent


def dsm_loss(model, data, masks, generator, device):
    """The training objective on one batch (sum over rows / rows)."""
    tr, sde = model.trainer, model.sde
    n = data.shape[0]
    one = torch.ones(1, device=device)
    lam_min, lam_max = sde.lambda_t(tr.eps * one), sde.lambda_t(one)
    u = torch.rand(n, 1, generator=generator).to(device)
    t = sde.time_of_lambda(lam_min * (lam_max / lam_min) ** u)
    noise = torch.randn(data.shape, generator=generator).to(device)
    x1 = noise * (1 - masks) + data * masks
    x_t = model.forward_diffusion_sample(data, t, x1, masks)
    out = model.model(x=x_t, t=t, c=masks)
    score = model.output_scale_function(t, out)
    return tr.loss_fn(score, t, x1, masks) / n


def nll(model, data, mask, timesteps, device, differentiable):
    # Training: one Hutchinson probe. log p is linear in the divergence integral, so
    # this is an unbiased estimate of the NLL and of its gradient, at a third of the
    # backward passes of the exact divergence. Held-out evaluation: exact.
    div = "hutchinson" if differentiable else "exact"
    lp = model.log_prob(data, mask, timesteps=timesteps, device=device,
                        differentiable=differentiable, divergence=div,
                        hutchinson_samples=1, batch_size=4096, endpoint="fixed")
    return -lp.mean()


def heldout(model, held, held_masks, device):
    g = torch.Generator().manual_seed(4242)            # fixed noise: comparable over steps
    with torch.no_grad():
        dsm = float(sum(dsm_loss(model, held[i:i + 512], held_masks[i:i + 512], g, device)
                        for i in range(0, len(held), 512)) / (len(held) // 512))
    lik = float(nll(model, held, LIK_MASK.to(device), 100, device, False))
    post = float(nll(model, held, POST_MASK.to(device), 100, device, False))
    return {"dsm": dsm, "nll_lik": lik, "nll_post": post}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default="menu_edm")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-timesteps", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--dsm-tolerance", type=float, default=0.02)
    parser.add_argument("--tag", default="ml")
    args = parser.parse_args()

    src = model_dir(args.family, 0.5, 65536, args.seed, args.source)
    out = model_dir(args.family, 0.5, 65536, args.seed, f"{args.source}_{args.tag}")
    if os.path.exists(os.path.join(out, "final.pt")):
        print(f"exists: {out}")
        return
    os.makedirs(out, exist_ok=True)
    device = "cuda"
    model = SBIm.load(os.path.join(src, "final.pt"), device=device)
    model.model.to(device)
    # The DSM term must be the source's own objective (train.py: EDM weights).
    model.trainer.time_loss_weight = "edm" if "edm" in args.source else None
    model.trainer.sigma_data = 1.0
    model.trainer.mask_loss_weights = []
    model.trainer.device = device
    model.trainer.eps = 1e-3                      # Trainer sets it only inside train()
    menu = ConditionalMenu(model.schema, roles={"t": [0, 1], "x": [2, 3, 4]},
                           patterns=[("p(theta|x)", "x"), ("p(x|theta)", "t")],
                           bernoulli_fraction=1 / 3)

    fam = Family(args.family, 0.5)
    theta, x = fam.simulate(65536, torch.Generator().manual_seed(1000 * args.seed + 17))
    th_h, x_h = fam.simulate(4096, torch.Generator().manual_seed(1000 * args.seed + 31337))
    train = torch.cat([theta, x], 1)
    held = torch.cat([th_h, x_h], 1).to(device)
    # Fixed, network-independent base density for the flow (PFODE endpoint="fixed").
    model.set_base_moments(train)
    # Validation-only guard, never used for selection: the exact held-out NLL. A
    # normalised density cannot beat it in expectation (Gibbs' inequality).
    sig = float(model.sde.marginal_prob_std(torch.tensor(1e-3)))
    exact_lik = float(-fam.log_lik(th_h, x_h, sig).mean())
    torch.manual_seed(77)
    held_masks = menu.sample(held.shape).to(device)

    opt = torch.optim.Adam(model.model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    model.model.eval()
    start = heldout(model, held, held_masks, device)
    history = [{"step": 0, **start, "exact_nll_lik_reference": exact_lik}]
    best_score = start["nll_lik"] + start["nll_post"]
    best_state, best_step = copy.deepcopy(model.model.state_dict()), 0
    print(history[-1], flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train), (args.batch,), generator=gen)
        batch = train[idx].to(device)
        masks = menu.sample(batch.shape).to(device)
        loss_dsm = dsm_loss(model, batch, masks, gen, device)
        loss_lik = nll(model, batch, LIK_MASK.to(device), args.train_timesteps, device, True)
        loss_post = nll(model, batch, POST_MASK.to(device), args.train_timesteps, device, True)
        loss = loss_dsm + args.beta * (loss_lik + loss_post)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            h = heldout(model, held, held_masks, device)
            h.update(step=step, seconds=time.time() - t0, train_dsm=float(loss_dsm),
                     train_nll_lik=float(loss_lik), train_nll_post=float(loss_post))
            history.append(h)
            print(h, flush=True)
            score = h["nll_lik"] + h["nll_post"]
            if score < best_score and h["dsm"] <= start["dsm"] * (1 + args.dsm_tolerance):
                best_score, best_step = score, step
                best_state = copy.deepcopy(model.model.state_dict())
    model.model.load_state_dict(best_state)
    model.save(out, name="final")
    with open(os.path.join(out, "finetune.json"), "w") as f:
        json.dump({"args": vars(args), "history": history, "best_step": best_step}, f, indent=1)
    print(f"done {out}: best step {best_step}; held-out NLL lik "
          f"{start['nll_lik']:.4f} -> {[h for h in history if h['step'] == best_step][0]['nll_lik']:.4f}")


if __name__ == "__main__":
    main()
