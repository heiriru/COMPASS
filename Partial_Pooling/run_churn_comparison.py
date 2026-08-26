#!/usr/bin/env python3
"""Churned (stochastic-predictor) DPM-Solver-2 on the partial-pooling benchmark.

Runs ``infer_partial_pooling.py`` unmodified, with one method swapped on
``MultiObsSampler`` for the duration of the process: the DPM sampler's
probability-flow predictor is replaced by its stochastic counterpart.

The churn step, and why it is not "PF-ODE plus noise"
-----------------------------------------------------
In noise-scale coordinates the VE probability-flow ODE and the reverse-time SDE
are

    PF-ODE      dx = -  sigma s(x, sigma) dsigma
    reverse SDE dx = -2 sigma s(x, sigma) dsigma + sqrt(2 sigma |dsigma|) dw

so the SDE carries *twice* the drift as well as the noise. Churn reproduces both
at once by stepping **up** in noise and then integrating the ODE down a
correspondingly longer arc (Karras et al. 2022, Alg. 2):

    sigma_hat = sigma_i + eta (sigma_i - sigma_{i+1})
    x        <- x + sqrt(sigma_hat^2 - sigma_i^2) * noise
    x        <- DPM2(x, sigma_hat -> sigma_{i+1})

The arc is ``(1 + eta)`` times the plain step, so ``eta = 1`` gives the doubled
drift and the matching noise variance ``2 sigma |dsigma|`` -- the exact reverse
SDE. ``eta > 1`` is the Karras eq. (6) family: the excess drift
``eta sigma |dsigma| s`` against injected variance ``2 eta sigma |dsigma|`` is
Langevin-balanced, so it leaves the marginals invariant for an exact score.

Cost: **no extra score evaluations**. The relaxation rides on the second-order
step the predictor was taking anyway, unlike a Langevin corrector, which buys the
same relaxation with a full compositional evaluation per iteration.

Isolation
---------
Nothing in ``compass``, ``infer_partial_pooling.py`` or any other benchmark
module is edited. ``MultiObsSampler._dpm_sampler`` is swapped for the duration of
one run and restored afterwards, and results are written under a separate
artifact root whose ``data`` and ``checkpoints`` are symlinks to the real ones --
so existing recovery artifacts cannot be reused, overwritten, or confused with
these.

Usage:
    python Partial_Pooling/run_churn_comparison.py \
        --preset small --inference-method dpm2_gauss_jacobian_newton \
        --churn-eta 2 --datasets 5 --subjects 20 --timesteps 50 --draws 4096

    # the reference arm, unpatched (eta = 0 leaves the stock sampler in place)
    python Partial_Pooling/run_churn_comparison.py \
        --preset small --inference-method langevin_fnpse --churn-eta 0 ...
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit(3)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import infer_partial_pooling as infer  # noqa: E402

ROOT = Path(__file__).resolve().parent
REAL_ARTIFACTS = ROOT / "artifacts"

# Karras et al. cap the per-step noise-up ratio for stability.
GAMMA_CAP = math.sqrt(2.0) - 1.0


def churned_dpm_sampler(eta, s_noise=1.0):
    """Build a ``_dpm_sampler`` replacement: noise up by ``eta``, then DPM2 down.

    Corrector requests are refused rather than ignored: this sampler exists to
    answer a question about a predictor, and silently dropping a corrector would
    make its answers unreadable.
    """
    import torch

    def sampler(self, data, condition_mask, idx, order=2, snr=0.1,
                corrector_steps_interval=5, corrector_steps=5,
                final_corrector_steps=3, terminal_corrector_steps=0):
        if corrector_steps or final_corrector_steps or terminal_corrector_steps:
            raise ValueError(
                "churned_dpm_sampler is a corrector-free predictor; got "
                f"corrector_steps={corrector_steps}, "
                f"final_corrector_steps={final_corrector_steps}, "
                f"terminal_corrector_steps={terminal_corrector_steps}. "
                "Pass --dpm-corrector-steps 0."
            )
        step = {1: self._dpm_solver_1_step, 2: self._dpm_solver_2_step,
                3: self._dpm_solver_3_step}[order]
        latent = 1 - condition_mask
        ceiling = float(self.sde.lambda_t(torch.ones(1, device=data.device)))

        if self.save_trajectory:
            self.data_t = torch.zeros(data.shape[0], self.timesteps,
                                      data.shape[1], data.shape[2])
            self.data_t[:, 0, :, :] = data

        for i in range(self.timesteps - 1):
            t_now = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i + 1].reshape(-1, 1)
            sigma_now = float(self.sde.lambda_t(t_now).reshape(-1)[0])
            sigma_next = float(self.sde.lambda_t(t_next).reshape(-1)[0])

            t_start = t_now
            if eta > 0.0:
                gamma = min(eta * (sigma_now - sigma_next) / sigma_now, GAMMA_CAP)
                sigma_hat = min(sigma_now * (1.0 + gamma), ceiling)
                spread = math.sqrt(max(sigma_hat**2 - sigma_now**2, 0.0))
                if spread > 0.0:
                    # _shared_noise keeps the hierarchy coordinates identical
                    # across observation rows, as everywhere else in the sampler.
                    data = data + s_noise * spread * self._shared_noise(data) * latent
                    t_start = self.sde.time_of_lambda(
                        torch.as_tensor(sigma_hat, dtype=t_now.dtype,
                                        device=t_now.device)
                    ).reshape(1, 1)

            data = step(data, t_start, t_next, condition_mask, idx)

            if self.save_trajectory:
                self.data_t[:, i + 1] = data

        return data.detach()

    return sampler


@contextlib.contextmanager
def patched_dpm_sampler(builder):
    """Swap ``MultiObsSampler._dpm_sampler`` for one run, then restore it."""
    from compass.MultiObsSampler import MultiObsSampler
    original = MultiObsSampler._dpm_sampler
    MultiObsSampler._dpm_sampler = builder
    try:
        yield
    finally:
        MultiObsSampler._dpm_sampler = original


@contextlib.contextmanager
def forced_jacobian_refresh(refresh):
    """Force ``sample(jacobian_refresh=...)``, which the CLI does not expose.

    ``correction="gauss_jacobian"`` rebuilds the per-observation curvature with
    one forward-mode JVP per latent coordinate -- ten here -- and each runs under
    the MATH attention backend because the fused kernels have no forward-mode AD
    rule. That is the whole 23x gap against ``gauss_hierarchical`` on this model.

    ``jacobian_refresh`` reuses the curvature across consecutive evaluations, and
    the cache is built for it: only the *lambda-free* information
    ``Lambda_j(t) - lambda^-2 I`` is held, while the exact lambda-dependent part
    is rebuilt every time. So a lagged refresh lags only the network's opinion
    about ``Sigma_0,j`` -- the very quantity ``gauss_hierarchical`` freezes for
    the entire trajectory -- rather than mismatching noise scales.

    ``infer_partial_pooling.py`` never passes the argument, so every previous run
    used the default of 1. Wrapping ``sample`` injects it without editing either
    ``compass`` or the benchmark script.
    """
    from compass.MultiObsSampler import MultiObsSampler
    original = MultiObsSampler.sample

    def sample(self, *args, **kwargs):
        kwargs["jacobian_refresh"] = int(refresh)
        return original(self, *args, **kwargs)

    MultiObsSampler.sample = sample
    try:
        yield
    finally:
        MultiObsSampler.sample = original


def scratch_root(base, eta, refresh=1):
    """An artifact root that shares data and checkpoints but nothing else.

    ``data`` and ``checkpoints`` are symlinked to the real tree so the 400,000
    simulations and the 5.96M-parameter checkpoint are reused rather than copied;
    every output directory is fresh, so no existing recovery artifact can be
    reused, overwritten, or mistaken for a churned one.
    """
    suffix = f"artifacts_churn_eta{eta:g}"
    if int(refresh) != 1:
        suffix += f"_refresh{int(refresh)}"
    root = Path(base) / suffix
    root.mkdir(parents=True, exist_ok=True)
    for name in ("data", "checkpoints"):
        link = root / name
        if not link.exists():
            link.symlink_to(REAL_ARTIFACTS / name, target_is_directory=True)
    return root


def main(argv=None):
    ahead = argparse.ArgumentParser(add_help=False)
    ahead.add_argument("--churn-eta", type=float, default=2.0)
    ahead.add_argument("--jacobian-refresh", type=int, default=1)
    ahead.add_argument("--scratch-base", type=Path, default=ROOT)
    known, rest = ahead.parse_known_args(argv)

    if known.jacobian_refresh < 1:
        raise SystemExit("--jacobian-refresh must be at least 1.")
    root = scratch_root(known.scratch_base, known.churn_eta,
                        known.jacobian_refresh)
    forwarded = list(rest) + ["--root", str(root)]
    args = infer.parser().parse_args(forwarded)

    if args.dpm_corrector_steps:
        raise SystemExit(
            "--dpm-corrector-steps must be 0: this script measures a "
            "corrector-free stochastic predictor."
        )
    if known.churn_eta and args.inference_method == "langevin_fnpse":
        raise SystemExit(
            "langevin_fnpse never enters _dpm_sampler, so churn would be inert; "
            "run it with --churn-eta 0 as the reference arm."
        )

    # Matches infer_partial_pooling.main's ordering: GPU selection and the
    # numeric imports happen only after the arguments are validated.
    from runtime import configure_runtime
    cpu_limit = configure_runtime()
    sys.path.insert(0, str(ROOT.parent))

    label = (f"churn eta={known.churn_eta:g}" if known.churn_eta
             else "stock sampler (no churn)")
    print(f"[{args.inference_method}] {label}, artifacts under {root}",
          flush=True)

    stack = contextlib.ExitStack()
    with stack:
        if known.churn_eta:
            stack.enter_context(
                patched_dpm_sampler(churned_dpm_sampler(known.churn_eta))
            )
        if known.jacobian_refresh != 1:
            stack.enter_context(forced_jacobian_refresh(known.jacobian_refresh))
        infer.run(args, cpu_limit)

    # Record what produced these artifacts; the run signature cannot see churn.
    note = root / "churn_manifest.json"
    entries = json.loads(note.read_text()) if note.exists() else {}
    entries[args.inference_method] = {
        "churn_eta": known.churn_eta,
        "jacobian_refresh": known.jacobian_refresh,
        "preset": args.preset,
        "datasets": args.datasets,
        "subjects": args.subjects,
        "draws": args.draws,
        "timesteps": args.timesteps,
        "map_estimator": args.map_estimator,
        "dpm_corrector_steps": args.dpm_corrector_steps,
    }
    note.write_text(json.dumps(entries, indent=2, sort_keys=True))
    print(f"Wrote {note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
