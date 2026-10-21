"""Trace an arruda_schur trajectory on the partial-pooling network, term by term."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
sys.path[:0] = [str(common.COMPASS)]
import Partial_Pooling.runtime as rt  # noqa: E402
rt.configure_cpu_limit(3)
rt.configure_runtime = lambda max_threads=3, **_: rt.configure_cpu_limit(max_threads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from compass import ScoreBasedInferenceModel  # noqa: E402
from compass.experimental.latent_attention import load_latent_attention  # noqa: E402
from compass.experimental.marginal_schur import MarginalSchurSampler  # noqa: E402
from Partial_Pooling import infer_partial_pooling as ipp  # noqa: E402
from Partial_Pooling.artifact_names import test_data_path  # noqa: E402
from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.model_pipeline import load_normalizers  # noqa: E402
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402
from Partial_Pooling.train_encoder_summary import TAG, encode, load_encoder  # noqa: E402
import pp  # noqa: E402

mode, estimator = sys.argv[1], sys.argv[2]
correction = sys.argv[3] if len(sys.argv) > 3 else "arruda_schur"
paths, config = BenchmarkPaths.default(), get_config("small")
ckpt = pp.checkpoint_of(mode, paths)
model = ScoreBasedInferenceModel.load(str(ckpt), device="cuda", arch="tokenized")
if mode != "none":
    load_latent_attention(model, ckpt.parent)
model.model.eval()
encoder, item_norm, z_norm = load_encoder(paths.data.parent / "encoders" / f"{TAG}.pt", "cuda")
test = torch.load(test_data_path(config, paths), map_location="cpu", weights_only=False)["payload"]["sde"]
theta_norm = load_normalizers(config, paths)["sde_joint_theta"]
n = 100
z = encode({"observations": test["observations"][0:1, :n]}, encoder, item_norm, z_norm, "cuda", 1024)[0]
truth = theta_norm.transform(torch.cat((test["globals"][0:1], test["locals"][0, :1]), 1))[0, :7]

sampler = MarginalSchurSampler(model, estimator=estimator, ancestral_locals=False)
model.multi_obs_sampler = sampler
log = []
original = sampler._marginal_scores


def traced(scores, x, t, condition_mask, variance):
    marginal, precision = original(scores, x, t, condition_mask, variance)
    theta_h = x[:1, :, :7]
    prior = sampler._diffused_gaussian_prior_score(theta_h, t).double()[0]
    log.append(dict(t=float(t), lam=variance ** 0.5,
                    g=theta_h[0].double().mean(0).cpu().numpy(),
                    raw=scores[:, :, :7].double().sum(0).mean(0).cpu().numpy(),
                    marg=marginal.sum(0).mean(0).cpu().numpy(),
                    prior=((1 - n) * prior).mean(0).cpu().numpy(),
                    loc=x[:, :, 7:10].double().abs().mean().item()))
    return marginal, precision


sampler._marginal_scores = traced
prior = ipp._normalized_global_prior(theta_norm, torch)
local_prior = ipp._normalized_local_prior(theta_norm, torch)
cm = torch.cat((torch.zeros(10), torch.ones(16))).cuda()
torch.manual_seed(0)
out = model.sample(x=z.cuda(), condition_mask=cm, timesteps=50, num_samples=4, multi_obs_inference=True,
                   hierarchy=list(range(7)), prior=prior, local_prior=local_prior, correction=correction,
                   order=2, corrector_steps=0, final_corrector_steps=0, device="cuda", verbose=False,
                   method="dpm", capture_attention=False, jacobian_refresh=5, jacobian_refresh_low_noise=20,
                   jacobian_refresh_lambda=0.3)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("truth (normalized):", truth.numpy())
for row in log[::4] + log[-2:]:
    print(f"t={row['t']:.3f} lam={row['lam']:6.3f} |l|={row['loc']:5.2f} g={row['g']}")
    print(f"      raw sum S_g {row['raw']}\n      marginal sum   {row['marg']}\n      (1-n) prior    {row['prior']}")
print("final g:", out[0, :, :7].mean(0).cpu().numpy() if out.dim() == 3 else out)
