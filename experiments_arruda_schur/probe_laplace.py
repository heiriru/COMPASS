"""Why does arruda_laplace run away on the partial-pooling 'full' network? Decompose the score."""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from Partial_Pooling.artifact_names import test_data_path  # noqa: E402
from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.model_pipeline import load_normalizers  # noqa: E402
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402
from Partial_Pooling.train_encoder_summary import TAG, encode, load_encoder  # noqa: E402
import pp  # noqa: E402

torch.set_grad_enabled(False)
paths = BenchmarkPaths.default()
config = get_config("small")
mode = sys.argv[1] if len(sys.argv) > 1 else "full"
ckpt = pp.checkpoint_of(mode, paths)
model = ScoreBasedInferenceModel.load(str(ckpt), device="cpu", arch="tokenized")
if mode != "none":
    load_latent_attention(model, ckpt.parent)
model.model.eval()
encoder, item_norm, z_norm = load_encoder(paths.data.parent / "encoders" / f"{TAG}.pt", "cpu")
test = torch.load(test_data_path(config, paths), map_location="cpu", weights_only=False)["payload"]["sde"]
theta_norm = load_normalizers(config, paths)["sde_joint_theta"]
n = 100
z = encode({"observations": test["observations"][0:1, :n]}, encoder, item_norm, z_norm, "cpu", 1024)[0]
truth = theta_norm.transform(torch.cat((test["globals"][0].expand(n, 7), test["locals"][0, :n]), 1))

S = 4
state = torch.zeros(n, S, 26, dtype=torch.float32)
state[:, :, :10] = truth.unsqueeze(1).float()
state[:, :, 10:] = z.unsqueeze(1).float()
mask = torch.zeros(n, S, 26, dtype=torch.float32)
mask[..., 10:] = 1


def setup(fd):
    s = MarginalSchurSampler(model, estimator="laplace")
    s.hierarchy, s.local_latent_indices = list(range(7)), [7, 8, 9]
    s.full_gaussian_features = list(range(10))
    s.correction = "arruda_schur"
    s._configure_jacobian_schedule(fd, None, 0.3)
    s.jacobian_refresh, s.jacobian_curvature_floor = 1, 1e-3
    s.local_prior_std = torch.ones(3)
    s.score_network_calls = s.evaluated_subject_rows = 0
    s._reset_marginal_state()
    return s


np.set_printoptions(precision=2, suppress=True, linewidth=160)
for t in (0.02, 0.1, 0.3, 0.6):
    tt = torch.tensor([t])
    lam2 = float(model.sde.lambda_t(tt)) ** 2
    noisy = state.clone()
    torch.manual_seed(0)
    noisy[:, :, :10] += (lam2 ** 0.5) * torch.randn(1, S, 10).expand(n, -1, -1) * torch.tensor([1.0] * 7 + [0.0] * 3)
    noisy[:, :, 7:10] += (lam2 ** 0.5) * torch.randn(n, S, 3, dtype=torch.float32)
    print(f"\n=== t={t}  lambda={lam2 ** 0.5:.3f}")
    for fd in (1e-2, 3e-2):
        s = setup(fd)
        raw = s._raw_row_scores(noisy, tt, mask).double()
        direct = raw[:, :, :7].sum(0).mean(0)
        base, J = s._fd_scores_and_jacobian(noisy, tt, mask, list(range(10)))
        eig, vec = s._clamped_eigh(-J[..., 7:, 7:], lam2, s._local_curvature_bound(lam2))
        inv = (vec / eig.unsqueeze(-2)) @ vec.mT
        schur = torch.einsum("nsgl,nsl->nsg", J[..., :7, 7:] @ inv, base[:, :, 7:10]).sum(0).mean(0)
        logdet = s._logdet_gradient(noisy, tt, mask, lam2, eig, s._local_curvature_bound(lam2))
        ld = (-0.5 * logdet).sum(0).mean(0)
        print(f"fd={fd:g}  sum_j S_g       {direct.numpy()}")
        print(f"          sum_j Schur term {schur.numpy()}")
        print(f"          sum_j -1/2 grad logdet {ld.numpy()}   (per-subject sd {(-0.5 * logdet).std(0).mean(0).numpy()})")
        print(f"          eig H_ll range [{eig.min():.3g}, {eig.max():.3g}]  lambda^-2={1 / lam2:.3g}")
