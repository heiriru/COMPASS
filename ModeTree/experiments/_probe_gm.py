import sys
from pathlib import Path
ROOT = Path("/export/home/rheinric/COMPASS")
sys.path.insert(0, str(ROOT / "ModeTree"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Gaussian_mixtures_compositional_score_modelling"))
from modetree.runtime import configure_cpu_limit
configure_cpu_limit(3)
import time, numpy as np, torch
import model as M, train, oracle, datasets
from compass.priors import GaussianMixturePrior
from modetree.compass_adapter import CompassRowSource

mix = M.DEFAULT
ck = ROOT / "Gaussian_mixtures_compositional_score_modelling/artifacts/models/tokenized_h32d3wide_n200000_six_weighted_log_sigma/Model_checkpoint.pt"
net = train.load_to(ck, "cpu")
print("params", sum(p.numel() for p in net.model.parameters()))
orc = oracle.OracleModel(mix, sigma=train.SDE_KWARGS["sigma"])
prior = GaussianMixturePrior([0.5,0.5], torch.tensor([[-mix.mu_0],[mix.mu_0]],dtype=torch.float64),
                             torch.tensor([[[mix.tau_g**2]],[[mix.tau_g**2]]],dtype=torch.float64))
x = np.array([[1.0],[ -0.5]])
for src,name in ((net,"learned"),(orc,"oracle")):
    rs = CompassRowSource(src, x, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES, prior)
    print(name, "sigma", rs.schedule.sigma)
    G = torch.linspace(-6,6,801, dtype=torch.float64)
    st = torch.stack([G, mix.a*G + mix.b*1.0], -1).unsqueeze(1).repeat(1,2,1)  # (B,n,2)
    t0=time.perf_counter()
    sc = rs.row_scores(st, 0.1)
    print(name, sc.shape, "%.2fs"%(time.perf_counter()-t0), sc[400])
