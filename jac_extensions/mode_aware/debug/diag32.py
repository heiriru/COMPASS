import os, sys
sys.argv=["x","report"]
sys.path.insert(0, "/export/data/rheinric/COMPASS-jacext/jac_extensions")
import gpu; gpu.claim(1500)
sys.path.insert(0, "/export/data/rheinric/COMPASS-jacext/jac_extensions/mode_aware")
import gm_mode_aware as G
import numpy as np, torch
import datasets, methods, model as M
from compass.MultiObsSampler import MultiObsSampler
import compass.experimental.mode_aware as MA
mix=M.DEFAULT; net=G.network("hierarchical"); prior,lp=methods._priors(mix)
rng=np.random.default_rng([2026,32,0,0]); x=datasets.make_dataset("asymmetric",32,rng,mix)["x"]
data=torch.as_tensor(np.asarray(x).reshape(-1,1),dtype=torch.float32); mask=torch.tensor([0.,0,1])
kw=dict(local_prior=lp, device=G.DEVICE, **G.SETTINGS)
orig=MultiObsSampler.sample
def logged(self,*a,**k):
    out=orig(self,*a,**k); g=out[0,:,0].double().cpu()
    print("  sample prior=",k.get("prior"),"start",k.get("start_lambda"),"-> g mean %.3f sd %.3f"%(g.mean(),g.std()), "patched", MultiObsSampler._diffused_gaussian_prior_score.__name__, flush=True)
    return out
MultiObsSampler.sample=logged
for arm in ("mode","mode_comp"):
    print(arm)
    torch.manual_seed(0)
    with methods.mixture_prior_score(mix, enabled=(arm!="mode_comp")):
        out,info=MA.mode_aware_sample(MultiObsSampler(net),data,mask,[0],2000,prior=prior,discovery_samples=1000,draws_per_mode=1000,
            log_prior_fn=G.log_prior_fn(mix), mode_prior_fn=(G.component_prior(mix) if arm=="mode_comp" else None),
            discovery_context=((lambda: methods.mixture_prior_score(mix)) if arm=="mode_comp" else None), **kw)
    print(" centres",info["centres"].ravel(),"masses",info["masses"])
