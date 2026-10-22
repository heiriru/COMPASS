import os, sys
os.environ["JACEXT_DEVICE"]="cpu"
sys.argv=["x","report"]
sys.path.insert(0, "/export/data/rheinric/COMPASS-jacext/jac_extensions/mode_aware")
import gm_mode_aware as G
import numpy as np, torch
import datasets, methods, model as M
from compass.MultiObsSampler import MultiObsSampler
from compass.experimental.mode_aware import fit_gmm
mix=M.DEFAULT; net=G.network("oracle"); prior,lp=methods._priors(mix)
rng=np.random.default_rng([2026,8,0,0]); x=datasets.make_dataset("asymmetric",8,rng,mix)["x"]
w,means,_=mix.global_posterior(x); print("exact weights",w,"branch means",means, "branch sd", mix.posterior_variance(8)**0.5 if hasattr(mix,'posterior_variance') else '')
data=torch.as_tensor(np.asarray(x).reshape(-1,1),dtype=torch.float32); mask=torch.tensor([0.,0,1])
kw=dict(local_prior=lp, device="cpu", **G.SETTINGS)
with methods.mixture_prior_score(mix):
    s=MultiObsSampler(net); torch.manual_seed(0)
    d=s.sample(world_size=1,data=data,condition_mask=mask,hierarchy=[0],prior=prior,num_samples=1000,verbose=False,**kw)
    g=d[0,:,0].double()
    print("discovery quantiles", np.quantile(g.numpy(),[.05,.25,.5,.75,.95]).round(3), "frac>0", float((g>0).double().mean()))
    lab,mu,cov,wt=fit_gmm(g.reshape(-1,1),3); print("gmm", mu.ravel(), cov.ravel()**.5, wt)
from compass.experimental.mode_aware import mode_aware_sample
import metrics
torch.manual_seed(0)
out,info=mode_aware_sample(MultiObsSampler(net),data,mask,[0],2000,prior=prior,discovery_samples=1000,draws_per_mode=1000,
    log_prior_fn=G.log_prior_fn(mix),mode_prior_fn=G.component_prior(mix),discovery_context=lambda: methods.mixture_prior_score(mix),**kw)
sm=out.numpy(); sm[:,:,2]=np.asarray(x).reshape(-1,1)
r=metrics.evaluate(sm,mix,x)
print("mode_comp", {k:round(r[k],3) for k in ("w1_over_sigma","branch_mass_error","width_ratio","local_w1_over_sigma")}, "masses",info["masses"].numpy().round(3),"ess",np.round(info["ess_fraction"],3),"centres",info["centres"].ravel().numpy().round(2))
g=out[0,:,0].double().numpy(); print("pos branch mean/sd", g[g>0].mean().round(3), g[g>0].std().round(3), "neg", g[g<0].mean().round(3), g[g<0].std().round(3))
