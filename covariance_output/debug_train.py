import os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
sys.path.insert(0, "/export/data/rheinric/COMPASS-covout/tests")
from test_covariance_output import tokenized, hierarchy, MASK
from compass import ConditionalMenu
from compass.covariance_output import CovarianceObjective, NetworkCovarianceHead, enable_covariance_output
torch.set_num_threads(3)
h=hierarchy(); z=h.simulate(20000, torch.Generator().manual_seed(1)).float(); theta,x=z[:,:3],z[:,3:]
_,gain,cov=h.conditional(MASK); target=torch.linalg.inv(cov)
for menu_on in (False, True):
    m=tokenized(); head=enable_covariance_output(m, rank=3, node_scale=z.std(0))
    kw={}
    if menu_on:
        kw["mask_sampler"]=ConditionalMenu(m.schema, {"theta":range(3),"x":range(3,5)}, [("posterior",["x"])], bernoulli_fraction=0.1)
    m.train(theta=theta,x=x,theta_val=theta[:1000],x_val=x[:1000],batch_size=256,max_epochs=30,lr=2e-3,verbose=False,path=tempfile.mkdtemp(),device="cpu",covariance_objective=CovarianceObjective(head),**kw)
    net=NetworkCovarianceHead(m,[0,1,2],[3,4])
    for lam in (0.1,0.5,2.0):
        y=theta[:512]+lam*torch.randn(512,3)
        K=net.information(y,x[:512],torch.full((512,),lam)).mean(0)
        print("menu",menu_on,"lam",lam,"relerr %.3f"%float((K-target).norm()/target.norm()), "diag K", K.diagonal().numpy().round(2), "target", target.diagonal().numpy().round(2))
