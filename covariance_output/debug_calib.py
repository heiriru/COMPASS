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
for epochs in (30, 150):
    m=tokenized(); head=enable_covariance_output(m, rank=3, node_scale=z.std(0))
    menu=ConditionalMenu(m.schema, {"theta":range(3),"x":range(3,5)}, [("posterior",["x"])], bernoulli_fraction=0.1)
    m.train(theta=theta,x=x,theta_val=theta[:1000],x_val=x[:1000],batch_size=256,max_epochs=epochs,lr=2e-3,verbose=False,path=tempfile.mkdtemp(),device="cpu",covariance_objective=CovarianceObjective(head),mask_sampler=menu)
    net=NetworkCovarianceHead(m,[0,1,2],[3,4])
    for lam in (0.1,0.5,2.0):
        N=4000; th=theta[:N]; xx=x[:N]; y=th+lam*torch.randn(N,3)
        t=m.sde.time_of_lambda(torch.full((N,1),lam))
        st=torch.zeros(N,5); st[:,:3]=y; st[:,3:]=xx
        with torch.no_grad():
            s=m.output_scale_function(t,m.model(x=st,t=t,c=MASK.expand(N,-1)))[:,:3].double()
        D=y.double()+lam**2*s; r=th.double()-D
        Kemp=torch.linalg.inv(torch.cov(r.T))-torch.eye(3,dtype=torch.float64)/lam**2
        # exact denoiser residual -> exact K
        mean=xx.double()@gain.T.double(); Ct=cov.double()
        Dex=mean+(Ct@torch.linalg.inv(Ct+lam**2*torch.eye(3,dtype=torch.float64))@(y.double()-mean).T).T
        Kex=torch.linalg.inv(torch.cov((th.double()-Dex).T))-torch.eye(3,dtype=torch.float64)/lam**2
        K=net.information(y,xx,torch.full((N,),lam)).mean(0)
        f=lambda A:A.diagonal().numpy().round(2)
        print(f"ep{epochs} lam {lam}: K_head {f(K)} | K_emp(net residual) {f(Kemp)} | K_emp(exact residual) {f(Kex)} | target {f(target)} | score rel err {float((D-Dex).norm()/ (Dex-y.double()).norm()):.3f}")
