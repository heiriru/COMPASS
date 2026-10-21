#!/bin/bash
# GPU stage of diag/single_obs.py, family by family once its MCMC file exists.
cd /export/home/rheinric/COMPASS
R=validation/a4_waic/results/single_obs
for f in offset noise tutorial population population_wide; do
  until [ -f $R/${f}_mcmc.npz ]; do sleep 60; done
  [ -f $R/${f}_net.npz ] || .COMPASS/bin/python -m validation.a4_waic.diag.single_obs net --family $f \
      > validation/a4_waic/logs/single_obs_net_$f.log 2>&1
done
