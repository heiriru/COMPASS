#!/bin/bash
# GPU stage of diag/composed_recovery.py for every family (needs no reference files).
cd /export/home/rheinric/COMPASS
for f in offset noise population population_wide; do
  .COMPASS/bin/python -m validation.a4_waic.diag.composed_recovery net --family $f \
      > validation/a4_waic/logs/crec_net_$f.log 2>&1
done
