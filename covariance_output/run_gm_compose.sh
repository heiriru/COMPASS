#!/bin/bash
# GM composition: GH pilot vs JAC-FD (r9) vs JAC-netcov, exact mixture prior score, on GPU 6.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-7} ROADMAP_OUT=/export/data/rheinric/covout/gm
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
cd /export/data/rheinric/COMPASS-covout/experiments_roadmap2
L=/export/data/rheinric/covout/logs
for s in 7 8; do
  for a in base cov cov_det; do
    M="gauss_hierarchical_mixture_prior gauss_jacobian_mixture_prior_fd"
    [ $a != base ] && M="$M gauss_jacobian_mixture_prior_netcov"
    [ -f $ROADMAP_OUT/${a}_s$s/scaling/eps0.001_cmp/summary.json ] && continue
    taskset -c 60-62 $PY -u gm.py compose --arm $a --seed $s --observations 8 16 32 64 --tag-suffix _cmp \
        --methods $M > $L/cmp_${a}_s$s.log 2>&1
  done
done
echo ALL DONE
