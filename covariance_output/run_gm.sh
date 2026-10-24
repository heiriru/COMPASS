#!/bin/bash
# GM benchmark for the network covariance output (GPU 6, exclusive at launch).
export CUDA_VISIBLE_DEVICES=${GPU:-6} ROADMAP_OUT=/export/data/rheinric/covout/gm
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
P="taskset -c 60-62 $PY"
cd /export/data/rheinric/COMPASS-covout/experiments_roadmap2
L=/export/data/rheinric/covout/logs
c=48
for s in 7 8; do for a in cov cov_det; do
  [ -f $ROADMAP_OUT/${a}_s$s/Model.pt ] || taskset -c $c-$((c+2)) $PY -u gm.py train --arm $a --seed $s > $L/train_${a}_s$s.log 2>&1 &
  c=$((c+3))
done; done; wait
for s in 7 8; do for a in base cov cov_det; do
  $P -u gm.py score --arm $a --seed $s > $L/score_${a}_s$s.log 2>&1
  $P -u gm.py info --arm $a --seed $s > $L/info_${a}_s$s.log 2>&1
done; done
for s in 7 8; do
  $P -u gm.py compose --arm base --seed $s --observations 8 16 32 64 --tag-suffix _covout \
     --methods gauss_hierarchical_mixture_prior gauss_jacobian_mixture_prior > $L/compose_base_s$s.log 2>&1
  for a in cov cov_det; do
    $P -u gm.py compose --arm $a --seed $s --observations 8 16 32 64 --tag-suffix _covout \
       --methods gauss_hierarchical_mixture_prior gauss_jacobian_mixture_prior gauss_jacobian_mixture_prior_netcov > $L/compose_${a}_s$s.log 2>&1
  done
done
echo ALL DONE
