#!/bin/bash
# Pushing C2 hierarchical + gauss_jacobian_FD past mean r 0.870 (30 datasets, seed 7, 100 subjects).
#   rec:  baseline and Rg, 400 draws, 100 steps        (tag a400t100)
#   A:    Rg, 400 draws, 50 steps                       (tag a400t50)
#   E:    Rg, 100 draws, 50 steps, prior truncated to 3.5 sd by rejection (tag support3.5)
# Each process gets a free GPU from autocvd (starts staggered); 3 CPUs each.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
job() {  # name cpus precondition tag draws steps start count [extra]
  local name=$1 cpus=$2 pre=$3 tag=$4 draws=$5 steps=$6 start=$7 count=$8; shift 8
  taskset -c $cpus $PY -u pp.py infer --mode hierarchical ${pre:+--precondition $pre} --method gauss_jacobian_FD \
      --subjects 100 --datasets $count --dataset-start $start --draws $draws --timesteps $steps --tag $tag "$@" \
      2>&1 | grep --line-buffered -v -i warn > $L/push_$name.log
}
job base_a 0-2   ""  a400t100 400 100 0 10 &  sleep 60
job base_b 3-5   ""  a400t100 400 100 10 10 & sleep 60
job base_c 6-8   ""  a400t100 400 100 20 10 & sleep 60
job rg_a   9-11  Rg  a400t100 400 100 0 10 &  sleep 60
job rg_b   12-14 Rg  a400t100 400 100 10 10 & sleep 60
job rg_c   15-17 Rg  a400t100 400 100 20 10 & sleep 60
job rgA_a  18-20 Rg  a400t50  400 50 0 15 &   sleep 60
job rgA_b  21-23 Rg  a400t50  400 50 15 15 &  sleep 60
job rgE    24-26 Rg  support3.5 100 50 0 30 --support-sd 3.5 &
wait
echo all done
