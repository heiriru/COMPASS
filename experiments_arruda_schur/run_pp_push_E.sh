#!/bin/bash
# The E arm of run_pp_push.sh, relaunched alone: autocvd waits until a GPU is free.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
taskset -c 24-26 /export/home/rheinric/COMPASS/.COMPASS/bin/python -u pp.py infer --mode hierarchical --precondition Rg \
  --method gauss_jacobian_FD --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50 \
  --tag support3.5 --support-sd 3.5 2>&1 | grep --line-buffered -v -i warn > $L/push_rgE.log
echo all done
