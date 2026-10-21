#!/bin/bash
# Which part of TR helps log_sigma_nu? C2 hierarchical attention + T only / R only / Rg (residual on
# globals only), each composed with gauss_jacobian_FD under the rerun protocol (seed 7, 100 subjects,
# 100 draws, 50 steps, datasets 0-29). GPU 8; one 3-CPU slot per arm.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=${GPU:-8} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
arm() {  # $1 precondition, $2 cpu slot
  taskset -c $2 $PY -u pp.py train --mode hierarchical --precondition $1 > $L/train_pp_$1.log 2>&1 &&
  taskset -c $2 $PY -u pp.py infer --mode hierarchical --precondition $1 --method gauss_jacobian_FD \
      --subjects 100 --datasets 30 --tag rerun 2>&1 | grep --line-buffered -v -i warn > $L/infer_pp_$1.log
}
arm T 0-2 & arm R 3-5 & arm Rg 6-8 & wait
echo all done
