#!/bin/bash
# Wide-prior Rg net at n = 1, 4, 25 subjects (n = 100 is the main run), 30 datasets, GPU 9.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
arm() {  # $1 subjects, $2 cores
  taskset -c $2 $PY -u pp.py infer --mode hierarchical --precondition Rg --train-prior-scale 1.5 \
    --method gauss_jacobian_FD --subjects $1 --datasets 30 --draws 100 --timesteps 50 --tag rerun \
    2>&1 | grep --line-buffered -v -i warn > $L/infer_pp_Rg_wide1.5_s$1.log
}
arm 1 9-11 & sleep 40; arm 4 39-41 & sleep 40; arm 25 27-29 & wait
echo all done
