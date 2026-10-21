#!/bin/bash
# Ablation of the 0.928 run: same 1.5x-wide training prior, same simulations and seed, but the plain
# hierarchical-attention net (no Rg residual output). Inference identical: 30 datasets, n=100, 100 draws, 50 steps. GPU 6 by request.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
taskset -c 9-11 $PY -u pp.py train --mode hierarchical --train-prior-scale 1.5 \
  2>&1 | grep --line-buffered -v -i warn > $L/train_pp_wide1.5_plain.log
taskset -c 9-11 $PY -u pp.py infer --mode hierarchical --train-prior-scale 1.5 \
  --method gauss_jacobian_FD --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50 \
  --tag rerun 2>&1 | grep --line-buffered -v -i warn > $L/infer_pp_wide1.5_plain.log
echo all done
