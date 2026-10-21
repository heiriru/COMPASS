#!/bin/bash
# Idea 1: train the Rg net on globals drawn from the prior with 1.5x its sd (same 100k simulations),
# then compose with the wider training prior divided out per subject (set_training_prior).
# Inference matches the rerun arms: 30 datasets, seed 7, 100 draws, 50 steps. GPU 2 by request.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
taskset -c 27-29 $PY -u pp.py train --mode hierarchical --precondition Rg --train-prior-scale 1.5 \
  2>&1 | grep --line-buffered -v -i warn > $L/train_pp_Rg_wide1.5.log
taskset -c 27-29 $PY -u pp.py infer --mode hierarchical --precondition Rg --train-prior-scale 1.5 \
  --method gauss_jacobian_FD --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50 \
  --tag rerun 2>&1 | grep --line-buffered -v -i warn > $L/infer_pp_Rg_wide1.5.log
echo all done
