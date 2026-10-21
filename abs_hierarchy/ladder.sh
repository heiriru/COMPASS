#!/bin/bash
# Low-noise ablation ladder on the lean hierarchical-attention net (precondition.py).
# T = log-lambda time input; TF = + training floor eps 1e-4; R = residual output alone; TFR = all.
set -u
cd /export/home/rheinric/COMPASS/abs_hierarchy
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-8} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
train() {  # cores tag flags...
  local cores=$1 tag=$2; shift 2
  taskset -c $cores $PY train_abs.py --model lean --mode hierarchical --tag $tag "$@" > logs/train_lean_hierarchical_$tag.log 2>&1
}
train 0-2 T --time & train 3-5 TF --time --eps 1e-4 & wait
train 0-2 R --residual & train 3-5 TFR --time --eps 1e-4 --residual & wait
echo trained
