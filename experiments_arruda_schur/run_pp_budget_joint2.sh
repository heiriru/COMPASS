#!/bin/bash
# Two follow-ups to the failed joint arm B, in parallel on one GPU (9):
#   jointft_n65536  warm start from arm A (encoder + score net), joint fine-tune lr 3e-4, clip 1
#   jointbf_n65536  from scratch, lr 3e-4, clip 1, posterior-only menu (BayesFlow-style)
# Same shared 65,536-sim set; inference identical to run_pp_budget.sh (n = 100, 30 datasets).
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
INFER="--mode hierarchical --precondition Rg --train-prior-scale 1.5 --method gauss_jacobian_FD \
  --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50 --tag rerun"
arm() {  # $1 variant, $2 cores
  taskset -c $2 $PY -u pp_budget.py joint --variant $1 2>&1 | grep --line-buffered -v -i warn > $L/budget_$1_train.log
  taskset -c $2 $PY -u pp.py infer $INFER --variant $1 \
    --encoder artifacts/pp/hierarchical_Rg_wide1.5_$1/checkpoints/encoder.pt \
    2>&1 | grep --line-buffered -v -i warn > $L/budget_$1_infer.log
  echo "$1 done"
}
arm jointft_n65536 39-41 & arm jointbf_n65536 42-44 & wait
echo all done
