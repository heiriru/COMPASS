#!/bin/bash
# The fair (65,536-sim) runs without the Rg residual on the globals: the plain hierarchical net,
# 1.5x-wide prior, same shared set, inference identical (n = 100, 30 datasets). GPU 8.
#   n65536          A's pretrained encoder (reused: it does not depend on Rg), plain score net
#   jointft_n65536  warm start from that plain A, joint fine-tune (lr 3e-4, clip 1)
#   jointbf_n65536  from scratch, lr 3e-4, clip 1, posterior-only menu
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=8 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
ENC=artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt
INFER="--mode hierarchical --train-prior-scale 1.5 --method gauss_jacobian_FD \
  --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50 --tag rerun"
joint() {  # $1 variant, $2 cores
  taskset -c $2 $PY -u pp_budget.py joint --precondition "" --variant $1 \
    2>&1 | grep --line-buffered -v -i warn > $L/budget_plain_$1_train.log
  taskset -c $2 $PY -u pp.py infer $INFER --variant $1 \
    --encoder artifacts/pp/hierarchical_wide1.5_$1/checkpoints/encoder.pt \
    2>&1 | grep --line-buffered -v -i warn > $L/budget_plain_$1_infer.log
  echo "plain $1 done"
}
arm_a() {
  taskset -c 45-47 $PY -u pp.py train --mode hierarchical --train-prior-scale 1.5 \
    --train-subset 65536 --encoder $ENC --variant n65536 \
    2>&1 | grep --line-buffered -v -i warn > $L/budget_plain_A_train.log
  # the warm start needs A's checkpoint: start it, then A's inference alongside
  joint jointft_n65536 51-53 &
  taskset -c 45-47 $PY -u pp.py infer $INFER --encoder $ENC --variant n65536 \
    2>&1 | grep --line-buffered -v -i warn > $L/budget_plain_A_infer.log
  echo "plain A done"
  wait
}
arm_a & joint jointbf_n65536 48-50 & wait
echo all done
