#!/bin/bash
# The 0.928 run (hierarchical + Rg + 1.5x-wide prior, gauss_jacobian_FD) at the upstream budget:
# one shared set of 65,536 wide-prior simulations (pp_budget.py data, already generated).
#   A  "n65536":       encoder pretrained (held-out-trial NLL) on the set, then the score net on it
#   B  "joint_n65536": encoder + score net trained jointly on the denoising loss only
# Both arms in parallel on GPU 9 by request. Inference identical to run_pp_wide.sh.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
ENC=artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt
INFER="--mode hierarchical --precondition Rg --train-prior-scale 1.5 --method gauss_jacobian_FD \
  --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50 --tag rerun"

arm_a() {
  taskset -c 30-32 $PY -u pp_budget.py pretrain 2>&1 | grep --line-buffered -v -i warn > $L/budget_A_pretrain.log
  taskset -c 30-32 $PY -u pp.py train --mode hierarchical --precondition Rg --train-prior-scale 1.5 \
    --train-subset 65536 --encoder $ENC --variant n65536 \
    2>&1 | grep --line-buffered -v -i warn > $L/budget_A_train.log
  taskset -c 30-32 $PY -u pp.py infer $INFER --encoder $ENC --variant n65536 \
    2>&1 | grep --line-buffered -v -i warn > $L/budget_A_infer.log
  echo "A done"
}
arm_b() {
  taskset -c 33-35 $PY -u pp_budget.py joint 2>&1 | grep --line-buffered -v -i warn > $L/budget_B_joint.log
  taskset -c 33-35 $PY -u pp.py infer $INFER \
    --encoder artifacts/pp/hierarchical_Rg_wide1.5_joint_n65536/checkpoints/encoder.pt --variant joint_n65536 \
    2>&1 | grep --line-buffered -v -i warn > $L/budget_B_infer.log
  echo "B done"
}
arm_a & arm_b & wait
echo all done
