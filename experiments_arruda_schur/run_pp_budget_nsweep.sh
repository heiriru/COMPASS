#!/bin/bash
# The 0.895 network (budget arm A, n65536) at n = 1, 4, 25 subjects (n = 100 is the main run),
# 30 datasets, same settings as run_pp_wide_nsweep.sh. GPU 8 (GPU 9 busy with another user).
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=8 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
ENC=artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt
arm() {  # $1 subjects, $2 cores
  taskset -c $2 $PY -u pp.py infer --mode hierarchical --precondition Rg --train-prior-scale 1.5 \
    --method gauss_jacobian_FD --subjects $1 --datasets 30 --draws 100 --timesteps 50 --tag rerun \
    --encoder $ENC --variant n65536 2>&1 | grep --line-buffered -v -i warn > $L/budget_A_infer_s$1.log
}
arm 1 30-32 & sleep 40; arm 4 33-35 & sleep 40; arm 25 36-38 & wait
echo all done
