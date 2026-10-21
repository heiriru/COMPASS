#!/bin/bash
# C2 partial pooling, hierarchical attention + T (log-lambda time input) + R (residual output),
# composed with gauss_jacobian_FD; matched to the rerun protocol (seed 7, 100 subjects, 100 draws,
# 50 steps, datasets 0-29). GPU 8; training on slot 0-2, inference split over slots 0-2 / 3-5.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=${GPU:-8} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
taskset -c 0-2 $PY -u pp.py train --mode hierarchical --precondition TR > $L/train_pp_TR.log 2>&1 || { echo train failed; exit 1; }
inf() { taskset -c $1 $PY -u pp.py infer --mode hierarchical --precondition TR --method gauss_jacobian_FD \
          --subjects 100 --datasets 15 --dataset-start $2 --tag rerun 2>&1 | grep -v -i warn > $L/infer_pp_TR_$2.log; }
inf 0-2 0 & inf 3-5 15 & wait
echo all done
