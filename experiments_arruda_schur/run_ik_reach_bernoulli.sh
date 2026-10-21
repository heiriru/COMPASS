#!/bin/bash
# ik_reach.py with the plain Bernoulli(0.33) mask (menu=False), GPU 8: sym on slot 0-2, shift on 3-5.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=${GPU:-8} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=.
arm() {  # $1 variant, $2 cpu slot
  taskset -c $2 $PY -u ik_reach.py train --variant $1 --menu bernoulli > $L/train_ik_reach_$1_bernoulli.log 2>&1 &&
  taskset -c $2 $PY -u ik_reach.py run --variant $1 --menu bernoulli 2>&1 | grep -v -i warn > $L/run_ik_reach_$1_bernoulli.log
}
arm sym 0-2 & arm shift 3-5 & wait
echo all done
