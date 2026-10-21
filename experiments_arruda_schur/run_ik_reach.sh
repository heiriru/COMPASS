#!/bin/bash
# ik_reach.py end to end on GPU 8: slot 0-2 sym net -> runs; slot 3-5 shift net -> runs; then grade.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=${GPU:-8} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=.
arm() {  # $1 variant, $2 cpu slot
  taskset -c $2 $PY -u ik_reach.py train --variant $1 > $L/train_ik_reach_$1.log 2>&1 &&
  taskset -c $2 $PY -u ik_reach.py run --variant $1 2>&1 | grep -v -i warn > $L/run_ik_reach_$1.log
}
arm sym 0-2 & arm shift 3-5 & wait
taskset -c 0-2 $PY -u ik_reach.py grade 2>&1 | grep -v -i warn > $L/grade_ik_reach.log
echo all done
