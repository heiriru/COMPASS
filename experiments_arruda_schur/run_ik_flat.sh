#!/bin/bash
# ik_flat.py end to end on GPU 8: slot 0-2 data+references -> fixed net -> fixed runs; slot 3-5 global net -> runs.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
export CUDA_VISIBLE_DEVICES=${GPU:-8} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTHONPATH=.
fixed() {
  taskset -c 0-2 $PY -u ik_flat.py data 2>&1 | grep -v -i warn > $L/ik_flat_data.log
  taskset -c 0-2 $PY -u ik_flat.py train --variant fixed > $L/train_ik_flat_fixed.log 2>&1 &&
  taskset -c 0-2 $PY -u ik_flat.py run --variant fixed 2>&1 | grep -v -i warn > $L/run_ik_flat_fixed.log
}
glob() {
  IK_SD_H=0.05 taskset -c 3-5 $PY -u ik_flat.py train --variant global > $L/train_ik_flat_global.log 2>&1 &&
  IK_SD_H=0.05 taskset -c 3-5 $PY -u ik_flat.py run --variant global 2>&1 | grep -v -i warn > $L/run_ik_flat_global.log
}
fixed & glob & wait
taskset -c 0-2 $PY -u ik_flat.py grade 2>&1 | grep -v -i warn > $L/grade_ik_flat.log
echo all done
