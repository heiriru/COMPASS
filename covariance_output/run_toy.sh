#!/bin/bash
export PYTHONPATH=/export/data/rheinric/COMPASS-covout/src JACEXT_GPU=${GPU:-6,7,8,9} NEED_MB=1500
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=/export/data/rheinric/covout/logs
cd /export/data/rheinric/COMPASS-covout/covariance_output
taskset -c 51-53 $PY -u toy.py train --arm cov > $L/toy_train_cov.log 2>&1 &
taskset -c 54-56 $PY -u toy.py train --arm cov_det > $L/toy_train_cov_det.log 2>&1 &
wait
taskset -c 51-53 $PY -u toy.py run --arm cov > $L/toy_run_cov.log 2>&1 &
taskset -c 54-56 $PY -u toy.py run --arm cov_det > $L/toy_run_cov_det.log 2>&1 &
wait
$PY toy.py report
