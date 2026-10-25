#!/bin/bash
while kill -0 3469049 2>/dev/null; do sleep 60; done
export PYTHONPATH=/export/data/rheinric/COMPASS-covout/src JACEXT_GPU=7,8,9 NEED_MB=4000
cd /export/data/rheinric/COMPASS-covout/covariance_output
taskset -c 54-56 /export/home/rheinric/COMPASS/.COMPASS/bin/python -u toy.py run --arm cov_det >> /export/data/rheinric/covout/logs/toy_run_cov_det.log 2>&1
/export/home/rheinric/COMPASS/.COMPASS/bin/python toy.py report > /export/data/rheinric/covout/logs/toy_report.log 2>&1
echo TOY DONE >> /export/data/rheinric/covout/logs/toy_run_cov_det.log
