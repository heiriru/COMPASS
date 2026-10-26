#!/bin/bash
while kill -0 3637316 2>/dev/null; do sleep 60; done
export PYTHONPATH=/export/data/rheinric/COMPASS-covout/src JACEXT_GPU=7,8,9 NEED_MB=4000
cd /export/data/rheinric/COMPASS-covout/covariance_output
taskset -c 51-53 /export/home/rheinric/COMPASS/.COMPASS/bin/python -u toy.py run --arm cov >> /export/data/rheinric/covout/logs/toy_run_cov.log 2>&1
/export/home/rheinric/COMPASS/.COMPASS/bin/python toy.py report > /export/data/rheinric/covout/logs/toy_report.log 2>&1
echo TOY COV DONE >> /export/data/rheinric/covout/logs/toy_run_cov.log
