#!/bin/bash
# Capacity test at 1k (shared time conditioning), inserted into the Fig. 4b queue
# at a job boundary; then Fig. 4b resumes (done cells are skipped).
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
JOB=$1   # harness job left running when the old driver was stopped
while kill -0 "$JOB" 2>/dev/null; do sleep 20; done
$PY report.py fig4b > /dev/null
M="compass_shared_t256 compass_shared_t64"
$PY -u run_benchmark.py fig4a --methods $M --budgets 1000 > results/logs/driver_capacity_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $M --budgets 1000 > results/logs/driver_capacity_fig4b.log 2>&1
$PY -u run_benchmark.py fig4b --methods score_transformer compass_legacy_full_fixeps --resample-from compass_legacy_full >> results/logs/driver_fig4b_fixeps.log 2>&1
