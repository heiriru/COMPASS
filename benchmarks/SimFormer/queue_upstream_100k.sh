#!/bin/bash
# Remaining cells after the 2026-09-30 stop: upstream v1.1 at 100k (train, then resample with eps forwarded).
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
U="compass_upstream_v11 compass_upstream_v11_fixeps --resample-from compass_upstream_v11"
$PY -u run_benchmark.py fig4a --methods $U --budgets 100000 >> results/logs/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 100000 >> results/logs/driver_upstream_fig4b.log 2>&1
