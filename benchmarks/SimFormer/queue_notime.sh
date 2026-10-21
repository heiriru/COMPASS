#!/bin/bash
# After the running round-2 fig4a driver: round 2 on fig4b, the no-time-
# modulation arms at 1k, then NLL/coverage (extra4a), then upstream 100k.
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
L=results/logs
JOB=$1
while kill -0 "$JOB" 2>/dev/null; do sleep 30; done
$PY -u run_benchmark.py fig4b --methods compass_fx_lowrank8 compass_fx_wd --budgets 1000 >> $L/driver_improve_fig4b.log 2>&1
N="compass_fx_notime_input compass_fx_notime_add compass_fx_notime_none"
$PY -u run_benchmark.py fig4a --methods $N --budgets 1000 >> $L/driver_notime_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $N --budgets 1000 >> $L/driver_notime_fig4b.log 2>&1
$PY -u run_benchmark.py extra4a --resample-from compass_legacy_full_fixeps compass_legacy_full >> $L/driver_extra4a.log 2>&1
U="compass_upstream_v11 compass_upstream_v11_fixeps --resample-from compass_upstream_v11"
$PY -u run_benchmark.py fig4a --methods $U --budgets 100000 >> $L/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 100000 >> $L/driver_upstream_fig4b.log 2>&1
