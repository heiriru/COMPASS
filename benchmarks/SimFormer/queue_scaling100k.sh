#!/bin/bash
# 100k for the single best small arm (the three were indistinguishable at 10k),
# then NLL/coverage (extra4a), then upstream 100k.
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
L=results/logs
S="compass_fx_notime_add"
$PY -u run_benchmark.py fig4a --methods $S --budgets 100000 >> $L/driver_scaling_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $S --budgets 100000 >> $L/driver_scaling_fig4b.log 2>&1
$PY -u run_benchmark.py extra4a --resample-from compass_legacy_full_fixeps compass_legacy_full >> $L/driver_extra4a.log 2>&1
U="compass_upstream_v11 compass_upstream_v11_fixeps --resample-from compass_upstream_v11"
$PY -u run_benchmark.py fig4a --methods $U --budgets 100000 >> $L/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 100000 >> $L/driver_upstream_fig4b.log 2>&1
