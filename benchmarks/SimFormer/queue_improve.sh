#!/bin/bash
# 1) improvement arms at 1k, 2) paper-appendix evaluations (eval only),
# 3) remaining upstream v1.1 100k cells. One job at a time, one GPU.
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
L=results/logs
JOB=$1   # upstream GL 100k job left running
while kill -0 "$JOB" 2>/dev/null; do sleep 30; done
$PY report.py fig4a > /dev/null 2>&1
for ph in fig4a fig4b; do
  $PY -u run_benchmark.py $ph --methods compass_fx_euler500 --budgets 1000 --resample-from compass_legacy_full_fixeps compass_legacy_full >> $L/driver_improve_$ph.log 2>&1
  $PY -u run_benchmark.py $ph --methods compass_fx_final compass_fx_condemb compass_fx_final_condemb --budgets 1000 >> $L/driver_improve_$ph.log 2>&1
done
$PY -u run_benchmark.py extra4b --resample-from compass_legacy_full_fixeps compass_legacy_full >> $L/driver_extra4b.log 2>&1
$PY -u run_benchmark.py extra4a --resample-from compass_legacy_full_fixeps compass_legacy_full >> $L/driver_extra4a.log 2>&1
U="compass_upstream_v11 compass_upstream_v11_fixeps --resample-from compass_upstream_v11"
$PY -u run_benchmark.py fig4a --methods $U --budgets 100000 >> $L/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 100000 >> $L/driver_upstream_fig4b.log 2>&1
