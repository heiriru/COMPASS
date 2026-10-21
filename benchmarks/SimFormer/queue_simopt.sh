#!/bin/bash
# After the last input-scaling driver: COMPASS with Simformer's optimizer +
# final weights, and additionally uniform time sampling, at 10k and 100k (8 tasks);
# then resume extra4a and upstream 100k. Done cells are skipped on restart.
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
L=results/logs
JOB=$1
while [ -n "$JOB" ] && kill -0 "$JOB" 2>/dev/null; do sleep 30; done
S="compass_fx_simopt compass_fx_simopt_unif"
for B in 10000 100000; do
  $PY -u run_benchmark.py fig4a --methods $S --budgets $B >> $L/driver_simopt_fig4a.log 2>&1
  $PY -u run_benchmark.py fig4b --methods $S --budgets $B >> $L/driver_simopt_fig4b.log 2>&1
done
$PY -u run_benchmark.py extra4a --resample-from compass_legacy_full_fixeps compass_legacy_full >> $L/driver_extra4a.log 2>&1
U="compass_upstream_v11 compass_upstream_v11_fixeps --resample-from compass_upstream_v11"
$PY -u run_benchmark.py fig4a --methods $U --budgets 100000 >> $L/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 100000 >> $L/driver_upstream_fig4b.log 2>&1
