#!/bin/bash
# After the last mask6mix job: EDM input scaling alone and with full attention
# (Simformer masks), 8 tasks x 1k/10k/100k; then resume extra4a and upstream 100k.
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
L=results/logs
JOB=$1
while kill -0 "$JOB" 2>/dev/null; do sleep 30; done
S="compass_fx_inscale compass_fx_inscale_fullattn"
for B in 1000 10000 100000; do
  $PY -u run_benchmark.py fig4a --methods $S --budgets $B >> $L/driver_inscale_fig4a.log 2>&1
  $PY -u run_benchmark.py fig4b --methods $S --budgets $B >> $L/driver_inscale_fig4b.log 2>&1
done
$PY -u run_benchmark.py extra4a --resample-from compass_legacy_full_fixeps compass_legacy_full >> $L/driver_extra4a.log 2>&1
U="compass_upstream_v11 compass_upstream_v11_fixeps --resample-from compass_upstream_v11"
$PY -u run_benchmark.py fig4a --methods $U --budgets 100000 >> $L/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 100000 >> $L/driver_upstream_fig4b.log 2>&1
