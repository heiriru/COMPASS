#!/bin/bash
# (1) full attention (latents attend latents) on the large and small trunks,
# (2) mask6mix training masks vs the Simformer law: 1k first, then 10k/100k for
# the large full-attention arm and mask6mix; then resume extra4a and upstream 100k.
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
L=results/logs
A="compass_fx_fullattn compass_fx_fullattn_small compass_fx_mask6mix"
$PY -u run_benchmark.py fig4a --methods $A --budgets 1000 >> $L/driver_archmask_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $A --budgets 1000 >> $L/driver_archmask_fig4b.log 2>&1
S="compass_fx_fullattn compass_fx_mask6mix"
for B in 10000 100000; do
  $PY -u run_benchmark.py fig4a --methods $S --budgets $B >> $L/driver_archmask_fig4a.log 2>&1
  $PY -u run_benchmark.py fig4b --methods $S --budgets $B >> $L/driver_archmask_fig4b.log 2>&1
done
$PY -u run_benchmark.py extra4a --resample-from compass_legacy_full_fixeps compass_legacy_full >> $L/driver_extra4a.log 2>&1
U="compass_upstream_v11 compass_upstream_v11_fixeps --resample-from compass_upstream_v11"
$PY -u run_benchmark.py fig4a --methods $U --budgets 100000 >> $L/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 100000 >> $L/driver_upstream_fig4b.log 2>&1
