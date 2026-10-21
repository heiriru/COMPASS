#!/bin/bash
# Replaces queue_capacity.sh after its first driver: capacity test (4b) ->
# upstream v1.1 at 1k/10k (Fig 4a, 4b; each cell trained then resampled with
# eps forwarded) -> training step-time check -> rest of Fig 4b -> upstream 100k.
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
PY=../../.COMPASS/bin/python
JOB=$1   # capacity fig4a driver already running
while kill -0 "$JOB" 2>/dev/null; do sleep 20; done
L=results/logs
$PY -u run_benchmark.py fig4b --methods compass_shared_t256 compass_shared_t64 --budgets 1000 > $L/driver_capacity_fig4b.log 2>&1
U="compass_upstream_v11 compass_upstream_v11_fixeps --resample-from compass_upstream_v11"
$PY -u run_benchmark.py fig4a --methods $U --budgets 1000 10000 > $L/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 1000 10000 > $L/driver_upstream_fig4b.log 2>&1
for n in 4 20; do for cfg in "upstream stock" "upstream batch" "ours batch"; do
  taskset -c $($PY -c "import sys; sys.path.insert(0,'.'); from run_benchmark import pick_idle_cpus; print(','.join(map(str,pick_idle_cpus(3))))") \
    $PY diagnostics/step_time.py $cfg $n 200 2>&1 | grep s_per_step >> diagnostics/out/step_time.txt
done; done
$PY -u run_benchmark.py fig4b --methods score_transformer compass_legacy_full_fixeps --resample-from compass_legacy_full >> $L/driver_fig4b_fixeps.log 2>&1
$PY -u run_benchmark.py fig4a --methods $U --budgets 100000 >> $L/driver_upstream_fig4a.log 2>&1
$PY -u run_benchmark.py fig4b --methods $U --budgets 100000 >> $L/driver_upstream_fig4b.log 2>&1
