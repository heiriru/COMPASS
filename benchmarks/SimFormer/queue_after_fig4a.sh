#!/bin/bash
# Runs after the fig4a driver exits: eps ablation, then fig4b. Strictly sequential.
while kill -0 924011 2>/dev/null; do sleep 30; done
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
../../.COMPASS/bin/python -u run_benchmark.py fig4a --methods compass_legacy_full_eps5 --budgets 10000 --tasks two_moons gaussian_mixture >> results/logs/driver_fig4a.log 2>&1
../../.COMPASS/bin/python -u run_benchmark.py fig4b > results/logs/driver_fig4b.log 2>&1
