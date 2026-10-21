#!/bin/bash
# Sequential: Fig. 4a resampled with the eps fix, then Fig. 4b (fixed COMPASS arm).
cd /export/home/rheinric/COMPASS/benchmarks/SimFormer
../../.COMPASS/bin/python -u run_benchmark.py fig4a --methods compass_legacy_full_fixeps --resample-from compass_legacy_full > results/logs/driver_fig4a_fixeps.log 2>&1
../../.COMPASS/bin/python -u run_benchmark.py fig4b --methods score_transformer compass_legacy_full_fixeps --resample-from compass_legacy_full > results/logs/driver_fig4b_fixeps.log 2>&1
