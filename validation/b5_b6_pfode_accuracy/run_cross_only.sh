#!/usr/bin/env bash
# Recompute only the cross-family block for the nu = 0.5 gauss/bimodal networks.
cd "$(dirname "$0")/../.."
PY=.COMPASS/bin/python
LOG=validation/b5_b6_pfode_accuracy/logs
for family in gauss bimodal; do
  for budget in 4096 16384 65536; do
    for seed in 0 1 2; do
      $PY -m validation.b5_b6_pfode_accuracy.evaluate --family $family --nu 0.5 --budget $budget --seed $seed --cross-only > "$LOG/cross_${family}_${budget}_${seed}.log" 2>&1
      echo "$family $budget $seed -> exit $?"
    done
  done
done
