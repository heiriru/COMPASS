#!/usr/bin/env bash
# E-C9.2: cross-evaluate the best-validation checkpoint of every nu = 0.5 B5/B6 network.
cd "$(dirname "$0")/../.."
PY=.COMPASS/bin/python
mkdir -p validation/c9_network_seeds/logs
for family in gauss student bimodal; do for budget in 4096 16384 65536; do for seed in 0 1 2; do
  $PY -m validation.b5_b6_pfode_accuracy.evaluate --family $family --nu 0.5 --budget $budget \
      --seed $seed --cross-only --best-checkpoint > validation/c9_network_seeds/logs/best_${family}_${budget}_${seed}.log 2>&1
  echo "$family $budget $seed -> exit $?"
done; done; done
