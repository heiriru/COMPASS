#!/usr/bin/env bash
# Posterior checks for baseline and log-lambda 2^16 networks (waits for loglam models).
cd "$(dirname "$0")/../.."
PY=.COMPASS/bin/python
LOG=validation/likelihood_improvement/logs
ROOT=/export/data/rheinric/compass_validation/b5b6/models
for family in gauss student bimodal; do for seed in 0 1 2; do
  $PY -m validation.likelihood_improvement.posterior_check --family $family --seed $seed \
      > "$LOG/post_${family}_s${seed}.log" 2>&1
  until [ -f "$ROOT/${family}_nu0.5/n65536_s${seed}_loglam/final.pt" ]; do sleep 30; done
  $PY -m validation.likelihood_improvement.posterior_check --family $family --seed $seed \
      --variant loglam > "$LOG/post_${family}_s${seed}_loglam.log" 2>&1
  echo "$family $seed done"
done; done
