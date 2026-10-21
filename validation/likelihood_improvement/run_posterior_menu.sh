#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
PY=.COMPASS/bin/python
LOG=validation/likelihood_improvement/logs
ROOT=/export/data/rheinric/compass_validation/b5b6/models
for family in gauss student bimodal; do for seed in 0 1 2; do
  until [ -f "$ROOT/${family}_nu0.5/n65536_s${seed}_menu/final.pt" ]; do sleep 30; done
  $PY -m validation.likelihood_improvement.posterior_check --family $family --seed $seed \
      --variant menu > "$LOG/post_${family}_s${seed}_menu.log" 2>&1
  echo "$family $seed done"
done; done
