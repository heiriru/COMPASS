#!/bin/bash
# The two 200k arms missing from the mask/time-sampling ladder, on ONE pinned GPU.
#
# Existing at 200k: legacy_h16d2_n200000 (mixture + bernoulli, the published
# baseline, endpoint 0.097) and legacy_h16d2_n200000_six_weighted_log_sigma.
# Missing: log_sigma alone (isolates the time-sampling change) and the UNIFORM
# six menu (the one the API call was asked for).
#
# train_model.py has no autocvd, so CUDA_VISIBLE_DEVICES is pinned by the caller.
set -u
IK=/export/home/rheinric/COMPASS/inverse_kinematics
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$IK/artifacts/logs

for MENU in bernoulli six; do
  echo "=== [$(date -Is)] start legacy/h16d2 200k, mask=$MENU, time=log_sigma" >&2
  "$PY" "$IK/train_model.py" \
      --config h16d2 --arch legacy --simulations 200000 \
      --mask-menu "$MENU" --time-sampling log_sigma \
      --device cuda \
      > "$LOGS/train-200k-${MENU}-log_sigma.log" 2>&1
  echo "=== [$(date -Is)] exit $? for mask=$MENU" >&2
done
