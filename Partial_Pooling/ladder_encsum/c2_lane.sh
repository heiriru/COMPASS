#!/usr/bin/env bash
# C2 = setenc-d16-w64x2s4-h96d6-log_sigma-te64-mask6mix, the best K=1 arm on
# mean r (0.842 at 30 datasets under gauss_hierarchical only).
#
# Three runs, all at the screen protocol (100 subjects, 100 draws, 50 timesteps)
# so they land on the same axis as the published 0.842:
#   1. gauss_jacobian  at 30 datasets  -- sampler robustness
#   2. Langevin F-NPSE at 30 datasets  -- sampler robustness
#   3. gauss_hierarchical, datasets 30..99, as a shard -- completes the
#      100-dataset paper-matched run by reusing the existing first 30, which
#      were produced at identical settings.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder_encsum/logs
NAME=setenc-d16-w64x2s4-h96d6-log_sigma-te64-mask6mix
COMMON="--name $NAME --subjects 100 --draws 100 --timesteps 50"

echo "$(date -Is) C2 lane start"

$PY "$PP/infer_encoder_summary.py" --method dpm2_gauss_jacobian_newton \
    --datasets 30 $COMMON > "$LOGS/c2-jacobian-30.log" 2>&1 &
pid1=$!
sleep 120

$PY "$PP/infer_encoder_summary.py" --method langevin_fnpse \
    --datasets 30 $COMMON > "$LOGS/c2-fnpse-30.log" 2>&1 &
pid2=$!
sleep 120

$PY "$PP/infer_encoder_summary.py" --method dpm2_gauss_hierarchical \
    --dataset-start 30 --datasets 70 --shard d030-099 $COMMON \
    > "$LOGS/c2-hier-100.log" 2>&1 &
pid3=$!

wait $pid1; echo "$(date -Is) jacobian exit $?"
wait $pid2; echo "$(date -Is) fnpse exit $?"
wait $pid3; echo "$(date -Is) hier-70 exit $?"
echo "$(date -Is) C2 LANE COMPLETE"
