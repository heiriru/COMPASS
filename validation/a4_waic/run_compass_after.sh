#!/usr/bin/env bash
# Wait until every network of a family/budget/seed exists, then run COMPASS WAIC for each true
# model (launches staggered so autocvd never hands one GPU to two jobs).
#   bash validation/a4_waic/run_compass_after.sh FAMILY BUDGET SEED "LV LP SP RM" REPS DRAWS DIV [TAG]
cd "$(dirname "$0")/../.."
FAM=$1; BUDGET=$2; SEED=$3; MODELS=$4; REPS=$5; DRAWS=$6; DIV=$7; TAG=$8
ROOT=/export/data/rheinric/compass_validation/a4_waic/models
LOG=validation/a4_waic/logs
for m in $MODELS; do
  until [ -f "$ROOT/$FAM/$m/n${BUDGET}_s${SEED}/meta.json" ]; do sleep 60; done
done
for t in $MODELS; do
  log=$LOG/compass_${FAM}_n${BUDGET}_s${SEED}_${t}${TAG:+_$TAG}.log
  .COMPASS/bin/python -m validation.a4_waic.run_compass --family $FAM --level trained --true $t \
      --reps $REPS --draws $DRAWS --waic-divergence $DIV --budget $BUDGET --net-seed $SEED \
      ${TAG:+--tag $TAG} > $log 2>&1 &
  until grep -q "Selected physical GPU" $log 2>/dev/null; do sleep 10; done
  sleep 90
done
wait
