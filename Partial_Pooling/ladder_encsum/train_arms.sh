#!/usr/bin/env bash
# Train the summary-space arms for the two questions on the table:
#   (1) can the frozen-summary hierarchical model be improved substantially?
#   (2) does spending te=256's budget on score capacity instead pay off?
#
# Every arm reuses the same frozen 16-D encoder and log_sigma time sampling.
# autocvd inside each script claims one free GPU, so the launches are staggered
# far enough apart that a starting job registers its memory before the next
# one looks.
set -u
cd /export/home/rheinric/COMPASS
PY=.COMPASS/bin/python
LOGS=Partial_Pooling/ladder_encsum/logs
MAX_CONCURRENT=4
STAGGER=100

# label : hidden : depth : heads : te : recipe : subset
ARMS=(
  "A1-h48d2-te256-mask6mix:48:2:4:256:mask6mix:100000"
  "B1-h48d2-te64-mask6mix:48:2:4:64:mask6mix:100000"
  "C0-h64d4-te64-mask6mix:64:4:8:64:mask6mix:100000"
  "C1-h64d4-te256-mask6mix:64:4:8:256:mask6mix:100000"
  "C2-h96d6-te64-mask6mix:96:6:8:64:mask6mix:100000"
  "C3-h96d6-te64-six_w6:96:6:8:64:six_w6:100000"
  "D1-h96d6-te64-mask6mix-400k:96:6:8:64:mask6mix:400000"
)

for arm in "${ARMS[@]}"; do
  IFS=: read -r label h d heads te recipe subset <<< "$arm"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 20; done
  echo "$(date -Is) launching $label"
  (
    $PY Partial_Pooling/train_encoder_summary.py \
      --train-subset "$subset" --hidden-size "$h" --depth "$d" --heads "$heads" \
      --time-embedding-size "$te" --mask-recipe "$recipe" \
      --epochs 500 --patience 20 --time-sampling log_sigma
    status=$?
    echo "$(date -Is) $label finished exit $status"
  ) > "$LOGS/train-$label.log" 2>&1 &
  sleep "$STAGGER"
done
wait
echo "$(date -Is) all training arms done"
