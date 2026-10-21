#!/bin/bash
# Run a list of (arm, method) evaluations with a bounded number of concurrent
# GPUs, so the shared host keeps free devices for other users. Jobs already
# running are counted against the limit.
set -u
L=/export/home/rheinric/COMPASS/Partial_Pooling/ladder
MAX=${MAX_CONCURRENT:-5}

SHAPE="--hidden-size 16 --depth 2 --num-heads 2 --mlp-ratio 4"
declare -A FLAGS=(
  [legacy_h16d2]="--train-subset 100000 --arch legacy $SHAPE"
  [tok_scalar_h16d2]="--train-subset 100000 --arch tokenized --tokenization scalar $SHAPE"
  [tok_trial_h16d2]="--train-subset 100000 --arch tokenized --tokenization trial $SHAPE"
)

running() { pgrep -fc "run_churn_comparison.py" || true; }

for job in "$@"; do
  arm=${job%%:*}; method=${job##*:}
  while [ "$(running)" -ge "$MAX" ]; do sleep 60; done
  echo "=== [$(date -Is)] queueing $arm/$method (running: $(running))" >&2
  nohup "$L/eval_lane.sh" "$arm" "$method" -- ${FLAGS[$arm]} \
      >> "$L/logs/eval.status" 2>&1 &
  sleep 90
done
wait
