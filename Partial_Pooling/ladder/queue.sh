#!/bin/bash
# Queue recovery evaluations with a bounded GPU concurrency.
# Each argument is  ARM:METHOD  where ARM is a key of FLAGS below.
set -u
L=/export/home/rheinric/COMPASS/Partial_Pooling/ladder
MAX=${MAX_CONCURRENT:-5}
BASE="--train-subset 100000 --mlp-ratio 4"

declare -A FLAGS=(
  [legacy_h16d2]="$BASE --arch legacy --hidden-size 16 --depth 2 --num-heads 2"
  [tok_scalar_h16d2]="$BASE --arch tokenized --tokenization scalar --hidden-size 16 --depth 2 --num-heads 2"
  [tok_scalar_h32d4]="$BASE --arch tokenized --tokenization scalar --hidden-size 32 --depth 4 --num-heads 4"
  [tok_scalar_h48d4]="$BASE --arch tokenized --tokenization scalar --hidden-size 48 --depth 4 --num-heads 4"
  [tok_scalar_h64d4]="$BASE --arch tokenized --tokenization scalar --hidden-size 64 --depth 4 --num-heads 8"
  [tok_trial_h16d2]="$BASE --arch tokenized --tokenization trial --hidden-size 16 --depth 2 --num-heads 2"
  [tok_trial_h16d3]="$BASE --arch tokenized --tokenization trial --hidden-size 16 --depth 3 --num-heads 2"
  [tok_trial_h16d4]="$BASE --arch tokenized --tokenization trial --hidden-size 16 --depth 4 --num-heads 2"
  [tok_trial_h32d2]="$BASE --arch tokenized --tokenization trial --hidden-size 32 --depth 2 --num-heads 4"
  [tok_trial_h32d3]="$BASE --arch tokenized --tokenization trial --hidden-size 32 --depth 3 --num-heads 4"
  [tok_trial_h32d4]="$BASE --arch tokenized --tokenization trial --hidden-size 32 --depth 4 --num-heads 4"
  [tok_trial_h48d2]="$BASE --arch tokenized --tokenization trial --hidden-size 48 --depth 2 --num-heads 4"
  [tok_trial_h48d3]="$BASE --arch tokenized --tokenization trial --hidden-size 48 --depth 3 --num-heads 4"
  [tok_trial_h48d4]="$BASE --arch tokenized --tokenization trial --hidden-size 48 --depth 4 --num-heads 4"
  [tok_trial_h64d2]="$BASE --arch tokenized --tokenization trial --hidden-size 64 --depth 2 --num-heads 8"
  [tok_trial_h64d3]="$BASE --arch tokenized --tokenization trial --hidden-size 64 --depth 3 --num-heads 8"
  [tok_trial_h64d4]="$BASE --arch tokenized --tokenization trial --hidden-size 64 --depth 4 --num-heads 8"
)

running() { pgrep -fc "run_churn_comparison.py" || true; }

for job in "$@"; do
  arm=${job%%:*}; method=${job##*:}
  [ -z "${FLAGS[$arm]+x}" ] && { echo "unknown arm $arm" >&2; exit 2; }
  while [ "$(running)" -ge "$MAX" ]; do sleep 60; done
  echo "=== [$(date -Is)] queueing $arm/$method (running: $(running))" >&2
  nohup "$L/eval_lane.sh" "$arm" "$method" -- ${FLAGS[$arm]} \
      >> "$L/logs/eval.status" 2>&1 &
  sleep 90
done
wait
