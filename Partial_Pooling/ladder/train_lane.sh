#!/bin/bash
# Ladder training lane. Every arm sees the same 100,000 simulations (the first
# 100,000 rows of the preset-small split), the same normalization constants and
# the same 100 test datasets as the frozen 5,957,984-parameter baseline, so the
# only thing that varies between arms is the network.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder/logs

COMMON=(--preset small --train-subset 100000 --device cuda)

run_arm() {
  local name=$1; shift
  echo "=== [$(date -Is)] starting $name" >&2
  "$PY" "$PP/train_models.py" "${COMMON[@]}" "$@" >"$LOGS/$name.log" 2>&1
  echo "=== [$(date -Is)] $name exit $?" >&2
}

case "$1" in
  legacy)  run_arm legacy_h16d2 --arch legacy \
             --hidden-size 16 --depth 2 --num-heads 2 --mlp-ratio 4 ;;
  scalar)  run_arm tok_scalar_h16d2 --arch tokenized --tokenization scalar \
             --hidden-size 16 --depth 2 --num-heads 2 --mlp-ratio 4 ;;
  trial)   run_arm tok_trial_h16d2 --arch tokenized --tokenization trial \
             --hidden-size 16 --depth 2 --num-heads 2 --mlp-ratio 4 ;;
esac
