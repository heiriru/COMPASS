#!/usr/bin/env bash
# Attribution across datasets, seeds and n. One GPU, sequential: the point is
# whether "the rule, not the network" survives replication, not throughput.
set -u
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
SCRIPT=/export/home/rheinric/COMPASS/inverse_kinematics/experiments/composition_attribution.py
OUT=/export/home/rheinric/COMPASS/inverse_kinematics/artifacts/composition_attribution
COMMON="--single-draws 20000 --draws 4000 --reference-draws 200000 --single-reference-draws 100000"

run () {  # run <name> <extra args...>
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $name"
  $PY "$SCRIPT" $COMMON "$@" --output "$OUT/$name" 2>&1 \
    | grep -Ev '^\s*$' | sed 's/^/    /'
  echo "=== [$(date +%H:%M:%S)] exit $? for $name"
}

run selected_n8_seed99  --dataset selected --select-x1 0.15 --observations 8  --seed 99
run selected_n8_seed101 --dataset selected --select-x1 0.15 --observations 8  --seed 101
run selected_n16_seed99 --dataset selected --select-x1 0.15 --observations 16 --seed 99
run exact_zero_n8_seed99 --dataset exact_zero --observations 8 --seed 99
run prior_n8_seed3      --dataset prior --observations 8 --seed 3
