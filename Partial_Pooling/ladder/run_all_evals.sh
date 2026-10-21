#!/bin/bash
# All ladder arms x all three inference methods, staggered so autocvd hands each
# process its own free GPU.
set -u
L=/export/home/rheinric/COMPASS/Partial_Pooling/ladder

SHAPE="--hidden-size 16 --depth 2 --num-heads 2 --mlp-ratio 4"
declare -A FLAGS=(
  [legacy_h16d2]="--train-subset 100000 --arch legacy $SHAPE"
  [tok_scalar_h16d2]="--train-subset 100000 --arch tokenized --tokenization scalar $SHAPE"
  [tok_trial_h16d2]="--train-subset 100000 --arch tokenized --tokenization trial $SHAPE"
)

for arm in tok_trial_h16d2 tok_scalar_h16d2 legacy_h16d2; do
  for method in jacobian_newton hier_churn2 fnpse; do
    nohup "$L/eval_lane.sh" "$arm" "$method" -- ${FLAGS[$arm]} \
        >> "$L/logs/eval.status" 2>&1 &
    sleep 90
  done
done
wait
