#!/bin/bash
# The full 30-dataset screen: every trained grouped arm against all three
# composition rules, so the mask-menu lesson is not repeated -- a representation
# change has to be checked on every sampler the benchmark deploys, not just the
# one it was selected on.
set -u
LANE=/export/home/rheinric/COMPASS/Partial_Pooling/ladder_group/eval_lane.sh
for K in 5 10 20; do
  for METHOD in hier_churn2 jacobian_newton fnpse; do
    bash "$LANE" "$K" "$METHOD"
  done
done
echo "EVAL LANE COMPLETE $(date -Is)"
