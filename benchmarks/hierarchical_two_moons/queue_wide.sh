#!/usr/bin/env bash
# Wider global training prior (1.5x sd): train both networks (in parallel), then evaluate
# their arms in the small-sigma regime and on the full prior. One GPU (caller's CUDA_VISIBLE_DEVICES).
set -uo pipefail
cd "$(dirname "$0")"
./run.sh train --methods attn_hier_w1.5:gauss_jacobian_FD > logs/train_attn_hier_w1.5.log 2>&1 &
./run.sh train --methods legacy_w1.5:gauss_jacobian_FD > logs/train_legacy_w1.5.log 2>&1 &
wait
for regime in "--sigma-max 0.6" ""; do
  tag=${regime:+smax}; tag=${tag:-full}
  ./run.sh run $regime --methods attn_hier_w1.5:gauss_jacobian_FD > logs/wide_${tag}_attn.log 2>&1 &
  ./run.sh run $regime --methods legacy_w1.5:gauss_jacobian_FD > logs/wide_${tag}_ljac.log 2>&1 &
  ./run.sh run $regime --methods legacy_w1.5:gauss_hierarchical > logs/wide_${tag}_lgh.log 2>&1 &
done
wait
./run.sh report --sigma-max 0.6 > logs/wide_report.log 2>&1
./run.sh report >> logs/wide_report.log 2>&1
echo all done
