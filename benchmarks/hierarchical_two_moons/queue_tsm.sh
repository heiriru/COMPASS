#!/usr/bin/env bash
# Target score matching on the globals: train attn_hier_tsm, score-check it against the exact
# per-group score, then run its arms on the max(sigma) <= 0.6 datasets. GPU from the caller.
set -uo pipefail
cd "$(dirname "$0")"
B=$(pwd); C=$(cd ../.. && pwd)
./run.sh train --methods attn_hier_tsm:gauss_jacobian_FD > logs/train_attn_hier_tsm.log 2>&1
PYTHONPATH=$B/.vendor:$B/external/hsbibm:$C/src:$C/Arruda/paper_benchmarks:$C/ModeTree/experiments \
  $C/.COMPASS/bin/python -u checks/score_check.py attn_hier_tsm 10 25 100 > logs/score_check_attn_hier_tsm.log 2>&1 &
./run.sh run --sigma-max 0.6 --methods attn_hier_tsm:arruda_direct > logs/tsm_smax_direct.log 2>&1 &
./run.sh run --sigma-max 0.6 --methods attn_hier_tsm:gauss_jacobian_FD > logs/tsm_smax_jac.log 2>&1 &
wait
echo all done
