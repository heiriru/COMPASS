#!/bin/bash
# Stop each running recovery job once it has completed 30 test datasets.
#
# run_churn_comparison writes one "<method>-dataset-<i>.pt" per dataset as that
# dataset finishes, so the results already on disk are complete and nothing is
# recomputed by stopping: when 30 files exist, datasets 0-29 are done. The
# scorer's --datasets 30 then truncates every arm -- including the finished
# 100-dataset reference runs -- to exactly those same 30, so the comparison
# stays like-for-like.
#
# The job is killed between datasets rather than mid-dataset wherever possible,
# and a half-written file would simply be the 31st, which --datasets 30 ignores.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
TARGET=${1:-30}

declare -A JOBS=(
  ["$PP/ladder_paper/tok_trial_h64d2/artifacts_churn_eta8"]="--scratch-base $PP/ladder_paper/tok_trial_h64d2 --churn-eta 8 --inference-method dpm2_gauss_hierarchical"
  ["$PP/ladder_paper/tok_trial_h64d2/artifacts_churn_eta8_refresh20"]="--scratch-base $PP/ladder_paper/tok_trial_h64d2 --churn-eta 8 --jacobian-refresh 20 --inference-method dpm2_gauss_jacobian_newton"
  ["$PP/ladder_paper/tok_trial_h64d2/artifacts_churn_eta4"]="--scratch-base $PP/ladder_paper/tok_trial_h64d2 --churn-eta 4 --inference-method dpm2_gauss_hierarchical"
  ["$PP/ladder_followup/mask6/artifacts_churn_eta2"]="--scratch-base $PP/ladder_followup/mask6 --churn-eta 2"
  ["$PP/ladder_followup/te64/artifacts_churn_eta2"]="--scratch-base $PP/ladder_followup/te64 --churn-eta 2"
  ["$PP/ladder_followup/pph/artifacts_churn_eta2"]="--scratch-base $PP/ladder_followup/pph --churn-eta 2"
)

while true; do
  remaining=0
  for dir in "${!JOBS[@]}"; do
    pattern=${JOBS[$dir]}
    pid=$(pgrep -f -- "run_churn_comparison.py $pattern" | head -1)
    [ -z "$pid" ] && continue
    remaining=$((remaining + 1))
    count=$(find "$dir" -name '*dataset-*.pt' 2>/dev/null | wc -l)
    if [ "$count" -ge "$TARGET" ]; then
      echo "[$(date -Is)] $dir reached $count datasets; stopping pid $pid"
      kill "$pid"
    fi
  done
  [ "$remaining" -eq 0 ] && { echo "[$(date -Is)] all jobs stopped"; break; }
  sleep 30
done
