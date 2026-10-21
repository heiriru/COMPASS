#!/bin/bash
# Resume the paused follow-up arm evaluations once the churn sweep has stopped.
#
# The recovery jobs are CPU/IO bound, not GPU bound, so running six at once
# divides the same cores six ways instead of finishing anything sooner. The arm
# evaluations are SIGSTOPped while the churn sweep -- which is much further
# along -- runs to its 30-dataset target, then SIGCONTed here. SIGSTOP freezes
# the process with its state and GPU allocation intact, so nothing is recomputed.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling

while pgrep -f "run_churn_comparison.py --scratch-base $PP/ladder_paper/tok_trial_h64d2" >/dev/null; do
  sleep 60
done
echo "[$(date -Is)] churn sweep finished; resuming arm evaluations"
for arm in mask6 te64 pph; do
  pid=$(pgrep -f "run_churn_comparison.py --scratch-base .*ladder_followup/$arm " | head -1)
  if [ -n "$pid" ]; then
    kill -CONT "$pid" && echo "[$(date -Is)] resumed $arm (pid $pid)"
  fi
done
