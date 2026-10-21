#!/bin/bash
# Wait for the lane, then assemble the paper-matched comparison and plot it
# into its own artifact root.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=$PP/churn100_paper

while [ "$(grep -c ' exit ' $L/logs/lane.status 2>/dev/null)" -lt 4 ]; do
  sleep 120
done

echo "=== [$(date -Is)] lane finished"
grep ' exit ' $L/logs/lane.status

for arm in jacobian_newton hier_churn2 hier_stock fnpse; do
  n=$(ls $L/$arm/artifacts_churn_eta*/partial_pooling_recovery/small/*/*-dataset-*.pt 2>/dev/null | wc -l)
  echo "  $arm: $n dataset artifacts"
  [ "$n" -ne 100 ] && echo "  !! $arm did not complete 100 datasets"
done

taskset -c 8-10 "$PY" "$PP/build_churn_comparison.py" \
  --output-root "$PP/artifacts_churn_comparison_paper" \
  --arm churn100_paper/jacobian_newton/artifacts_churn_eta2_refresh20:dpm2_gauss_jacobian_newton \
  --arm churn100_paper/hier_churn2/artifacts_churn_eta2:dpm2_gauss_hierarchical:churn2 \
  --arm churn100_paper/hier_stock/artifacts_churn_eta0:dpm2_gauss_hierarchical \
  --arm churn100_paper/fnpse/artifacts_churn_eta0:langevin_fnpse
echo "=== [$(date -Is)] build exit $?"
