#!/bin/bash
# Wait for both lanes, then assemble the 100-dataset comparison and plot it.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=$PP/churn100

while [ "$(cat $L/logs/laneA.status $L/logs/laneB.status 2>/dev/null | grep -c ' exit ')" -lt 4 ]; do
  sleep 60
done

echo "=== [$(date -Is)] lanes finished"
grep ' exit ' $L/logs/laneA.status $L/logs/laneB.status

for arm in jacobian_newton hier_churn2 hier_stock fnpse; do
  n=$(ls $L/$arm/artifacts_churn_eta*/partial_pooling_recovery/small/*/*-dataset-*.pt 2>/dev/null | wc -l)
  echo "  $arm: $n dataset artifacts"
  if [ "$n" -ne 100 ]; then echo "  !! $arm did not complete 100 datasets"; fi
done

taskset -c 8-10 "$PY" "$PP/build_churn_comparison.py" \
  --output-root "$PP/artifacts_churn_comparison_100" \
  --arm churn100/jacobian_newton/artifacts_churn_eta2_refresh20:dpm2_gauss_jacobian_newton \
  --arm churn100/hier_churn2/artifacts_churn_eta2:dpm2_gauss_hierarchical:churn2 \
  --arm churn100/hier_stock/artifacts_churn_eta0:dpm2_gauss_hierarchical \
  --arm churn100/fnpse/artifacts_churn_eta0:langevin_fnpse
echo "=== [$(date -Is)] build exit $?"
