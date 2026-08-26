#!/bin/bash
# One lane of the 100-dataset churn comparison. Arms run sequentially inside a
# lane; lanes run in parallel on separate free GPUs (autocvd picks them).
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/churn100/logs

# Shared with the 5-dataset run this reproduces; only --datasets changes.
COMMON=(--preset small --datasets 100 --subjects 20 --draws 256 --timesteps 50
        --gaussian-precision-samples 256 --dpm-corrector-steps 0
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep)

run_arm() {  # name, scratch-base subdir, then run_churn_comparison args
  local name=$1 base=$2; shift 2
  echo "=== [$(date -Is)] starting $name" >&2
  "$PY" "$PP/run_churn_comparison.py" \
      --scratch-base "$PP/churn100/$base" "$@" "${COMMON[@]}" \
      >"$LOGS/$name.log" 2>&1
  echo "=== [$(date -Is)] $name exit $?" >&2
}

case "$1" in
  A)
    run_arm jacobian_newton jacobian_newton \
        --churn-eta 2 --jacobian-refresh 20 \
        --inference-method dpm2_gauss_jacobian_newton
    run_arm hier_stock hier_stock \
        --churn-eta 0 --inference-method dpm2_gauss_hierarchical
    ;;
  B)
    run_arm hier_churn2 hier_churn2 \
        --churn-eta 2 --inference-method dpm2_gauss_hierarchical
    run_arm fnpse fnpse \
        --churn-eta 0 --inference-method langevin_fnpse
    ;;
esac
