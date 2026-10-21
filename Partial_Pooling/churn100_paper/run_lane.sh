#!/bin/bash
# Paper-matched run: 100 datasets x 100 subjects x 100 draws (Arruda et al. 2025,
# Case Study 4 partial pooling). One lane, one GPU, arms sequential.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/churn100_paper/logs

# Identical to the 20-subject run except --subjects and --draws.
COMMON=(--preset small --datasets 100 --subjects 100 --draws 100 --timesteps 50
        --gaussian-precision-samples 256 --dpm-corrector-steps 0
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep)

run_arm() {
  local name=$1 base=$2; shift 2
  echo "=== [$(date -Is)] starting $name" >&2
  "$PY" "$PP/run_churn_comparison.py" \
      --scratch-base "$PP/churn100_paper/$base" "$@" "${COMMON[@]}" \
      >"$LOGS/$name.log" 2>&1
  echo "=== [$(date -Is)] $name exit $?" >&2
}

run_arm jacobian_newton jacobian_newton \
    --churn-eta 2 --jacobian-refresh 20 \
    --inference-method dpm2_gauss_jacobian_newton
run_arm hier_churn2 hier_churn2 \
    --churn-eta 2 --inference-method dpm2_gauss_hierarchical
run_arm hier_stock hier_stock \
    --churn-eta 0 --inference-method dpm2_gauss_hierarchical
run_arm fnpse fnpse \
    --churn-eta 0 --inference-method langevin_fnpse
