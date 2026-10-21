#!/bin/bash
# Paper-matched protocol for the 5,957,984-parameter legacy baseline, so every
# inference method has both arms at 100 subjects. Mirrors churn100_paper's
# COMMON exactly; the jacobian_newton arm already exists there and is not rerun.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder/logs
MODEL="--train-subset 100000 --arch legacy --hidden-size 16 --depth 2 --num-heads 2 --mlp-ratio 4"
COMMON=(--preset small --datasets 100 --subjects 100 --draws 100 --timesteps 50
        --gaussian-precision-samples 256 --dpm-corrector-steps 0
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep)

run() {
  local name=$1; shift
  echo "=== [$(date -Is)] starting paperlegacy/$name" >&2
  "$PY" "$PP/run_churn_comparison.py" \
      --scratch-base "$PP/ladder_paper/legacy_h16d2" "$@" $MODEL "${COMMON[@]}" \
      >"$LOGS/paperlegacy-$name.log" 2>&1
  echo "=== [$(date -Is)] paperlegacy/$name exit $?" >&2
}

case "$1" in
  jac)   run jacobian_newton --churn-eta 2 --jacobian-refresh 20 \
             --inference-method dpm2_gauss_jacobian_newton ;;
  hier)  run hier_churn2 --churn-eta 2 --inference-method dpm2_gauss_hierarchical ;;
  fnpse) run fnpse --churn-eta 0 --inference-method langevin_fnpse ;;
esac
