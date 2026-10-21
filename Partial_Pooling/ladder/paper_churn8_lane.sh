#!/bin/bash
# Churn sweep at the paper-matched protocol (100 datasets x 100 subjects x 100
# draws) for the 463,364-parameter trial-h64/d2 arm. eta=2 already exists in
# ladder_paper/tok_trial_h64d2; this lane adds eta=8 (and eta=4 as the midpoint)
# so the global calibration error can be read as a function of churn rather than
# at a single setting.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder/logs
mkdir -p "$LOGS"
MODEL="--train-subset 100000 --arch tokenized --tokenization trial \
       --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4"
COMMON=(--preset small --datasets 100 --subjects 100 --draws 100 --timesteps 50
        --gaussian-precision-samples 256 --dpm-corrector-steps 0
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep)

run() {
  local name=$1; shift
  echo "=== [$(date -Is)] starting paperchurn/$name" >&2
  "$PY" "$PP/run_churn_comparison.py" \
      --scratch-base "$PP/ladder_paper/tok_trial_h64d2" "$@" $MODEL "${COMMON[@]}" \
      >"$LOGS/paperchurn-$name.log" 2>&1
  echo "=== [$(date -Is)] paperchurn/$name exit $?" >&2
}

case "$1" in
  hier8)  run hier_churn8  --churn-eta 8 --inference-method dpm2_gauss_hierarchical ;;
  jac8)   run jacobian_churn8 --churn-eta 8 --jacobian-refresh 20 \
              --inference-method dpm2_gauss_jacobian_newton ;;
  hier4)  run hier_churn4  --churn-eta 4 --inference-method dpm2_gauss_hierarchical ;;
  jac4)   run jacobian_churn4 --churn-eta 4 --jacobian-refresh 20 \
              --inference-method dpm2_gauss_jacobian_newton ;;
esac
