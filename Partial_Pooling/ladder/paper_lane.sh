#!/bin/bash
# Paper-matched protocol, mirroring churn100_paper/run_lane.sh exactly:
# 100 test datasets x 100 subjects x 100 posterior draws, which is the
# N_TEST / N_SUBJECTS / N_SAMPLES triple in
# bayesflow-org/diffusion-experiments case_study4/settings.py.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder/logs
MODEL="--train-subset 100000 --arch tokenized --tokenization trial \
       --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4"
COMMON=(--preset small --datasets 100 --subjects 100 --draws 100 --timesteps 50
        --gaussian-precision-samples 256 --dpm-corrector-steps 0
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep)

run() {
  local name=$1; shift
  echo "=== [$(date -Is)] starting paper/$name" >&2
  "$PY" "$PP/run_churn_comparison.py" \
      --scratch-base "$PP/ladder_paper/tok_trial_h64d2" "$@" $MODEL "${COMMON[@]}" \
      >"$LOGS/paper-$name.log" 2>&1
  echo "=== [$(date -Is)] paper/$name exit $?" >&2
}

case "$1" in
  jac)  run jacobian_newton --churn-eta 2 --jacobian-refresh 20 \
            --inference-method dpm2_gauss_jacobian_newton ;;
  hier) run hier_churn2 --churn-eta 2 --inference-method dpm2_gauss_hierarchical ;;
  fnpse) run fnpse --churn-eta 0 --inference-method langevin_fnpse ;;
esac
