#!/bin/bash
# 30-dataset screen for one grouped arm and one inference method, on the same
# protocol docs/followup_results.md used for the flat arms it is measured
# against: 100 subjects x 100 draws x 50 steps, kde_global_then_local_ascent.
#
# 30 datasets is a screen, not the headline. The plan promotes only the winner
# to the 100-dataset paper-matched run, because a global metric has one sample
# per dataset and 30 of them is too noisy to read at the hundredths.
#
# Usage: eval_lane.sh <K> <method> [datasets]
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder_group/logs
mkdir -p "$LOGS"

K=$1; METHOD=$2; DATASETS=${3:-30}

BACKBONE="--arch tokenized --tokenization group --train-subjects ${K} \
          --train-size 100000 --validation-size 5000 \
          --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4 \
          --mask-menu six_mixed --mask-bernoulli-fraction 0.25"

case "$METHOD" in
  hier_churn2)     CHURN="--churn-eta 2 --inference-method dpm2_gauss_hierarchical" ;;
  jacobian_newton) CHURN="--churn-eta 2 --jacobian-refresh 20 --inference-method dpm2_gauss_jacobian_newton" ;;
  fnpse)           CHURN="--churn-eta 0 --inference-method langevin_fnpse" ;;
  *) echo "unknown method $METHOD" >&2; exit 2 ;;
esac

# --map-estimator kde_global_then_local_ascent is required, not preferred: the
# joint_score path reuses a (subjects, 10, 10) covariance and a flat [theta, x]
# row layout, neither of which survives grouping, and refuses outright.
COMMON=(--preset small --datasets "$DATASETS" --subjects 100 --draws 100 --timesteps 50
        --gaussian-precision-samples 256 --dpm-corrector-steps 0
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep)

echo "=== [$(date -Is)] starting gK${K}/${METHOD} (${DATASETS} datasets)" >&2
"$PY" "$PP/run_churn_comparison.py" \
    --scratch-base "$PP/ladder_group/gK${K}" \
    $CHURN $BACKBONE "${COMMON[@]}" \
    >"$LOGS/eval-gK${K}-${METHOD}.log" 2>&1
status=$?
echo "=== [$(date -Is)] gK${K}/${METHOD} exit ${status}" >&2
