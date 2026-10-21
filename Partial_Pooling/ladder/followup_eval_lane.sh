#!/bin/bash
# Paper-matched evaluation (100 test datasets x 100 subjects x 100 draws) for
# one follow-up arm and one inference method, on exactly the protocol the
# 463,364-parameter reference and the 5,957,984-parameter legacy baseline were
# measured with in ladder_paper/.
#
# Usage: followup_eval_lane.sh <arm> <method>
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder/logs
mkdir -p "$LOGS"

ARM=$1; METHOD=$2; DATASETS=${3:-100}; STEPS=${4:-50}
BACKBONE="--train-subset 100000 --arch tokenized --tokenization trial \
          --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4"

case "$ARM" in
  control)  EXTRA="" ;;
  mask6)    EXTRA="--mask-menu six" ;;
  mask6mix) EXTRA="--mask-menu six_mixed --mask-bernoulli-fraction 0.25" ;;
  mask6pw4) EXTRA="--mask-menu six --mask-posterior-weight 4" ;;
  mask6pw8) EXTRA="--mask-menu six --mask-posterior-weight 8" ;;
  te64)     EXTRA="--time-embedding-size 64" ;;
  pph)      EXTRA="--per-position-head" ;;
  te64pph)  EXTRA="--time-embedding-size 64 --per-position-head" ;;
  best)     EXTRA="--mask-menu six --time-embedding-size 64 --per-position-head" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

case "$METHOD" in
  jacobian_newton) CHURN="--churn-eta 2 --jacobian-refresh 20 --inference-method dpm2_gauss_jacobian_newton" ;;
  hier_churn2)     CHURN="--churn-eta 2 --inference-method dpm2_gauss_hierarchical" ;;
  fnpse)           CHURN="--churn-eta 0 --inference-method langevin_fnpse" ;;
  damping)         CHURN="--churn-eta 0 --inference-method langevin_damping" ;;
  gauss_damping)   CHURN="--churn-eta 2 --inference-method dpm2_gauss_damping" ;;
  hybrid)          CHURN="--churn-eta 2 --inference-method dpm2_hybrid" ;;
  hybrid_damping)  CHURN="--churn-eta 2 --inference-method dpm2_hybrid_damping" ;;
  *) echo "unknown method $METHOD" >&2; exit 2 ;;
esac

COMMON=(--preset small --datasets "$DATASETS" --subjects 100 --draws 100 --timesteps "$STEPS"
        --gaussian-precision-samples 256 --dpm-corrector-steps 0
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep)

echo "=== [$(date -Is)] starting followup/$ARM/$METHOD" >&2
"$PY" "$PP/run_churn_comparison.py" \
    --scratch-base "$PP/ladder_followup/$ARM${STEPS:+-t$STEPS}" \
    $CHURN $BACKBONE $EXTRA "${COMMON[@]}" \
    >"$LOGS/followup-$ARM-$METHOD-t$STEPS.log" 2>&1
echo "=== [$(date -Is)] followup/$ARM/$METHOD exit $?" >&2
