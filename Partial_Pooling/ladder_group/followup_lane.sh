#!/bin/bash
# Two things gate a quotable claim, in the order they gate it.
#
# 1. Partition sensitivity. The exact Bayesian posterior cannot depend on which
#    subjects share a composed group, so any dependence is an artifact this
#    representation introduces. Eight random re-assignments of the 100 test
#    subjects into 5 groups of 20, on gauss_hierarchical because it is the
#    cheapest rule (2.5 min/dataset) and the three rules agreed to within 0.003
#    at K=20 anyway.
#
#    Note: --grouping-seed permutes all 100 subjects and then reshapes, so it
#    varies *both* which subjects share a group and their slot order at once.
#    The plan asks for these separately; running them together first is the
#    stronger test, since stability under the combined perturbation implies
#    stability under either alone. Only if this shows instability is it worth
#    building a slot-only permutation to say which half caused it.
#
# 2. The 100-dataset paper-matched confirmation on the winning arm. 30-dataset
#    global numbers are exactly what docs/followup_results.md warns must not be
#    read at the hundredths -- one sample per dataset.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder_group/logs
mkdir -p "$LOGS"

BACKBONE="--arch tokenized --tokenization group --train-subjects 20 \
          --train-size 100000 --validation-size 5000 \
          --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4 \
          --mask-menu six_mixed --mask-bernoulli-fraction 0.25"
COMMON="--preset small --subjects 100 --draws 100 --timesteps 50 \
        --gaussian-precision-samples 256 --dpm-corrector-steps 0 \
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep"

for SEED in 1 2 3 4 5 6 7 8; do
  echo "=== [$(date -Is)] partition seed ${SEED}"
  $PY "$PP/run_churn_comparison.py" \
      --scratch-base "$PP/ladder_group/gK20_part${SEED}" \
      --churn-eta 2 --inference-method dpm2_gauss_hierarchical \
      --grouping-seed "${SEED}" --datasets 30 \
      $BACKBONE $COMMON > "${LOGS}/part-seed${SEED}.log" 2>&1
  status=$?
  echo "=== [$(date -Is)] partition seed ${SEED} exit ${status}"
done
echo "PARTITION STUDY COMPLETE $(date -Is)"

for SPEC in "hier:--churn-eta 2 --inference-method dpm2_gauss_hierarchical" \
            "fnpse:--churn-eta 0 --inference-method langevin_fnpse"; do
  TAG=${SPEC%%:*}; RULE=${SPEC#*:}
  echo "=== [$(date -Is)] 100-dataset ${TAG}"
  $PY "$PP/run_churn_comparison.py" \
      --scratch-base "$PP/ladder_group/gK20_paper" \
      $RULE --datasets 100 \
      $BACKBONE $COMMON > "${LOGS}/paper100-${TAG}.log" 2>&1
  status=$?
  echo "=== [$(date -Is)] 100-dataset ${TAG} exit ${status}"
done
echo "FOLLOWUP LANE COMPLETE $(date -Is)"
