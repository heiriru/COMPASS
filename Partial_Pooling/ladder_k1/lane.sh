#!/bin/bash
# The K=1 baseline, re-established at full power, and the first arm of the new
# direction.
#
# 1. mask6mix at 100 datasets under gauss_hierarchical. Our best K=1 number
#    (0.799) is a 30-dataset figure, and the grouped work showed 30-dataset
#    estimates can be badly off -- gK20's shrinkage slope moved 0.437 -> 0.617
#    between 30 and 100 datasets. Until this run exists we do not know how far
#    K=1 actually is from the paper's 0.870.
#
# 2. mask6mix + time_sampling=log_sigma, trained then evaluated the same way.
#    log_sigma draws noise scales log-uniform instead of the 50/50 mixture every
#    arm so far used; under a VESDE at sigma=25 the uniform half puts only ~1%
#    of draws below sigma_m = 0.1, which is exactly the regime where posteriors
#    resolve.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder_k1/logs
BACKBONE="--train-subset 100000 --arch tokenized --tokenization trial \
          --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4 \
          --mask-menu six_mixed --mask-bernoulli-fraction 0.25"
COMMON="--preset small --subjects 100 --draws 100 --timesteps 50 \
        --gaussian-precision-samples 256 --dpm-corrector-steps 0 \
        --map-estimator kde_global_then_local_ascent --skip-observation-sweep"

echo "=== [$(date -Is)] mask6mix 100-dataset baseline"
$PY "$PP/run_churn_comparison.py" --scratch-base "$PP/ladder_k1/mask6mix" \
    --churn-eta 2 --inference-method dpm2_gauss_hierarchical --datasets 100 \
    $BACKBONE $COMMON > "$LOGS/mask6mix-100.log" 2>&1
status=$?; echo "=== [$(date -Is)] mask6mix 100-dataset exit ${status}"

echo "=== [$(date -Is)] log_sigma train"
$PY "$PP/train_models.py" --preset small --device cuda \
    --time-sampling log_sigma --patience 50 \
    $BACKBONE > "$LOGS/logsigma-train.log" 2>&1
status=$?; echo "=== [$(date -Is)] log_sigma train exit ${status}"

echo "=== [$(date -Is)] log_sigma 100-dataset"
$PY "$PP/run_churn_comparison.py" --scratch-base "$PP/ladder_k1/logsigma" \
    --churn-eta 2 --inference-method dpm2_gauss_hierarchical --datasets 100 \
    --time-sampling log_sigma \
    $BACKBONE $COMMON > "$LOGS/logsigma-100.log" 2>&1
status=$?; echo "=== [$(date -Is)] log_sigma 100-dataset exit ${status}"
echo "K1 LANE COMPLETE $(date -Is)"
