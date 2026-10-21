#!/bin/bash
# Is the K non-monotonicity in the shrinkage slope (0.68 at K=5, 0.37 at K=10,
# 0.44 at K=20) a property of K, or just training-seed noise?
#
# We have exactly one training run per K, so the two are currently confounded:
# all three arms trained 98 epochs, took their best at epoch 48, and sit within
# 2% of each other on per-node validation loss. Nothing distinguishes them on
# the training side, which is precisely why the slope difference cannot be
# attributed to K without measuring seed variance first.
#
# Two extra K=10 arms at different *initialisation* seeds -- --init-seed, not
# --seed. root_seed is a data axis: changing it regenerates the simulations, so
# it would compare two different experiments rather than the same one run twice.
# The first attempt at this used --seed and died instantly on a data-signature
# mismatch, and the lane reported "exit 0" because $(date -Is) had already reset
# $? by the time it was read. If their slopes bracket 0.37
# tightly, the K effect is real; if they scatter across 0.37-0.68, the
# "K does two different things" reading in the progress doc is wrong and must
# be retracted.
#
# Waits for the partition study and the 100-dataset confirmations to finish, so
# it never competes with work that was actually asked for.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder_group/logs
cd /export/home/rheinric/COMPASS

for SEED in 20250719 20250720; do
  echo "=== [$(date -Is)] K=10 seed ${SEED}"
  $PY "$PP/train_models.py" \
      --preset small --device cuda --init-seed "${SEED}" \
      --arch tokenized --tokenization group --train-subjects 10 \
      --train-size 100000 --validation-size 5000 \
      --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4 \
      --mask-menu six_mixed --mask-bernoulli-fraction 0.25 \
      --patience 50 --force > "${LOGS}/seed${SEED}-train.log" 2>&1
  status=$?
  echo "=== [$(date -Is)] K=10 seed ${SEED} train exit ${status}"
  $PY "$PP/run_churn_comparison.py" \
      --scratch-base "$PP/ladder_group/gK10_seed${SEED}" \
      --churn-eta 2 --inference-method dpm2_gauss_hierarchical --datasets 30 \
      --preset small --init-seed "${SEED}" \
      --arch tokenized --tokenization group --train-subjects 10 \
      --train-size 100000 --validation-size 5000 \
      --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4 \
      --mask-menu six_mixed --mask-bernoulli-fraction 0.25 \
      --subjects 100 --draws 100 --timesteps 50 \
      --gaussian-precision-samples 256 --dpm-corrector-steps 0 \
      --map-estimator kde_global_then_local_ascent --skip-observation-sweep \
      > "${LOGS}/seed${SEED}-eval.log" 2>&1
  status=$?
  echo "=== [$(date -Is)] K=10 seed ${SEED} eval exit ${status}"
done
echo "SEED STUDY COMPLETE $(date -Is)"
