#!/bin/bash
# Globals-matched Group arms: 100,000 groups each, i.e. the same 100,000 draws
# of the group-level parameters as the flat mask6mix control. The number of
# global draws is the quantity the group scales are learned from, so this is the
# arm that tests the hypothesis; the subject-budget-matched arms are a separate
# lane.
#
# Every arm is the mask6mix recipe with only K varying: h64/te256, depth 2 (so
# depth_within=2, depth_across=2, depth_readout=0), 8 heads, mlp_ratio 4,
# six_mixed masks at bernoulli fraction 0.25 -- 761,797 parameters at K=20,
# under the 1,000,000 cap.
#
# Patience 50, not the preset's 20. Measured, not guessed: the flat mask6mix
# control ran 90 epochs and took its best validation at epoch 70, while the first
# K=5 attempt at patience 20 stopped at epoch 40 with its best at epoch 20. The
# two curves have the same shape (both fall steeply for three epochs then grind,
# improving 8.3% and 7.7% respectively thereafter) and the same ~2.8%
# epoch-to-epoch validation noise, so patience 20 was firing on that noise and
# cutting the grouped arms short relative to the control. The plan requires
# optimizer updates to be held comparable across arms; an arm stopped at 40
# epochs against the control's 90 would fail the hypothesis test for a reason
# that has nothing to do with the representation.
#
# --force because patience is part of the model signature, so these are formally
# different arms from the patience-20 attempt archived in logs_patience20/.
#
# autocvd picks one genuinely free GPU and waits if there is none, so this is
# safe to leave running on the shared host.
set -u
cd /export/home/rheinric/COMPASS
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=Partial_Pooling/ladder_group/logs

for K in 5 10 20; do
    echo "=== K=${K} globals-matched  $(date -Is)"
    $PY Partial_Pooling/train_models.py \
        --preset small --device cuda \
        --arch tokenized --tokenization group --train-subjects "${K}" \
        --train-size 100000 --validation-size 5000 \
        --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4 \
        --mask-menu six_mixed --mask-bernoulli-fraction 0.25 \
        --patience 50 --force \
        > "${LOGS}/gK${K}_globals.log" 2>&1
    status=$?
    echo "=== K=${K} exit ${status}  $(date -Is)"
    if [ $status -ne 0 ]; then
        echo "--- tail of ${LOGS}/gK${K}_globals.log ---"
        tail -25 "${LOGS}/gK${K}_globals.log"
    fi
done
echo "TRAIN LANE COMPLETE $(date -Is)"
