#!/bin/bash
# Step 3 (|x1| variant), after step 2 finishes: train, sample the COMPASS rules, rule-vs-network, plot.
# One GPU at a time via autocvd inside each script.
builtin cd /export/home/rheinric/COMPASS
A=Arruda/artifacts/ik
while kill -0 423629 2>/dev/null; do sleep 60; done
CKPT=inverse_kinematics/artifacts/models/legacy_h16d2_n200000_six_weighted_log_sigma_absx1/Model_checkpoint.pt
.COMPASS/bin/python -u inverse_kinematics/experiments/train_abs_x1.py > $A/absx1/train.log 2>&1 || exit 1
IK_OBSERVE_ABS_X1=1 IK_CHECKPOINT=$CKPT .COMPASS/bin/python -u $A/analysis/rule_vs_network.py \
    $A/absx1/data_a0.6.npy $A/absx1_compass $A/exact/absx1_a0.6.npz > $A/absx1/rule_vs_network.log 2>&1
for m in gauss_hierarchical gauss_jacobian langevin_fnpse; do
  .COMPASS/bin/python -u Arruda/compare_compass_ik.py --stage sample --methods $m --checkpoint $CKPT \
      --data $A/absx1/data_a0.6.npy --out $A/absx1_compass > $A/absx1/sample_$m.log 2>&1
done
.COMPASS/bin/python -u Arruda/compare_compass_ik.py --stage plot --data $A/absx1/data_a0.6.npy --out $A/absx1_compass \
    --arruda $A/absx1 --reference $A/exact/absx1_a0.6.npz --reference-label "exact noise-free posterior" --suffix _exact \
    > $A/absx1/plot.log 2>&1
