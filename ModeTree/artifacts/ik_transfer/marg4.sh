PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
E=/export/home/rheinric/COMPASS/ModeTree/experiments/exp5_ik_marginalization.py
M=/export/home/rheinric/COMPASS/inverse_kinematics/artifacts/models
D=/export/home/rheinric/COMPASS/ModeTree/artifacts/ik_transfer
A="--observations 8 16 32 --times 0.03 0.01 0.003 0.001 --grid-points 801 --row-draws 4000 --samples 256 --max-draws 60000000 --regime bimodal --estimator topk"
$PY $E $A --ik-checkpoint $M/ikKbpre_h16d2_n200000_s7/Model_checkpoint.pt \
   --ik-wrapper preconditioned --out-suffix _Kbpre_topk > $D/marg_topk.log 2>&1
$PY $E $A --ik-checkpoint $M/ikKa_h16d2_n200000_s7/Model_checkpoint.pt \
   --ik-wrapper plain --out-suffix _Ka_topk >> $D/marg_topk.log 2>&1
echo "MARG TOPK COMPLETE"
