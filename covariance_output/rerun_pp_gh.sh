#!/bin/bash
L=/export/data/rheinric/covout/logs
until grep -q "covout_cov_s7 done" $L/run_pp.log; do sleep 60; done
MAIN=/export/home/rheinric/COMPASS; cd $MAIN/experiments_arruda_schur
PY=$MAIN/.COMPASS/bin/python; LP=/export/data/rheinric/COMPASS-covout/covariance_output/launch_pp.py
ENC=artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt
COMMON="--mode hierarchical --precondition Rg --train-prior-scale 1.5 --encoder $ENC"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
g=$($PY -c "
import sys, os; sys.path.insert(0, '/export/data/rheinric/COMPASS-covout/jac_extensions'); import gpu
os.environ['JACEXT_GPU']='7,8,9'; gpu.claim(5000, verbose=False); print(os.environ['CUDA_VISIBLE_DEVICES'])")
echo "$(date '+%F %T') rerun covout_cov_s7 GH on GPU $g" >> $L/run_pp.log
CUDA_VISIBLE_DEVICES=$g taskset -c 39-41 $PY -u $LP pp.py infer $COMMON --subjects 100 --datasets 30 --dataset-start 0 \
  --draws 100 --timesteps 50 --method gauss_hierarchical --variant covout_cov_s7 --checkpoint final --tag final \
  2>&1 | grep --line-buffered -v -i warn > $L/pp/covout_cov_s7_infer_gauss_hierarchical_rerun.log
echo "$(date '+%F %T') rerun covout_cov_s7 GH done" >> $L/run_pp.log
