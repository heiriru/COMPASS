#!/bin/bash
# Partial pooling: covariance output arms (cov = joint, cov_det = readout only), then
# GH (pilot) vs JAC-FD r9 vs JAC-netcov on each net. Main-tree driver, this branch's src.
set -u
MAIN=/export/home/rheinric/COMPASS
cd $MAIN/experiments_arruda_schur
PY=$MAIN/.COMPASS/bin/python
LP=/export/data/rheinric/COMPASS-covout/covariance_output/launch_pp.py
L=/export/data/rheinric/covout/logs/pp; mkdir -p $L
ENC=artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt
COMMON="--mode hierarchical --precondition Rg --train-prior-scale 1.5 --encoder $ENC"
INFER="--subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
claim() {   # a GPU among 7-9 with only my processes and >= $1 MB free
  $PY - "$1" <<'PYEOF'
import sys, time
sys.path.insert(0, "/export/data/rheinric/COMPASS-covout/jac_extensions")
import gpu, os
os.environ["JACEXT_GPU"] = "7,8,9"
gpu.claim(int(sys.argv[1]), verbose=False)
print(os.environ["CUDA_VISIBLE_DEVICES"])
PYEOF
}
lane() {  # $1 arm (cov|cov_det), $2 seed, $3 cores
  local V=covout_$1_s$2 mode=joint
  [ $1 = cov_det ] && mode=detached
  local g; g=$(claim 3000); echo "$(date '+%F %T') $V train on GPU $g"
  CUDA_VISIBLE_DEVICES=$g COVOUT_TRAIN=$mode taskset -c $3 $PY -u $LP pp.py train $COMMON \
      --train-subset 65536 --variant $V --train-seed $2 2>&1 | grep --line-buffered -v -i warn > $L/${V}_train.log
  for m in gauss_hierarchical gauss_jacobian_FD_r9 netcov; do
    g=$(claim 3000); echo "$(date '+%F %T') $V infer $m on GPU $g"
    if [ $m = netcov ]; then
      CUDA_VISIBLE_DEVICES=$g COVOUT_NETCOV=1 taskset -c $3 $PY -u $LP pp.py infer $COMMON $INFER \
        --method gauss_jacobian_FD_r1 --variant $V --checkpoint final --tag netcov 2>&1 | grep --line-buffered -v -i warn > $L/${V}_infer_$m.log
    else
      CUDA_VISIBLE_DEVICES=$g taskset -c $3 $PY -u $LP pp.py infer $COMMON $INFER \
        --method $m --variant $V --checkpoint final --tag final 2>&1 | grep --line-buffered -v -i warn > $L/${V}_infer_$m.log
    fi
  done
  echo "$(date '+%F %T') $V done"
}
lane cov 7 39-41 & sleep 90
lane cov_det 7 42-44 & sleep 90
lane cov 8 45-47 & sleep 90
lane cov_det 8 66-68 & wait
echo "$(date '+%F %T') ALL DONE"
