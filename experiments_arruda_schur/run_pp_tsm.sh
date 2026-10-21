#!/bin/bash
# Test 1 (TSM on the 6 globals other than beta_raw) and test 4 (final-epoch vs best-validation
# weights) on the budget-A recipe: hierarchical + Rg + 1.5x-wide prior, 65,536 sims, pretrained
# encoder, gauss_jacobian_FD, 30 datasets x 100 subjects, inference seed 7.
# Arms {base, tsm} x training seeds {7, 8, 9}; each model is inferred from both checkpoints
# (recovery/rerun = best-validation, recovery/final = final epoch).
# GPU rule: a lane claims only a GPU that carries no other user's compute process (checked twice,
# 60 s apart, under a lock), with at most 2 of these lanes per GPU.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs/tsm
mkdir -p $L/slots
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
ENC=artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt
COMMON="--mode hierarchical --precondition Rg --train-prior-scale 1.5 --encoder $ENC"
INFER="--method gauss_jacobian_FD --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50"
ME=$(whoami)

eligible() {  # GPU indices with no other user's compute process, < 2 lane slots, >= 4 GB free
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits |
  while IFS=', ' read i u m t; do
    local other=0
    for pid in $(grep "$u" <<< "$apps" | awk -F', ' '{print $2}'); do
      [ "$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')" != "$ME" ] && other=1
    done
    local mine; mine=$(ls $L/slots | grep -c "^gpu$i\.")
    [ $other -eq 0 ] && [ "$mine" -lt 2 ] && [ $((t - m)) -ge 4000 ] && echo $i
  done
}
claim() {  # $1 = lane name; prints the claimed GPU index
  while true; do
    exec 9>$L/claim.lock; flock 9
    local first; first=$(eligible | head -1)
    if [ -n "$first" ]; then
      sleep 60
      if eligible | grep -qx "$first"; then touch $L/slots/gpu$first.$1; flock -u 9; echo $first; return; fi
    fi
    flock -u 9; sleep 120
  done
}
run() {  # $1 lane, $2 cores, $3 arm (base|tsm), $4 training seed
  local V=n65536_${3}_s$4 extra=""; [ "${3#tsm}" != "$3" ] && extra="--tsm"
  local g; g=$(claim "$1.$V")
  echo "$(date '+%F %T') $V on GPU $g"
  ( export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
    taskset -c $2 $PY -u pp.py train $COMMON --train-subset 65536 --variant $V --train-seed $4 $extra \
      2>&1 | grep --line-buffered -v -i warn > $L/${V}_train.log
    for kind in best final; do
      tag=rerun; [ $kind = final ] && tag=final
      taskset -c $2 $PY -u pp.py infer $COMMON $INFER --variant $V --checkpoint $kind --tag $tag \
        2>&1 | grep --line-buffered -v -i warn > $L/${V}_infer_$kind.log
    done )
  rm -f $L/slots/gpu$g.$1.$V
  echo "$(date '+%F %T') $V done"
}

# smoke: 1 epoch + 1 tiny dataset; abort everything if it fails
g=$(claim smoke)
echo "$(date '+%F %T') smoke on GPU $g"
export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
taskset -c 51-53 $PY -u smoke_tsm.py train > $L/smoke_train.log 2>&1 &&
taskset -c 51-53 $PY -u smoke_tsm.py infer > $L/smoke_infer.log 2>&1
ok=$?
rm -f $L/slots/gpu$g.smoke; unset CUDA_VISIBLE_DEVICES
if [ $ok -ne 0 ] || ! grep -q "global std-RMSE" $L/smoke_infer.log; then echo "smoke FAILED"; exit 1; fi
echo "$(date '+%F %T') smoke ok"

# Base s7-s9 finished 2026-10-20. The first "tsm" arm diverged (blend weight read sigma off the
# noisy globals: huge targets at high noise); "tsm2" is the fixed pessimistic blend (tsm_targets.py).
lane1() { run l1 51-53 tsm2 7; }
lane2() { run l2 54-56 tsm2 8; }
lane3() { run l3 57-59 tsm2 9; }
lane1 & sleep 5; lane2 & sleep 5; lane3 & wait
echo "$(date '+%F %T') all done"
