#!/bin/bash
# Roadmap 2.4: does time_loss_weight="edm" help or hurt the partial-pooling six-menu net?
# Arm "edm" = the budget-A base recipe (hierarchical + Rg + 1.5x-wide prior, 65,536 sims, pretrained
# encoder, six_mixed menu, patience 20, unseeded validation) with ONLY the EDM loss weight added,
# so the existing base arms n65536_base_s{7,8,9} (recovery/final) are the paired control.
# Same inference protocol: gauss_jacobian_FD, 30 datasets x 100 subjects, 100 draws, seed 7,
# final-epoch weights. Library code: a frozen snapshot of the roadmap branch's src.
# GPU rule (as run_pp_tsm.sh): claim only a GPU with no other user's compute process, >= 4 GB free,
# at most 2 of these lanes per GPU.
set -u
MAIN=/export/home/rheinric/COMPASS
cd $MAIN/experiments_arruda_schur
L=/export/data/rheinric/roadmap2/logs/edm
mkdir -p $L/slots
PY=$MAIN/.COMPASS/bin/python
LAUNCH=/export/home/rheinric/COMPASS-wt-2x/experiments_roadmap2/launch.py
export COMPASS_SRC=/export/data/rheinric/roadmap2/src_edm_test
ENC=artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt
COMMON="--mode hierarchical --precondition Rg --train-prior-scale 1.5 --encoder $ENC"
INFER="--method gauss_jacobian_FD --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50"
ME=$(whoami)

eligible() {
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
claim() {
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
run() {  # $1 lane, $2 cores, $3 seed
  local V=n65536_edm_s$3
  local g; g=$(claim "$1.$V")
  echo "$(date '+%F %T') $V train on GPU $g"
  ( export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
    TRAIN_KW='{"time_loss_weight": "edm", "validation_noise_seed": null}' \
      taskset -c $2 $PY -u $LAUNCH pp.py train $COMMON --train-subset 65536 --variant $V --train-seed $3 \
      2>&1 | grep --line-buffered -v -i warn > $L/${V}_train.log )
  rm -f $L/slots/gpu$g.$1.$V
  g=$(claim "$1.$V")
  echo "$(date '+%F %T') $V infer on GPU $g"
  ( export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
    taskset -c $2 $PY -u $LAUNCH pp.py infer $COMMON $INFER --variant $V --checkpoint final --tag final \
      2>&1 | grep --line-buffered -v -i warn > $L/${V}_infer_final.log )
  rm -f $L/slots/gpu$g.$1.$V
  echo "$(date '+%F %T') $V done"
}
run e1 33-35 7 & sleep 5; run e2 36-38 8 & sleep 5; run e3 39-41 9 & wait
echo "$(date '+%F %T') all done"
