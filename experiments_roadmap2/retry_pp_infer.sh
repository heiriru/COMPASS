#!/bin/bash
# Re-run (resumable) pp.py inference for the EDM arms until every seed has all 30 datasets.
# Waits for the main queue (run_pp_edm.sh) to exit first. Same GPU rule, with 5 GB free.
set -u
MAIN=/export/home/rheinric/COMPASS
cd $MAIN/experiments_arruda_schur
L=/export/data/rheinric/roadmap2/logs/edm
PY=$MAIN/.COMPASS/bin/python
LAUNCH=/export/home/rheinric/COMPASS-wt-2x/experiments_roadmap2/launch.py
export COMPASS_SRC=/export/data/rheinric/roadmap2/src_edm_test
ENC=artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt
COMMON="--mode hierarchical --precondition Rg --train-prior-scale 1.5 --encoder $ENC"
INFER="--method gauss_jacobian_FD --subjects 100 --datasets 30 --dataset-start 0 --draws 100 --timesteps 50"
ME=$(whoami)
while pgrep -f run_pp_edm.sh > /dev/null; do sleep 120; done
eligible() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits |
  while IFS=', ' read i u m t; do
    local other=0
    for pid in $(grep "$u" <<< "$apps" | awk -F', ' '{print $2}'); do
      [ "$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')" != "$ME" ] && other=1
    done
    [ $other -eq 0 ] && [ $((t - m)) -ge 5000 ] && echo $i
  done
}
for seed in 7 8 9; do
  V=n65536_edm_s$seed
  D=artifacts/pp/hierarchical_Rg_wide1.5_$V/recovery/final/gauss_jacobian_FD-s100-t50
  while [ "$(ls $D/dataset-*.pt 2>/dev/null | wc -l)" -lt 30 ]; do
    g=""; while [ -z "$g" ]; do g=$(eligible | head -1); [ -z "$g" ] && sleep 120; done
    echo "$(date '+%F %T') $V retry infer on GPU $g ($(ls $D/dataset-*.pt 2>/dev/null | wc -l)/30 done)"
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
      taskset -c 39-41 $PY -u $LAUNCH pp.py infer $COMMON $INFER --variant $V --checkpoint final --tag final \
      2>&1 | grep --line-buffered -v -i warn >> $L/${V}_infer_final_retry.log
    sleep 30
  done
done
echo "$(date '+%F %T') retry done"
