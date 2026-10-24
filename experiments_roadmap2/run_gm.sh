#!/bin/bash
# Roadmap 2.8/2.9/2.10 on the Gaussian-mixture hierarchy: 6 arms x seeds {7, 8}; each job trains,
# scores against the oracle, and composes (sample eps 1e-3; also 1e-5 for base/lo/edm_lo).
# GPU rule: only GPUs with no other user's compute process, >= 2.5 GB free, <= 2 lanes per GPU.
set -u
L=/export/data/rheinric/roadmap2/logs/gm; mkdir -p $L/slots
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
DRIVER=/export/data/rheinric/roadmap2/gm_driver.py   # frozen copy of experiments_roadmap2/gm.py
export COMPASS_SRC=/export/data/rheinric/roadmap2/src_gm
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
    [ $other -eq 0 ] && [ "$mine" -lt 2 ] && [ $((t - m)) -ge 2500 ] && echo $i
  done
}
claim() {
  while true; do
    exec 9>$L/claim.lock; flock 9
    local first; first=$(eligible | head -1)
    if [ -n "$first" ]; then
      sleep 45
      if eligible | grep -qx "$first"; then touch $L/slots/gpu$first.$1; flock -u 9; echo $first; return; fi
    fi
    flock -u 9; sleep 90
  done
}
job() {  # $1 lane, $2 cores, $3 arm, $4 seed
  local name=$3_s$4 g
  g=$(claim "$1.$name"); echo "$(date '+%F %T') $name on GPU $g"
  export CUDA_VISIBLE_DEVICES=$g
  local R=/export/data/rheinric/roadmap2/gm/$name
  [ -f $R/Model.pt ] || taskset -c $2 $PY -u $DRIVER train --arm $3 --seed $4 > $L/${name}_train.log 2>&1
  taskset -c $2 $PY -u $DRIVER score --arm $3 --seed $4 > $L/${name}_score.log 2>&1
  taskset -c $2 $PY -u $DRIVER compose --arm $3 --seed $4 --sample-eps 1e-3 > $L/${name}_compose_eps1e-3.log 2>&1
  case $3 in base|lo|edm_lo)
    taskset -c $2 $PY -u $DRIVER compose --arm $3 --seed $4 --sample-eps 1e-5 > $L/${name}_compose_eps1e-5.log 2>&1;; esac
  rm -f $L/slots/gpu$g.$1.$name; unset CUDA_VISIBLE_DEVICES
  echo "$(date '+%F %T') $name done"
}
lane() { local id=$1 cores=$2; shift 2; while [ $# -ge 2 ]; do job $id $cores $1 $2; shift 2; done; }
lane g1 15-17 base 7 tsm_g 7 lo 7 &
sleep 3; lane g2 18-20 fisher 7 tsm_gl 7 edm_lo 7 &
sleep 3; lane g3 21-23 base 8 tsm_g 8 lo 8 &
sleep 3; lane g4 24-26 fisher 8 tsm_gl 8 edm_lo 8 &
wait; echo "$(date '+%F %T') all done"
