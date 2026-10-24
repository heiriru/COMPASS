#!/bin/bash
# Third compose pass for the n = 32 cells, which OOM whenever another ~5 GB job shares the GPU.
# One lane, a GPU with no other user and >= 9 GB free.
set -u
L=/export/data/rheinric/roadmap2/logs/gm; mkdir -p $L/slots3
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
DRIVER=/export/data/rheinric/roadmap2/gm_driver.py
export COMPASS_SRC=/export/data/rheinric/roadmap2/src_gm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ME=$(whoami)
eligible() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits |
  while IFS=', ' read i u m t; do
    local other=0
    for pid in $(grep "$u" <<< "$apps" | awk -F', ' '{print $2}'); do
      [ "$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')" != "$ME" ] && other=1
    done
    local mine; mine=$(ls $L/slots3 | grep -c "^gpu$i\.")
    [ $other -eq 0 ] && [ "$mine" -lt 1 ] && [ $((t - m)) -ge 9000 ] && echo $i
  done
}
claim() {
  while true; do
    exec 9>$L/claim3.lock; flock 9
    local first; first=$(eligible | head -1)
    if [ -n "$first" ]; then
      sleep 45
      if eligible | grep -qx "$first"; then touch $L/slots3/gpu$first.$1; flock -u 9; echo $first; return; fi
    fi
    flock -u 9; sleep 90
  done
}
job() {  # $1 lane $2 cores $3 arm $4 seed $5 eps
  local name=$3_s$4 g
  g=$(claim "$1.$name.$5"); echo "$(date '+%F %T') $name eps $5 on GPU $g"
  CUDA_VISIBLE_DEVICES=$g taskset -c $2 $PY -u $DRIVER compose --arm $3 --seed $4 --sample-eps $5 \
    > $L/${name}_compose_eps$5.log 2>&1
  rm -f $L/slots3/gpu$g.$1.$name.$5
  echo "$(date '+%F %T') $name eps $5 done ($(grep -c '!! ' $L/${name}_compose_eps$5.log) failed cells)"
}
lane() { local id=$1 cores=$2; shift 2; while [ $# -ge 3 ]; do job $id $cores $1 $2 $3; shift 3; done; }
lane c1 15-17 base 7 1e-3 fisher 7 1e-3 lo 7 1e-3 edm_lo 7 1e-3 tsm_gl 7 1e-3 tsm_g 7 1e-3 base 8 1e-3 fisher 8 1e-3 lo 8 1e-3 edm_lo 8 1e-3 tsm_gl 8 1e-3 tsm_g 8 1e-3 &
wait; echo "$(date '+%F %T') all done"
