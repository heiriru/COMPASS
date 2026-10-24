#!/bin/bash
# Roadmap 2.9 two moons: train attn_hier_tsm_gl, score-check it, run it on the sigma_max 0.6 datasets.
# GPU rule: only a GPU with no other user's compute process and >= 3 GB free.
set -u
L=/export/data/rheinric/roadmap2/logs/two_moons; mkdir -p $L
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
D=/export/data/rheinric/roadmap2/two_moons_driver.py
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
    [ $other -eq 0 ] && [ $((t - m)) -ge 3000 ] && echo $i
  done
}
gpu() { local g=""; while [ -z "$g" ]; do g=$(eligible | head -1); [ -z "$g" ] && sleep 120; done; sleep 45; eligible | grep -qx "$g" && echo $g || gpu; }
g=$(gpu); echo "$(date '+%F %T') train on GPU $g"
CUDA_VISIBLE_DEVICES=$g taskset -c 0-2 $PY -u $D train > $L/train.log 2>&1
g=$(gpu); echo "$(date '+%F %T') score + run on GPU $g"
CUDA_VISIBLE_DEVICES=$g taskset -c 0-2 $PY -u $D score 10 25 100 > $L/score_check.log 2>&1
CUDA_VISIBLE_DEVICES=$g taskset -c 0-2 $PY -u $D run --sigma-max 0.6 --Ns 1 10 25 100 \
  --methods attn_hier_tsm_gl:gauss_jacobian_FD > $L/run_smax_jac.log 2>&1
echo "$(date '+%F %T') done"
