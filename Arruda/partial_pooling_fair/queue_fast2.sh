#!/bin/bash
# ours-fast = refresh 9 + dataset batching + fp32/1024-chunk eigh (MultiObsSampler.set_eigh_precision),
# with OOM splitting per chunk. ONE GPU, claimed only when no other user's compute process is on it
# (checked twice, 60 s apart; re-checked after the timing probe). Steps: time D=4 vs D=8 on the
# claimed GPU (profile_sampler.py), take the faster, run joint + joint_wide on 100 datasets, score.
set -u
cd /export/home/rheinric/COMPASS/Arruda/partial_pooling_fair
ME=$(whoami)
F="warn\|INFO\|gradients\|enable_grad\|^$\|disable autograd\|^with torch\|^\s*\.\.\.\|Summarizing\|Sampling:"
eligible() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | while IFS=', ' read i u; do
    local other=0
    for pid in $(grep "$u" <<< "$apps" | awk -F', ' '{print $2}'); do
      [ "$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')" != "$ME" ] && other=1
    done
    [ $other -eq 0 ] && echo $i
  done
}
claim() {
  while true; do
    g=$(eligible | head -1)
    if [ -n "$g" ]; then sleep 60; eligible | grep -qx "$g" && return; fi
    sleep 120
  done
}
claim
echo "GPU $g claimed $(date)"
export CUDA_VISIBLE_DEVICES=$g KERAS_BACKEND=torch
[ -s logs/probe_fast.log ] || for D in 4 8; do
  PYTHONPATH=/export/home/rheinric/COMPASS/Arruda/.vendor:/export/home/rheinric/COMPASS taskset -c 51-53 \
    /export/home/rheinric/COMPASS/.COMPASS/bin/python -u profile_sampler.py $D ours-fast 2>&1 | grep -E "^D=" | tee -a logs/probe_fast.log
done
# the probe is only valid if nobody else joined the GPU meanwhile
eligible | grep -qx "$g" || echo "WARNING: another user joined GPU $g during the probe" | tee -a logs/probe_fast.log
best=$(awk '/^D=/{gsub("D=|:","",$1); gsub("[(]","",$4); print $4, $1}' logs/probe_fast.log | tail -2 | sort -n | head -1 | cut -d' ' -f2)
echo "chunk $best"
export FAIR_CHUNK=$best
for net in joint joint_wide; do
  echo "== ours-fast $net $(date)"
  taskset -c 51-53 ./run.sh infer ours-fast $net 2>&1 | grep -v -i "$F" > logs/infer_ours-fast_$net.log
  tail -1 logs/infer_ours-fast_$net.log
  eligible | grep -qx "$g" || echo "WARNING: another user on GPU $g during $net (timing not clean)"
done
./run.sh score 2>&1 | grep -v -i "$F" > logs/score_fast2.log
echo "all done $(date)"
