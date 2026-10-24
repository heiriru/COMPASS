#!/bin/bash
# Speed-ups 1 (Jacobian refresh 9) and 2 (datasets batched along the draw axis; exact, see
# check-batched) on the trained joint nets. ONE GPU, claimed only when no other user's compute
# process is on it (checked twice, 60 s apart). Runs, in order:
#   ours-batched     joint       (original settings: isolates the batching speed-up)
# 4 datasets per call (probe 2026-10-22: speed saturates at 3-4; 10 and 5 ran out of memory).
#   ours-batched-r9  joint       (both speed-ups)
#   ours-batched-r9  joint_wide
# then the score table. On CUDA OOM a run retries with 2 datasets per call.
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
while true; do
  g=$(eligible | head -1)
  if [ -n "$g" ]; then sleep 60; eligible | grep -qx "$g" && break; fi
  sleep 120
done
echo "GPU $g claimed $(date)"
export CUDA_VISIBLE_DEVICES=$g FAIR_CHUNK=4
run() {
  local log=logs/infer_$1_$2.log
  echo "== $1 $2 $(date)"
  taskset -c 51-53 ./run.sh infer $1 $2 2>&1 | grep -v -i "$F" > $log
  if grep -q "out of memory" $log; then
    echo "   OOM at chunk 4, retrying with 2"
    FAIR_CHUNK=2 taskset -c 51-53 ./run.sh infer $1 $2 2>&1 | grep -v -i "$F" > $log
  fi
  tail -1 $log
}
run ours-batched joint
run ours-batched-r9 joint
run ours-batched-r9 joint_wide
./run.sh score 2>&1 | grep -v -i "$F" > logs/score_fast.log
echo "all done $(date)"
