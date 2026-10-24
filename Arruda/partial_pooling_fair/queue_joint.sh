#!/bin/bash
# Apples-to-apples arm: one upstream-architecture net (SetTransformer16 + DiffusionModel, 500 epochs,
# batch 128) on the joint s(g, l | x_j), trained on the same 65,536 simulations upstream's two nets
# consume, composed by gauss_jacobian_FD (g and l). Two training priors (joint, joint_wide) in
# parallel, then bridge checks, n=100 and the n-sweep (4, 25), then the score table.
# GPU rule: only a GPU with no other user's compute process (checked twice, 60 s apart).
set -u
cd /export/home/rheinric/COMPASS/Arruda/partial_pooling_fair
ME=$(whoami)
F="warn\|INFO\|gradients\|enable_grad\|^$\|disable autograd\|^with torch\|^\s*\.\.\.\|Summarizing\|Sampling:"
mkdir -p logs/slots
eligible() {
  local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | while IFS=', ' read i u; do
    local other=0
    for pid in $(grep "$u" <<< "$apps" | awk -F', ' '{print $2}'); do
      [ "$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')" != "$ME" ] && other=1
    done
    [ $other -eq 0 ] && [ "$(ls logs/slots | grep -c "^gpu$i\.")" -eq 0 ] && echo $i
  done
}
claim() {
  while true; do
    exec 9>logs/claim.lock; flock 9
    local first; first=$(eligible | head -1)
    if [ -n "$first" ]; then
      sleep 60
      if eligible | grep -qx "$first"; then touch logs/slots/gpu$first.$1; flock -u 9; echo $first; return; fi
    fi
    flock -u 9; sleep 120
  done
}
run() { echo "== $* $(date)"; ./run.sh "$@" 2>&1 | grep -v -i "$F" > logs/$(echo "$*" | tr ' ' _).log; }
export F; export -f run
arm() {  # $1 = joint | joint_wide, $2 = cores
  local g; g=$(claim $1)
  echo "$1 on GPU $g $(date)"
  ( export CUDA_VISIBLE_DEVICES=$g
    taskset -c $2 bash -c "run train $1; run check-bridge $1; run infer ours $1"
    if [ $1 = joint ]; then for n in 4 25; do taskset -c $2 bash -c "run infer ours joint $n"; done; fi )
  rm -f logs/slots/gpu$g.$1
  echo "$1 done $(date)"
}
arm joint 51-53 & sleep 5; arm joint_wide 54-56 & wait
./run.sh score 2>&1 | grep -v -i "$F" > logs/score_joint.log
echo "all done $(date)"
