#!/bin/bash
# After the three trainings (started separately on GPU 9) finish: bridge check on the trained
# net, the 2x2 cells at n = 100, the n-sweep (4, 25) for the Test-1 cells, then the score table.
# Inference runs on GPU 9 (the training GPU), one job at a time.
set -u
cd /export/home/rheinric/COMPASS/Arruda/partial_pooling_fair
M=/export/data/rheinric/compass_pp_fair/models
until [ -f $M/partial_pooling_global.keras ] && [ -f $M/partial_pooling_global_wide.keras ] && [ -f $M/partial_pooling_local.keras ]; do sleep 60; done
export CUDA_VISIBLE_DEVICES=9
F="warn\|INFO\|gradients\|enable_grad\|^$\|disable autograd\|^with torch\|^\s*\.\.\.\|Summarizing\|Sampling:"
run() { echo "== $* $(date)"; ./run.sh "$@" 2>&1 | grep -v -i "$F" > logs/$(echo "$*" | tr ' ' _).log; }
run check-bridge global
run infer bf global
run infer bf global_wide
run infer ours global
run infer ours global_wide
for n in 4 25; do run infer bf global $n; run infer ours global $n; done
./run.sh score 2>&1 | grep -v -i "$F" > logs/score.log
echo "all done $(date)"
