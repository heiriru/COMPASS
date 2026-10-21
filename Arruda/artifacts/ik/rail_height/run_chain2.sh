#!/usr/bin/env bash
# After run_chain.sh (pid 2369547): the Arruda sampler via Arruda/run.sh (its BayesFlow env) on the same GPU, then the figure again.
set -u
builtin cd /export/home/rheinric/COMPASS
RH=Arruda/artifacts/ik/rail_height
while kill -0 2369547 2>/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=9 taskset -c 12-14 bash Arruda/run.sh --benchmark ik --stage sample --ik-regime file \
  --ik-data $RH/data_a0.6_tol0.05.npy --ik-reference $RH/reference_a0.6_tol0.05.npz --ik-true-angle 0.6 --tag rail_a0.6 \
  --weights-dir Arruda/artifacts/ik --observations 1 2 4 8 16 32 64 128 --methods clean:1 clean:0.1 diffused:1 \
  > $RH/arruda.log 2>&1
taskset -c 12-14 .COMPASS/bin/python Arruda/compare_compass_ik.py --stage plot --data $RH/data_a0.6_tol0.05.npy --true-angle 0.6 \
  --out Arruda/artifacts/ik/rail_a0.6_compass --arruda Arruda/artifacts/ik/rail_a0.6 > Arruda/artifacts/ik/rail_a0.6_compass/plot.log 2>&1
echo CHAIN2_DONE
