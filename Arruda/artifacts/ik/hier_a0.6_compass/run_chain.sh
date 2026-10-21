#!/usr/bin/env bash
# Everything on one GPU, one job after another (user request): wait for the Arruda run, then the
# three COMPASS methods sequentially, then the figure.
set -u
builtin cd /export/home/rheinric/COMPASS
while kill -0 681741 2>/dev/null; do sleep 20; done
for m in langevin_fnpse gauss_hierarchical gauss_jacobian; do
  CUDA_VISIBLE_DEVICES=4 taskset -c 12-14 .COMPASS/bin/python Arruda/compare_compass_ik.py --stage sample --methods $m \
    > Arruda/artifacts/ik/hier_a0.6_compass/sample_$m.log 2>&1
done
taskset -c 12-14 .COMPASS/bin/python Arruda/compare_compass_ik.py --stage plot > Arruda/artifacts/ik/hier_a0.6_compass/plot.log 2>&1
echo CHAIN_DONE
