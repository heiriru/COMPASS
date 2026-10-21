#!/usr/bin/env bash
# One GPU, one job after another: 32M-draw reference, Arruda (3 variants), the three COMPASS rules, figure.
set -u
builtin cd /export/home/rheinric/COMPASS
PY=.COMPASS/bin/python; RH=Arruda/artifacts/ik/rail_height; GPU=9
CUDA_VISIBLE_DEVICES=$GPU taskset -c 12-14 $PY Arruda/artifacts/ik/analysis/rail_height_reference.py a0.6_tol0.05 64 > $RH/reference_a0.6_tol0.05_32M.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU taskset -c 12-14 $PY Arruda/run_arruda.py --benchmark ik --stage sample --ik-regime file \
  --ik-data $RH/data_a0.6_tol0.05.npy --ik-reference $RH/reference_a0.6_tol0.05.npz --ik-true-angle 0.6 --tag rail_a0.6 \
  --weights-dir Arruda/artifacts/ik --observations 1 2 4 8 16 32 64 128 --methods clean:1 clean:0.1 diffused:1 \
  > $RH/arruda.log 2>&1
mkdir -p Arruda/artifacts/ik/rail_a0.6_compass
for m in langevin_fnpse gauss_hierarchical gauss_jacobian; do
  CUDA_VISIBLE_DEVICES=$GPU taskset -c 12-14 $PY Arruda/compare_compass_ik.py --stage sample --methods $m \
    --data $RH/data_a0.6_tol0.05.npy --true-angle 0.6 --out Arruda/artifacts/ik/rail_a0.6_compass --arruda Arruda/artifacts/ik/rail_a0.6 \
    > Arruda/artifacts/ik/rail_a0.6_compass/sample_$m.log 2>&1
done
taskset -c 12-14 $PY Arruda/compare_compass_ik.py --stage plot --data $RH/data_a0.6_tol0.05.npy --true-angle 0.6 \
  --out Arruda/artifacts/ik/rail_a0.6_compass --arruda Arruda/artifacts/ik/rail_a0.6 > Arruda/artifacts/ik/rail_a0.6_compass/plot.log 2>&1
echo CHAIN_DONE
