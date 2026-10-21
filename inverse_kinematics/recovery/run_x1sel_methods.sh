#!/bin/bash
# Arruda (paper setting) + COMPASS gauss_hierarchical, Langevin F-NPSE, gauss_jacobian_FD on the
# |x1| < 0.15 set, mirroring Arruda/artifacts/ik/rail_height/run_chain*.sh. Each step skips finished n.
set -u
builtin cd /export/home/rheinric/COMPASS
PY=.COMPASS/bin/python; D=Arruda/artifacts/ik/x1sel; GPU=${GPU:-8}
CUDA_VISIBLE_DEVICES=$GPU bash Arruda/run.sh --benchmark ik --stage sample --ik-regime file \
  --ik-data $D/data_a0.6_x1sel0.15.npy --ik-reference $D/reference_exact.npz --ik-true-angle 0.6 --tag x1sel_a0.6 \
  --weights-dir Arruda/artifacts/ik --observations 1 2 4 8 16 32 64 128 --methods clean:1 > $D/arruda.log 2>&1
for m in gauss_hierarchical langevin_fnpse; do
  CUDA_VISIBLE_DEVICES=$GPU $PY Arruda/compare_compass_ik.py --stage sample --methods $m \
    --data $D/data_a0.6_x1sel0.15.npy --true-angle 0.6 --out Arruda/artifacts/ik/x1sel_a0.6_compass \
    --arruda Arruda/artifacts/ik/x1sel_a0.6 > $D/sample_$m.log 2>&1
done
CUDA_VISIBLE_DEVICES=$GPU $PY -u inverse_kinematics/recovery/run_rail_fd.py $D/data_a0.6_x1sel0.15.npy \
  Arruda/artifacts/ik/x1sel_a0.6_compass > $D/sample_gauss_jacobian_FD.log 2>&1
echo X1SEL_DONE
