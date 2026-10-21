#!/bin/bash
# sigma_h = 10, true angle_1 = 0.25 (wide_height.py). Two independent lanes, each resumable:
#   lane A (GPU_A): COMPASS training -> gauss_hierarchical + gauss_jacobian_FD -> Langevin F-NPSE
#   lane B (GPU_B): Arruda/BayesFlow training (paper setting, 1000 epochs) -> Arruda samples
# WAIT_A / WAIT_B: pid of a training already running for that lane (waited for, not restarted).
#   GPU_A=<card> GPU_B=<card> bash inverse_kinematics/recovery/run_wide_height.sh
set -u
builtin cd /export/home/rheinric/COMPASS
export IK_SD_H=10 IK_VESDE_SIGMA=60
PY=.COMPASS/bin/python; R=inverse_kinematics/recovery; W=$R/artifacts/wide_height; A=Arruda/artifacts/ik/wide_sdh10
CKPT=inverse_kinematics/artifacts/models/legacy_h16d2_n200000_six_weighted_log_sigma_sdh10/Model_checkpoint.pt
GPU_A=${GPU_A:?}; GPU_B=${GPU_B:?}; mkdir -p $A $W

lane_a() {
  if [ -n "${WAIT_A:-}" ]; then while kill -0 $WAIT_A 2>/dev/null; do sleep 30; done
  elif [ ! -f $(dirname $CKPT)/training.json ]; then
    CUDA_VISIBLE_DEVICES=$GPU_A taskset -c 33-35 $PY -u $R/wide_height.py train > $W/train_compass.log 2>&1; fi
  [ -f $(dirname $CKPT)/training.json ] || { echo "COMPASS training failed"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU_A taskset -c 33-35 $PY -u $R/wide_height.py network >> $W/network.log 2>&1
  for d in 0 1 2 3 4 5 6 7 8 9; do
    CUDA_VISIBLE_DEVICES=$GPU_A taskset -c 33-35 $PY -u Arruda/compare_compass_ik.py --stage sample \
      --methods langevin_fnpse --ns 1 2 8 32 128 --data $W/d$d/data.npy --true-angle 0.25 \
      --out $W/d$d/langevin --checkpoint $CKPT >> $W/langevin.log 2>&1
  done
  echo LANE_A_DONE
}

lane_b() {
  if [ -n "${WAIT_B:-}" ]; then while kill -0 $WAIT_B 2>/dev/null; do sleep 30; done
  elif [ ! -f $A/training_config.json ]; then
    CUDA_VISIBLE_DEVICES=$GPU_B taskset -c 36-38 bash Arruda/run.sh --benchmark ik --stage train --tag wide_sdh10 \
      > $W/train_arruda.log 2>&1; fi
  [ -f $A/training_config.json ] || { echo "Arruda training failed"; return 1; }
  for d in 0 1 2 3 4 5 6 7 8 9; do
    [ -f $A/d$d/samples_n128_bridge1.npy ] && continue
    CUDA_VISIBLE_DEVICES=$GPU_B taskset -c 36-38 bash Arruda/run.sh --benchmark ik --stage sample --ik-regime file \
      --ik-data $W/d$d/data.npy --ik-reference $W/d$d/reference_exact.npz --ik-true-angle 0.25 \
      --tag wide_sdh10/d$d --weights-dir $A --observations 1 2 8 32 128 --methods clean:1 >> $W/arruda.log 2>&1
  done
  echo LANE_B_DONE
}

lane_a & lane_b & wait
$PY $R/wide_height.py plot > $W/plot.log 2>&1
echo WIDE_DONE
