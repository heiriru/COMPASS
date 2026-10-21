#!/usr/bin/env bash
# 30-dataset recovery screen for the summary-space arms.
#
# The protocol matches the raw-trial K=1 control exactly -- 100 subjects,
# 100 draws, 50 timesteps -- so the numbers land on the same axis as the
# 0.801 reference. The first arm (A0) was originally run at 80 timesteps,
# so it is re-screened here rather than carried over.
set -u
cd /export/home/rheinric/COMPASS
PY=.COMPASS/bin/python
LOGS=Partial_Pooling/ladder_encsum/logs
TAG=setenc-d16-w64x2s4
MAX_CONCURRENT=4
STAGGER=120
DATASETS=${DATASETS:-30}

ARMS=(
  "A0-h48d2-te256-six_w6:$TAG-h48d2-log_sigma"
  "A1-h48d2-te256-mask6mix:$TAG-h48d2-log_sigma-mask6mix"
  "B1-h48d2-te64-mask6mix:$TAG-h48d2-log_sigma-te64-mask6mix"
  "C0-h64d4-te64-mask6mix:$TAG-h64d4-log_sigma-te64-mask6mix"
  "C1-h64d4-te256-mask6mix:$TAG-h64d4-log_sigma-mask6mix"
  "C2-h96d6-te64-mask6mix:$TAG-h96d6-log_sigma-te64-mask6mix"
  "C3-h96d6-te64-six_w6:$TAG-h96d6-log_sigma-te64"
  "D1-h96d6-te64-mask6mix-400k:$TAG-h96d6-log_sigma-te64-mask6mix-n400k"
)

for arm in "${ARMS[@]}"; do
  IFS=: read -r label name <<< "$arm"
  ckpt="Partial_Pooling/artifacts/encoder_summary/checkpoints/${name}_checkpoint.pt"
  if [ ! -f "$ckpt" ]; then
    echo "$(date -Is) SKIP $label -- no checkpoint at $ckpt"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]; do sleep 30; done
  echo "$(date -Is) screening $label"
  (
    $PY Partial_Pooling/infer_encoder_summary.py \
      --method dpm2_gauss_hierarchical --name "$name" \
      --subjects 100 --datasets "$DATASETS" --draws 100 --timesteps 50
    status=$?
    echo "$(date -Is) $label finished exit $status"
  ) > "$LOGS/screen-$label.log" 2>&1 &
  sleep "$STAGGER"
done
wait
echo "$(date -Is) all screens done"
