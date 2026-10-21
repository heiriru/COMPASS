#!/bin/bash
# Rerun of the plain jointft inference (OOMed twice on a shared GPU 8). Waits for a GPU with no
# compute process at all (checked twice, 60 s apart, so a card that is just between jobs is skipped),
# then runs there alone.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
free_gpu() {  # first GPU index with no compute apps and < 100 MiB used
  local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | while IFS=', ' read i u m; do
    if [ "$m" -lt 100 ] && ! grep -q "$u" <<< "$busy"; then echo $i; break; fi; done
}
while true; do
  g=$(free_gpu)
  if [ -n "$g" ]; then sleep 60; [ "$(free_gpu)" = "$g" ] && break; fi
  sleep 60
done
echo "launching on GPU $g at $(date)"
export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
taskset -c 51-53 /export/home/rheinric/COMPASS/.COMPASS/bin/python -u pp.py infer --mode hierarchical \
  --train-prior-scale 1.5 --method gauss_jacobian_FD --subjects 100 --datasets 30 --dataset-start 0 \
  --draws 100 --timesteps 50 --tag rerun --variant jointft_n65536 \
  --encoder artifacts/pp/hierarchical_wide1.5_jointft_n65536/checkpoints/encoder.pt \
  2>&1 | grep --line-buffered -v -i warn > $L/budget_plain_jointft_n65536_infer.log
echo "done $(date)"
