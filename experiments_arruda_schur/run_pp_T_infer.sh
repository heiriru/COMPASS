#!/bin/bash
# Re-run the T arm's inference (it OOM'd on the shared GPU 8); no CUDA_VISIBLE_DEVICES, so pp.py
# picks a free GPU via autocvd. Same protocol as run_pp_ablation.sh.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
taskset -c 0-2 /export/home/rheinric/COMPASS/.COMPASS/bin/python -u pp.py infer --mode hierarchical --precondition T \
  --method gauss_jacobian_FD --subjects 100 --datasets 30 --tag rerun 2>&1 | grep --line-buffered -v -i warn > $L/infer_pp_T.log
echo all done
