#!/bin/bash
# Resume T / R / Rg inference after the home-quota crash (finished datasets are skipped).
# No CUDA_VISIBLE_DEVICES: pp.py picks a free GPU per arm via autocvd; starts are staggered.
set -u
cd /export/home/rheinric/COMPASS/experiments_arruda_schur
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
L=/export/home/rheinric/COMPASS/abs_hierarchy/logs
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
arm() {  # $1 precondition, $2 cpu slot
  taskset -c $2 $PY -u pp.py infer --mode hierarchical --precondition $1 --method gauss_jacobian_FD \
      --subjects 100 --datasets 30 --tag rerun 2>&1 | grep --line-buffered -v -i warn > $L/infer_pp_$1.log
}
arm T 0-2 & sleep 90; arm R 3-5 & sleep 90; arm Rg 6-8 & wait
echo all done
