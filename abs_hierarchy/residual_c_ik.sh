#!/bin/bash
# Residual output (precondition.py --residual) on step C (curved: l ~ N(g^2 + 0.4 g, .5^2)) and on IK |x1|.
# Slot 0-2: C stock hier -> C hier_R -> C MCMC + composed runs -> IK stock hier (train + run).
# Slot 3-5: IK hier_R (train + run).   Both on GPU 8.
set -u
cd /export/home/rheinric/COMPASS/abs_hierarchy
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-8} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
IKD=/export/home/rheinric/COMPASS/experiments_arruda_schur
ik() {  # cores args...
  local cores=$1; shift
  env --chdir=$IKD PYTHONPATH=. taskset -c $cores $PY -u ik_absx1.py "$@"
}
slot_ik_R() {
  ik 3-5 train --residual > logs/train_ik_absx1_R.log 2>&1 &&
  ik 3-5 run --residual 2>&1 | grep -v -i warn > logs/run_ik_absx1_R.log
}
slot_c() {
  taskset -c 0-2 $PY -u train_abs.py --model curved --mode hierarchical > logs/train_curved_hierarchical.log 2>&1
  taskset -c 0-2 $PY -u train_abs.py --model curved --mode hierarchical --tag R --residual > logs/train_curved_hierarchical_R.log 2>&1
  for m in hierarchical hierarchical_R; do
    taskset -c 0-2 $PY -u run.py --model curved --mode $m --methods gauss_jacobian_FD arruda_direct 2>&1 | grep -v -i warn > logs/run_curved_$m.log
  done
  echo "C done"
  ik 0-2 train > logs/train_ik_absx1.log 2>&1 &&
  ik 0-2 run 2>&1 | grep -v -i warn > logs/run_ik_absx1.log
}
slot_ik_R & slot_c & wait
ik 0-2 grade > logs/grade_ik_absx1.log 2>&1
echo all done
