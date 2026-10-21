#!/bin/bash
# Grade the ladder nets: score error vs the exact score, then hier. attn + gauss_jacobian_FD at n=4..128.
set -u
cd /export/home/rheinric/COMPASS/abs_hierarchy
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
S=/export/home/rheinric/.cache/vscode-tmp/claude-2733/-export-home-rheinric/db525932-5839-4227-ac18-f0c09407dc8e/scratchpad
export CUDA_VISIBLE_DEVICES=${GPU:-8} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export PYTHONPATH=.:../Gaussian_mixtures_compositional_score_modelling:../experiments_arruda_schur
TAGS="${TAGS:-T TF R TFR}"
nets="hierarchical"; for t in $TAGS; do nets="$nets hierarchical_$t"; done
taskset -c 0-2 $PY $S/score_ladder.py $nets 2>&1 | grep -v -i warn > logs/lean_ladder_score.log
run() { taskset -c $1 $PY run.py --model lean --methods gauss_jacobian_FD "${@:2}" 2>&1 | grep -v -i warn; }
run 0-2 --mode hierarchical_T --suffix _e4 --sample-eps 1e-4 > logs/lean_ladder_run_T_e4.log &
for t in $TAGS; do run 3-5 --mode hierarchical_$t > logs/lean_ladder_run_$t.log; done
wait
$PY $S/metrics_ladder.py lean > logs/lean_ladder_metrics.log 2>&1
echo evaluated
