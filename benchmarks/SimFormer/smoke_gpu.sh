#!/bin/bash
# GPU smoke test: tiny budgets, one autocvd-selected GPU.
B=/export/home/rheinric/COMPASS/benchmarks/SimFormer
cd $B
GPU=$(/export/home/rheinric/COMPASS/.COMPASS/bin/python -c "from autocvd import autocvd; print(autocvd(num_gpus=1, interval=1)[0])" 2>/dev/null | tail -1)
echo "GPU=$GPU"
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
P=$B/.simformer/bin/python
taskset -c 0-2 $P bridge/run_scoresbi.py name=smoke method=compass_legacy task=two_moons task.num_simulations=1000 ++method.compass.max_steps=200 save_model=false > smoke_a.log 2>&1
echo "a rc=$?"
taskset -c 0-2 $P bridge/run_scoresbi.py name=smoke method=compass_tokenized task=tree_all_cond task.num_simulations=1000 ++method.compass.max_steps=200 eval=c2st eval.metric.c2st.num_evaluations=4 eval.metric.c2st.num_samples=500 save_model=false > smoke_b.log 2>&1
echo "b rc=$?"
taskset -c 0-2 $P bridge/run_scoresbi.py name=smoke method=score_transformer task=two_moons task.num_simulations=1000 method.train.min_number_steps=300 save_model=false > smoke_c.log 2>&1
echo "c rc=$?"
