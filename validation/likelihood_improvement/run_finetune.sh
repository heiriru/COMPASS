#!/usr/bin/env bash
# ML fine-tuning of the single menu+EDM network (fixed PF-ODE base), then B6 eval + posterior check.
#   bash validation/likelihood_improvement/run_finetune.sh 0
cd "$(dirname "$0")/../.."
PY=.COMPASS/bin/python
LOG=validation/likelihood_improvement/logs
SEED=${1:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {
  local family=$1
  $PY -m validation.likelihood_improvement.finetune_ml --family $family --seed $SEED \
      > "$LOG/finetune_${family}_s${SEED}.log" 2>&1
  $PY -m validation.b5_b6_pfode_accuracy.evaluate --family $family --budget 65536 --seed $SEED \
      --variant menu_edm_ml > "$LOG/eval_ml_${family}_s${SEED}.log" 2>&1
  $PY -m validation.likelihood_improvement.posterior_check --family $family --seed $SEED \
      --variant menu_edm_ml > "$LOG/post_${family}_s${SEED}_menu_edm_ml.log" 2>&1
  echo "$family seed $SEED -> exit $?"
}
for family in gauss student bimodal; do run $family & sleep 120; done
wait
echo "all fine-tune jobs finished"
