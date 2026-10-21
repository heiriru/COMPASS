#!/usr/bin/env bash
# Capacity / training-length / data arms on the single menu+EDM network.
#   bash validation/likelihood_improvement/run_capacity.sh 3
cd "$(dirname "$0")/../.."
WORKERS=${1:-3}
PY=.COMPASS/bin/python
LOG=validation/likelihood_improvement/logs
BASE="--mask-menu inference --loss-weight edm"
jobs=()
for seed in 0 1 2; do for family in gauss student bimodal; do
  jobs+=("$family|$seed|65536|--hidden-size 256|menu_edm_h256")
  jobs+=("$family|$seed|65536|--depth 8|menu_edm_d8")
  jobs+=("$family|$seed|65536|--patience 100 --max-epochs 1500|menu_edm_p100e1500")
  jobs+=("$family|$seed|262144||menu_edm")
done; done
worker() {
  local w=$1
  for ((i = w; i < ${#jobs[@]}; i += WORKERS)); do
    IFS='|' read -r family seed budget extra variant <<< "${jobs[$i]}"
    tag="${variant}_${family}_s${seed}_n${budget}"
    $PY -m validation.b5_b6_pfode_accuracy.train --family $family --budget $budget --seed $seed \
        $BASE $extra > "$LOG/train_cap_$tag.log" 2>&1
    $PY -m validation.b5_b6_pfode_accuracy.evaluate --family $family --budget $budget --seed $seed \
        --variant $variant > "$LOG/eval_cap_$tag.log" 2>&1
    $PY -m validation.likelihood_improvement.posterior_check --family $family --seed $seed \
        --variant $variant --budget $budget > "$LOG/post_cap_$tag.log" 2>&1
    echo "[worker $w] $tag -> exit $?"
  done
}
for ((w = 0; w < WORKERS; w++)); do worker $w & sleep 150; done
wait
echo "all capacity jobs finished"
