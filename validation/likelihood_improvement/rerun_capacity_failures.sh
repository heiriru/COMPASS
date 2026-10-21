#!/usr/bin/env bash
# Re-run the OOM-failed capacity evaluations / posterior checks one at a time.
cd "$(dirname "$0")/../.."
PY=.COMPASS/bin/python
LOG=validation/likelihood_improvement/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for job in "gauss|0|menu_edm_d8" "gauss|2|menu_edm_d8" "gauss|2|menu_edm_p100e1500"; do
  IFS='|' read -r f s v <<< "$job"
  $PY -m validation.b5_b6_pfode_accuracy.evaluate --family $f --budget 65536 --seed $s --variant $v \
      > "$LOG/eval_cap_${v}_${f}_s${s}_n65536.log" 2>&1; echo "eval $job -> $?"
done
for job in "gauss|0|menu_edm_d8" "gauss|2|menu_edm_d8" "gauss|2|menu_edm_p100e1500" \
           "gauss|0|menu_edm_h256" "gauss|1|menu_edm_h256" "gauss|2|menu_edm_h256" \
           "student|0|menu_edm_h256" "student|1|menu_edm_h256" "student|2|menu_edm_h256" \
           "bimodal|0|menu_edm_h256" "bimodal|1|menu_edm_h256" "bimodal|2|menu_edm_h256"; do
  IFS='|' read -r f s v <<< "$job"
  $PY -m validation.likelihood_improvement.posterior_check --family $f --seed $s --variant $v \
      > "$LOG/post_cap_${v}_${f}_s${s}_n65536.log" 2>&1; echo "post $job -> $?"
done
