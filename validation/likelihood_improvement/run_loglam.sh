#!/usr/bin/env bash
# Train + evaluate the log-lambda time-embedding networks (2^16 sims, 3 families x 3 seeds).
#   bash validation/likelihood_improvement/run_loglam.sh 3
cd "$(dirname "$0")/../.."
WORKERS=${1:-3}
PY=.COMPASS/bin/python
LOG=validation/likelihood_improvement/logs
jobs=()
for family in gauss student bimodal; do for seed in 0 1 2; do jobs+=("$family $seed"); done; done
worker() {
  local w=$1
  for ((i = w; i < ${#jobs[@]}; i += WORKERS)); do
    set -- ${jobs[$i]}
    $PY -m validation.b5_b6_pfode_accuracy.train --family $1 --budget 65536 --seed $2 \
        --time-embedding log_lambda > "$LOG/train_$1_s$2.log" 2>&1
    $PY -m validation.b5_b6_pfode_accuracy.evaluate --family $1 --budget 65536 --seed $2 \
        --variant loglam > "$LOG/eval_$1_s$2.log" 2>&1
    echo "[worker $w] $1 seed $2 -> exit $?"
  done
}
for ((w = 0; w < WORKERS; w++)); do worker $w & sleep 120; done
wait
echo "all loglam jobs finished"
