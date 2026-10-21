#!/usr/bin/env bash
# Train all B5/B6 networks with N parallel workers; each job picks a free GPU via autocvd.
#   bash validation/b5_b6_pfode_accuracy/run_training.sh 3
# Jobs that already have final.pt are skipped, so the script can be re-run safely.
cd "$(dirname "$0")/../.."
WORKERS=${1:-3}
PY=.COMPASS/bin/python
LOG=validation/b5_b6_pfode_accuracy/logs
mkdir -p "$LOG"

jobs=()
for budget in 65536 16384 4096; do
  for family in gauss student bimodal; do
    for seed in 0 1 2; do
      jobs+=("--family $family --nu 0.5 --budget $budget --seed $seed")
    done
  done
done
for nu in 0.05 0.1 0.2; do                       # E-B5.3 noise sweep (gauss, 2^16, seed 0)
  jobs+=("--family gauss --nu $nu --budget 65536 --seed 0")
done

worker() {
  local w=$1
  for ((i = w; i < ${#jobs[@]}; i += WORKERS)); do
    local args=${jobs[$i]}
    local tag=$(echo "$args" | tr -d '-' | tr ' ' '_')
    $PY -m validation.b5_b6_pfode_accuracy.train $args > "$LOG/train_$tag.log" 2>&1
    echo "[worker $w] $args -> exit $?"
  done
}
for ((w = 0; w < WORKERS; w++)); do worker $w & done
wait
echo "all training jobs finished"
