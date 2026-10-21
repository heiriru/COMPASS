#!/usr/bin/env bash
# Train all WAIC-validation networks with N parallel workers (each picks a free GPU via autocvd).
#   bash validation/a4_waic/run_training.sh 4 [seed]
# Jobs with an existing final.pt are skipped, so the script can be re-run.
cd "$(dirname "$0")/../.."
WORKERS=${1:-4}
SEED=${2:-0}
PY=.COMPASS/bin/python
LOG=validation/a4_waic/logs
mkdir -p "$LOG"
jobs=("--family population --model LV --budget 131072"
      "--family population --model LP --budget 131072"
      "--family population --model SP --budget 131072"
      "--family population --model RM --budget 131072"
      "--family noise --model G --budget 65536"
      "--family noise --model T --budget 65536"
      "--family offset --model M1 --budget 65536"
      "--family offset --model M0 --budget 65536")
worker() {
  local w=$1
  for ((i = w; i < ${#jobs[@]}; i += WORKERS)); do
    local args="${jobs[$i]} --seed $SEED"
    local tag=$(echo "$args" | tr -d '-' | tr ' ' '_')
    $PY -m validation.a4_waic.train $args > "$LOG/train_$tag.log" 2>&1
    echo "[worker $w] $args -> exit $?"
    sleep 20   # let autocvd see the previous job's GPU memory before the next pick
  done
}
for ((w = 0; w < WORKERS; w++)); do worker $w & sleep 60; done
wait
echo "all training jobs finished"
