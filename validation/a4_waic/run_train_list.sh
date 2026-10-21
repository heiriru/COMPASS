#!/usr/bin/env bash
# Train a list of WAIC-validation networks, one job per worker, launches staggered so autocvd
# never hands one GPU to two jobs.   bash validation/a4_waic/run_train_list.sh jobs.txt
# Each line of jobs.txt: arguments for validation.a4_waic.train (existing final.pt are skipped).
cd "$(dirname "$0")/../.."
LOG=validation/a4_waic/logs
mapfile -t jobs < "$1"
for args in "${jobs[@]}"; do
  [ -z "$args" ] && continue
  tag=$(echo "$args" | tr -d '-' | tr ' ' '_')
  .COMPASS/bin/python -m validation.a4_waic.train $args > "$LOG/train_$tag.log" 2>&1 &
  until grep -q "Selected physical GPU\|exists:" "$LOG/train_$tag.log" 2>/dev/null; do sleep 10; done
  sleep 60
done
wait
echo "all training jobs finished"
