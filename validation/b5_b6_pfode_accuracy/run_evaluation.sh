#!/usr/bin/env bash
# Evaluate every trained B5/B6 network (GPU via autocvd) and the per-family oracles (CPU).
#   bash validation/b5_b6_pfode_accuracy/run_evaluation.sh 3
# Finished results are skipped, so the script can be re-run safely.
cd "$(dirname "$0")/../.."
WORKERS=${1:-3}
PY=.COMPASS/bin/python
LOG=validation/b5_b6_pfode_accuracy/logs
ROOT=/export/data/rheinric/compass_validation/b5b6/models

for family in gauss student bimodal; do
  $PY -m validation.b5_b6_pfode_accuracy.evaluate --family $family --oracle > "$LOG/eval_oracle_$family.log" 2>&1 &
done

jobs=()
for dir in $ROOT/*/n*_s*; do
  [ -f "$dir/final.pt" ] || continue
  tag=$(basename "$(dirname "$dir")"); run=$(basename "$dir")
  family=${tag%_nu*}; nu=${tag#*_nu}; budget=${run%_s*}; budget=${budget#n}; seed=${run#*_s}
  jobs+=("--family $family --nu $nu --budget $budget --seed $seed")
done
worker() {
  local w=$1
  for ((i = w; i < ${#jobs[@]}; i += WORKERS)); do
    local args=${jobs[$i]}
    local tag=$(echo "$args" | tr -d '-' | tr ' ' '_')
    $PY -m validation.b5_b6_pfode_accuracy.evaluate $args > "$LOG/eval_$tag.log" 2>&1
    echo "[worker $w] $args -> exit $?"
  done
}
for ((w = 0; w < WORKERS; w++)); do worker $w & sleep 120; done  # stagger so autocvd sees the previous GPU in use
wait
echo "all evaluation jobs finished"
