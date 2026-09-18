#!/usr/bin/env bash
# Regenerate every artifact.  Experiments 1-5 are independent and run in
# parallel (each caps itself at three threads); the timing measurement runs
# alone afterwards, because its whole point is an uncontended wall clock.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$ROOT/.COMPASS/bin/python}"
export PYTHONPATH="$ROOT/ModeTree:$ROOT/src"
export CUDA_VISIBLE_DEVICES=""
OUT="$ROOT/ModeTree/artifacts"
mkdir -p "$OUT"

for script in experiment1_known_modes compare_methods ablations \
              experiment_2d experiment_hierarchical experiment_score_error; do
    "$PY" -u "$ROOT/ModeTree/experiments/$script.py" > "$OUT/$script.log" 2>&1 &
done
wait

"$PY" -u "$ROOT/ModeTree/experiments/timing.py"    > "$OUT/timing.log"    2>&1
"$PY" -u "$ROOT/ModeTree/experiments/figures.py"   > "$OUT/figures.log"   2>&1
"$PY" -u "$ROOT/ModeTree/experiments/summarize.py" > "$OUT/summarize.log" 2>&1
echo "all artifacts regenerated in $OUT"
