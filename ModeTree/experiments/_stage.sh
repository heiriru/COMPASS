#!/bin/bash
# Run one or more stages of exp6 unbuffered, appending to a per-stage log.
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
OUT=/export/home/rheinric/COMPASS/ModeTree/artifacts/gm_oracle
mkdir -p "$OUT"
for stage in "$@"; do
  echo "### START $stage $(date -Is)" >> "$OUT/$stage.log"
  PYTHONUNBUFFERED=1 "$PY" /export/home/rheinric/COMPASS/ModeTree/experiments/exp6_gm_oracle_bridges.py --stage "$stage" >> "$OUT/$stage.log" 2>&1
  echo "### DONE $stage rc=$? $(date -Is)" >> "$OUT/$stage.log"
done
