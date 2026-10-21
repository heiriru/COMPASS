#!/bin/bash
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
OUT=/export/home/rheinric/COMPASS/ModeTree/artifacts/gm_oracle
MU=$1; shift
mkdir -p "$OUT"
for stage in "$@"; do
  echo "### START $stage mu=$MU $(date -Is)" >> "$OUT/${stage}_mu${MU}.log"
  PYTHONUNBUFFERED=1 "$PY" /export/home/rheinric/COMPASS/ModeTree/experiments/exp6_gm_oracle_bridges.py \
     --stage "$stage" --mu0 "$MU" --n 128 >> "$OUT/${stage}_mu${MU}.log" 2>&1
  echo "### DONE $stage rc=$? $(date -Is)" >> "$OUT/${stage}_mu${MU}.log"
done
