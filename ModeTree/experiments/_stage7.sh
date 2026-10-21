#!/bin/bash
# Run exp7 stages unbuffered, appending to a per-stage log.
#   _stage7.sh <mu0|std> <stage> [stage ...]
# Each invocation is one process; the CPU cap of 3 threads is applied inside
# the script itself (configure_cpu_limit(3)), so a lane is 3 threads.
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
OUT=/export/home/rheinric/COMPASS/ModeTree/artifacts/gm_topology
SCRIPT=/export/home/rheinric/COMPASS/ModeTree/experiments/exp7_gm_topology.py
MU=$1; shift
mkdir -p "$OUT"
if [ "$MU" = "std" ]; then EXTRA=(); SUF=""; else EXTRA=(--mu0 "$MU" --n 128); SUF="_mu${MU}"; fi
for stage in "$@"; do
  echo "### START $stage $MU $(date -Is)" >> "$OUT/${stage}${SUF}.log"
  PYTHONUNBUFFERED=1 "$PY" "$SCRIPT" --stage "$stage" "${EXTRA[@]}" \
      >> "$OUT/${stage}${SUF}.log" 2>&1
  echo "### DONE $stage rc=$? $(date -Is)" >> "$OUT/${stage}${SUF}.log"
done
