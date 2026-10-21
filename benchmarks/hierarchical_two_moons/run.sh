#!/usr/bin/env bash
# Entry point: ./run.sh <reference|train|run|report|all> [options]  (see benchmark.py)
# COMPASS arms run in the COMPASS env; BayesFlow arms additionally need the
# vendored BayesFlow 2.0.13 (Arruda/.vendor) and KERAS_BACKEND=torch, so `all`
# runs the two families as separate processes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPASS="$(cd "$HERE/../.." && pwd)"
PY="$COMPASS/.COMPASS/bin/python"
BASE_PATH="$HERE/.vendor:$HERE/external/hsbibm:$COMPASS/src${PYTHONPATH:+:$PYTHONPATH}"

compass() { PYTHONPATH="$BASE_PATH" "$PY" -u "$HERE/benchmark.py" "$@"; }
bayesflow() {
  KERAS_BACKEND=torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="$COMPASS/Arruda/.vendor:$BASE_PATH" "$PY" -u "$HERE/benchmark.py" "$@"
}

if [[ "${1:-}" == "all" ]]; then
  shift
  compass reference "$@"
  compass train --family compass "$@"
  bayesflow train --family bayesflow "$@"
  compass run --family compass "$@"
  bayesflow run --family bayesflow "$@"
  compass report "$@"
elif [[ " $* " == *bayesflow* ]]; then
  bayesflow "$@"
else
  compass "$@"
fi
