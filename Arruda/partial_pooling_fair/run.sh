#!/usr/bin/env bash
# fair.py on COMPASS's torch env with the vendored BayesFlow 2.0.13 (as Arruda/paper_benchmarks/run_bayesflow.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KERAS_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/Arruda/.vendor:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/Arruda/partial_pooling_fair"
exec "$ROOT/.COMPASS/bin/python" -u fair.py "$@"
