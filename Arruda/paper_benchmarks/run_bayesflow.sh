#!/usr/bin/env bash
# BayesFlow 2.0.13 (vendored under Arruda/.vendor) on COMPASS's torch env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KERAS_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/Arruda/.vendor:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.COMPASS/bin/python" "$ROOT/Arruda/paper_benchmarks/bayesflow_arruda.py" "$@"
