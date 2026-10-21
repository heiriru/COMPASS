#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KERAS_BACKEND=torch
export PYTHONPATH="$ROOT/Arruda/.vendor:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/.COMPASS/bin/python" "$ROOT/Arruda/run_arruda.py" "$@"
