#!/usr/bin/env bash
# The full benchmark, one process at a time (every step pins itself to 3 CPU cores).
# GPU is chosen by the caller (CUDA_VISIBLE_DEVICES) or by autocvd inside benchmark.py.
set -uo pipefail
cd "$(dirname "$0")"
./run.sh reference
./run.sh train --family compass
./run.sh train --family bayesflow
./run.sh run --family compass
./run.sh run --family bayesflow
./run.sh report
