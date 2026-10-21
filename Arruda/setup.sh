#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/Arruda/.vendor"
mkdir -p "$VENDOR"
"$ROOT/.COMPASS/bin/pip" install --no-cache-dir --target "$VENDOR" "bayesflow==2.0.13"
