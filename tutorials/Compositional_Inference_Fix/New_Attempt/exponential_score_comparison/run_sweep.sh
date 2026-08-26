#!/usr/bin/env bash
# Run the capacity sweep one cell at a time on a single GPU.
#
# This host is shared, so the sweep must not spread itself across every card.
# One process at a time, one GPU, in the order below; `capacity.py` appends each
# row as it finishes and skips rows already in artifacts/capacity.csv, so this
# script is safe to re-run and resumes where a kill left off.
#
# The first six cells hold the simulation budget at 200K and vary the
# architecture (how small?); the last three vary the budget at the architecture
# chosen from those (how much data?). CONFIG_FOR_DATA_LADDER is separate so the
# ladder can be re-pointed without touching the size sweep.
#
# Usage:
#   ./run_sweep.sh            # GPU 0
#   GPU=3 ./run_sweep.sh      # somewhere else
#   FRESH=1 ./run_sweep.sh    # discard partial checkpoints and the CSV first
#
# FRESH matters after an interrupted run: the trainer writes
# Model_checkpoint.pt on every validation improvement, so a killed cell leaves a
# *partially trained* checkpoint behind, and train_or_load would load it as
# though it were finished. Deleting the directory is the only way to be sure a
# reported row reflects the full recipe.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/export/home/rheinric/COMPASS/.COMPASS/bin/python}"
GPU="${GPU:-0}"
LOGS="$HERE/artifacts/logs"

CONFIG_FOR_DATA_LADDER="${CONFIG_FOR_DATA_LADDER:-h8d1}"
SIZE_CONFIGS=(h128d6 h32d3 h16d2 h16d1 h8d1 h4d1)
DATA_BUDGETS=(10000 50000 500000)
REFERENCE_SAMPLES=200000

CELLS=()
for config in "${SIZE_CONFIGS[@]}"; do
    CELLS+=("$config $REFERENCE_SAMPLES")
done
for budget in "${DATA_BUDGETS[@]}"; do
    CELLS+=("$CONFIG_FOR_DATA_LADDER $budget")
done

if [[ "${FRESH:-0}" == "1" ]]; then
    echo "FRESH: removing partial checkpoints and artifacts/capacity.csv"
    rm -rf "$HERE/artifacts/models" "$HERE/artifacts/capacity.csv"
fi
mkdir -p "$LOGS"

echo "Running ${#CELLS[@]} cells sequentially on GPU $GPU"
for cell in "${CELLS[@]}"; do
    read -r config samples <<<"$cell"
    echo "=== $(date '+%H:%M:%S')  $config @ $samples simulations ==="
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$HERE" \
        "$PYTHON" "$HERE/capacity.py" --config "$config" --train-samples "$samples" \
        > "$LOGS/${config}_${samples}.log" 2>&1
    status=$?
    if [[ $status -ne 0 ]]; then
        # Leave the partial checkpoint for inspection but make the failure loud:
        # a silently skipped cell would show up as a missing row much later.
        echo "!!! $config @ $samples failed (exit $status); see $LOGS/${config}_${samples}.log"
    else
        tail -2 "$LOGS/${config}_${samples}.log"
    fi
done
echo "=== $(date '+%H:%M:%S')  sweep finished ==="
