#!/usr/bin/env bash
# Stage 2 (inference) and stage 3 (plots), strictly after stage 1 (training).
#
# Nothing here runs while a model is still training. The script blocks until
# run_sweep.sh has written every row of artifacts/capacity.csv, then runs the
# comparisons one at a time on one GPU, then draws the figures. Each stage sees
# a complete, immutable input from the stage before it, so a figure can never be
# built from a half-finished sweep or a partly-written .npz.
#
# Usage:
#   GPU=1 ./run_inference_and_plots.sh
#   GPU=1 WAIT=0 ./run_inference_and_plots.sh    # training already finished
#
# Check the GPU is idle first -- this host is shared:
#   nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/export/home/rheinric/COMPASS/.COMPASS/bin/python}"
GPU="${GPU:-0}"
LOGS="$HERE/artifacts/logs"
ARTIFACTS="$HERE/artifacts"
EXPECTED_CELLS="${EXPECTED_CELLS:-9}"
HEADLINE="${HEADLINE:-h16d2}"
CONTROL="${CONTROL:-h128d6}"
SEEDS="${SEEDS:-1 2}"

mkdir -p "$LOGS"

# ---------------------------------------------------------------------------
# Stage 1 gate: block until every training cell has finished.
# ---------------------------------------------------------------------------
if [[ "${WAIT:-1}" == "1" ]]; then
    echo "=== $(date '+%H:%M:%S')  waiting for training to finish ==="
    while true; do
        rows=0
        if [[ -f "$ARTIFACTS/capacity.csv" ]]; then
            rows=$(( $(wc -l < "$ARTIFACTS/capacity.csv") - 1 ))
        fi
        if [[ $rows -ge $EXPECTED_CELLS ]]; then
            echo "  training complete: $rows/$EXPECTED_CELLS cells"
            break
        fi
        # A dead sweep with missing rows must not silently become "done".
        if ! pgrep -f "exponential_score_comparison/run_sweep.sh" > /dev/null; then
            echo "!!! run_sweep.sh is not running and only $rows/$EXPECTED_CELLS cells are done."
            echo "!!! Refusing to run inference on an incomplete sweep. Restart it with:"
            echo "!!!   GPU=$GPU $HERE/run_sweep.sh"
            exit 1
        fi
        sleep 60
    done
fi

# No training process may still be touching a checkpoint we are about to read.
if pgrep -f "exponential_score_comparison/capacity.py" > /dev/null; then
    echo "!!! a capacity.py training process is still alive; aborting."
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 2: inference, one run at a time.
# ---------------------------------------------------------------------------
run() {
    local name="$1"; shift
    echo "=== $(date '+%H:%M:%S')  inference: $name ==="
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$HERE" \
        "$PYTHON" -u "$HERE/compare.py" "$@" > "$LOGS/${name}.log" 2>&1
    local status=$?
    if [[ $status -ne 0 ]]; then
        echo "!!! $name failed (exit $status); see $LOGS/${name}.log"
        return 1
    fi
    tail -2 "$LOGS/${name}.log"
}

# The headline: the smallest architecture that clears the fidelity bar.
run "compare_${HEADLINE}" --config "$HEADLINE" --output-dir "$ARTIFACTS"

# Repeat seeds. A seed is a different true g *and* a different set of 30
# observations, so it redraws the whole problem -- and since the posterior is
# pressed against a wall at min_j x_j, a minimum of 30 draws, that wall moves a
# lot between datasets. One dataset cannot tell a method difference from one
# unlucky wall position.
for seed in $SEEDS; do
    run "compare_${HEADLINE}_seed${seed}" --config "$HEADLINE" --seed "$seed" \
        --output-dir "$ARTIFACTS/seed${seed}"
done

# The control: at 0.04 relative single-observation score error, essentially all
# remaining composed-score error is the composition rule itself. Reduced draw
# count because 5.2M parameters x 30 observations x 3000 draws is 90K rows in
# one composed call and does not fit an 11 GB card; this makes its *sampling*
# statistics noisier and not directly comparable on W1, while leaving the
# composed-score panel (which uses --score-states) unaffected.
run "compare_${CONTROL}_control" --config "$CONTROL" --num-samples 1000 \
    --output-dir "$ARTIFACTS/control_${CONTROL}"

# ---------------------------------------------------------------------------
# Stage 3: figures, from the finished inference outputs only.
# ---------------------------------------------------------------------------
echo "=== $(date '+%H:%M:%S')  plots ==="
PYTHONPATH="$HERE" "$PYTHON" "$HERE/plot_capacity.py" \
    --config h8d1 > "$LOGS/plot_capacity.log" 2>&1 \
    && tail -1 "$LOGS/plot_capacity.log" \
    || echo "!!! plot_capacity failed; see $LOGS/plot_capacity.log"

echo "=== $(date '+%H:%M:%S')  pipeline finished ==="
ls -1 "$ARTIFACTS"/*.png "$ARTIFACTS"/*.csv 2>/dev/null
