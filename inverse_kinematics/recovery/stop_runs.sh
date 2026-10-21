#!/bin/bash
# Stop every recovery job cleanly (lane shells first, so no next step starts; then the workers).
# Safe at any moment: finished batches/datasets are already on disk, partial writes are .tmp files.
for pat in "bash -c .*recovery/run_" "bash -c for n in .*recovery/run_" "python -u .*recovery/run_reference.py" \
           "python -u .*recovery/run_network.py" "python -u .*recovery/bimodality.py" "python .*recovery/plot_recovery.py"; do
    for pid in $(pgrep -u "$USER" -f "$pat"); do
        [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null
    done
done
sleep 5
pgrep -u "$USER" -fa "recovery/(run_|bimodality|plot_recovery)" || echo "all recovery jobs stopped $(date)"
