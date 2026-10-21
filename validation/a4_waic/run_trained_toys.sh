#!/usr/bin/env bash
# Trained-network WAIC runs for the offset and noise families, one GPU process at a time per
# family; the second launch waits until the first has claimed its GPU (autocvd collisions).
cd "$(dirname "$0")/../.."
PY=.COMPASS/bin/python
LOG=validation/a4_waic/logs
run() {  # family true
  PYTHONPATH=. $PY -m validation.a4_waic.run_compass --family $1 --level trained --true $2 --reps 200 --draws 1000 > $LOG/trained_$1_$2.log 2>&1
}
wait_claimed() { until grep -q "Selected physical GPU" $1 2>/dev/null; do sleep 10; done; sleep 90; }
(run offset M1; run offset M0) &
wait_claimed $LOG/trained_offset_M1.log
(run noise G; run noise T) &
wait
