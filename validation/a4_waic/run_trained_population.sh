#!/usr/bin/env bash
# Population trained-network WAIC runs (Hutchinson-1 PF-ODE divergence, 500 draws); each launch
# waits until the previous one has claimed its GPU, so autocvd never hands one GPU to two jobs.
cd "$(dirname "$0")/../.."
LOG=validation/a4_waic/logs
for t in LV LP SP RM; do
  PYTHONPATH=. .COMPASS/bin/python -m validation.a4_waic.run_compass --family population --level trained \
      --true $t --reps 30 --draws 500 --waic-divergence hutchinson > $LOG/trained_population_$t.log 2>&1 &
  until grep -q "Selected physical GPU" $LOG/trained_population_$t.log 2>/dev/null; do sleep 10; done
  sleep 90
done
wait
