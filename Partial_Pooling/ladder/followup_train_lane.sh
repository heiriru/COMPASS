#!/bin/bash
# Follow-up arms on top of the Stage-4 winner (trial tokens, h64/d2/x8/m4,
# 463,364 parameters, 100,000 simulations). Every arm keeps that backbone, the
# same simulations, the same normalization constants and the same test datasets,
# so the only thing that varies is the intervention named in the arm.
#
#   mask6     task 1: train on the six conditionals p(g,l|x), p(g|l,x),
#             p(l|g,x), p(x|g,l), p(x,l|g), p(x,g|l) instead of Bernoulli(0.33)
#   mask6mix  the same six, 75/25 against the Bernoulli draw
#   te64      task 3: diffusion-time embedding 256 -> 64 (463,364 -> 192,836)
#   pph       task 4: per-scalar output readout for the 10 theta dimensions
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder/logs
mkdir -p "$LOGS"

BACKBONE=(--preset small --train-subset 100000 --device cuda
          --arch tokenized --tokenization trial
          --hidden-size 64 --depth 2 --num-heads 8 --mlp-ratio 4)

run_arm() {
  local name=$1; shift
  echo "=== [$(date -Is)] starting $name" >&2
  "$PY" "$PP/train_models.py" "${BACKBONE[@]}" "$@" >"$LOGS/train-$name.log" 2>&1
  echo "=== [$(date -Is)] $name exit $?" >&2
}

case "$1" in
  control)  run_arm control ;;
  mask6)    run_arm mask6    --mask-menu six ;;
  mask6mix) run_arm mask6mix --mask-menu six_mixed --mask-bernoulli-fraction 0.25 ;;
  mask6pw4) run_arm mask6pw4 --mask-menu six --mask-posterior-weight 4 ;;
  mask6pw8) run_arm mask6pw8 --mask-menu six --mask-posterior-weight 8 ;;
  te64)     run_arm te64     --time-embedding-size 64 ;;
  pph)      run_arm pph      --per-position-head ;;
  te64pph)  run_arm te64pph  --time-embedding-size 64 --per-position-head ;;
  best)     run_arm best     --mask-menu six --time-embedding-size 64 --per-position-head ;;
  *) echo "unknown arm $1" >&2; exit 2 ;;
esac
