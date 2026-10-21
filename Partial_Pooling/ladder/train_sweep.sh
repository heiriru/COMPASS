#!/bin/bash
# Stage 4: parameter-budget sweep on the lossless IID-trial architecture, with
# the Stage-2 (scalar-token) axis kept measured at matched budgets and the
# Stage-2 token-offset ablation the plan asks for.
#
# Training is cheap here (10-25 min per arm), so the whole grid is trained and
# ranked on held-out denoising loss; only the best arms then pay for the
# ~2-3 hour recovery evaluation.
set -u
PP=/export/home/rheinric/COMPASS/Partial_Pooling
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
LOGS=$PP/ladder/logs
MAX=${MAX_CONCURRENT:-5}

running() { pgrep -fc "train_models.py" || true; }

launch() {  # name, extra flags...
  local name=$1; shift
  while [ "$(running)" -ge "$MAX" ]; do sleep 30; done
  echo "=== [$(date -Is)] starting $name" >&2
  ( "$PY" "$PP/train_models.py" --preset small --train-subset 100000 --device cuda \
        "$@" >"$LOGS/$name.log" 2>&1
    echo "=== [$(date -Is)] $name exit $?" >&2 ) &
  sleep 45
}

heads_for() {  # largest head count up to 8 keeping head_dim >= 8
  case $1 in 16) echo 2;; 32) echo 4;; 48) echo 4;; 64) echo 8;; *) echo 1;; esac
}

for H in 16 32 48 64; do
  for D in 2 3 4; do
    [ "$H" = 16 ] && [ "$D" = 2 ] && continue   # already trained
    launch "tok_trial_h${H}d${D}" --arch tokenized --tokenization trial \
        --hidden-size "$H" --depth "$D" --num-heads "$(heads_for $H)" --mlp-ratio 4
  done
done

# Stage-2 axis at matched budgets, so the trunk question stays answerable at
# every scale rather than only at the smallest one.
for H in 32 48 64; do
  launch "tok_scalar_h${H}d4" --arch tokenized --tokenization scalar \
      --hidden-size "$H" --depth 4 --num-heads "$(heads_for $H)" --mlp-ratio 4
done

# The plan's Stage-2 open question, tested as an ablation rather than a default.
launch "tok_trial_h16d2_offset" --arch tokenized --tokenization trial \
    --hidden-size 16 --depth 2 --num-heads 2 --mlp-ratio 4 --token-offset

wait
