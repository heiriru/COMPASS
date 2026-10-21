#!/bin/bash
# One lane of partial-pooling speed-up runs, sequential, on the GPU given as $1.
# usage: queue_pp.sh <gpu> <variant[:seed]> ...
gpu=$1; shift
cd "$(dirname "$0")"
for item in "$@"; do
  variant=${item%%:*}; seed=7; [[ $item == *:* ]] && seed=${item##*:}
  tag=speed_${variant//\//o}; [[ $seed != 7 ]] && tag=${tag}_seed$seed
  echo "=== $(date +%H:%M) $variant seed $seed -> $tag"
  CUDA_VISIBLE_DEVICES=$gpu ../.COMPASS/bin/python -u bench_pp.py "$variant" --mode hierarchical \
    --precondition Rg --train-prior-scale 1.5 --datasets 10 --seed "$seed" --tag "$tag" 2>&1 \
    | grep --line-buffered -v -i "warn\|_mle" | grep --line-buffered "^\[\|Error\|error"
done
echo "=== lane done $(date +%H:%M)"
