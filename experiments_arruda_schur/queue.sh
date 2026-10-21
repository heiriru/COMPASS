#!/usr/bin/env bash
# queue.sh GPU WAIT_PID CMD...  -- run each ';;'-separated command on GPU once WAIT_PID has exited.
gpu=$1; wait_pid=$2; shift 2
cd "$(dirname "$0")"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
IFS=';;' read -ra _ <<< ""
cmds="$*"
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  echo "$(date -Is) GPU $gpu: $cmd"
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd"
  echo "$(date -Is) GPU $gpu: exit $?"
done <<< "$(echo "$cmds" | tr '|' '\n')"
