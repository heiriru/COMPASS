#!/usr/bin/env bash
# pipeline.sh MODE "METHODS..." [MODEL=abs] : train (skip if cached) then run inference for MODE.
cd "$(dirname "$0")"
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
mode=$1; methods=$2; model=${3:-abs}
$PY -u train_abs.py --model "$model" --mode "$mode" >> "logs/train_${model}_$mode.log" 2>&1 || exit 1
$PY -u run.py --model "$model" --mode "$mode" --methods $methods >> "logs/run_${model}_$mode.log" 2>&1
