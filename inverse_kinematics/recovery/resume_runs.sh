#!/bin/bash
# Resume the n = 64 / 128 recovery runs after an interruption; finished work is skipped.
# Run from anywhere: bash COMPASS/inverse_kinematics/recovery/resume_runs.sh
builtin cd /export/home/rheinric/COMPASS || exit 1
R=inverse_kinematics/recovery; P=.COMPASS/bin/python; A=$R/artifacts
AH="--warmup 4000 --draws 500 --thin 20"
reference() {   # layout n extra-args...
    local layout=$1 n=$2; shift 2
    local dir=$A/${layout}_n$n
    if grep -q repaired $dir/reference_diagnostics.json 2>/dev/null; then echo "$layout n=$n: done"; return; fi
    if [ ! -f $dir/reference.npz ]; then
        $P -u $R/run_reference.py --kind hier --layout $layout --n $n --count 100 --batch 10 --resumable "$@" || return
    fi
    $P -u $R/run_reference.py --kind hier --layout $layout --n $n --count 100 "$@" --repair 2
}
c=6
for n in 64 128; do
    nohup taskset -c $c-$((c+2)) bash -c "$(declare -f reference); A=$A R=$R P=$P; reference angle $n" > $R/logs/reference_angle_n$n.log 2>&1 &
    c=$((c+3))
    nohup taskset -c $c-$((c+2)) bash -c "$(declare -f reference); A=$A R=$R P=$P; reference angle_height $n $AH" > $R/logs/reference_angle_height_n$n.log 2>&1 &
    c=$((c+3))
done
# GPUs: autocvd picks free cards (run_network.py calls it when CUDA_VISIBLE_DEVICES is unset).
for layout in angle angle_height; do
    nohup taskset -c $c-$((c+2)) bash -c "for n in 64 128; do $P -u $R/run_network.py --kind hier --layout $layout --n \$n --count 100; done" \
        > $R/logs/network_${layout}_n64_128.log 2>&1 &
    c=$((c+3))
done
echo "resumed $(date); logs in $R/logs/"
