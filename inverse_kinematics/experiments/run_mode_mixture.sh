#!/usr/bin/env bash
# The mode-mixture experiment on the arm. One GPU, sequential.
#
#   main   the method: K discovered by continuation, Laplace components from
#          the score Jacobian, branch weights from the score path integral
#   k1     the internal control: the same machinery with K forced to 1, which
#          is a Laplace version of the existing Gaussian rules
#   em     the attribution control: the same discovered K and the same
#          composition, but the components refit to the network's own draws by
#          maximum likelihood, so a failure that survives it is not the Laplace
#          approximation's fault
#   branch weights read off up the ladder, where the path between modes is
#          short, instead of at the working noise scale
set -u
PY=/export/home/rheinric/COMPASS/.COMPASS/bin/python
SCRIPT=/export/home/rheinric/COMPASS/inverse_kinematics/experiments/mode_mixture.py
OUT=/export/home/rheinric/COMPASS/inverse_kinematics/artifacts/mode_mixture
COMMON="--observations 8 --seed 99 --single-draws 20000 --draws 4000 \
        --reference-draws 200000 --single-reference-draws 100000"

run () {  # run <name> <extra args...>
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $name"
  CUDA_VISIBLE_DEVICES=${GPU:-1} $PY "$SCRIPT" $COMMON "$@" --output "$OUT/$name" 2>&1 \
    | grep -Ev '^\s*$' | sed 's/^/    /'
  echo "=== [$(date +%H:%M:%S)] done $name"
}

run main   --sigma-weights 0.40 --methods gauss_jacobian gauss_hierarchical
run k1     --sigma-weights 0.40 --force-k1 --methods
run em     --sigma-weights 0.40 --component-source em --methods
run branch --sigma-weights 0.25 --sigma-branch 0.40 --methods

# Replicates of the working configuration. One dataset is a pilot, not a result.
replicate () { run "$@" --component-source em --methods gauss_jacobian gauss_hierarchical; }
