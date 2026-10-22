#!/bin/bash
# conservative vs plain GM nets: train, Jacobian accuracy, JAC composition (sequential, one GPU slot)
export PYTHONPATH=/export/data/rheinric/COMPASS-jacext/src
P=/export/home/rheinric/COMPASS/.COMPASS/bin/python
cd /export/data/rheinric/COMPASS-jacext/jac_extensions/conservative
for seed in 7 13; do for arm in plain cons; do
  $P -u gm_conservative.py train --arm $arm --seed $seed
  $P -u gm_conservative.py accuracy --arm $arm --seed $seed
  $P -u gm_conservative.py compose --arm $arm --seed $seed
done; done
$P gm_conservative.py report
