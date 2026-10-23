#!/bin/bash
export PYTHONPATH=/export/data/rheinric/COMPASS-jacext/src
P=/export/home/rheinric/COMPASS/.COMPASS/bin/python
cd /export/data/rheinric/COMPASS-jacext/jac_extensions/conservative
for seed in 7 13; do $P -u gm_conservative.py compose --arm cons --seed $seed; done
$P gm_conservative.py report
