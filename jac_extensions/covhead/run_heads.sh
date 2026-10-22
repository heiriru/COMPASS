#!/bin/bash
# head training + accuracy for both GM networks (GPU picked by gpu.py)
export PYTHONPATH=/export/data/rheinric/COMPASS-jacext/src
P=/export/home/rheinric/COMPASS/.COMPASS/bin/python
cd /export/data/rheinric/COMPASS-jacext/jac_extensions/covhead
for mode in hierarchical none; do
  $P -u gm_head.py train --mode $mode --steps 20000
  $P -u gm_head.py accuracy --mode $mode
done
