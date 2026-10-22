#!/bin/bash
export PYTHONPATH=/export/data/rheinric/COMPASS-jacext/src
P=/export/home/rheinric/COMPASS/.COMPASS/bin/python
cd /export/data/rheinric/COMPASS-jacext/jac_extensions/mode_aware
for net in oracle hierarchical; do $P -u gm_mode_aware.py run --net $net; done
