#!/bin/bash
export PYTHONPATH=/export/data/rheinric/COMPASS-jacext/src JACEXT_DEVICE=cpu
cd /export/data/rheinric/COMPASS-jacext/jac_extensions/mode_aware
/export/home/rheinric/COMPASS/.COMPASS/bin/python -u gm_mode_aware.py run --net hierarchical
