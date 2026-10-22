#!/bin/bash
# GM composition: production hierarchical net, JAC with FD Jacobian vs with the covariance head
export PYTHONPATH=/export/data/rheinric/COMPASS-jacext/src JACEXT_DEVICE=cpu
P=/export/home/rheinric/COMPASS/.COMPASS/bin/python
cd /export/data/rheinric/COMPASS-jacext/jac_extensions/conservative
H=/export/data/rheinric/COMPASS-jacext/jac_extensions/covhead/artifacts/gm_hierarchical/head.pt
$P -u gm_conservative.py compose --arm prod_hierarchical --tag jacext_covhead --head $H --device cpu
$P -u gm_conservative.py compose --arm prod_hierarchical --tag jacext_fdref --device cpu
