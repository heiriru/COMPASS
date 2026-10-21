#!/bin/bash
# Step 2: rule vs network on the rail set, then on hier_a0.6. One GPU (autocvd), sequential.
builtin cd /export/home/rheinric/COMPASS
A=Arruda/artifacts/ik
.COMPASS/bin/python -u $A/analysis/rule_vs_network.py $A/rail_height/data_a0.6_tol0.05.npy $A/rail_a0.6_compass $A/exact/rail_a0.6.npz > $A/exact/rule_vs_network_rail.log 2>&1
.COMPASS/bin/python -u $A/analysis/rule_vs_network.py hier $A/hier_a0.6_compass $A/exact/hier_a0.6.npz > $A/exact/rule_vs_network_hier.log 2>&1
