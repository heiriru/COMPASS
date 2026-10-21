#!/bin/bash
# Remaining exp7 work, in three lanes with explicit prerequisite waits.
# Each lane runs one process at a time; each process caps itself at 3 threads.
D=/export/home/rheinric/COMPASS/ModeTree/artifacts/gm_topology
S=/export/home/rheinric/COMPASS/ModeTree/experiments/_stage7.sh

waitfor() { while [ ! -f "$D/$1" ]; do sleep 30; done; }

laneA() {   # standard: topology profile, then the cross-comparison
  waitfor alpha_simple.json
  "$S" std topoprofile
  waitfor fnpe_ladder.json
  "$S" std cross
}

laneB() {   # mu0 = 7.5: the mandatory multimodal setting, end to end
  waitfor alpha_simple_mu7.5.json
  "$S" 7.5 topoprofile fnpe_step fnpe_ladder cross
}

laneA & laneB & wait
