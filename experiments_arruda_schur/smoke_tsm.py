"""Smoke test of the TSM arm: 1 training epoch (pp.py train --tsm), then one 4-subject dataset
from the final-epoch weights. Writes to the variant ``smoke_tsm`` only."""
import sys

import pp

pp.C2["epochs"] = 1
ENC = "artifacts/pp/budget/encoders/setenc-d16-w64x2s4-wide1.5-n65536.pt"
COMMON = ["--mode", "hierarchical", "--precondition", "Rg", "--train-prior-scale", "1.5",
          "--encoder", ENC, "--variant", "smoke_tsm"]
step = sys.argv[1]
if step == "train":
    sys.argv = ["pp.py", "train", *COMMON, "--train-subset", "65536", "--tsm", "--train-seed", "1"]
else:
    sys.argv = ["pp.py", "infer", *COMMON, "--method", "gauss_jacobian_FD", "--subjects", "4",
                "--datasets", "1", "--draws", "25", "--timesteps", "50", "--tag", "smoke", "--checkpoint", "final"]
pp.main()
