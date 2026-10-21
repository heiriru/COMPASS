"""C2ST (harness metric: sbi c2st, random forest) of saved samples vs sbibm references."""
import os, sys; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from sbi.analysis.sbc import c2st
refs = np.load(f"diagnostics/out/{sys.argv[1]}_refs.npz")["ref"]
for f in sys.argv[2:]:
    s = np.load(f)
    v = [float(c2st(torch.tensor(r), torch.tensor(q), classifier="rf")) for r, q in zip(refs, s)]
    print(f"{os.path.basename(f):32s} C2ST per obs {np.round(v, 3)}  mean {np.mean(v):.3f}")
