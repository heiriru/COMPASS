"""Entry point: the official ``scoresbi`` Hydra script with COMPASS registered.

Run inside the ``.simformer`` env from ``benchmarks/SimFormer`` (results land in
``./results/<name>/``). Any ``scoresbi`` override works, plus ``method=compass_*``
from ``bridge/config/method``.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Their Hydra config_path resolves to a top-level config package.
sys.path.insert(0, str(HERE.parent / "simformer_official" / "src" / "scoresbibm"))

import compass_method  # noqa: E402,F401  (registers the compass method)
from scoresbibm.scripts.hydra_script import score_sbi  # noqa: E402

if __name__ == "__main__":
    sys.argv.append(f"hydra.searchpath=[file://{HERE / 'config'}]")
    score_sbi()
