from runtime import configure_runtime
CPU_LIMIT = configure_runtime()
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Partial_Pooling.cli import run_stage
if __name__ == "__main__":
    run_stage("create_training_data", CPU_LIMIT)
