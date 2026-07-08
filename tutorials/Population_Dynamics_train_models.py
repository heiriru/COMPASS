#!/usr/bin/env python
# coding: utf-8
"""Train/load Population Dynamics COMPASS checkpoints from saved normalized data."""
from __future__ import annotations

from autocvd import autocvd

from Population_Dynamics import MODEL_DIR, train_models_from_saved_data


def main() -> None:
    autocvd(num_gpus=1, interval=1)
    train_models_from_saved_data()
    print(f"Ready checkpoints under: {MODEL_DIR}")


if __name__ == "__main__":
    main()
